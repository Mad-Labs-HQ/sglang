"""Fit /v1/systemone calibration profiles on labeled rows, against a running server.

Each input line is a System One request body with the gold answers added:

    {"state": ..., "questions": {"q": {...}}, "labels": {"q": "billing"}}

A label is the option name of a choice, the level index of a score, and true or
false for a noul. The tool asks the server for the reads of the config's base
mode, fits a temperature per choice or score group and Platt scaling per noul
group, and writes the config with the fitted profiles added. A group is a
question signature, else the questions of one type and option count, else one
type; one is fitted only with --min-rows rows and kept only when its
out-of-fold NLL is lower than without it.

    python -m sglang.srt.entrypoints.systemone.fit_calibration \\
        --url http://127.0.0.1:30000 --config label_free.json --base label_free \\
        --data labeled.jsonl --min-rows 50 --folds 5 --out fitted.json
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List, Tuple

import requests

from sglang.srt.entrypoints.openai.serving_decisions import PROMPT_FORMAT_VERSION
from sglang.srt.entrypoints.systemone.calibration import (
    CalibrationConfig,
    FittedProfiles,
    Profile,
    combine_reads,
    decode_config,
    encode_config,
    question_signature,
    read_log_probabilities,
    reads_fingerprint,
)
from sglang.srt.entrypoints.systemone.calibration_fit import (
    cross_validated_nll,
    fit_platt,
    fit_temperature,
)

# System One question types, as /v1/decisions names them in signatures.
KINDS = {"noul": "yes_no", "choice": "choice", "score": "score"}


def _gold_index(question: dict, label) -> int:
    if question["type"] == "choice":
        return list(question["criteria"]).index(label)
    if question["type"] == "score":
        return int(label)
    return 0 if label is True else 1


def question_signature_of(question: dict) -> str:
    """The signature the server computes for this question's view."""
    kind = KINDS[question["type"]]
    if kind == "choice":
        names, details = list(question["criteria"]), list(question["criteria"].values())
    elif kind == "score":
        names = [str(i) for i in range(len(question["criteria"]))]
        details = list(question["criteria"])
    else:
        criteria = question.get("criteria") or {}
        names, details = ["yes", "no"], [criteria.get("true"), criteria.get("false")]
    return question_signature(kind, question.get("instructions"), names, details)


def _answer_log_q(answer: dict) -> List[float]:
    """Base-mode log probabilities of an answer, recomputed from its reads."""
    names = answer["x_reads"][0]["order"]
    return combine_reads(
        [
            read_log_probabilities([read["logprobs"][name] for name in names])
            for read in answer["x_reads"]
        ]
    )


def _collect(
    url: str, base: str, rows: List[dict]
) -> List[Tuple[str, str, int, List[float], int]]:
    """(signature, kind, options, log_q, gold) of every labeled question."""
    session = requests.Session()
    out = []
    for row in rows:
        body = {
            "state": row["state"],
            "model": "fit",
            "questions": row["questions"],
            "x_calibration": base,
            "x_return_reads": True,
        }
        response = session.post(f"{url}/v1/systemone", json=body, timeout=600)
        response.raise_for_status()
        answers = response.json()["answers"]
        for question_id, label in row["labels"].items():
            question = row["questions"][question_id]
            log_q = _answer_log_q(answers[question_id])
            out.append(
                (
                    question_signature_of(question),
                    KINDS[question["type"]],
                    len(log_q),
                    log_q,
                    _gold_index(question, label),
                )
            )
    return out


def _fit_groups(
    items: List[Tuple[str, str, int, List[float], int]], min_rows: int, folds: int
) -> List[Profile]:
    """Profiles for every group with enough rows whose fit helps out of fold."""
    groups: Dict[str, Tuple[str, List[Tuple[List[float], int]]]] = {}
    for signature, kind, options, log_q, gold in items:
        for key in (
            f"signature:{signature}",
            f"bucket:{kind}:{options}",
            f"type:{kind}",
        ):
            groups.setdefault(key, (kind, []))[1].append((log_q, gold))
    profiles = []
    for key, (kind, members) in groups.items():
        log_q = [m[0] for m in members]
        gold = [m[1] for m in members]
        # A single gold class gives nothing to fit.
        if len(members) < min_rows or len(set(gold)) < 2:
            continue
        fit = fit_platt if kind == "yes_no" else fit_temperature
        before, after = cross_validated_nll(log_q, gold, fit, folds=folds)
        if not after < before:
            continue
        profiles.append(
            Profile(
                key=key,
                params=fit(log_q, gold),
                rows=len(members),
                oof_nll_base=before,
                oof_nll_fitted=after,
            )
        )
    return profiles


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", required=True)
    parser.add_argument("--config", required=True, help="the config the server runs")
    parser.add_argument("--base", required=True, choices=["raw", "label_free"])
    parser.add_argument("--data", required=True)
    parser.add_argument("--min-rows", type=int, required=True)
    parser.add_argument("--folds", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.config, "rb") as f:
        config: CalibrationConfig = decode_config(f.read())
    with open(args.data) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    info = requests.get(f"{args.url}/server_info", timeout=30).json()
    items = _collect(args.url, args.base, rows)
    profiles = _fit_groups(items, args.min_rows, args.folds)
    fitted = FittedProfiles(
        model=info["model_path"],
        model_revision=info["revision"],
        prompt_format_version=PROMPT_FORMAT_VERSION,
        base=args.base,
        reads_fingerprint=reads_fingerprint(args.base, config.label_free),
        profiles=profiles,
    )
    out = CalibrationConfig(
        default_mode="fitted", label_free=config.label_free, fitted=fitted
    )
    with open(args.out, "wb") as f:
        f.write(encode_config(out))
    for profile in profiles:
        print(
            f"{profile.key}: {profile.params} rows={profile.rows} out-of-fold NLL "
            f"{profile.oof_nll_base:.4f} -> {profile.oof_nll_fitted:.4f}"
        )
    print(f"{len(items)} labeled answers, {len(profiles)} profiles -> {args.out}")


if __name__ == "__main__":
    main()
