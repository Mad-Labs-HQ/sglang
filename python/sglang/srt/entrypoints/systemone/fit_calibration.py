"""Fit per-question calibrations for /v1/systemone from labeled rows, as a client.

Each input line is a System One request body with the gold answers added:

    {"state": ..., "questions": {"team": {...}}, "labels": {"team": "billing"}}

A label is the option name of a choice, the level index of a score, and true or
false for a noul. A question id names a question: every row that uses an id
must ask the same question, and one calibration is fitted per id.

The tool asks the server each row with the read setup in --read-setup (the
server's default reads when absent), and takes each answer's
x_read_probabilities, its probabilities before any calibration. For every
question id with at least --min-rows labeled answers and two different gold
answers it fits the candidates for its type (temperature and Platt scaling for
noul, temperature and vector scaling for choice and score) and keeps the one
with the lowest out-of-fold NLL, unless leaving the answers as they are does
better. The output maps question ids to calibrations, each stamped with the
x_fingerprint of the answers it was fitted on:

    {"read_setup": {...} or null,
     "calibrations": {"team": {"type": "vector", "scale": [...], "bias": [...],
                               "fitted_on": "..."}},
     "report": {"team": {"rows": 300, "out_of_fold_nll": {...}}}}

Send a calibration as its question's x_calibration, with the same x_read_setup.
The server refuses it once the model or the reads it was fitted on change.

    python -m sglang.srt.entrypoints.systemone.fit_calibration \\
        --url http://127.0.0.1:30000 --data labeled.jsonl --read-setup reads.json \\
        --min-rows 50 --folds 5 --concurrency 8 --out calibrations.json
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import msgspec
import requests

from sglang.srt.entrypoints.systemone.calibration import ReadSetup
from sglang.srt.entrypoints.systemone.calibration_fit import choose_calibration

# System One question types, as /v1/decisions names them.
KINDS = {"noul": "yes_no", "choice": "choice", "score": "score"}


def gold_index(question: Dict[str, Any], label: Any) -> int:
    if question["type"] == "choice":
        return list(question["criteria"]).index(label)
    if question["type"] == "score":
        return int(label)
    if not isinstance(label, bool):
        raise ValueError(f"a noul label must be true or false, not {label!r}")
    return 0 if label else 1


def uncalibrated(questions: Dict[str, Any]) -> Dict[str, Any]:
    """The questions without calibrations, so none is checked or applied."""
    return {
        qid: {k: v for k, v in question.items() if k != "x_calibration"}
        for qid, question in questions.items()
    }


def ask(
    session: requests.Session,
    url: str,
    row: Dict[str, Any],
    read_setup: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    body = {
        "state": row["state"],
        "model": "fit",
        "questions": uncalibrated(row["questions"]),
    }
    if read_setup is not None:
        body["x_read_setup"] = read_setup
    response = session.post(f"{url}/v1/systemone", json=body, timeout=600)
    if response.status_code != 200:
        raise RuntimeError(f"{response.status_code}: {response.text}")
    return response.json()["answers"]


def collect(
    url: str,
    rows: List[Dict[str, Any]],
    read_setup: Optional[Dict[str, Any]],
    concurrency: int,
) -> Dict[str, Dict[str, Any]]:
    """Per question id: its kind, fingerprint, and (log probabilities, gold) rows."""
    session = requests.Session()
    with ThreadPoolExecutor(concurrency) as pool:
        answers = list(pool.map(lambda row: ask(session, url, row, read_setup), rows))
    groups: Dict[str, Dict[str, Any]] = {}
    for row, row_answers in zip(rows, answers):
        for qid, label in row.get("labels", {}).items():
            question = row["questions"][qid]
            answer = row_answers[qid]
            group = groups.setdefault(
                qid,
                {
                    "kind": KINDS[question["type"]],
                    "fingerprint": answer["x_fingerprint"],
                    "log_q": [],
                    "gold": [],
                },
            )
            if answer["x_fingerprint"] != group["fingerprint"] or (
                KINDS[question["type"]] != group["kind"]
            ):
                raise ValueError(
                    f"question id {qid!r} asks different kinds of question or "
                    "gets answers with different fingerprints across rows"
                )
            group["log_q"].append(
                [
                    math.log(max(p, 1e-300))
                    for p in answer["x_read_probabilities"].values()
                ]
            )
            group["gold"].append(gold_index(question, label))
    return groups


def fit(groups: Dict[str, Dict[str, Any]], min_rows: int, folds: int) -> Dict[str, Any]:
    calibrations, report = {}, {}
    for qid, group in groups.items():
        rows = len(group["gold"])
        if rows < min_rows or len(set(group["gold"])) < 2:
            report[qid] = {"rows": rows, "skipped": "too few rows or one gold answer"}
            continue
        calibration, nll_by_candidate = choose_calibration(
            group["kind"], group["log_q"], group["gold"], folds
        )
        report[qid] = {"rows": rows, "out_of_fold_nll": nll_by_candidate}
        if calibration is not None:
            calibrations[qid] = {**calibration, "fitted_on": group["fingerprint"]}
    return {"calibrations": calibrations, "report": report}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", required=True)
    parser.add_argument("--data", required=True, help="labeled rows, one JSON per line")
    parser.add_argument(
        "--read-setup",
        help="JSON file with the x_read_setup to ask with; the server's default reads when absent",
    )
    parser.add_argument("--min-rows", type=int, required=True)
    parser.add_argument("--folds", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    read_setup = None
    if args.read_setup is not None:
        with open(args.read_setup, "rb") as f:
            read_setup = msgspec.to_builtins(
                msgspec.json.decode(f.read(), type=ReadSetup)
            )
    with open(args.data) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    groups = collect(args.url.rstrip("/"), rows, read_setup, args.concurrency)
    out = {"read_setup": read_setup, **fit(groups, args.min_rows, args.folds)}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    for qid, entry in out["report"].items():
        chosen = out["calibrations"].get(qid, {}).get("type", "none")
        nlls = entry.get("out_of_fold_nll")
        detail = (
            ", ".join(f"{k} {v:.4f}" for k, v in nlls.items())
            if nlls
            else entry["skipped"]
        )
        print(f"{qid}: {chosen} ({entry['rows']} rows; out-of-fold NLL {detail})")
    print(f"{len(out['calibrations'])} calibrations -> {args.out}")


if __name__ == "__main__":
    main()
