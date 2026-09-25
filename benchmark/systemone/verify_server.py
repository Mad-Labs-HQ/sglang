"""Check what a server with a calibration config answers, end to end.

``export`` writes the fit half of every task as labeled rows for
``python -m sglang.srt.entrypoints.systemone.fit_calibration``. ``check`` asks
the server the test half in a calibration mode, recomputes every answer from
its returned reads with the server's calibration config, reports the largest
difference, and scores the answers as served.

    python benchmark/systemone/verify_server.py export --rows 600 --out labeled_fit.jsonl
    python benchmark/systemone/verify_server.py check --url http://127.0.0.1:30000 \\
        --config fitted.json --mode fitted --rows 600 --concurrency 8 --out served_fitted.md
"""

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor

import requests
from tasks import TASKS, load_task

from sglang.srt.entrypoints.systemone.calibration import (
    apply_params,
    combine_reads,
    decode_config,
    find_profile,
    read_log_probabilities,
)
from sglang.srt.entrypoints.systemone.calibration_fit import (
    expected_calibration_error,
    nll,
    top_label,
)
from sglang.srt.entrypoints.systemone.fit_calibration import (
    KINDS,
    question_signature_of,
)


def gold_label(row):
    question = row["question"]
    if question["type"] == "choice":
        return list(question["criteria"])[row["gold"]]
    if question["type"] == "score":
        return row["gold"]
    return row["gold"] == 0


def export(args):
    with open(args.out, "w") as f:
        for task in TASKS:
            for row in load_task(task, args.rows):
                if row["split"] != "fit":
                    continue
                f.write(
                    json.dumps(
                        {
                            "state": row["state"],
                            "questions": {"q": row["question"]},
                            "labels": {"q": gold_label(row)},
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


def served_log_q(answer):
    if answer["type"] == "noul":
        p = answer["noul"]
        return [math.log(max(p, 1e-300)), math.log(max(1 - p, 1e-300))]
    return [math.log(max(p, 1e-300)) for p in answer["probabilities"].values()]


def expected_log_q(config, mode, question, answer):
    """The answer's log probabilities recomputed from its reads."""
    names = answer["x_reads"][0]["order"]
    log_q = combine_reads(
        [
            read_log_probabilities([read["logprobs"][n] for n in names])
            for read in answer["x_reads"]
        ]
    )
    if mode == "fitted":
        profile = find_profile(
            config.fitted,
            KINDS[question["type"]],
            len(names),
            question_signature_of(question),
        )
        if profile is not None:
            log_q = apply_params(profile.params, log_q)
    return log_q


def check(args):
    with open(args.config, "rb") as f:
        config = decode_config(f.read())
    session = requests.Session()
    lines = [
        f"Served answers in mode `{args.mode}` on the test half.",
        "",
        "| task | acc | ECE | NLL | max abs diff vs reads |",
        "| --- | --- | --- | --- | --- |",
    ]
    worst = 0.0
    answers_out = open(args.out.rsplit(".", 1)[0] + ".answers.jsonl", "w")
    for task in TASKS:
        rows = [r for r in load_task(task, args.rows) if r["split"] == "test"]

        def ask(row):
            body = {
                "state": row["state"],
                "model": "verify",
                "questions": {"q": row["question"]},
                "x_calibration": args.mode,
                "x_return_reads": True,
            }
            response = session.post(f"{args.url}/v1/systemone", json=body, timeout=600)
            response.raise_for_status()
            return response.json()["answers"]["q"]

        with ThreadPoolExecutor(args.concurrency) as pool:
            answers = list(pool.map(ask, rows))
        for row, answer in zip(rows, answers):
            answers_out.write(json.dumps({"id": row["id"], "answer": answer}) + "\n")
        values = [served_log_q(a) for a in answers]
        diff = max(
            abs(math.exp(s) - math.exp(e))
            for row, a, v in zip(rows, answers, values)
            for s, e in zip(v, expected_log_q(config, args.mode, row["question"], a))
        )
        worst = max(worst, diff)
        gold = [r["gold"] for r in rows]
        confidence, correct = top_label(values, gold)
        if rows[0]["kind"] == "noul":
            ece = expected_calibration_error(
                [math.exp(v[0]) for v in values], [g == 0 for g in gold]
            )
        else:
            ece = expected_calibration_error(confidence, correct)
        applied = {a["x_calibration"] for a in answers}
        lines.append(
            f"| {task} | {sum(correct) / len(correct):.3f} | {ece:.3f} | {nll(values, gold):.3f} "
            f"| {diff:.2e} ({', '.join(sorted(applied))}) |"
        )
        print(lines[-1], flush=True)
    answers_out.close()
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    if worst > 1e-9:
        raise SystemExit(f"served answers differ from their reads by up to {worst}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("export")
    p.add_argument("--rows", type=int, required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("check")
    p.add_argument("--url", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--mode", required=True, choices=["label_free", "fitted"])
    p.add_argument("--rows", type=int, required=True)
    p.add_argument("--concurrency", type=int, required=True)
    p.add_argument("--out", required=True)
    args = parser.parse_args()
    {"export": export, "check": check}[args.command](args)


if __name__ == "__main__":
    main()
