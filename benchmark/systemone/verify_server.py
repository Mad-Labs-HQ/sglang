"""Check what a server answers with client calibrations, end to end.

``export`` writes the fit half of every task as labeled rows for
``python -m sglang.srt.entrypoints.systemone.fit_calibration``, with the task
name as the question id. ``check`` asks the server the test half, attaching the
calibration fitted for each task if a calibrations file is given, recomputes
every answer from its returned reads and calibration, reports the largest
difference, and scores the answers as served.

    python benchmark/systemone/verify_server.py export --rows 600 --out labeled_fit.jsonl
    python benchmark/systemone/verify_server.py check --url http://127.0.0.1:30000 \\
        --calibrations calibrations.json --rows 600 --concurrency 8 --out served_calibrated.md
"""

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor

import requests
from tasks import TASKS, load_task

from sglang.srt.entrypoints.systemone.calibration import (
    apply_calibration,
    combine_reads,
    read_log_probabilities,
)
from sglang.srt.entrypoints.systemone.calibration_fit import (
    expected_calibration_error,
    nll,
    top_label,
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
                            "questions": {task: row["question"]},
                            "labels": {task: gold_label(row)},
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


def expected_log_q(answer, calibration):
    """The answer's log probabilities recomputed from its reads and calibration."""
    names = answer["x_reads"][0]["order"]
    log_q = combine_reads(
        [
            read_log_probabilities([read["logprobs"][n] for n in names])
            for read in answer["x_reads"]
        ]
    )
    if calibration is not None:
        log_q = apply_calibration(calibration, log_q)
    return log_q


def check(args):
    calibrations, read_setup = {}, None
    if args.calibrations:
        with open(args.calibrations) as f:
            fitted = json.load(f)
        calibrations, read_setup = fitted["calibrations"], fitted["read_setup"]
    session = requests.Session()
    lines = [
        "Served answers on the test half"
        + (f", calibrated by {args.calibrations}." if args.calibrations else "."),
        "",
        "| task | acc | ECE | NLL | calibration | max abs diff vs reads |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    worst = 0.0
    answers_out = open(args.out.rsplit(".", 1)[0] + ".answers.jsonl", "w")
    for task in TASKS:
        rows = [r for r in load_task(task, args.rows) if r["split"] == "test"]
        calibration = calibrations.get(task)

        def ask(row):
            question = dict(row["question"])
            if calibration is not None:
                question["x_calibration"] = calibration
            body = {
                "state": row["state"],
                "model": "verify",
                "questions": {task: question},
                "x_return_reads": True,
            }
            if read_setup is not None:
                body["x_read_setup"] = read_setup
            response = session.post(f"{args.url}/v1/systemone", json=body, timeout=600)
            if response.status_code != 200:
                raise RuntimeError(f"{response.status_code}: {response.text}")
            return response.json()["answers"][task]

        with ThreadPoolExecutor(args.concurrency) as pool:
            answers = list(pool.map(ask, rows))
        for row, answer in zip(rows, answers):
            answers_out.write(json.dumps({"id": row["id"], "answer": answer}) + "\n")
        values = [served_log_q(a) for a in answers]
        diff = max(
            abs(math.exp(s) - math.exp(e))
            for a, v in zip(answers, values)
            for s, e in zip(v, expected_log_q(a, calibration))
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
        applied = ", ".join(sorted({a["x_calibration"] for a in answers}))
        lines.append(
            f"| {task} | {sum(correct) / len(correct):.3f} | {ece:.3f} "
            f"| {nll(values, gold):.3f} | {applied} | {diff:.2e} |"
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
    p.add_argument(
        "--calibrations",
        help="fit tool output; the server's default reads, uncalibrated, when absent",
    )
    p.add_argument("--rows", type=int, required=True)
    p.add_argument("--concurrency", type=int, required=True)
    p.add_argument("--out", required=True)
    args = parser.parse_args()
    {"export": export, "check": check}[args.command](args)


if __name__ == "__main__":
    main()
