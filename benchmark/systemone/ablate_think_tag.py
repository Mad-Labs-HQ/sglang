"""Compare K2-Horizon's empty reasoning blocks as the position answers are read from.

The decision routes answer after ``<ifm|think>\\n</ifm|think>``, the block the
chat template renders for a reply without reasoning. K2-Horizon also has
``think_fast`` and ``think_faster`` blocks for lower reasoning efforts. This
takes each test row's prompt from /v1/decisions, swaps the empty block, and
scores the same labels (with case variants for yes or no) through /v1/score.

    python benchmark/systemone/ablate_think_tag.py --url http://127.0.0.1:30000 \\
        --rows 600 --out benchmark/systemone/results/k2-horizon-3.7b-rtx5070ti
"""

import argparse
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor

import requests
from tasks import TASKS, load_task
from transformers import AutoTokenizer

from sglang.srt.entrypoints.systemone.calibration import read_log_probabilities
from sglang.srt.entrypoints.systemone.calibration_fit import (
    expected_calibration_error,
    nll,
    top_label,
)

BLOCKS = {
    "think": "<ifm|think>\n</ifm|think>",
    "think_fast": "<ifm|think_fast>\n</ifm|think_fast>",
    "think_faster": "<ifm|think_faster>\n</ifm|think_faster>",
}


def to_decision_question(question):
    """The /v1/decisions form of a System One question."""
    if question["type"] == "noul":
        criteria = question.get("criteria") or {}
        return {
            "id": "q",
            "type": "yes_no",
            "question": question["instructions"],
            **{
                k: v
                for k, v in (
                    ("yes", criteria.get("true")),
                    ("no", criteria.get("false")),
                )
                if v
            },
        }
    if question["type"] == "choice":
        return {
            "id": "q",
            "type": "choice",
            "question": question["instructions"],
            "options": [
                {"name": n} if d is None else {"name": n, "description": d}
                for n, d in question["criteria"].items()
            ],
        }
    return {
        "id": "q",
        "type": "score",
        "question": question["instructions"],
        "levels": question["criteria"],
    }


def score_row(session, url, tokenizer, row):
    body = {
        "input": row["state"],
        "questions": [to_decision_question(row["question"])],
        "return_prompt_token_ids": True,
    }
    answer = session.post(f"{url}/v1/decisions", json=body, timeout=600).json()[
        "answers"
    ]["q"]
    prompt = tokenizer.decode(answer["prompt_token_ids"])
    assert prompt.endswith(BLOCKS["think"]), prompt[-60:]
    groups = [[label] for label in answer["label_token_ids"]]
    if row["kind"] == "noul":
        # Case variants of yes and no, as the label-free reads count them.
        for group, word in zip(groups, ("yes", "no")):
            for variant in (word.capitalize(), word.upper()):
                ids = tokenizer.encode(variant, add_special_tokens=False)
                if len(ids) == 1:
                    group.append(ids[0])
    out = {}
    for name, block in BLOCKS.items():
        text = prompt[: -len(BLOCKS["think"])] + block
        ids = tokenizer.encode(text, add_special_tokens=False)
        flat = [t for g in groups for t in g]
        response = session.post(
            f"{url}/v1/score",
            json={
                "query": [],
                "items": [ids],
                "label_token_ids": [flat],
                "return_token_logprobs": True,
            },
            timeout=600,
        ).json()
        lps = response["token_logprobs"][0]
        per_option, k = [], 0
        for g in groups:
            per_option.append(lps[k : k + len(g)])
            k += len(g)
        out[name] = read_log_probabilities(per_option)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    args = parser.parse_args()
    info = requests.get(f"{args.url}/server_info", timeout=30).json()
    tokenizer = AutoTokenizer.from_pretrained(
        info["tokenizer_path"], revision=info["revision"]
    )
    session = requests.Session()
    report = {}
    lines = ["| task | block | acc | ECE | NLL |", "| --- | --- | --- | --- | --- |"]
    for task in args.tasks:
        rows = [r for r in load_task(task, args.rows) if r["split"] == "test"]
        with ThreadPoolExecutor(args.concurrency) as pool:
            scored = list(
                pool.map(lambda r: score_row(session, args.url, tokenizer, r), rows)
            )
        gold = [r["gold"] for r in rows]
        report[task] = {}
        for name in BLOCKS:
            values = [s[name] for s in scored]
            confidence, correct = top_label(values, gold)
            if rows[0]["kind"] == "noul":
                ece = expected_calibration_error(
                    [math.exp(v[0]) for v in values], [g == 0 for g in gold]
                )
            else:
                ece = expected_calibration_error(confidence, correct)
            result = {
                "accuracy": sum(correct) / len(correct),
                "ece": ece,
                "nll": nll(values, gold),
            }
            report[task][name] = result
            lines.append(
                f"| {task} | {name} | {result['accuracy']:.3f} | {ece:.3f} | {result['nll']:.3f} |"
            )
        print("\n".join(lines[-3:]), flush=True)
    with open(os.path.join(args.out, "think_tag_ablation.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(args.out, "think_tag_ablation.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
