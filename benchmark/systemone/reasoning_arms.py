"""Think, then read the answer labels: the System 2 reference for System One reads.

For each test row the model first reasons from the /v1/decisions prompt with its
reasoning block open (``<ifm|think>``, ``<ifm|think_fast>``, or
``<ifm|think_faster>`` for high, medium, or low effort), up to a token budget, sampled as the model card recommends. The block
is then closed and the same labels are read through /v1/score, so the answers
carry probabilities comparable with the single-pass read on the same rows.
Reports accuracy, calibration, reasoning tokens, and seconds per question.

    python benchmark/systemone/reasoning_arms.py --url http://127.0.0.1:30000 \\
        --effort low --budget 1024 --rows-per-task 100 --concurrency 8 \\
        --out benchmark/systemone/results/k2-horizon-3.7b-rtx5070ti
"""

import argparse
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from ablate_think_tag import BLOCKS, to_decision_question
from tasks import TASKS, load_task
from transformers import AutoTokenizer

from sglang.srt.entrypoints.systemone.calibration import read_log_probabilities
from sglang.srt.entrypoints.systemone.calibration_fit import (
    expected_calibration_error,
    nll,
    top_label,
)

OPEN = {
    "high": "<ifm|think>\n",
    "medium": "<ifm|think_fast>\n",
    "low": "<ifm|think_faster>\n",
}
CLOSE = {
    "high": "</ifm|think>",
    "medium": "</ifm|think_fast>",
    "low": "</ifm|think_faster>",
}
# Rows per task are the first test rows, and /v1/decisions labels at most 26 options.
TASKS_WITH_DECISIONS = [t for t in TASKS if t != "banking77"]


def label_groups(tokenizer, kind, label_ids):
    groups = [[label] for label in label_ids]
    if kind == "noul":
        for group, word in zip(groups, ("yes", "no")):
            for variant in (word.capitalize(), word.upper()):
                ids = tokenizer.encode(variant, add_special_tokens=False)
                if len(ids) == 1:
                    group.append(ids[0])
    return groups


def read_labels(session, url, ids, groups):
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
    lps, per_option, k = response["token_logprobs"][0], [], 0
    for g in groups:
        per_option.append(lps[k : k + len(g)])
        k += len(g)
    return read_log_probabilities(per_option)


def run_row(session, url, tokenizer, row, effort, budget):
    body = {
        "input": row["state"],
        "questions": [to_decision_question(row["question"])],
        "return_prompt_token_ids": True,
    }
    answer = session.post(f"{url}/v1/decisions", json=body, timeout=600).json()[
        "answers"
    ]["q"]
    prompt = tokenizer.decode(answer["prompt_token_ids"])
    groups = label_groups(tokenizer, row["kind"], answer["label_token_ids"])
    system_one = read_labels(session, url, answer["prompt_token_ids"], groups)

    opened = prompt[: -len(BLOCKS["think"])] + OPEN[effort]
    started = time.perf_counter()
    generated = session.post(
        f"{url}/generate",
        json={
            "input_ids": tokenizer.encode(opened, add_special_tokens=False),
            "sampling_params": {
                "max_new_tokens": budget,
                "temperature": 1.0,
                "top_p": 0.95,
                "stop": [CLOSE[effort]],
            },
        },
        timeout=3600,
    ).json()
    reasoning = generated["text"]
    meta = generated["meta_info"]
    truncated = meta["finish_reason"]["type"] == "length"
    closed = opened + reasoning.split(CLOSE[effort])[0] + CLOSE[effort]
    system_two = read_labels(
        session, url, tokenizer.encode(closed, add_special_tokens=False), groups
    )
    seconds = time.perf_counter() - started
    return {
        "system_one": system_one,
        "system_two": system_two,
        "reasoning_tokens": meta["completion_tokens"],
        "truncated": truncated,
        "seconds": seconds,
    }


def summarize(kind, values, gold):
    confidence, correct = top_label(values, gold)
    if kind == "noul":
        ece = expected_calibration_error(
            [math.exp(v[0]) for v in values], [g == 0 for g in gold]
        )
    else:
        ece = expected_calibration_error(confidence, correct)
    return {
        "accuracy": sum(correct) / len(correct),
        "ece": ece,
        "nll": nll(values, gold),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", required=True)
    parser.add_argument("--effort", required=True, choices=list(OPEN))
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--rows-per-task", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    info = requests.get(f"{args.url}/server_info", timeout=30).json()
    tokenizer = AutoTokenizer.from_pretrained(
        info["tokenizer_path"], revision=info["revision"]
    )
    session = requests.Session()
    report = {}
    lines = [
        f"Reasoning effort {args.effort}, budget {args.budget} tokens, {args.rows_per_task} test rows per task.",
        "",
        "| task | one-pass acc | think-then-read acc | one-pass ECE | think-then-read ECE | one-pass NLL | think-then-read NLL | reasoning tokens (mean) | truncated | s/question |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for task in TASKS_WITH_DECISIONS:
        rows = [r for r in load_task(task, 600) if r["split"] == "test"][
            : args.rows_per_task
        ]
        started = time.perf_counter()
        with ThreadPoolExecutor(args.concurrency) as pool:
            results = list(
                pool.map(
                    lambda r: run_row(
                        session, args.url, tokenizer, r, args.effort, args.budget
                    ),
                    rows,
                )
            )
        wall = time.perf_counter() - started
        gold = [r["gold"] for r in rows]
        kind = rows[0]["kind"]
        one = summarize(kind, [r["system_one"] for r in results], gold)
        two = summarize(kind, [r["system_two"] for r in results], gold)
        tokens = sum(r["reasoning_tokens"] for r in results) / len(results)
        truncated = sum(r["truncated"] for r in results) / len(results)
        report[task] = {
            "one_pass": one,
            "think_then_read": two,
            "reasoning_tokens_mean": tokens,
            "truncated": truncated,
            "wall_seconds": wall,
            "seconds_per_question_mean": sum(r["seconds"] for r in results)
            / len(results),
        }
        lines.append(
            f"| {task} | {one['accuracy']:.3f} | {two['accuracy']:.3f} | {one['ece']:.3f} | {two['ece']:.3f} "
            f"| {one['nll']:.3f} | {two['nll']:.3f} | {tokens:.0f} | {truncated:.2f} "
            f"| {report[task]['seconds_per_question_mean']:.2f} |"
        )
        print(lines[-1], flush=True)
    name = f"reasoning_{args.effort}"
    with open(os.path.join(args.out, f"{name}.json"), "w") as f:
        json.dump(
            {"effort": args.effort, "budget": args.budget, "tasks": report}, f, indent=2
        )
    with open(os.path.join(args.out, f"{name}.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
