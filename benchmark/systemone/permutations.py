"""Cyclic rotations versus other option orders for choice questions, at equal budgets.

A read shows a choice question's options in some order, labeled A, B, C by
position, and reads the label probabilities. The client reorders the options
itself and asks each order as a single raw read, so any set of orders can be
compared. Sets of k orders, identity first:

    rotations   the orders x_read_setup {"choice_orders": "rotations"} reads: all
                cyclic shifts for k = n, else the k most discordant
    williams    the same for the rows of a Williams square, whose n rows put each
                option once per position and, for even n, each ordered neighbor
                pair next to each other once
    random      random permutations

Each set's reads are combined by geometric mean, as the server combines them,
and scored on the test half against gold labels and against a reference: the
average over every order for 4 options, or over 36 other random orders.

    python benchmark/systemone/permutations.py --url http://127.0.0.1:30000 \\
        --tasks ag_news emotion banking77 --rows 600 --concurrency 4 \\
        --out results/k2-horizon-3.7b-rtx5070ti/permutations.md
"""

import argparse
import itertools
import json
import math
import random
from concurrent.futures import ThreadPoolExecutor

import requests
from tasks import load_task

from sglang.srt.entrypoints.systemone.calibration import choice_orders, combine_reads
from sglang.srt.entrypoints.systemone.calibration_fit import (
    expected_calibration_error,
    nll,
    top_label,
)

RAW_READS = {
    "choice_orders": "rotations",
    "choice_max_orders": 1,
    "choice_name_variants": False,
    "noul_orders": 1,
    "noul_case_variants": False,
}
BUDGETS = (2, 3, 4, 6, 8)
REFERENCE_RANDOM = 36


def random_orders(n, k, rng):
    orders = [tuple(range(n))]
    while len(orders) < k:
        order = list(range(n))
        rng.shuffle(order)
        orders.append(tuple(order))
    return orders


def row_plan(row, n):
    """Every order a row needs, and which of them each set uses."""
    rng = random.Random(f"arm:{row['id']}")
    sets = {}
    for k in BUDGETS:
        if k > n:
            continue
        sets[("rotations", k)] = list(choice_orders("rotations", n, k))
        sets[("williams", k)] = list(choice_orders("williams", n, k))
        sets[("random", k)] = random_orders(n, max(BUDGETS), rng)[:k]
    if n <= 4:
        sets[("reference", None)] = list(itertools.permutations(range(n)))
    elif n <= 8:
        reference_rng = random.Random(f"reference:{row['id']}")
        sets[("reference", None)] = random_orders(
            n, REFERENCE_RANDOM + 1, reference_rng
        )[1:]
    orders = sorted({order for chosen in sets.values() for order in chosen})
    return sets, orders


def ask_orders(session, url, row, orders):
    """Log probabilities of every option, in request order, for each order asked."""
    names = list(row["question"]["criteria"])
    questions = {}
    for i, order in enumerate(orders):
        question = dict(row["question"])
        question["criteria"] = {
            names[j]: row["question"]["criteria"][names[j]] for j in order
        }
        questions[f"o{i}"] = question
    body = {
        "state": row["state"],
        "model": "orders",
        "questions": questions,
        "x_read_setup": RAW_READS,
    }
    response = session.post(f"{url}/v1/systemone", json=body, timeout=1200)
    response.raise_for_status()
    answers = response.json()["answers"]
    return {
        order: [
            math.log(max(answers[f"o{i}"]["probabilities"][name], 1e-300))
            for name in names
        ]
        for i, order in enumerate(orders)
    }


def evaluate(task, rows, reads_by_row, sets_by_row):
    gold = [row["gold"] for row in rows]
    names = sorted(
        {key for sets in sets_by_row for key in sets}, key=lambda k: (k[0], k[1] or 0)
    )
    combined = {
        key: [
            combine_reads([reads[o] for o in sets[key]])
            for reads, sets in zip(reads_by_row, sets_by_row)
        ]
        for key in names
    }
    reference = combined.get(("reference", None))
    lines = []
    for key in names:
        if key[0] == "reference":
            continue
        values = combined[key]
        confidence, correct = top_label(values, gold)
        line = {
            "task": task,
            "set": key[0],
            "k": key[1],
            "accuracy": sum(correct) / len(correct),
            "nll": nll(values, gold),
            "ece": expected_calibration_error(confidence, correct),
        }
        if reference is not None:
            line["diff_from_reference"] = sum(
                abs(math.exp(a) - math.exp(b))
                for v, r in zip(values, reference)
                for a, b in zip(v, r)
            ) / sum(len(v) for v in values)
            line["top_agrees_with_reference"] = sum(
                max(range(len(v)), key=v.__getitem__)
                == max(range(len(r)), key=r.__getitem__)
                for v, r in zip(values, reference)
            ) / len(values)
        lines.append(line)
    if reference is not None:
        confidence, correct = top_label(reference, gold)
        lines.append(
            {
                "task": task,
                "set": "reference",
                "k": len(sets_by_row[0][("reference", None)]),
                "accuracy": sum(correct) / len(correct),
                "nll": nll(reference, gold),
                "ece": expected_calibration_error(confidence, correct),
            }
        )
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    session = requests.Session()
    results = []
    for task in args.tasks:
        rows = [r for r in load_task(task, args.rows) if r["split"] == "test"]
        n = len(rows[0]["question"]["criteria"])
        plans = [row_plan(row, n) for row in rows]
        with ThreadPoolExecutor(args.concurrency) as pool:
            reads_by_row = list(
                pool.map(
                    lambda rp: ask_orders(session, args.url, rp[0], rp[1][1]),
                    zip(rows, plans),
                )
            )
        task_lines = evaluate(task, rows, reads_by_row, [p[0] for p in plans])
        results.extend(task_lines)
        for line in task_lines:
            print(json.dumps(line), flush=True)
    header = "| task | orders | k | accuracy | NLL | ECE | mean abs diff from reference | top agrees with reference |"
    lines = [header, "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in results:
        lines.append(
            f"| {r['task']} | {r['set']} | {r['k']} | {r['accuracy']:.3f} | {r['nll']:.3f} | {r['ece']:.3f} "
            f"| {r.get('diff_from_reference', float('nan')):.4f} | {r.get('top_agrees_with_reference', float('nan')):.3f} |"
        )
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
