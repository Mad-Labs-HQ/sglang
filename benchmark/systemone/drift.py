"""How much answers move between two runs of the same reads: batch and cache drift.

Compares the answers ``verify_server.py check`` saved against the same answers
recomputed from the reads ``collect.py`` saved, for the default reads of a
reads config. The two runs differ only in which requests shared a batch and
which prefixes came from the cache, so any difference is numerical drift.

    python benchmark/systemone/drift.py --run runs/k2-horizon-3.7b \\
        --served results/k2-horizon-3.7b-rtx5070ti/served_label_free.answers.jsonl \\
        --config ../../examples/runtime/systemone/reads.json \\
        --out results/k2-horizon-3.7b-rtx5070ti/drift.md
"""

import argparse
import json
import math
import os

from analyze import TASK_ORDER, load, log_q, strategy_for
from verify_server import served_log_q

from sglang.srt.entrypoints.systemone.calibration import decode_config


def rotations_of(reads) -> int:
    """The rotation count a reads config's default choice reads equal."""
    if reads.choice_max_orders == 1:
        return 1
    if reads.choice_orders != "rotations" or reads.choice_max_orders == "all":
        raise ValueError("the collection has only up to 8 rotations of each question")
    return reads.choice_max_orders


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", required=True)
    parser.add_argument("--served", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    with open(args.config, "rb") as f:
        label_free = decode_config(f.read()).default_reads
    chosen = {
        "noul_case_variants": label_free.noul_case_variants,
        "choice_name_variants": label_free.choice_name_variants,
        "noul_orders": label_free.noul_orders,
        # The collection read choice questions in rotations.
        "choice_rotations": rotations_of(label_free),
        "batch_prior": 0.0,
        "content_free": 0.0,
    }
    with open(args.served) as f:
        served = {item["id"]: item["answer"] for item in map(json.loads, f)}
    lines = [
        "| task | rows | mean abs diff | p99 abs diff | max abs diff | top answer flips |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for task in TASK_ORDER:
        if not os.path.exists(os.path.join(args.run, f"{task}.jsonl")):
            continue
        rows, _ = load(args.run, task)
        kind = rows[0]["kind"]
        strategy = strategy_for(kind, chosen)
        diffs, flips, n = [], 0, 0
        for row in rows:
            if row["id"] not in served:
                continue
            n += 1
            cold = log_q(row["answer"], kind, strategy.reads)
            warm = served_log_q(served[row["id"]])
            diffs.extend(abs(math.exp(a) - math.exp(b)) for a, b in zip(cold, warm))
            flips += max(range(len(cold)), key=cold.__getitem__) != max(
                range(len(warm)), key=warm.__getitem__
            )
        diffs.sort()
        lines.append(
            f"| {task} | {n} | {sum(diffs) / len(diffs):.4f} "
            f"| {diffs[int(0.99 * (len(diffs) - 1))]:.4f} | {diffs[-1]:.4f} | {flips / n:.3f} |"
        )
        print(lines[-1], flush=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
