"""Markdown tables comparing server variants from bench_speed.py output.

    python benchmark/systemone/speed_report.py \\
        --speed results/k2-horizon-3.7b-rtx5070ti/speed.jsonl \\
        --out results/k2-horizon-3.7b-rtx5070ti/speed.md
"""

import argparse
import json
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--speed", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    with open(args.speed) as f:
        rows = [json.loads(line) for line in f]
    labels = list(dict.fromkeys(r["label"] for r in rows))
    points = defaultdict(dict)
    for r in rows:
        key = (r["concurrency"], r["mode"], r["state_tokens"], r["questions"])
        points[key][r["label"]] = r

    sections = []
    for concurrency in sorted({k[0] for k in points}):
        for mode in ("raw", "label_free"):
            keys = sorted(k for k in points if k[0] == concurrency and k[1] == mode)
            if not keys:
                continue
            header = ["state tokens", "questions"]
            for label in labels:
                header += [
                    f"{label} p50 s",
                    f"{label} q/s",
                    f"{label} prefilled / request",
                ]
            lines = [
                f"### {mode}, concurrency {concurrency}",
                "",
                "| " + " | ".join(header) + " |",
                "| " + " | ".join("---" for _ in header) + " |",
            ]
            for key in keys:
                cells = [str(key[2]), str(key[3])]
                for label in labels:
                    r = points[key].get(label)
                    if r is None:
                        cells += ["", "", ""]
                        continue
                    cells += [
                        f"{r['p50_s']:.3f}",
                        f"{r['questions_per_s']:.1f}",
                        f"{r['prefill_computed_tokens'] / r['requests']:.0f}",
                    ]
                lines.append("| " + " | ".join(cells) + " |")
            sections.append("\n".join(lines))
    with open(args.out, "w") as f:
        f.write("\n\n".join(sections) + "\n")
    print("\n\n".join(sections))


if __name__ == "__main__":
    main()
