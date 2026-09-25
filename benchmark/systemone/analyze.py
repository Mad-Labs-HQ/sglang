"""Score every calibration strategy on the reads collected by collect.py.

Strategies are computed offline with the functions the server uses, from one
collection: which reads to combine (choice rotations, noul orders), whether
label variants count, and what is applied after (running batch prior,
content-free prior, fitted temperature or Platt scaling). Metrics are on the
``test`` half of each task; fitted parameters and batch priors see the ``fit``
half first, as a server would see traffic before labeled rows.

Label-free components are then chosen by the rule fixed before the run: one is
kept when, added to those already kept, it lowers the mean test ECE over the
tasks it applies to and costs no task more than 1 point of accuracy.

    python benchmark/systemone/analyze.py --run benchmark/systemone/runs/k2-horizon-3.7b \\
        --out benchmark/systemone/results/k2-horizon-3.7b-rtx5070ti
"""

import argparse
import json
import math
import os
from typing import Dict, List, NamedTuple, Optional

from sglang.srt.entrypoints.systemone.calibration import (
    BatchPriorOn,
    BatchPriors,
    apply_params,
    combine_reads,
    log_normalize,
    read_log_probabilities,
    rotation_offsets,
)
from sglang.srt.entrypoints.systemone.calibration_fit import (
    auroc,
    brier,
    cross_validated_nll,
    decidable_share,
    expected_calibration_error,
    fit_platt,
    fit_temperature,
    nll,
    quadratic_weighted_kappa,
    ranked_probability_score,
    selective_accuracy,
    top_label,
)

TASK_ORDER = [
    "boolq",
    "sst2",
    "ag_news_sports",
    "ag_news",
    "emotion",
    "banking77",
    "sst5",
    "yelp",
]
MAX_ACCURACY_LOSS = 0.01
BATCH_PRIOR_MIN_COUNT = 8
BATCH_PRIOR_MAX_KEYS = 4096


class Reads(NamedTuple):
    """Which collected reads a strategy combines."""

    # Choice rotations, or noul orders; 1 is the /v1/decisions read.
    n: int
    # Count case variants (noul) or option-name variants (choice).
    variants: bool


class Strategy(NamedTuple):
    reads: Reads
    # Exponent of the running batch prior divided out, 0 for none.
    batch_prior: float
    # Exponent of the content-free prior divided out, 0 for none.
    content_free: float
    # Fit a temperature (choice, score) or Platt scaling (noul) on the fit half.
    fitted: bool

    def name(self) -> str:
        parts = [f"reads={self.reads.n}"]
        if self.reads.variants:
            parts.append("variants")
        if self.batch_prior:
            parts.append(f"batch_prior={self.batch_prior}")
        if self.content_free:
            parts.append(f"content_free={self.content_free}")
        if self.fitted:
            parts.append("fitted")
        return " ".join(parts)


RAW = Strategy(Reads(1, False), 0.0, 0.0, False)


def load(run_dir: str, task: str):
    with open(os.path.join(run_dir, f"{task}.jsonl")) as f:
        rows = [json.loads(line) for line in f]
    with open(os.path.join(run_dir, f"{task}.content_free.jsonl")) as f:
        content_free = {item["key"]: item["answer"] for item in map(json.loads, f)}
    return rows, content_free


def question_key(question) -> str:
    return json.dumps(question, sort_keys=True, ensure_ascii=False)


def option_names(row) -> List[str]:
    return row["answer"]["x_reads"][0]["order"]


def selected_reads(answer, kind: str, reads: Reads) -> List[dict]:
    all_reads = answer["x_reads"]
    if kind == "choice":
        names = all_reads[0]["order"]
        offsets = set(rotation_offsets(len(names), reads.n))
        return [r for r in all_reads if names.index(r["order"][0]) in offsets]
    return all_reads[: reads.n]


def log_q(answer, kind: str, reads: Reads) -> List[float]:
    """Combined option log probabilities of an answer's selected reads."""
    names = answer["x_reads"][0]["order"]
    per_read = []
    for read in selected_reads(answer, kind, reads):
        logprobs = read["logprobs"]
        per_read.append(
            read_log_probabilities(
                [
                    logprobs[name] if reads.variants else logprobs[name][:1]
                    for name in names
                ]
            )
        )
    return combine_reads(per_read)


def strategy_rows(rows, content_free, kind: str, strategy: Strategy):
    """Test-half log probabilities and gold indices under a strategy."""
    values = [log_q(row["answer"], kind, strategy.reads) for row in rows]
    if strategy.content_free:
        values = [
            log_normalize(
                [
                    v - strategy.content_free * c
                    for v, c in zip(
                        value,
                        log_q(
                            content_free[question_key(row["question"])],
                            kind,
                            strategy.reads,
                        ),
                    )
                ]
            )
            for value, row in zip(values, rows)
        ]
    if strategy.batch_prior:
        priors = BatchPriors(
            BatchPriorOn(
                strength=strategy.batch_prior,
                min_count=BATCH_PRIOR_MIN_COUNT,
                max_keys=BATCH_PRIOR_MAX_KEYS,
            )
        )
        # Rows arrive fit half first, then test half, as collected.
        values = [
            priors.apply(question_key(row["question"]), value)
            for value, row in zip(values, rows)
        ]
    gold = [row["gold"] for row in rows]
    fit = [i for i, row in enumerate(rows) if row["split"] == "fit"]
    test = [i for i, row in enumerate(rows) if row["split"] == "test"]
    fit_note = None
    if strategy.fitted:
        fitter = fit_platt if kind == "noul" else fit_temperature
        fit_q, fit_gold = [values[i] for i in fit], [gold[i] for i in fit]
        before, after = cross_validated_nll(fit_q, fit_gold, fitter, folds=5)
        params = fitter(fit_q, fit_gold)
        # The fit tool keeps a profile only when it helps out of fold.
        if after < before:
            values = [apply_params(params, value) for value in values]
            fit_note = params
        else:
            fit_note = "kept identity"
    return [values[i] for i in test], [gold[i] for i in test], fit_note


def metrics(kind: str, values, gold) -> Dict[str, float]:
    confidence, correct = top_label(values, gold)
    out = {
        "accuracy": sum(correct) / len(correct),
        "nll": nll(values, gold),
        "brier": brier(values, gold),
        "selective_acc_50": selective_accuracy(confidence, correct, 0.5),
        "decidable_at_5pct_error": decidable_share(confidence, correct, 0.05),
    }
    if kind == "noul":
        p_yes = [math.exp(v[0]) for v in values]
        is_yes = [g == 0 for g in gold]
        out["ece"] = expected_calibration_error(p_yes, is_yes)
        out["auroc"] = auroc(p_yes, is_yes)
    else:
        out["ece"] = expected_calibration_error(confidence, correct)
    if kind == "score":
        expected = [
            math.fsum(i * math.exp(v) for i, v in enumerate(row)) for row in values
        ]
        out["mae_expected_score"] = math.fsum(
            abs(e - g) for e, g in zip(expected, gold)
        ) / len(gold)
        out["rps"] = ranked_probability_score(values, gold)
        pred = [max(range(len(v)), key=v.__getitem__) for v in values]
        out["qwk"] = quadratic_weighted_kappa(pred, gold, len(values[0]))
    return out


def read_stats(rows, kind: str) -> Dict[str, float]:
    """Label mass of the /v1/decisions read, and how often reads disagree."""
    mass = [row["answer"]["x_label_mass"] for row in rows]
    flips = 0
    for row in rows:
        names = row["answer"]["x_reads"][0]["order"]
        tops = set()
        for read in row["answer"]["x_reads"]:
            lps = read["logprobs"]
            tops.add(max(names, key=lambda n: lps[n][0]))
        flips += len(tops) > 1
    raw_mass = []
    for row in rows:
        lps = row["answer"]["x_reads"][0]["logprobs"]
        raw_mass.append(math.fsum(math.exp(v[0]) for v in lps.values()))
    return {
        "label_mass_labels_only": sum(raw_mass) / len(raw_mass),
        "label_mass_with_variants": sum(mass) / len(mass),
        "read_disagreement": flips / len(rows),
        "reads_collected": len(rows[0]["answer"]["x_reads"]),
    }


def evaluate(data, task: str, strategy: Strategy):
    rows, content_free, kind = data[task]
    values, gold, fit_note = strategy_rows(rows, content_free, kind, strategy)
    out = metrics(kind, values, gold)
    if fit_note is not None:
        out["fit"] = str(fit_note)
    return out


def strategy_for(kind: str, config: Dict[str, object]) -> Strategy:
    """The unfitted strategy a label-free config gives questions of a kind."""
    n = {"noul": config["noul_orders"], "choice": config["choice_rotations"]}.get(
        kind, 1
    )
    variants = {
        "noul": config["noul_case_variants"],
        "choice": config["choice_name_variants"],
    }.get(kind, False)
    return Strategy(
        Reads(int(n), bool(variants)),
        float(config["batch_prior"]),
        float(config["content_free"]),
        False,
    )


def choose_label_free(data, tasks):
    """Greedy selection in a fixed order, under the rule in the module docstring."""
    chosen: Dict[str, object] = {
        "noul_case_variants": False,
        "choice_name_variants": False,
        "noul_orders": 1,
        "choice_rotations": 1,
        "batch_prior": 0.0,
        "content_free": 0.0,
    }
    log = []

    def trial(field: str, value, kinds) -> Optional[dict]:
        candidate = dict(chosen, **{field: value})
        deltas = []
        for task in tasks:
            kind = data[task][2]
            if kind not in kinds:
                continue
            before = evaluate(data, task, strategy_for(kind, chosen))
            after = evaluate(data, task, strategy_for(kind, candidate))
            deltas.append(
                {
                    "task": task,
                    "ece": after["ece"] - before["ece"],
                    "accuracy": after["accuracy"] - before["accuracy"],
                }
            )
        mean_ece = sum(d["ece"] for d in deltas) / len(deltas)
        worst_accuracy = min(d["accuracy"] for d in deltas)
        keep = mean_ece < 0 and worst_accuracy >= -MAX_ACCURACY_LOSS
        log.append(
            {
                "component": f"{field}={value}",
                "mean_ece_delta": mean_ece,
                "worst_accuracy_delta": worst_accuracy,
                "kept": keep,
                "per_task": deltas,
            }
        )
        return candidate if keep else None

    for field, values, kinds in (
        ("noul_case_variants", [True], ("noul",)),
        ("noul_orders", [2], ("noul",)),
        ("choice_name_variants", [True], ("choice",)),
        ("choice_rotations", [2, 4, 8], ("choice",)),
        ("batch_prior", [0.5, 0.75, 1.0], ("noul", "choice", "score")),
        ("content_free", [0.5, 1.0], ("noul", "choice", "score")),
    ):
        best = None
        for value in values:
            candidate = trial(field, value, kinds)
            if candidate is not None:
                # Among passing values, keep the one with the lowest mean ECE.
                if best is None or log[-1]["mean_ece_delta"] < best[1]:
                    best = (candidate, log[-1]["mean_ece_delta"])
        if best is not None:
            chosen = best[0]
    return chosen, log


def label_free_config(chosen: Dict[str, object]) -> dict:
    """A server calibration config with the chosen label-free components."""
    if chosen["content_free"]:
        raise ValueError("the server has no content-free prior; implement it first")
    batch_prior = (
        {
            "type": "on",
            "strength": chosen["batch_prior"],
            "min_count": BATCH_PRIOR_MIN_COUNT,
            "max_keys": BATCH_PRIOR_MAX_KEYS,
        }
        if chosen["batch_prior"]
        else {"type": "off"}
    )
    return {
        "default_mode": "label_free",
        "label_free": {
            "choice_rotations": chosen["choice_rotations"],
            "choice_name_variants": chosen["choice_name_variants"],
            "noul_orders": chosen["noul_orders"],
            "noul_case_variants": chosen["noul_case_variants"],
            "batch_prior": batch_prior,
        },
        "fitted": None,
    }


def table(rows: List[List[str]]) -> str:
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    lines = [
        "| " + " | ".join(c.ljust(w) for c, w in zip(r, widths)) + " |" for r in rows
    ]
    lines.insert(1, "| " + " | ".join("-" * w for w in widths) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    data = {}
    for task in TASK_ORDER:
        if os.path.exists(os.path.join(args.run, f"{task}.jsonl")):
            rows, content_free = load(args.run, task)
            data[task] = (rows, content_free, rows[0]["kind"])
    tasks = list(data)

    chosen, selection_log = choose_label_free(data, tasks)

    report = {"selection": {"chosen": chosen, "log": selection_log}, "tasks": {}}
    summary = [
        [
            "task",
            "kind",
            "strategy",
            "acc",
            "ECE",
            "NLL",
            "Brier",
            "sel.acc@50%",
            "decidable@5%",
            "extra",
        ]
    ]
    for task in tasks:
        rows, _, kind = data[task]
        label_free = strategy_for(kind, chosen)
        strategies = {
            "raw": RAW,
            "raw+fitted": RAW._replace(fitted=True),
            "label_free": label_free,
            "label_free+fitted": label_free._replace(fitted=True),
        }
        # Every single component alone on top of raw, for reference.
        if kind in ("noul", "choice"):
            strategies[
                "case variants only" if kind == "noul" else "name variants only"
            ] = RAW._replace(reads=Reads(1, True))
            max_reads = 2 if kind == "noul" else 8
            strategies[f"reads={max_reads} only"] = RAW._replace(
                reads=Reads(max_reads, False)
            )
        strategies["batch_prior=1.0 only"] = RAW._replace(batch_prior=1.0)
        strategies["content_free=1.0 only"] = RAW._replace(content_free=1.0)
        results = {name: evaluate(data, task, s) for name, s in strategies.items()}
        report["tasks"][task] = {
            "kind": kind,
            "reads": read_stats(rows, kind),
            "label_free_strategy": label_free.name(),
            "results": results,
        }
        for name, r in results.items():
            extra = ""
            if kind == "noul":
                extra = f"AUROC {r['auroc']:.3f}"
            elif kind == "score":
                extra = f"MAE {r['mae_expected_score']:.3f} RPS {r['rps']:.3f} QWK {r['qwk']:.3f}"
            summary.append(
                [
                    task,
                    kind,
                    name,
                    f"{r['accuracy']:.3f}",
                    f"{r['ece']:.3f}",
                    f"{r['nll']:.3f}",
                    f"{r['brier']:.3f}",
                    f"{r['selective_acc_50']:.3f}",
                    f"{r['decidable_at_5pct_error']:.3f}",
                    extra,
                ]
            )

    stats = [
        ["task", "labels-only mass", "mass with variants", "reads disagree", "reads"]
    ]
    for task in tasks:
        s = report["tasks"][task]["reads"]
        stats.append(
            [
                task,
                f"{s['label_mass_labels_only']:.3f}",
                f"{s['label_mass_with_variants']:.3f}",
                f"{s['read_disagreement']:.3f}",
                str(s["reads_collected"]),
            ]
        )
    selection_table = [["component", "mean ECE delta", "worst acc delta", "kept"]]
    for entry in selection_log:
        selection_table.append(
            [
                entry["component"],
                f"{entry['mean_ece_delta']:+.4f}",
                f"{entry['worst_accuracy_delta']:+.4f}",
                "yes" if entry["kept"] else "no",
            ]
        )
    markdown = "\n\n".join(
        [
            "## Label-free component selection",
            table(selection_table),
            f"Chosen: `{json.dumps(chosen)}`",
            "## Reads",
            table(stats),
            "## Metrics on the test half",
            table(summary),
        ]
    )
    with open(os.path.join(args.out, "label_free.json"), "w") as f:
        json.dump(label_free_config(chosen), f, indent=2)
        f.write("\n")
    with open(os.path.join(args.out, "efficacy.md"), "w") as f:
        f.write(markdown + "\n")
    with open(os.path.join(args.out, "efficacy.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(markdown)


if __name__ == "__main__":
    main()
