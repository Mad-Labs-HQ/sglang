"""Fitting and scoring calibration of System One answers on labeled rows.

Every function takes per-answer log probabilities over options (``log_q``, one
row per answer) and gold option indices, as plain floats and ints. Fitted
calibrations are dicts in the wire format a request carries them in.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from sglang.srt.entrypoints.systemone.calibration import (
    LOG_FLOOR,
    apply_calibration,
    log_sigmoid,
)

Calibration = Dict[str, Any]
Fit = Callable[[Sequence[Sequence[float]], Sequence[int]], Calibration]

# Search range of fitted temperatures. Published fits on chat models span
# about 1 to 12, so this leaves room on both sides.
MIN_TEMPERATURE = 0.05
MAX_TEMPERATURE = 20.0
# L2 penalty on Platt parameters, only to keep separable data finite.
PLATT_L2 = 1e-3
# L2 penalty pulling vector scaling toward the identity, per mean-NLL unit.
# Arbitrary; enough to keep options that are rarely gold from running off.
VECTOR_L2 = 1e-2


def nll(log_q: Sequence[Sequence[float]], gold: Sequence[int]) -> float:
    return -math.fsum(row[g] for row, g in zip(log_q, gold)) / len(gold)


def fit_temperature(
    log_q: Sequence[Sequence[float]], gold: Sequence[int]
) -> Calibration:
    """Temperature minimizing the NLL, by golden-section search over log T."""

    def loss(log_t: float) -> float:
        calibration = {"type": "temperature", "temperature": math.exp(log_t)}
        return nll([apply_calibration(calibration, row) for row in log_q], gold)

    lo, hi = math.log(MIN_TEMPERATURE), math.log(MAX_TEMPERATURE)
    ratio = (math.sqrt(5) - 1) / 2
    a, b = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
    fa, fb = loss(a), loss(b)
    for _ in range(80):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - ratio * (hi - lo)
            fa = loss(a)
        else:
            lo, a, fa = a, b, fb
            b = lo + ratio * (hi - lo)
            fb = loss(b)
    return {"type": "temperature", "temperature": math.exp((lo + hi) / 2)}


def fit_platt(log_q: Sequence[Sequence[float]], gold: Sequence[int]) -> Calibration:
    """Logistic regression of the first option on its log odds, by damped Newton steps."""
    z = [row[0] - row[1] for row in log_q]
    y = [1.0 if g == 0 else 0.0 for g in gold]
    a, b = 1.0, 0.0
    loss = _platt_loss(z, y, a, b)
    for _ in range(100):
        grad_a, grad_b = PLATT_L2 * a, PLATT_L2 * b
        h_aa, h_ab, h_bb = PLATT_L2, 0.0, PLATT_L2
        for zi, yi in zip(z, y):
            p = math.exp(log_sigmoid(a * zi + b))
            grad_a += (p - yi) * zi
            grad_b += p - yi
            w = p * (1 - p)
            h_aa += w * zi * zi
            h_ab += w * zi
            h_bb += w
        det = h_aa * h_bb - h_ab * h_ab
        if det <= 0:
            break
        step_a = (h_bb * grad_a - h_ab * grad_b) / det
        step_b = (h_aa * grad_b - h_ab * grad_a) / det
        # Where predictions saturate the Hessian is tiny and a full step overshoots,
        # so halve it until the loss falls.
        scale = 1.0
        while True:
            new_a, new_b = a - scale * step_a, b - scale * step_b
            new_loss = _platt_loss(z, y, new_a, new_b)
            if new_loss <= loss or scale < 1e-8:
                break
            scale /= 2
        if new_loss > loss:
            break
        converged = abs(new_a - a) + abs(new_b - b) < 1e-10
        a, b, loss = new_a, new_b, new_loss
        if converged:
            break
    return {"type": "platt", "a": a, "b": b}


def _platt_loss(z: Sequence[float], y: Sequence[float], a: float, b: float) -> float:
    """Penalized negative log likelihood that fit_platt minimizes."""
    loss = PLATT_L2 * (a * a + b * b) / 2
    for zi, yi in zip(z, y):
        t = a * zi + b
        loss -= yi * log_sigmoid(t) + (1 - yi) * log_sigmoid(-t)
    return loss


def fit_vector(log_q: Sequence[Sequence[float]], gold: Sequence[int]) -> Calibration:
    """A scale and bias per option minimizing the mean NLL plus an L2 pull toward
    the identity, by gradient descent with backtracking (the loss is convex)."""
    x = np.maximum(np.asarray(log_q, dtype=np.float64), LOG_FLOOR)
    n, k = x.shape
    y = np.zeros((n, k))
    y[np.arange(n), np.asarray(gold)] = 1.0
    scale, bias = np.ones(k), np.zeros(k)

    def loss_and_grad(s, b):
        z = x * s + b
        z = z - z.max(axis=1, keepdims=True)
        log_p = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        p = np.exp(log_p)
        loss = -(log_p * y).sum() / n + VECTOR_L2 / 2 * (
            ((s - 1) ** 2).sum() + (b**2).sum()
        )
        grad_s = ((p - y) * x).sum(axis=0) / n + VECTOR_L2 * (s - 1)
        grad_b = (p - y).sum(axis=0) / n + VECTOR_L2 * b
        return loss, grad_s, grad_b

    loss, grad_s, grad_b = loss_and_grad(scale, bias)
    step = 1.0
    for _ in range(2000):
        norm = float((grad_s**2).sum() + (grad_b**2).sum())
        if norm < 1e-14:
            break
        while step > 1e-12:
            new_s, new_b = scale - step * grad_s, bias - step * grad_b
            new_loss, new_grad_s, new_grad_b = loss_and_grad(new_s, new_b)
            # Armijo condition: accept a step that lowers the loss enough.
            if new_loss <= loss - 1e-4 * step * norm:
                break
            step /= 2
        else:
            break
        scale, bias, loss = new_s, new_b, new_loss
        grad_s, grad_b = new_grad_s, new_grad_b
        step *= 2
    return {"type": "vector", "scale": scale.tolist(), "bias": bias.tolist()}


def cross_validated_nll(
    log_q: Sequence[Sequence[float]],
    gold: Sequence[int],
    fit: Fit,
    folds: int,
) -> Tuple[float, float]:
    """Out-of-fold NLL before and after fitting, over folds taken by row index."""
    fitted_rows: List[List[float]] = [[] for _ in gold]
    for fold in range(folds):
        train = [i for i in range(len(gold)) if i % folds != fold]
        calibration = fit([log_q[i] for i in train], [gold[i] for i in train])
        for i in range(fold, len(gold), folds):
            fitted_rows[i] = apply_calibration(calibration, log_q[i])
    return nll(log_q, gold), nll(fitted_rows, gold)


# Calibrations tried for each question kind, as /v1/decisions names kinds.
CANDIDATES: Dict[str, Tuple[Tuple[str, Fit], ...]] = {
    "yes_no": (("temperature", fit_temperature), ("platt", fit_platt)),
    "choice": (("temperature", fit_temperature), ("vector", fit_vector)),
    "score": (("temperature", fit_temperature), ("vector", fit_vector)),
}


def choose_calibration(
    kind: str,
    log_q: Sequence[Sequence[float]],
    gold: Sequence[int],
    folds: int,
) -> Tuple[Optional[Calibration], Dict[str, float]]:
    """The candidate with the lowest out-of-fold NLL, fitted on all rows, or None
    when no candidate beats leaving the answers as they are."""
    report = {"none": nll(log_q, gold)}
    for name, fit in CANDIDATES[kind]:
        report[name] = cross_validated_nll(log_q, gold, fit, folds)[1]
    best = min(report, key=report.get)
    if best == "none":
        return None, report
    return dict(CANDIDATES[kind])[best](log_q, gold), report


def expected_calibration_error(
    confidence: Sequence[float], correct: Sequence[bool], bins: int = 15
) -> float:
    """Mean gap between confidence and accuracy over equal-width confidence bins."""
    totals = [[0, 0.0, 0.0] for _ in range(bins)]
    for c, ok in zip(confidence, correct):
        slot = totals[min(int(c * bins), bins - 1)]
        slot[0] += 1
        slot[1] += c
        slot[2] += float(ok)
    n = len(confidence)
    return math.fsum(abs(s[1] - s[2]) for s in totals if s[0]) / n


def top_label(log_q: Sequence[Sequence[float]], gold: Sequence[int]):
    """Confidence of the most probable option and whether it is the gold one."""
    confidence, correct = [], []
    for row, g in zip(log_q, gold):
        top = max(range(len(row)), key=row.__getitem__)
        confidence.append(math.exp(row[top]))
        correct.append(top == g)
    return confidence, correct


def brier(log_q: Sequence[Sequence[float]], gold: Sequence[int]) -> float:
    return math.fsum(
        math.fsum((math.exp(v) - (i == g)) ** 2 for i, v in enumerate(row))
        for row, g in zip(log_q, gold)
    ) / len(gold)


def auroc(scores: Sequence[float], positive: Sequence[bool]) -> float:
    """Probability that a random positive outscores a random negative, ties half."""
    ranked = sorted(zip(scores, positive))
    rank_sum, i = 0.0, 0
    while i < len(ranked):
        j = i
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        mean_rank = (i + j + 1) / 2
        rank_sum += mean_rank * sum(1 for k in range(i, j) if ranked[k][1])
        i = j
    n_pos = sum(1 for p in positive if p)
    n_neg = len(positive) - n_pos
    if not n_pos or not n_neg:
        return float("nan")
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def ranked_probability_score(
    log_q: Sequence[Sequence[float]], gold: Sequence[int]
) -> float:
    """Squared error of cumulative level probabilities, averaged over thresholds."""
    total = 0.0
    for row, g in zip(log_q, gold):
        cumulative, error = 0.0, 0.0
        for i, v in enumerate(row[:-1]):
            cumulative += math.exp(v)
            error += (cumulative - (1.0 if g <= i else 0.0)) ** 2
        total += error / max(len(row) - 1, 1)
    return total / len(gold)


def quadratic_weighted_kappa(
    pred: Sequence[int], gold: Sequence[int], levels: int
) -> float:
    observed = [[0.0] * levels for _ in range(levels)]
    for p, g in zip(pred, gold):
        observed[p][g] += 1
    n = len(gold)
    pred_hist = [sum(row) for row in observed]
    gold_hist = [sum(observed[i][j] for i in range(levels)) for j in range(levels)]
    num = den = 0.0
    for i in range(levels):
        for j in range(levels):
            weight = (i - j) ** 2 / (levels - 1) ** 2
            num += weight * observed[i][j]
            den += weight * pred_hist[i] * gold_hist[j] / n
    return 1.0 - num / den if den else float("nan")


def selective_accuracy(
    confidence: Sequence[float], correct: Sequence[bool], coverage: float
) -> float:
    """Accuracy on the most confident fraction of answers."""
    order = sorted(range(len(confidence)), key=lambda i: -confidence[i])
    kept = order[: max(1, round(coverage * len(order)))]
    return sum(1 for i in kept if correct[i]) / len(kept)


def decidable_share(
    confidence: Sequence[float], correct: Sequence[bool], max_error: float
) -> float:
    """Largest fraction of most confident answers whose error stays within max_error."""
    order = sorted(range(len(confidence)), key=lambda i: -confidence[i])
    best, wrong = 0, 0
    for kept, i in enumerate(order, start=1):
        wrong += not correct[i]
        if wrong / kept <= max_error:
            best = kept
    return best / len(order)
