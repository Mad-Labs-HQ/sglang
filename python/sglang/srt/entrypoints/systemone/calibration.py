"""Reads and calibration of System One answers from the full-vocabulary logprobs of their labels.

A question is answered from one or more reads: prompts that show its options in
some order and score some label tokens per option. A read setup chooses them:
orders of choice options (cyclic rotations or the rows of a Williams square),
yes and no in both orders, and case or option-name variants of the labels. The
raw setup is the single read of /v1/decisions. Reads are combined by geometric
mean, and then a calibration the client fitted for that question, if the
request carries one, is applied: a temperature, Platt scaling, or vector
scaling. The server keeps no calibration state; a request says how to read and
calibrate each question.

Pure functions over plain floats, shared by the server, the fit tool, and the
benchmarks in ``benchmark/systemone`` so all three compute the same numbers.
Calibrations are plain dicts in their wire format, as a request carries them.
"""

from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache
from typing import (
    Annotated,
    Any,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import msgspec

# Probability floor of an option within one read, so one read that gives an
# option no mass cannot veto it in the geometric mean. Arbitrary, well below
# any probability an answer reports.
LOG_FLOOR = math.log(1e-12)

# Orders a choice question may be read in when no reads config sets a cap; the
# most the benchmarks measured on large option lists.
BUILTIN_MAX_CHOICE_ORDERS = 8

ChoiceOrders = Literal["rotations", "williams"]
# A number of orders, or "all": one per option, the whole rotation cycle or
# Williams square.
MaxOrders = Union[Annotated[int, msgspec.Meta(ge=1)], Literal["all"]]


class ReadSetup(msgspec.Struct, forbid_unknown_fields=True, frozen=True):
    # How choice options are reordered across reads, the request order first.
    choice_orders: ChoiceOrders
    # How many orders a choice question is read in, at most its option count.
    choice_max_orders: MaxOrders
    # Also score each choice option by the first token of its name.
    choice_name_variants: bool
    # 1 reads yes then no, 2 also reads no then yes.
    noul_orders: Annotated[int, msgspec.Meta(ge=1, le=2)]
    # Also score Yes, YES, No, and NO where they are single tokens.
    noul_case_variants: bool


# The single read of /v1/decisions.
RAW_READS = ReadSetup(
    choice_orders="rotations",
    choice_max_orders=1,
    choice_name_variants=False,
    noul_orders=1,
    noul_case_variants=False,
)


class ReadsConfig(msgspec.Struct, forbid_unknown_fields=True, frozen=True):
    # Reads of requests that do not send x_read_setup, such as SDK clients.
    default_reads: ReadSetup
    # The most orders any choice question may be read in.
    max_choice_orders: Annotated[int, msgspec.Meta(ge=1)]


# The reads of a server launched without --decision-reads-config.
BUILTIN_READS_CONFIG = ReadsConfig(
    default_reads=RAW_READS, max_choice_orders=BUILTIN_MAX_CHOICE_ORDERS
)


def decode_config(data: bytes) -> ReadsConfig:
    """Parse a reads config, checking what the types cannot."""
    config = msgspec.json.decode(data, type=ReadsConfig)
    # A number within the cap, so clients that cannot choose reads are never refused.
    default_max = config.default_reads.choice_max_orders
    if default_max == "all" or default_max > config.max_choice_orders:
        raise ValueError(
            "default_reads.choice_max_orders must be a number no larger than "
            "max_choice_orders"
        )
    return config


def load_config(path: Optional[str]) -> ReadsConfig:
    """The reads config at path, or the built-in one when no path is set."""
    if path is None:
        return BUILTIN_READS_CONFIG
    with open(path, "rb") as f:
        return decode_config(f.read())


def choice_order_count(setup: ReadSetup, options: int) -> int:
    """How many orders a choice question with this many options is read in."""
    if setup.choice_max_orders == "all":
        return options
    return min(options, setup.choice_max_orders)


def reads_fingerprint(
    kind: str,
    options: int,
    setup: ReadSetup,
    model: str,
    model_revision: Optional[str],
    prompt_format_version: int,
) -> str:
    """Identity of what a calibration for this question was fitted on: the model,
    the prompt wording, and the reads it gets, however the read setup spells them."""
    count = choice_order_count(setup, options)
    relevant = {
        "yes_no": [setup.noul_orders, setup.noul_case_variants],
        # One order is the request order, whatever the scheme.
        "choice": [
            setup.choice_orders if count > 1 else "identity",
            count,
            setup.choice_name_variants,
        ],
        "score": [],
    }[kind]
    blob = json.dumps(
        [model, model_revision, prompt_format_version, kind, options, relevant],
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


class Read(msgspec.Struct, frozen=True):
    """One prompt of a question: its options in display order and its labels."""

    # Option index shown at each display position.
    order: Tuple[int, ...]
    # Label text shown at each display position.
    labels: Tuple[str, ...]


def identity_read(labels: Sequence[str]) -> Read:
    """The read of /v1/decisions: options in request order, labeled in order."""
    return Read(order=tuple(range(len(labels))), labels=tuple(labels))


def plan_reads(kind: str, labels: Sequence[str], setup: ReadSetup) -> List[Read]:
    """Reads of a question whose options carry these labels, identity first."""
    n = len(labels)
    if kind == "choice":
        # Options move between reads; the labels stay with their positions.
        return [
            Read(order=order, labels=tuple(labels))
            for order in choice_orders(
                setup.choice_orders, n, choice_order_count(setup, n)
            )
        ]
    if kind == "yes_no" and setup.noul_orders == 2:
        # Yes and no keep their labels, only the order they are shown in changes.
        return [
            identity_read(labels),
            Read(order=(1, 0), labels=(labels[1], labels[0])),
        ]
    # Levels are ordinal, so they are always shown in order.
    return [identity_read(labels)]


@lru_cache(maxsize=1024)
def choice_orders(scheme: str, n: int, count: int) -> Tuple[Tuple[int, ...], ...]:
    """count orders of n options, the request order first: every row of the
    scheme when count is n, else the most discordant subset, grown greedily so
    smaller counts are prefixes of larger ones.

    Both schemes are cyclic families, row s being row 0 shifted by s, so the
    Kendall distance between two rows depends only on their shift difference.
    """
    first = williams_row(n) if scheme == "williams" else list(range(n))
    rows = [[(v + shift) % n for v in first] for shift in range(n)]
    # Relabel so the first row reads in request order; distances are unchanged.
    relabel = {v: i for i, v in enumerate(first)}
    orders = [tuple(relabel[v] for v in row) for row in rows]
    if count >= n:
        return tuple(orders)
    by_shift = [kendall_distance(rows[0], rows[d]) for d in range(n)]
    chosen = [0]
    while len(chosen) < count:
        # The row farthest from its nearest chosen row, then from all of them.
        best = max(
            (s for s in range(n) if s not in chosen),
            key=lambda s: (
                min(by_shift[(s - c) % n] for c in chosen),
                sum(by_shift[(s - c) % n] for c in chosen),
                -s,
            ),
        )
        chosen.append(best)
    return tuple(orders[s] for s in chosen)


def williams_row(n: int) -> List[int]:
    """First row of a Williams square, 0, 1, n-1, 2, n-2, ... Its cyclic shifts
    put every option in every position once and, for even n, every ordered pair
    of options next to each other once."""
    row, low, high = [0], 1, n - 1
    while len(row) < n:
        row.append(low)
        low += 1
        if len(row) < n:
            row.append(high)
            high -= 1
    return row


def kendall_distance(a: Sequence[int], b: Sequence[int]) -> int:
    """Pairs of items that two orders of the same items put the other way round,
    counted as the inversions of a's items at their positions in b, by merge sort."""
    position = {v: i for i, v in enumerate(b)}
    items = [position[v] for v in a]
    count, width = 0, 1
    while width < len(items):
        merged = []
        for start in range(0, len(items), 2 * width):
            left = items[start : start + width]
            right = items[start + width : start + 2 * width]
            i = j = 0
            while i < len(left) and j < len(right):
                if left[i] <= right[j]:
                    merged.append(left[i])
                    i += 1
                else:
                    merged.append(right[j])
                    count += len(left) - i
                    j += 1
            merged.extend(left[i:])
            merged.extend(right[j:])
        items = merged
        width *= 2
    return count


def label_variants(kind: str, label: str, name: str, setup: ReadSetup) -> List[str]:
    """Texts other than the label whose first token also counts for an option."""
    if kind == "yes_no" and setup.noul_case_variants:
        return _distinct([label.capitalize(), label.upper()], exclude=label)
    if kind == "choice" and setup.choice_name_variants:
        return _distinct([name, name[:1].upper() + name[1:]], exclude=label)
    return []


def _distinct(texts: List[str], exclude: str) -> List[str]:
    return [text for text in dict.fromkeys(texts) if text and text != exclude]


def logsumexp(values: Sequence[float]) -> float:
    top = max(values)
    if top == -math.inf:
        return -math.inf
    return top + math.log(math.fsum(math.exp(v - top) for v in values))


def log_normalize(values: Sequence[float]) -> List[float]:
    total = logsumexp(values)
    return [v - total for v in values]


def read_log_probabilities(
    option_token_logprobs: Sequence[Sequence[float]],
) -> List[float]:
    """Log probability of each option within one read, from its tokens' logprobs."""
    return log_normalize([logsumexp(lps) for lps in option_token_logprobs])


def combine_reads(reads: Sequence[Sequence[float]]) -> List[float]:
    """Geometric mean over reads of per-option log probabilities, renormalized."""
    n = len(reads[0])
    mean = [
        math.fsum(max(read[i], LOG_FLOOR) for read in reads) / len(reads)
        for i in range(n)
    ]
    return log_normalize(mean)


def apply_calibration(
    calibration: Mapping[str, Any], log_q: Sequence[float]
) -> List[float]:
    """Calibrated log probabilities of a question's options, from its combined reads.

    ``temperature`` divides every log probability, ``platt`` rescales the log odds
    of the first of two options, and ``vector`` scales and shifts each option's
    log probability (diagonal Dirichlet calibration) before renormalizing.
    """
    kind = calibration["type"]
    if kind == "temperature":
        return log_normalize([v / calibration["temperature"] for v in log_q])
    if kind == "platt":
        if len(log_q) != 2:
            raise ValueError("Platt scaling needs exactly two options")
        z = calibration["a"] * (log_q[0] - log_q[1]) + calibration["b"]
        return [log_sigmoid(z), log_sigmoid(-z)]
    if kind == "vector":
        scale, bias = calibration["scale"], calibration["bias"]
        if len(scale) != len(log_q) or len(bias) != len(log_q):
            raise ValueError("vector scaling needs a scale and bias per option")
        return log_normalize(
            [s * max(v, LOG_FLOOR) + b for s, v, b in zip(scale, log_q, bias)]
        )
    raise ValueError(f"unknown calibration type {kind!r}")


def log_sigmoid(z: float) -> float:
    return -math.log1p(math.exp(-z)) if z >= 0 else z - math.log1p(math.exp(z))
