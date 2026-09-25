"""Reads and calibration of System One answers from the full-vocabulary logprobs of their labels.

A question is answered from one or more reads: prompts that show its options in
some order and score some label tokens per option. A read setup chooses them:
cyclic rotations of choice options, yes and no in both orders, and case or
option-name variants of the labels. The raw setup is the single read of
/v1/decisions. Reads are combined by geometric mean, and then a calibration the
client fitted for that question, if the request carries one, is applied: a
temperature, Platt scaling, or vector scaling. The server keeps no calibration
state; a request says how to read and calibrate each question.

Pure functions over plain floats, shared by the server, the fit tool, and the
benchmarks in ``benchmark/systemone`` so all three compute the same numbers.
Calibrations are plain dicts in their wire format, as a request carries them.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Annotated, Any, List, Mapping, Optional, Sequence, Tuple

import msgspec

# Probability floor of an option within one read, so one read that gives an
# option no mass cannot veto it in the geometric mean. Arbitrary, well below
# any probability an answer reports.
LOG_FLOOR = math.log(1e-12)

# Rotations a request may ask for when no reads config sets a cap; the most the
# benchmarks measured.
BUILTIN_MAX_CHOICE_ROTATIONS = 8


class ReadSetup(msgspec.Struct, forbid_unknown_fields=True, frozen=True):
    # Evenly spaced cyclic rotations of the options, identity first.
    choice_rotations: Annotated[int, msgspec.Meta(ge=1)]
    # Also score each choice option by the first token of its name.
    choice_name_variants: bool
    # 1 reads yes then no, 2 also reads no then yes.
    noul_orders: Annotated[int, msgspec.Meta(ge=1, le=2)]
    # Also score Yes, YES, No, and NO where they are single tokens.
    noul_case_variants: bool


# The single read of /v1/decisions.
RAW_READS = ReadSetup(
    choice_rotations=1,
    choice_name_variants=False,
    noul_orders=1,
    noul_case_variants=False,
)


class ReadsConfig(msgspec.Struct, forbid_unknown_fields=True, frozen=True):
    # Reads of requests that do not send x_read_setup, such as SDK clients.
    default_reads: ReadSetup
    # The most choice rotations a request may ask for.
    max_choice_rotations: Annotated[int, msgspec.Meta(ge=1)]


# The reads of a server launched without --decision-reads-config.
BUILTIN_READS_CONFIG = ReadsConfig(
    default_reads=RAW_READS, max_choice_rotations=BUILTIN_MAX_CHOICE_ROTATIONS
)


def decode_config(data: bytes) -> ReadsConfig:
    """Parse a reads config, checking what the types cannot."""
    config = msgspec.json.decode(data, type=ReadsConfig)
    if config.default_reads.choice_rotations > config.max_choice_rotations:
        raise ValueError("default_reads.choice_rotations is above max_choice_rotations")
    return config


def load_config(path: Optional[str]) -> ReadsConfig:
    """The reads config at path, or the built-in one when no path is set."""
    if path is None:
        return BUILTIN_READS_CONFIG
    with open(path, "rb") as f:
        return decode_config(f.read())


def reads_fingerprint(
    kind: str,
    setup: ReadSetup,
    model: str,
    model_revision: Optional[str],
    prompt_format_version: int,
) -> str:
    """Identity of what a calibration for this kind of question was fitted on:
    the model, the prompt wording, and the parts of the read setup that apply."""
    relevant = {
        "yes_no": [setup.noul_orders, setup.noul_case_variants],
        "choice": [setup.choice_rotations, setup.choice_name_variants],
        "score": [],
    }[kind]
    blob = json.dumps(
        [model, model_revision, prompt_format_version, kind, relevant],
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
        # A rotation relabels options by display position, so the labels stay put.
        return [
            Read(order=tuple((offset + i) % n for i in range(n)), labels=tuple(labels))
            for offset in rotation_offsets(n, setup.choice_rotations)
        ]
    if kind == "yes_no" and setup.noul_orders == 2:
        # Yes and no keep their labels, only the order they are shown in changes.
        return [
            identity_read(labels),
            Read(order=(1, 0), labels=(labels[1], labels[0])),
        ]
    # Levels are ordinal, so they are always shown in order.
    return [identity_read(labels)]


def rotation_offsets(n: int, rotations: int) -> List[int]:
    """Evenly spaced offsets, nested for doubling counts: those of k are among 2k's."""
    k = min(n, rotations)
    return [i * n // k for i in range(k)]


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
