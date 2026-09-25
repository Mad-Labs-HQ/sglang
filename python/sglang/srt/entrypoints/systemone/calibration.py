"""Calibration of System One answers from the full-vocabulary logprobs of their labels.

A question is answered from one or more reads: prompts that show its options in
some order and score some label tokens per option. ``raw`` is the single read of
/v1/decisions. ``label_free`` needs no labeled data: it averages cyclic option
rotations (choice), both yes and no orders (noul), and case or option-name
variants of each label, and can divide out a running prior per question.
``fitted`` applies a temperature or Platt scaling fitted on labeled rows by
``fit_calibration`` on top of its base mode.

Pure functions over plain floats, shared by the server, the fit tool, and the
benchmarks in ``benchmark/systemone`` so all three compute the same numbers.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict
from typing import Annotated, Any, List, Literal, Optional, Sequence, Tuple, Union

import msgspec

# Probability floor of an option within one read, so one read that gives an
# option no mass cannot veto it in the geometric mean. Arbitrary, well below
# any probability an answer reports.
LOG_FLOOR = math.log(1e-12)

Mode = Literal["raw", "label_free", "fitted"]
BaseMode = Literal["raw", "label_free"]


class BatchPriorOff(msgspec.Struct, tag="off", forbid_unknown_fields=True, frozen=True):
    pass


class BatchPriorOn(msgspec.Struct, tag="on", forbid_unknown_fields=True, frozen=True):
    # Exponent of the prior that is divided out, 1 for Batch Calibration.
    strength: Annotated[float, msgspec.Meta(gt=0, le=1)]
    # Answers of one question seen before its prior is applied.
    min_count: Annotated[int, msgspec.Meta(ge=1)]
    # Questions whose priors are kept, least recently used dropped first.
    max_keys: Annotated[int, msgspec.Meta(ge=1)]


class LabelFreeConfig(msgspec.Struct, forbid_unknown_fields=True, frozen=True):
    # Evenly spaced cyclic rotations of the options, identity first.
    choice_rotations: Annotated[int, msgspec.Meta(ge=1)]
    # Also score each choice option by the first token of its name.
    choice_name_variants: bool
    # 1 reads yes then no, 2 also reads no then yes.
    noul_orders: Annotated[int, msgspec.Meta(ge=1, le=2)]
    # Also score Yes, YES, No, and NO where they are single tokens.
    noul_case_variants: bool
    batch_prior: Union[BatchPriorOff, BatchPriorOn]


class TemperatureParams(
    msgspec.Struct, tag="temperature", forbid_unknown_fields=True, frozen=True
):
    temperature: Annotated[float, msgspec.Meta(gt=0)]


class PlattParams(msgspec.Struct, tag="platt", forbid_unknown_fields=True, frozen=True):
    a: float
    b: float


class Profile(msgspec.Struct, forbid_unknown_fields=True, frozen=True):
    # "signature:<sha256>", "bucket:<type>:<options>", or "type:<type>"
    key: str
    params: Union[TemperatureParams, PlattParams]
    # Provenance of the fit, reported by the fit tool.
    rows: int
    oof_nll_base: float
    oof_nll_fitted: float


class FittedProfiles(msgspec.Struct, forbid_unknown_fields=True, frozen=True):
    model: str
    # None when the server that was fitted on did not pin a revision.
    model_revision: Optional[str]
    prompt_format_version: int
    base: BaseMode
    # label_free_fingerprint of the reads the profiles were fitted on.
    reads_fingerprint: str
    profiles: List[Profile]


class CalibrationConfig(msgspec.Struct, forbid_unknown_fields=True, frozen=True):
    default_mode: Mode
    label_free: LabelFreeConfig
    fitted: Optional[FittedProfiles]


def decode_config(data: bytes) -> CalibrationConfig:
    """Parse a calibration config, checking what the types cannot."""
    config = msgspec.json.decode(data, type=CalibrationConfig)
    if config.default_mode == "fitted" and config.fitted is None:
        raise ValueError("default_mode 'fitted' needs fitted profiles")
    if config.fitted is not None:
        keys = [profile.key for profile in config.fitted.profiles]
        if len(set(keys)) != len(keys):
            raise ValueError("fitted profile keys must be distinct")
    return config


def load_config(
    path: Optional[str], model: str, model_revision: Optional[str]
) -> Optional[CalibrationConfig]:
    """The config at path checked against the served model, None when no path is set."""
    if path is None:
        return None
    with open(path, "rb") as f:
        config = decode_config(f.read())
    if config.fitted is not None:
        # Imported here to keep this module free of server imports at load time.
        from sglang.srt.entrypoints.openai.serving_decisions import (
            PROMPT_FORMAT_VERSION,
        )

        check_fitted_matches(
            fitted=config.fitted,
            label_free=config.label_free,
            model=model,
            model_revision=model_revision,
            prompt_format_version=PROMPT_FORMAT_VERSION,
        )
    return config


def encode_config(config: CalibrationConfig) -> bytes:
    return msgspec.json.format(msgspec.json.encode(config), indent=2) + b"\n"


def check_fitted_matches(
    fitted: FittedProfiles,
    label_free: LabelFreeConfig,
    model: str,
    model_revision: Optional[str],
    prompt_format_version: int,
) -> None:
    """Refuse profiles fitted for another model, revision, prompt, or set of reads."""
    for field, fitted_value, served_value in (
        ("model", fitted.model, model),
        ("model_revision", fitted.model_revision, model_revision),
        (
            "prompt_format_version",
            fitted.prompt_format_version,
            prompt_format_version,
        ),
        (
            "reads_fingerprint",
            fitted.reads_fingerprint,
            reads_fingerprint(fitted.base, label_free),
        ),
    ):
        if fitted_value != served_value:
            raise ValueError(
                f"the fitted profiles have {field} {fitted_value!r}, but this "
                f"server has {served_value!r}; fit them again for this server"
            )


def reads_fingerprint(base: BaseMode, label_free: LabelFreeConfig) -> str:
    """Identity of the reads a base mode makes, without the batch prior."""
    if base == "raw":
        return "raw"
    reads = msgspec.structs.asdict(label_free)
    del reads["batch_prior"]
    return "label_free:" + json.dumps(reads, sort_keys=True, separators=(",", ":"))


class Read(msgspec.Struct, frozen=True):
    """One prompt of a question: its options in display order and its labels."""

    # Option index shown at each display position.
    order: Tuple[int, ...]
    # Label text shown at each display position.
    labels: Tuple[str, ...]


def identity_read(labels: Sequence[str]) -> Read:
    """The read of /v1/decisions: options in request order, labeled in order."""
    return Read(order=tuple(range(len(labels))), labels=tuple(labels))


def plan_reads(
    kind: str, labels: Sequence[str], label_free: LabelFreeConfig
) -> List[Read]:
    """Label-free reads of a question whose options carry these labels, identity first."""
    n = len(labels)
    if kind == "choice":
        # A rotation relabels options by display position, so the labels stay put.
        return [
            Read(order=tuple((offset + i) % n for i in range(n)), labels=tuple(labels))
            for offset in rotation_offsets(n, label_free.choice_rotations)
        ]
    if kind == "yes_no" and label_free.noul_orders == 2:
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


def label_variants(
    kind: str, label: str, name: str, label_free: LabelFreeConfig
) -> List[str]:
    """Texts other than the label whose first token also counts for an option."""
    if kind == "yes_no" and label_free.noul_case_variants:
        return _distinct([label.capitalize(), label.upper()], exclude=label)
    if kind == "choice" and label_free.choice_name_variants:
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


def question_signature(
    kind: str, question: Any, names: Sequence[str], details: Sequence[Any]
) -> str:
    """Stable key of a question's wording and options, for priors and profiles."""
    blob = json.dumps(
        [kind, question, list(names), list(details)],
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=repr,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


class BatchPriors:
    """Running mean of each question's probabilities, divided out once seen enough."""

    def __init__(self, config: BatchPriorOn):
        self.config = config
        # signature -> (answers seen, summed probabilities)
        self._sums: OrderedDict[str, Tuple[int, List[float]]] = OrderedDict()

    def apply(self, signature: str, log_q: Sequence[float]) -> List[float]:
        """Record this answer, then divide out the prior when min_count are seen."""
        count, sums = self._sums.pop(signature, (0, [0.0] * len(log_q)))
        if len(sums) != len(log_q):
            count, sums = 0, [0.0] * len(log_q)
        count += 1
        sums = [s + math.exp(v) for s, v in zip(sums, log_q)]
        self._sums[signature] = (count, sums)
        while len(self._sums) > self.config.max_keys:
            self._sums.popitem(last=False)
        if count < self.config.min_count:
            return list(log_q)
        strength = self.config.strength
        return log_normalize(
            [
                v - strength * math.log(max(s / count, 1e-300))
                for v, s in zip(log_q, sums)
            ]
        )


def find_profile(
    fitted: FittedProfiles, kind: str, n_options: int, signature: str
) -> Optional[Profile]:
    """The most specific profile for a question: its signature, bucket, then type."""
    by_key = {profile.key: profile for profile in fitted.profiles}
    for key in (
        f"signature:{signature}",
        f"bucket:{kind}:{n_options}",
        f"type:{kind}",
    ):
        if key in by_key:
            return by_key[key]
    return None


def apply_params(
    params: Union[TemperatureParams, PlattParams], log_q: Sequence[float]
) -> List[float]:
    """Temperature over all options, or Platt scaling of the first of two."""
    if isinstance(params, TemperatureParams):
        return log_normalize([v / params.temperature for v in log_q])
    if len(log_q) != 2:
        raise ValueError("Platt scaling needs exactly two options")
    z = params.a * (log_q[0] - log_q[1]) + params.b
    return [log_sigmoid(z), log_sigmoid(-z)]


def log_sigmoid(z: float) -> float:
    return -math.log1p(math.exp(-z)) if z >= 0 else z - math.log1p(math.exp(z))
