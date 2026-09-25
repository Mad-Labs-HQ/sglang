"""Handler for the System One compatible decision API, on the /v1/decisions scoring path."""

from __future__ import annotations

import math
import string
from typing import Any, Dict, Iterator, List, Optional, Tuple

import msgspec
import orjson
from fastapi import Request
from fastapi.responses import ORJSONResponse

from sglang.srt.entrypoints.openai.serving_decisions import (
    PROMPT_FORMAT_VERSION,
    OpenAIServingDecisions,
    QuestionView,
    default_labels,
    encode_all,
    encode_labels,
    first_token_id,
    label_context,
    label_mass,
    label_token_id,
    render_question,
    render_text,
)
from sglang.srt.entrypoints.systemone.calibration import (
    Read,
    ReadsConfig,
    ReadSetup,
    apply_calibration,
    combine_reads,
    label_variants,
    plan_reads,
    read_log_probabilities,
    reads_fingerprint,
)
from sglang.srt.entrypoints.systemone.protocol import (
    SystemOneChoiceAnswer,
    SystemOneChoiceQuestion,
    SystemOneNoulAnswer,
    SystemOneNoulQuestion,
    SystemOneQuestion,
    SystemOneRead,
    SystemOneRequest,
    SystemOneResponse,
    SystemOneScoreAnswer,
    SystemOneUsage,
)

# Beyond A to Z, every option gets a two-letter label, in this fixed order.
_PAIR_LABELS = [a + b for a in string.ascii_uppercase for b in string.ascii_uppercase]


class EncodedRead(msgspec.Struct, frozen=True):
    """A read as scored: its token ids and which of them count for each option."""

    read: Read
    token_ids: List[int]
    # Per option index, positions in token_ids of its tokens and their texts.
    groups: List[List[int]]
    texts: List[List[str]]


class QuestionPlan(msgspec.Struct):
    """A question, its calibration, and its reads, filled in as they are encoded."""

    question_id: str
    view: QuestionView
    # The client's calibration in its wire format, None when it sent none.
    calibration: Optional[Dict[str, Any]]
    # What a calibration fitted on this question's answers is tied to.
    fingerprint: str
    reads: List[EncodedRead]


class SystemOneServing(OpenAIServingDecisions):
    """Answers System One questions with the rendering, label checks, and scoring of /v1/decisions."""

    route = "/v1/systemone"

    def __init__(
        self,
        chat_serving,
        reads_config: ReadsConfig,
        model: str,
        model_revision: Optional[str],
    ):
        super().__init__(chat_serving)
        self.reads_config = reads_config
        # What calibrations fitted on this server's answers are tied to.
        self.model = model
        self.model_revision = model_revision

    def _request_id_prefix(self) -> str:
        return "systemone-"

    def _validate_request(self, request: SystemOneRequest) -> Optional[str]:
        return (
            self._validate_server(request.model)
            or self._validate_reasoning(request.chat_template_kwargs)
            or self._validate_reads(request)
        )

    def _read_setup(self, request: SystemOneRequest) -> ReadSetup:
        if request.x_read_setup is None:
            return self.reads_config.default_reads
        return ReadSetup(**request.x_read_setup.model_dump())

    def _fingerprint(self, kind: str, setup: ReadSetup) -> str:
        return reads_fingerprint(
            kind=kind,
            setup=setup,
            model=self.model,
            model_revision=self.model_revision,
            prompt_format_version=PROMPT_FORMAT_VERSION,
        )

    def _validate_reads(self, request: SystemOneRequest) -> Optional[str]:
        """Refuse reads above the server's cap and calibrations fitted on other answers."""
        setup = self._read_setup(request)
        limit = self.reads_config.max_choice_rotations
        if setup.choice_rotations > limit:
            return (
                f"x_read_setup asks for {setup.choice_rotations} choice rotations, "
                f"but this server allows at most {limit}"
            )
        for question_id, question in request.questions.items():
            calibration = question.x_calibration
            if calibration is None or calibration.fitted_on is None:
                continue
            fingerprint = self._fingerprint(_view(question).kind, setup)
            if calibration.fitted_on != fingerprint:
                return (
                    f"question {question_id!r}: its calibration was fitted on "
                    f"answers with fingerprint {calibration.fitted_on!r}, but this "
                    f"model and read setup give {fingerprint!r}; fit it again on "
                    "answers from this server"
                )
        return None

    def _convert_to_internal_request(
        self,
        request: SystemOneRequest,
        raw_request: Request = None,
    ) -> Tuple[
        Iterator[Tuple[List[int], List[int]]],
        Tuple[SystemOneRequest, List[QuestionPlan]],
    ]:
        plans: List[QuestionPlan] = []
        # Lazy, so the async handler can yield to other requests between reads.
        encoded = self._encoded_reads(request, self._read_setup(request), plans)
        return encoded, (request, plans)

    def _encoded_reads(
        self,
        request: SystemOneRequest,
        setup: ReadSetup,
        plans: List[QuestionPlan],
    ) -> Iterator[Tuple[List[int], List[int]]]:
        """Prompt and token ids of every read, in request order, recording each in plans."""
        text = render_text(request.state)
        chat_template_kwargs = self._chat_template_kwargs(request.chat_template_kwargs)
        pair_labels = None
        for question_id, question in request.questions.items():
            view = _view(question)
            try:
                labels = default_labels(view)
                if view.kind == "choice" and len(view.names) > len(labels):
                    if pair_labels is None:
                        pair_labels = self._pair_labels(chat_template_kwargs)
                    if len(view.names) > len(pair_labels):
                        raise ValueError(
                            f"it has {len(view.names)} options, but the served "
                            "tokenizer and chat template can label at most "
                            f"{max(len(pair_labels), len(string.ascii_uppercase))}"
                        )
                    labels = pair_labels[: len(view.names)]
                if view.kind == "score":
                    _check_legend(question_id, view)
                calibration = question.x_calibration
                plan = QuestionPlan(
                    question_id=question_id,
                    view=view,
                    calibration=(
                        None
                        if calibration is None
                        else calibration.model_dump(exclude={"fitted_on"})
                    ),
                    fingerprint=self._fingerprint(view.kind, setup),
                    reads=[],
                )
                plans.append(plan)
                for read in plan_reads(view.kind, labels, setup):
                    prompt_ids, encoded = self._encode_read(
                        text=text,
                        view=view,
                        read=read,
                        setup=setup,
                        chat_template_kwargs=chat_template_kwargs,
                    )
                    plan.reads.append(encoded)
                    yield prompt_ids, encoded.token_ids
            except ValueError as e:
                raise ValueError(f"question {question_id!r}: {e}") from e

    def _encode_read(
        self,
        text: str,
        view: QuestionView,
        read: Read,
        setup: ReadSetup,
        chat_template_kwargs: Dict[str, Any],
    ) -> Tuple[List[int], EncodedRead]:
        """Prompt ids of a read, and its label tokens followed by any variant tokens."""
        shown = QuestionView(
            kind=view.kind,
            question=view.question,
            names=[view.names[i] for i in read.order],
            details=[view.details[i] for i in read.order],
        )
        content = render_question(text=text, view=shown, labels=list(read.labels))
        prompt, prompt_ids = self._encode_prompt(content, chat_template_kwargs)
        label_ids = encode_labels(
            tokenizer=self.tokenizer_manager.tokenizer,
            prompt=prompt,
            prompt_ids=prompt_ids,
            labels=list(read.labels),
            added_tokens=self.added_tokens,
        )
        groups: List[List[int]] = [[] for _ in read.order]
        texts: List[List[str]] = [[] for _ in read.order]
        for position, option in enumerate(read.order):
            groups[option].append(position)
            texts[option].append(read.labels[position])
        token_ids = list(label_ids)
        for option, variant, token_id in self._variant_tokens(
            view=view,
            read=read,
            setup=setup,
            prompt=prompt,
            prompt_ids=prompt_ids,
            label_ids=label_ids,
        ):
            groups[option].append(len(token_ids))
            texts[option].append(variant)
            token_ids.append(token_id)
        return prompt_ids, EncodedRead(
            read=read, token_ids=token_ids, groups=groups, texts=texts
        )

    def _variant_tokens(
        self,
        view: QuestionView,
        read: Read,
        setup: ReadSetup,
        prompt: str,
        prompt_ids: List[int],
        label_ids: List[int],
    ) -> List[Tuple[int, str, int]]:
        """(option, text, token) of each variant whose token no other option or label has."""
        variants = [
            (option, variant)
            for position, option in enumerate(read.order)
            for variant in label_variants(
                view.kind, read.labels[position], view.names[option], setup
            )
        ]
        if not variants:
            return []
        tokenizer = self.tokenizer_manager.tokenizer
        text, text_ids, _ = label_context(
            tokenizer, prompt, prompt_ids, self.added_tokens
        )
        # Case variants must be whole labels, name variants only need to start one.
        encode = label_token_id if view.kind == "yes_no" else first_token_id
        claims: Dict[int, Dict[int, str]] = {}
        for option, variant in variants:
            token_id = encode(tokenizer, text, text_ids, variant)
            if token_id is not None and token_id not in label_ids:
                claims.setdefault(token_id, {}).setdefault(option, variant)
        return [
            (option, variant, token_id)
            for token_id, owners in claims.items()
            if len(owners) == 1
            for option, variant in owners.items()
        ]

    def _pair_labels(self, chat_template_kwargs: Dict[str, Any]) -> List[str]:
        """Two-letter labels that are distinct single tokens at the answer position."""
        if not self.added_tokens:
            raise ValueError(
                "more than 26 options needs added tokens the server can read "
                "from the tokenizer, and the served tokenizer reports none"
            )
        tokenizer = self.tokenizer_manager.tokenizer
        contexts = []
        for message in ("x", "y"):
            prompt = self._answer_prompt(message, chat_template_kwargs)
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            contexts.append(
                label_context(tokenizer, prompt, prompt_ids, self.added_tokens)
            )
        (text, text_ids, shortcut), other = contexts
        # Labels must be checked on text that excludes the message. Otherwise
        # every label re-encodes the whole input on the event loop, which is
        # unbounded work for hundreds of options.
        if not (shortcut and other[2] and other[0] == text):
            raise ValueError(
                "more than 26 options needs an added token between the message "
                "and the answer position, after which the text tokenizes the same on "
                "its own, which this tokenizer and chat template do not provide"
            )
        labels, label_ids = [], set()
        for label in _PAIR_LABELS:
            token_id = label_token_id(tokenizer, text, text_ids, label)
            if token_id is not None and token_id not in label_ids:
                labels.append(label)
                label_ids.add(token_id)
        # Each final prompt is still checked as a whole in _encode_question.
        return labels

    async def _handle_non_streaming_request(
        self,
        adapted_request: Iterator[Tuple[List[int], List[int]]],
        processed: Tuple[SystemOneRequest, List[QuestionPlan]],
        raw_request: Request,
    ) -> ORJSONResponse:
        request, plans = processed
        prompts, label_token_ids = await encode_all(adapted_request)
        result = await self._score_prompts(
            prompts=prompts,
            label_token_ids=label_token_ids,
            raw_request=raw_request,
            temperature=1.0,
        )
        answers = {}
        start = 0
        for plan in plans:
            end = start + len(plan.reads)
            answers[plan.question_id] = _answer_question(
                plan=plan,
                scores=result.scores[start:end],
                token_logprobs=result.token_logprobs[start:end],
                return_reads=request.x_return_reads,
            )
            start = end
        response = SystemOneResponse(
            # The served model answered, whatever name the request used.
            model=self.tokenizer_manager.served_model_name,
            answers=answers,
            usage=SystemOneUsage(input_tokens=result.prompt_tokens),
        )
        content = response.model_dump()
        for answer in content["answers"].values():
            if answer["x_reads"] is None:
                del answer["x_reads"]
        return ORJSONResponse(content=content)


def _answer_question(
    plan: QuestionPlan,
    scores: List[List[float]],
    token_logprobs: List[List[float]],
    return_reads: bool,
):
    """The answer of one question from the scores of its reads and its calibration."""
    view = plan.view
    log_q = combine_reads(
        [
            read_log_probabilities(
                [[logprobs[k] for k in group] for group in read.groups]
            )
            for read, logprobs in zip(plan.reads, token_logprobs)
        ]
    )
    single = len(plan.reads) == 1 and all(len(g) == 1 for g in plan.reads[0].groups)
    # A single read of the labels alone reports exactly what /v1/decisions does.
    read_probabilities = scores[0] if single else [math.exp(v) for v in log_q]
    if plan.calibration is None:
        probabilities, applied = read_probabilities, "none"
    else:
        probabilities = [
            math.exp(v) for v in apply_calibration(plan.calibration, log_q)
        ]
        applied = plan.calibration["type"]
    reads = (
        [
            _read_report(view, read, logprobs)
            for read, logprobs in zip(plan.reads, token_logprobs)
        ]
        if return_reads
        else None
    )
    return _answer(
        view=view,
        probabilities=probabilities,
        read_probabilities=read_probabilities,
        # The first read shows the options in request order.
        mass=label_mass(token_logprobs[0]),
        question_id=plan.question_id,
        applied=applied,
        fingerprint=plan.fingerprint,
        reads=reads,
    )


def _read_report(
    view: QuestionView, read: EncodedRead, logprobs: List[float]
) -> SystemOneRead:
    names = view.names
    return SystemOneRead(
        order=[names[i] for i in read.read.order],
        labels=list(read.read.labels),
        texts={names[i]: read.texts[i] for i in range(len(names))},
        logprobs={
            names[i]: [logprobs[k] for k in read.groups[i]] for i in range(len(names))
        },
    )


def _view(question: SystemOneQuestion) -> QuestionView:
    if isinstance(question, SystemOneChoiceQuestion):
        return QuestionView(
            kind="choice",
            question=question.instructions,
            names=list(question.criteria),
            details=list(question.criteria.values()),
        )
    if isinstance(question, SystemOneNoulQuestion):
        criteria = question.criteria
        return QuestionView(
            kind="yes_no",
            question=question.instructions,
            names=["yes", "no"],
            details=[
                criteria.true if criteria else None,
                criteria.false if criteria else None,
            ],
        )
    return QuestionView(
        kind="score",
        question=question.instructions,
        names=[str(level) for level in range(len(question.criteria))],
        details=list(question.criteria),
    )


def _answer(
    view: QuestionView,
    probabilities: List[float],
    read_probabilities: List[float],
    mass: float,
    question_id: str,
    applied: str,
    fingerprint: str,
    reads: Optional[List[SystemOneRead]],
):
    if not all(
        math.isfinite(value) for value in [*probabilities, *read_probabilities, mass]
    ):
        # A server fault, reported as 500 rather than as a client error.
        raise RuntimeError(f"question {question_id!r} scored non-finite values")
    extensions = {
        "x_label_mass": mass,
        "x_read_probabilities": dict(zip(view.names, read_probabilities)),
        "x_calibration": applied,
        "x_fingerprint": fingerprint,
        "x_reads": reads,
    }
    # Reported as scored, like /v1/decisions, and normalized only for confidence.
    if view.kind == "yes_no":
        return SystemOneNoulAnswer(noul=probabilities[0], **extensions)
    probabilities_by_name = dict(zip(view.names, probabilities))
    if view.kind == "choice":
        return SystemOneChoiceAnswer(
            choice=view.names[probabilities.index(max(probabilities))],
            confidence=_choice_confidence(_normalized(probabilities)),
            probabilities=probabilities_by_name,
            **extensions,
        )
    return SystemOneScoreAnswer(
        score=math.fsum(i * p for i, p in enumerate(probabilities)),
        confidence=_score_confidence(_normalized(probabilities)),
        legend=_legend(view),
        probabilities=probabilities_by_name,
        **extensions,
    )


def _legend(view: QuestionView) -> Dict[str, Any]:
    return dict(zip(view.names, view.details))


def _check_legend(question_id: str, view: QuestionView) -> None:
    """Refuse, before scoring, levels that the response cannot echo in its legend."""
    answer = SystemOneScoreAnswer(
        score=0.0,
        confidence=0.0,
        legend=_legend(view),
        probabilities={},
        x_label_mass=0.0,
        x_read_probabilities={},
        x_calibration="none",
        x_fingerprint="",
    )
    response = SystemOneResponse(
        model="", answers={question_id: answer}, usage=SystemOneUsage(input_tokens=0)
    )
    try:
        # The same encoder and options as the real response.
        ORJSONResponse(content=response.model_dump())
    except orjson.JSONEncodeError as e:
        raise ValueError(f"a level cannot be returned in the legend: {e}") from e


def _normalized(probabilities: List[float]) -> List[float]:
    total = math.fsum(probabilities)
    if total <= 0:
        return [1.0 / len(probabilities)] * len(probabilities)
    return [p / total for p in probabilities]


def _choice_confidence(q: List[float]) -> float:
    """How far the top option stands above a uniform guess, from 0 to 1."""
    n = len(q)
    if n == 1:
        return 1.0
    return min(1.0, max(0.0, (n * max(q) - 1) / (n - 1)))


def _score_confidence(q: List[float]) -> float:
    """One minus the spread around the top level relative to a uniform spread, floored at 0."""
    n = len(q)
    if n == 1:
        return 1.0
    top = q.index(max(q))
    spread = math.fsum(p * abs(i - top) for i, p in enumerate(q))
    uniform_spread = math.fsum(abs(i - (n - 1) / 2) for i in range(n)) / n
    return max(0.0, 1 - spread / uniform_spread)
