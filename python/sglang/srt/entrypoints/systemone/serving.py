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
    BaseMode,
    BatchPriorOn,
    BatchPriors,
    CalibrationConfig,
    Mode,
    Read,
    apply_params,
    combine_reads,
    find_profile,
    identity_read,
    label_variants,
    plan_reads,
    question_signature,
    read_log_probabilities,
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
    """A question and its reads, filled in as the reads are encoded."""

    question_id: str
    view: QuestionView
    reads: List[EncodedRead]


class SystemOneServing(OpenAIServingDecisions):
    """Answers System One questions with the rendering, label checks, and scoring of /v1/decisions."""

    route = "/v1/systemone"

    def __init__(self, chat_serving, calibration: Optional[CalibrationConfig]):
        super().__init__(chat_serving)
        # None when the server was launched without --decision-calibration-config.
        self.calibration = calibration
        self.batch_priors = (
            BatchPriors(calibration.label_free.batch_prior)
            if calibration is not None
            and isinstance(calibration.label_free.batch_prior, BatchPriorOn)
            else None
        )

    def _request_id_prefix(self) -> str:
        return "systemone-"

    def _validate_request(self, request: SystemOneRequest) -> Optional[str]:
        return (
            self._validate_server(request.model)
            or self._validate_reasoning(request.chat_template_kwargs)
            or self._validate_calibration(self._mode(request))
        )

    def _mode(self, request: SystemOneRequest) -> Mode:
        if request.x_calibration is not None:
            return request.x_calibration
        return self.calibration.default_mode if self.calibration is not None else "raw"

    def _validate_calibration(self, mode: Mode) -> Optional[str]:
        if mode == "raw":
            return None
        if self.calibration is None:
            return (
                f"x_calibration {mode!r} needs a server launched with "
                "--decision-calibration-config"
            )
        if mode == "fitted" and self.calibration.fitted is None:
            return (
                "x_calibration 'fitted' needs fitted profiles in the server's "
                "calibration config"
            )
        return None

    def _base_mode(self, mode: Mode) -> BaseMode:
        """The mode whose reads a request scores: fitted profiles set their own."""
        if mode == "fitted":
            return self.calibration.fitted.base
        return mode

    def _convert_to_internal_request(
        self,
        request: SystemOneRequest,
        raw_request: Request = None,
    ) -> Tuple[
        Iterator[Tuple[List[int], List[int]]],
        Tuple[SystemOneRequest, Mode, List[QuestionPlan]],
    ]:
        views = [_view(question) for question in request.questions.values()]
        mode = self._mode(request)
        plans: List[QuestionPlan] = []
        # Lazy, so the async handler can yield to other requests between reads.
        encoded = self._encoded_reads(request, views, self._base_mode(mode), plans)
        return encoded, (request, mode, plans)

    def _encoded_reads(
        self,
        request: SystemOneRequest,
        views: List[QuestionView],
        base: BaseMode,
        plans: List[QuestionPlan],
    ) -> Iterator[Tuple[List[int], List[int]]]:
        """Prompt and token ids of every read, in request order, recording each in plans."""
        text = render_text(request.state)
        chat_template_kwargs = self._chat_template_kwargs(request.chat_template_kwargs)
        pair_labels = None
        for question_id, view in zip(request.questions, views):
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
                plan = QuestionPlan(question_id=question_id, view=view, reads=[])
                plans.append(plan)
                reads = (
                    [identity_read(labels)]
                    if base == "raw"
                    else plan_reads(view.kind, labels, self.calibration.label_free)
                )
                for read in reads:
                    prompt_ids, encoded = self._encode_read(
                        text=text,
                        view=view,
                        read=read,
                        base=base,
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
        base: BaseMode,
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
        if base == "label_free":
            for option, variant, token_id in self._variant_tokens(
                view=view,
                read=read,
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
        prompt: str,
        prompt_ids: List[int],
        label_ids: List[int],
    ) -> List[Tuple[int, str, int]]:
        """(option, text, token) of each variant whose token no other option or label has."""
        tokenizer = self.tokenizer_manager.tokenizer
        text, text_ids, _ = label_context(
            tokenizer, prompt, prompt_ids, self.added_tokens
        )
        # Case variants must be whole labels, name variants only need to start one.
        encode = label_token_id if view.kind == "yes_no" else first_token_id
        claims: Dict[int, Dict[int, str]] = {}
        for position, option in enumerate(read.order):
            for variant in label_variants(
                view.kind,
                read.labels[position],
                view.names[option],
                self.calibration.label_free,
            ):
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
        processed: Tuple[SystemOneRequest, Mode, List[QuestionPlan]],
        raw_request: Request,
    ) -> ORJSONResponse:
        request, mode, plans = processed
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
            answers[plan.question_id] = self._answer_question(
                plan=plan,
                mode=mode,
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
        self,
        plan: QuestionPlan,
        mode: Mode,
        scores: List[List[float]],
        token_logprobs: List[List[float]],
        return_reads: bool,
    ):
        """The answer of one question from the scores of its reads, in its mode."""
        view = plan.view
        applied = mode
        if mode == "raw":
            # Exactly the probabilities /v1/decisions reports.
            probabilities = scores[0]
        else:
            log_q = combine_reads(
                [
                    read_log_probabilities(
                        [[logprobs[k] for k in group] for group in read.groups]
                    )
                    for read, logprobs in zip(plan.reads, token_logprobs)
                ]
            )
            signature = question_signature(
                view.kind, view.question, view.names, view.details
            )
            if mode == "label_free" and self.batch_priors is not None:
                log_q = self.batch_priors.apply(signature, log_q)
            if mode == "fitted":
                profile = find_profile(
                    self.calibration.fitted, view.kind, len(view.names), signature
                )
                if profile is None:
                    applied = self.calibration.fitted.base
                else:
                    log_q = apply_params(profile.params, log_q)
            probabilities = [math.exp(value) for value in log_q]
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
            # The first read shows the options in request order.
            mass=label_mass(token_logprobs[0]),
            question_id=plan.question_id,
            applied=applied,
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
    mass: float,
    question_id: str,
    applied: Mode,
    reads: Optional[List[SystemOneRead]],
):
    if not all(math.isfinite(value) for value in [*probabilities, mass]):
        # A server fault, reported as 500 rather than as a client error.
        raise RuntimeError(f"question {question_id!r} scored non-finite values")
    extensions = {"x_label_mass": mass, "x_calibration": applied, "x_reads": reads}
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
        x_calibration="raw",
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
