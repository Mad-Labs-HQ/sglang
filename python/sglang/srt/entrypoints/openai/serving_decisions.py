from __future__ import annotations

import asyncio
import json
import logging
import math
import string
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Tuple

import msgspec
from fastapi import Request
from fastapi.responses import ORJSONResponse
from transformers import PreTrainedTokenizerBase

from sglang.srt.entrypoints.openai.protocol import (
    DecisionAnswer,
    DecisionChoiceQuestion,
    DecisionQuestion,
    DecisionRequest,
    DecisionResponse,
    DecisionScoreQuestion,
    DecisionText,
    UsageInfo,
    is_blank_decision_text,
)
from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
from sglang.srt.entrypoints.openai.serving_chat import _CHAT_TEMPLATE_CLIENT_ERRORS
from sglang.srt.environ import envs
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.runtime_context import get_exec, get_memory

if TYPE_CHECKING:
    from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat

logger = logging.getLogger(__name__)

# Version of the server-owned prompt wording and answer labels.
# Any change to a rendering /v1/decisions can produce needs a new version.
PROMPT_FORMAT_VERSION = 1

# Parser defaults that name the chat template kwarg toggling reasoning.
_PARSER_TOGGLE_MODES = (
    "thinking",
    "enable_thinking",
    "explicit_thinking",
    "explicit_enable_thinking",
)
# Answer text of a finished reply, rendered only to see what precedes an answer.
_REPLY_SENTINEL = "DECISION_ANSWER"
# Message fields that chat templates read reasoning from, tried in this order
# to render a reply whose reasoning is empty.
_REASONING_FIELDS = ("reasoning_content", "reasoning", "thinking", "think")
# Cached closing suffixes before the cache is cleared. There are a few closing
# lines, so only request chat_template_kwargs can grow it.
_MAX_CLOSING_SUFFIXES = 256


class QuestionView(msgspec.Struct, frozen=True):
    """A question as the renderer and scorer see it, shared by the decision routes."""

    # choice, score, or yes_no
    kind: str
    # None or blank when the question has no text of its own
    question: Any
    # Option names, level indices, or yes and no, in candidate order
    names: List[str]
    # Option descriptions, levels, or the yes and no descriptions
    details: List[Any]


class OpenAIServingDecisions(OpenAIServingBase):
    """Handler for /v1/decisions requests, answered by candidate scoring without generation"""

    # Named in setup refusals, so each decision route reports itself.
    route = "/v1/decisions"

    def __init__(self, chat_serving: OpenAIServingChat):
        super().__init__(chat_serving.tokenizer_manager)
        # Render the way the chat route does, and refuse where it renders differently.
        self.template_manager = chat_serving.template_manager
        self.default_chat_template_kwargs = chat_serving.default_chat_template_kwargs
        self.chat_encoding_spec = chat_serving.chat_encoding_spec
        self.prompt_text_is_lossy = chat_serving._prompt_text_round_trip_is_lossy
        tokenizer = self.tokenizer_manager.tokenizer
        # Other tokenizers skip the shortcut in encode_labels and check the full prompt.
        self.added_tokens = (
            {i: token for token, i in tokenizer.get_added_vocab().items()}
            if isinstance(tokenizer, PreTrainedTokenizerBase)
            else {}
        )
        # The configured reasoning parser, or the one the chat template suggests,
        # tells where reasoning blocks start and end and whether answers open one.
        parser = (
            chat_serving.reasoning_parser
            or self.template_manager.suggested_reasoning_parser
        )
        self.reasoning_markers = None
        self.answers_open_reasoning = False
        # The kwarg that turns reasoning on and off: the template's, else, when
        # detection finds no config, the one the configured or suggested parser names.
        config = self.template_manager.reasoning_config
        self.reasoning_toggle = config.toggle_param if config is not None else None
        # Such templates open a reasoning block in every generation prompt, so
        # answers are read after the empty block the template closes in its replies.
        self.always_reasons = config is not None and config.always_on
        # Text each generation prompt is extended by to close an empty reasoning
        # block, by closing line and chat template kwargs.
        self._closing_suffixes: Dict[str, Optional[str]] = {}
        if parser is not None:
            try:
                detector = ReasoningParser(
                    model_type=parser, tokenizer=tokenizer
                ).detector
            except ValueError as e:
                logger.warning(
                    "No reasoning block check for %s with parser '%s': %s",
                    self.route,
                    parser,
                    e,
                )
            else:
                mode = detector.reasoning_default
                if config is None and mode in _PARSER_TOGGLE_MODES:
                    self.reasoning_toggle = mode.removeprefix("explicit_")
                if detector.think_start_token and detector.think_end_token:
                    self.reasoning_markers = (
                        detector.think_start_token,
                        detector.think_end_token,
                    )
                    self.answers_open_reasoning = detector.reasoning_default == "always"

    def _request_id_prefix(self) -> str:
        return "decision-"

    def _validate_request(self, request: DecisionRequest) -> Optional[str]:
        error = self._validate_server(request.model)
        if error is not None:
            return error
        version = request.prompt_format_version
        if version is not None and version != PROMPT_FORMAT_VERSION:
            return (
                f"prompt_format_version {version} is not served, this server "
                f"uses version {PROMPT_FORMAT_VERSION}"
            )
        return self._validate_reasoning(request.chat_template_kwargs)

    def _validate_server(self, model: str) -> Optional[str]:
        """Refuse servers this route cannot render faithfully."""
        route = self.route
        if not self.tokenizer_manager.is_generation:
            return f"{route} requires a generation model"
        if self.tokenizer_manager.tokenizer is None:
            return f"{route} requires the server tokenizer"
        if self.chat_encoding_spec is not None:
            return (
                f"{route} requires a chat template, but this model's chat "
                f"route uses the {self.chat_encoding_spec!r} encoder"
            )
        if self.prompt_text_is_lossy:
            return (
                f"{route} places answer labels on the rendered chat text, "
                "which this tokenizer does not encode back to the same ids"
            )
        if self.template_manager.chat_template_name is not None:
            return (
                f"{route} renders the tokenizer's Jinja chat template, but "
                "this server uses the built-in chat template "
                f"{self.template_manager.chat_template_name!r}"
            )
        if get_exec().features.enable_mis:
            return f"{route} does not support --enable-mis"
        if get_exec().dllm.dllm_algorithm is not None:
            return (
                f"{route} does not support diffusion language models "
                "served with --dllm-algorithm"
            )
        _, adapter = self._parse_model_parameter(model)
        if adapter is not None:
            return (
                f"model names the LoRA adapter {adapter!r}, which {route} "
                "does not support"
            )
        return None

    def _validate_reasoning(
        self, chat_template_kwargs: Dict[str, Any]
    ) -> Optional[str]:
        """The answer position must follow the reasoning block, not sit inside it."""
        toggle = self.reasoning_toggle
        if toggle in chat_template_kwargs and chat_template_kwargs[toggle] is not False:
            return (
                f"chat_template_kwargs sets {toggle!r} to "
                f"{chat_template_kwargs[toggle]!r}, but decisions need it false or unset"
            )
        return None

    def _chat_template_kwargs(self, request_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Reasoning off, then the server defaults, then the request kwargs."""
        kwargs = {}
        if self.reasoning_toggle is not None:
            kwargs[self.reasoning_toggle] = False
        for key, value in self.default_chat_template_kwargs.items():
            kwargs.setdefault(key, value)
        kwargs.update(request_kwargs)
        return kwargs

    def _convert_to_internal_request(
        self,
        request: DecisionRequest,
        raw_request: Request = None,
    ) -> Tuple[Iterator[Tuple[List[int], List[int]]], DecisionRequest]:
        # Lazy, so the handler can let other requests run between questions.
        return self._encoded_questions(request), request

    def _encoded_questions(
        self, request: DecisionRequest
    ) -> Iterator[Tuple[List[int], List[int]]]:
        """Prompt and label ids for each question, in request order."""
        text = render_text(request.input)
        chat_template_kwargs = self._chat_template_kwargs(request.chat_template_kwargs)
        for question in request.questions:
            view = _decision_view(question)
            try:
                encoded = self._encode_question(
                    text=text,
                    view=view,
                    labels=default_labels(view),
                    chat_template_kwargs=chat_template_kwargs,
                )
            except ValueError as e:
                raise ValueError(f"question {question.id!r}: {e}") from e
            yield encoded

    def _encode_question(
        self,
        text: str,
        view: QuestionView,
        labels: List[str],
        chat_template_kwargs: Dict[str, Any],
    ) -> Tuple[List[int], List[int]]:
        content = render_question(text=text, view=view, labels=labels)
        prompt, prompt_ids = self._encode_prompt(content, chat_template_kwargs)
        label_ids = encode_labels(
            tokenizer=self.tokenizer_manager.tokenizer,
            prompt=prompt,
            prompt_ids=prompt_ids,
            labels=labels,
            added_tokens=self.added_tokens,
        )
        return prompt_ids, label_ids

    def _encode_prompt(
        self, content: str, chat_template_kwargs: Dict[str, Any]
    ) -> Tuple[str, List[int]]:
        """Answer prompt of one message, as text and ids, that fits the context."""
        prompt = self._answer_prompt(content, chat_template_kwargs)
        prompt_ids = self.tokenizer_manager.tokenizer.encode(
            prompt, add_special_tokens=False
        )
        # Refuse here because --allow-auto-truncate would cut off the answer position.
        context_len = self.tokenizer_manager.context_len
        if len(prompt_ids) + self.tokenizer_manager.num_reserved_tokens >= context_len:
            raise ValueError(
                f"the prompt has {len(prompt_ids)} tokens, which does not fit "
                f"the context length of {context_len} tokens"
            )
        return prompt, prompt_ids

    def _answer_prompt(self, content: str, chat_template_kwargs: Dict[str, Any]) -> str:
        """Chat text ending at the answer position: the generation prompt, else that
        prompt extended to the empty reasoning block the template's own reply closes."""
        prompt = self._apply_chat_template(content, chat_template_kwargs)
        # Look only after the message, whose last line is fixed text.
        closing = content.rsplit("\n", 1)[-1]
        try:
            self._check_generation_prompt(prompt, closing, chat_template_kwargs)
        except ValueError as refusal:
            if self.reasoning_markers is None:
                raise
            suffix = self._closing_suffix(closing, chat_template_kwargs)
            if suffix is None:
                raise ValueError(
                    f"{refusal}, and the chat template's own reply without "
                    "reasoning does not close an empty reasoning block before "
                    "the answer"
                ) from refusal
            return prompt + suffix
        return prompt

    def _check_generation_prompt(
        self, prompt: str, closing: str, chat_template_kwargs: Dict[str, Any]
    ) -> None:
        """Refuse a generation prompt whose answer would start inside reasoning."""
        if self.always_reasons:
            raise ValueError(
                f"{self.route} does not support chat templates that always "
                "reason before answering"
            )
        if self.reasoning_markers is None:
            return
        cut = prompt.rfind(closing)
        generation_prompt = prompt if cut < 0 else prompt[cut + len(closing) :]
        start, end = self.reasoning_markers
        opened = generation_prompt.rfind(start)
        closed = generation_prompt.rfind(end)
        if opened > closed:
            raise ValueError(
                "the chat template leaves a reasoning block open at the "
                "answer position, so this model is not supported with these "
                "chat_template_kwargs"
            )
        if self.answers_open_reasoning and closed < 0:
            raise ValueError(
                "the reasoning parser for this model expects answers to start "
                "with a reasoning block, and the chat template does not close "
                "one. Send chat_template_kwargs that turn thinking off, if the "
                "template supports it"
            )
        # The template's own finished reply shows whether answers start with
        # a reasoning block that the generation prompt leaves out.
        reply = self._render_reply(
            closing,
            chat_template_kwargs,
            {"role": "assistant", "content": _REPLY_SENTINEL},
        )
        begin = reply.rfind(closing) if reply is not None else -1
        answer = reply.find(_REPLY_SENTINEL, begin) if begin >= 0 else -1
        if answer >= 0 and reply[begin:answer].count(start) > (
            generation_prompt.count(start)
        ):
            raise ValueError(
                "the chat template starts every answer with a reasoning "
                "block, so this model is not supported"
            )

    def _closing_suffix(
        self, closing: str, chat_template_kwargs: Dict[str, Any]
    ) -> Optional[str]:
        """Cached _find_closing_suffix, since it depends only on its arguments."""
        key = "\0".join(
            (closing, json.dumps(chat_template_kwargs, sort_keys=True, default=repr))
        )
        if key not in self._closing_suffixes:
            # Request kwargs are unbounded, the closing lines are not.
            if len(self._closing_suffixes) >= _MAX_CLOSING_SUFFIXES:
                self._closing_suffixes.clear()
            self._closing_suffixes[key] = self._find_closing_suffix(
                closing, chat_template_kwargs
            )
        return self._closing_suffixes[key]

    def _find_closing_suffix(
        self, closing: str, chat_template_kwargs: Dict[str, Any]
    ) -> Optional[str]:
        """What a reply with empty reasoning adds to the generation prompt before its
        answer, or None unless it continues that prompt and closes one empty block."""
        try:
            generation = self._apply_chat_template(closing, chat_template_kwargs)
        except ValueError:
            return None
        for field in _REASONING_FIELDS:
            reply = self._render_reply(
                closing,
                chat_template_kwargs,
                {"role": "assistant", "content": _REPLY_SENTINEL, field: ""},
            )
            suffix = self._empty_block_suffix(closing, generation, reply)
            if suffix is not None:
                return suffix
        return None

    def _empty_block_suffix(
        self, closing: str, generation: str, reply: Optional[str]
    ) -> Optional[str]:
        """What the reply adds to the generation prompt, if it closes one empty block."""
        if reply is None:
            return None
        start, end = self.reasoning_markers
        generation_cut = generation.rfind(closing)
        reply_cut = reply.rfind(closing)
        if (
            generation_cut < 0
            or reply_cut < 0
            or generation[:generation_cut] != reply[:reply_cut]
        ):
            return None
        answer = reply.find(_REPLY_SENTINEL, reply_cut + len(closing))
        if answer < 0:
            return None
        generation_tail = generation[generation_cut + len(closing) :]
        reply_tail = reply[reply_cut + len(closing) : answer]
        if not reply_tail.startswith(generation_tail):
            return None
        opened = reply_tail.rfind(start)
        closed = reply_tail.rfind(end)
        if opened < 0 or closed < opened:
            return None
        if (
            reply_tail[opened + len(start) : closed].strip()
            or reply_tail[closed + len(end) :].strip()
        ):
            return None
        return reply_tail[len(generation_tail) :]

    def _apply_chat_template(
        self, content: str, chat_template_kwargs: Dict[str, Any]
    ) -> str:
        try:
            return self.tokenizer_manager.tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
                **chat_template_kwargs,
            )
        except _CHAT_TEMPLATE_CLIENT_ERRORS as e:
            raise ValueError(f"the chat template failed: {e}") from e

    def _render_reply(
        self,
        message: str,
        chat_template_kwargs: Dict[str, Any],
        reply: Dict[str, str],
    ) -> Optional[str]:
        """The message and a finished reply, or None when the template refuses them."""
        try:
            return self.tokenizer_manager.tokenizer.apply_chat_template(
                [{"role": "user", "content": message}, reply],
                tokenize=False,
                **chat_template_kwargs,
            )
        except _CHAT_TEMPLATE_CLIENT_ERRORS:
            # Callers treat a refused reply as showing nothing about the answer.
            return None

    async def _score(
        self,
        adapted_request: Iterator[Tuple[List[int], List[int]]],
        raw_request: Request,
        temperature: float = 1.0,
    ):
        """Encode every question, then score them all in one call."""
        prompts, label_token_ids = await encode_all(adapted_request)
        result = await self._score_prompts(
            prompts=prompts,
            label_token_ids=label_token_ids,
            raw_request=raw_request,
            temperature=temperature,
        )
        return prompts, label_token_ids, result

    async def _score_prompts(
        self,
        prompts: List[List[int]],
        label_token_ids: List[List[int]],
        raw_request: Request,
        temperature: float,
    ):
        """Score encoded prompts in one call, with full-vocabulary label logprobs."""
        await self._prime_shared_prefix(prompts, raw_request)
        return await self.tokenizer_manager.score_prompts(
            prompts=prompts,
            label_token_ids=label_token_ids,
            apply_softmax=True,
            request=raw_request,
            temperature=temperature,
            return_token_logprobs=True,
        )

    async def _prime_shared_prefix(
        self, prompts: List[List[int]], raw_request: Request
    ) -> None:
        """Prefill the prefix all prompts share once, so the scoring batch reads it
        from the radix cache instead of every prompt prefilling it again."""
        if len(prompts) < 2 or get_memory().disable_radix_cache:
            return
        prefix = shared_prefix(prompts)
        if len(prefix) < envs.SGLANG_DECISION_PREFIX_PRIME_MIN_TOKENS.get():
            return
        # Scoring a label at the end of the prefix is the cheapest prefill-only request.
        await self.tokenizer_manager.score_prompts(
            prompts=[prefix],
            label_token_ids=[[prefix[-1]]],
            apply_softmax=False,
            request=raw_request,
        )

    async def _handle_non_streaming_request(
        self,
        adapted_request: Iterator[Tuple[List[int], List[int]]],
        request: DecisionRequest,
        raw_request: Request,
    ) -> ORJSONResponse:
        prompts, label_token_ids, result = await self._score(
            adapted_request=adapted_request,
            raw_request=raw_request,
            temperature=request.temperature,
        )
        answers = {}
        for i, question in enumerate(request.questions):
            answer = _build_answer(
                question=question,
                view=_decision_view(question),
                probabilities=result.scores[i],
                token_logprobs=result.token_logprobs[i],
            )
            if request.return_prompt_token_ids:
                answer.prompt_token_ids = prompts[i]
                answer.label_token_ids = label_token_ids[i]
            answers[question.id] = answer
        response = DecisionResponse(
            model=request.model,
            prompt_format_version=PROMPT_FORMAT_VERSION,
            answers=answers,
            usage=UsageInfo(
                prompt_tokens=result.prompt_tokens,
                total_tokens=result.prompt_tokens,
            ),
        )
        return ORJSONResponse(content=response.model_dump(exclude_none=True))


def render_text(value: Optional[DecisionText]) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def shared_prefix(prompts: List[List[int]]) -> List[int]:
    """The longest token prefix every prompt starts with."""
    shortest = min(prompts, key=len)
    length = len(shortest)
    for prompt in prompts:
        length = next((i for i in range(length) if prompt[i] != shortest[i]), length)
    return shortest[:length]


async def encode_all(
    encoded: Iterator[Tuple[List[int], List[int]]],
) -> Tuple[List[List[int]], List[List[int]]]:
    """Collect prompt and label ids, letting other requests run between questions."""
    prompts, label_token_ids = [], []
    for prompt_ids, label_ids in encoded:
        prompts.append(prompt_ids)
        label_token_ids.append(label_ids)
        # Each question renders and tokenizes the whole input on the event loop.
        await asyncio.sleep(0)
    return prompts, label_token_ids


def _decision_view(question: DecisionQuestion) -> QuestionView:
    if isinstance(question, DecisionChoiceQuestion):
        return QuestionView(
            kind="choice",
            question=question.question,
            names=[option.name for option in question.options],
            details=[option.description for option in question.options],
        )
    if isinstance(question, DecisionScoreQuestion):
        return QuestionView(
            kind="score",
            question=question.question,
            names=[str(level) for level in range(len(question.levels))],
            details=list(question.levels),
        )
    return QuestionView(
        kind="yes_no",
        question=question.question,
        names=["yes", "no"],
        details=[question.yes, question.no],
    )


def default_labels(view: QuestionView) -> List[str]:
    """Single-token labels in candidate order: A to Z, level indices, or yes and no."""
    if view.kind == "choice":
        return list(string.ascii_uppercase[: len(view.names)])
    return list(view.names)


def render_question(text: str, view: QuestionView, labels: List[str]) -> str:
    """Prompt wording of PROMPT_FORMAT_VERSION."""
    # Every /v1/decisions question has text. A question without its own text
    # drops the question line, and a yes or no question keeps its lead in.
    question_text = (
        "" if is_blank_decision_text(view.question) else render_text(view.question)
    )
    if view.kind == "choice":
        lines = [f"Question: {question_text}"] if question_text else []
        for label, name, description in zip(labels, view.names, view.details):
            detail = render_text(description)
            lines.append(
                f"{label}: {name} - {detail}" if detail else f"{label}: {name}"
            )
        lines.append("Answer with the letter of one option only.")
    elif view.kind == "score":
        lines = [f"Question: {question_text}"] if question_text else []
        lines += [
            f"{label}: {render_text(level)}"
            for label, level in zip(labels, view.details)
        ]
        lines.append("Answer with the number of one level only.")
    else:
        lines = [
            f"Is the following true? {question_text}"
            if question_text
            else "Is the following true?"
        ]
        for label, description in zip(labels, view.details):
            detail = render_text(description)
            if detail:
                lines.append(f"{label}: {detail}")
        # Labels come in the order shown, which a calibration read can reverse.
        lines.append(f"Answer with {labels[0]} or {labels[1]} only.")
    return "\n".join([text, "", *lines])


def label_context(
    tokenizer: Any,
    prompt: str,
    prompt_ids: List[int],
    added_tokens: Dict[int, str],
) -> Tuple[str, List[int], bool]:
    """Text and ids after which labels are checked, and whether the shortcut applies."""
    # Added tokens are split off before tokenization.
    # The text after the last one tokenizes on its own,
    # so the check does not grow with the input.
    # When the prompt ends with an added token, each label starts a new segment,
    # which is how the model continues after that token.
    last = next(
        (i for i in reversed(range(len(prompt_ids))) if prompt_ids[i] in added_tokens),
        None,
    )
    if last is not None:
        token = added_tokens[prompt_ids[last]]
        start = prompt.rfind(token)
        suffix = prompt[start + len(token) :]
        if start >= 0 and (
            tokenizer.encode(suffix, add_special_tokens=False) == prompt_ids[last + 1 :]
        ):
            return suffix, prompt_ids[last + 1 :], True
    return prompt, prompt_ids, False


def label_token_id(
    tokenizer: Any, text: str, text_ids: List[int], label: str
) -> Optional[int]:
    """The token a label adds after the text, or None when it is not exactly one."""
    ids = tokenizer.encode(text + label, add_special_tokens=False)
    if len(ids) != len(text_ids) + 1 or ids[:-1] != text_ids:
        return None
    return ids[-1]


def first_token_id(
    tokenizer: Any, text: str, text_ids: List[int], continuation: str
) -> Optional[int]:
    """The first token a continuation adds after the text, or None when it adds none."""
    ids = tokenizer.encode(text + continuation, add_special_tokens=False)
    if len(ids) <= len(text_ids) or ids[: len(text_ids)] != text_ids:
        return None
    return ids[len(text_ids)]


def encode_labels(
    tokenizer: Any,
    prompt: str,
    prompt_ids: List[int],
    labels: List[str],
    added_tokens: Dict[int, str],
) -> List[int]:
    """Check that each label adds exactly one distinct token after the prompt."""
    text, text_ids, _ = label_context(tokenizer, prompt, prompt_ids, added_tokens)
    label_ids = []
    for label in labels:
        token_id = label_token_id(tokenizer, text, text_ids, label)
        if token_id is None or token_id in label_ids:
            raise ValueError(
                f"the answer label {label!r} is not one distinct token after the "
                "chat prompt for this tokenizer, so this model is not supported"
            )
        label_ids.append(token_id)
    return label_ids


def label_mass(token_logprobs: List[float]) -> float:
    return math.fsum(math.exp(logprob) for logprob in token_logprobs)


def _build_answer(
    question: DecisionQuestion,
    view: QuestionView,
    probabilities: List[float],
    token_logprobs: List[float],
) -> DecisionAnswer:
    names = view.names
    value = {}
    if view.kind == "choice":
        value["choice"] = names[probabilities.index(max(probabilities))]
    elif view.kind == "score":
        value["score"] = math.fsum(i * p for i, p in enumerate(probabilities))
    return DecisionAnswer(
        type=question.type,
        probabilities=dict(zip(names, probabilities)),
        label_mass=label_mass(token_logprobs),
        **value,
    )
