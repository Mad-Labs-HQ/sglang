import json
import logging
import math
import re
from typing import Any, List, Optional

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.utils import (
    get_schema_properties,
    infer_type_from_json_schema,
    safe_literal_eval,
)

logger = logging.getLogger(__name__)


def _reject_non_rfc_constant(literal: str):
    raise ValueError(f"non-RFC-8259 literal: {literal}")


def _reject_non_finite_number(literal: str) -> float:
    number = float(literal)
    if not math.isfinite(number):
        # float("1e400") is inf, and json.dumps would then write a bare
        # `Infinity` that no RFC-8259 parser accepts.
        raise ValueError(f"non-finite number: {literal}")
    return number


# Python's json accepts NaN/Infinity and happily writes them back out; nothing
# else does. Anything we re-serialize has to survive a strict client, so parse
# envelope payloads with a decoder that refuses them.
_STRICT_JSON = json.JSONDecoder(
    parse_constant=_reject_non_rfc_constant,
    parse_float=_reject_non_finite_number,
)


class Qwen3CoderDetector(BaseFormatDetector):
    def __init__(self):
        super().__init__()

        # Sentinel tokens
        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"
        self.tool_call_prefix: str = "<function="
        self.function_end_token: str = "</function>"
        self.parameter_prefix: str = "<parameter="
        self.parameter_end_token: str = "</parameter>"

        # Regex for non-streaming fallback
        self.tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
        self.tool_call_function_regex = re.compile(
            r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL
        )
        self.tool_call_parameter_regex = re.compile(
            r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)",
            re.DOTALL,
        )

        # Streaming State
        # Base class already initializes _buffer, we just use it directly
        # No need to check with hasattr - we control the lifecycle through inheritance

        # Index pointing to the next character to be processed in buffer
        self.parsed_pos: int = 0
        # Parameter count inside the current tool being processed, used to determine whether to add comma
        self.current_tool_param_count: int = 0
        # Flag indicating whether current tool has already sent '{'
        self.json_started: bool = False

        # [FIX] New state flag: mark whether inside tool_call structure block
        self.is_inside_tool_call: bool = False

        # Initialize attributes that were missing in the original PR
        self.current_func_name: Optional[str] = None

        # Incremental parameter streaming (vendored PR #21829, adapted).
        # A long string value is emitted in pieces as it arrives instead of
        # being withheld until </parameter>, so a tool call that authors a file
        # does not land as one giant delta after seconds of dead air.
        self._streaming_param_active: bool = False
        self._streaming_param_emitted: int = 0
        self._streaming_param_leading_checked: bool = False

    def has_tool_call(self, text: str) -> bool:
        return self.tool_call_start_token in text

    def _get_arguments_config(
        self, func_name: str, tools: Optional[list[Tool]]
    ) -> dict:
        """Extract argument configuration for a function."""
        if tools is None:
            return {}
        for config in tools:
            try:
                config_type = config.type
                config_function = config.function
                config_function_name = config_function.name
            except AttributeError:
                continue

            if config_type == "function" and config_function_name == func_name:
                try:
                    params = config_function.parameters
                except AttributeError:
                    return {}

                if isinstance(params, dict):
                    properties = get_schema_properties(params)
                    if properties or "properties" in params:
                        return properties
                    return params
                else:
                    return {}
        logger.warning(f"Tool '{func_name}' is not defined in the tools list.")
        return {}

    def _get_param_type(self, param_schema: Any) -> str:
        """Infer the parser conversion type from a JSON schema parameter."""
        inferred_type = infer_type_from_json_schema(param_schema)
        if inferred_type is None:
            return "string"
        return str(inferred_type).strip().lower()

    def _convert_param_value(
        self, param_value: str, param_name: str, param_config: dict, func_name: str
    ) -> Any:
        """Convert parameter value based on its type in the schema."""
        # Handle null value for any type
        if param_value.lower() == "null":
            return None

        if param_name not in param_config:
            if param_config != {}:
                logger.warning(
                    f"Parsed parameter '{param_name}' is not defined in the tool "
                    f"parameters for tool '{func_name}', directly returning the string value."
                )
            return param_value

        param_type = self._get_param_type(param_config[param_name])
        if param_type in ["string", "str", "text", "varchar", "char", "enum"]:
            return param_value
        elif (
            param_type.startswith("int")
            or param_type.startswith("uint")
            or param_type.startswith("long")
            or param_type.startswith("short")
            or param_type.startswith("unsigned")
        ):
            try:
                param_value = int(param_value)
            except Exception:
                logger.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' is not an integer in tool "
                    f"'{func_name}', degenerating to string."
                )
            return param_value
        elif param_type.startswith("num") or param_type.startswith("float"):
            try:
                maybe_convert = (
                    False if "." in param_value or "e" in param_value.lower() else True
                )
                param_value: float = float(param_value)
                if maybe_convert and param_value.is_integer():
                    param_value = int(param_value)
            except Exception:
                logger.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' is not a float in tool "
                    f"'{func_name}', degenerating to string."
                )
            return param_value
        elif param_type in ["boolean", "bool", "binary"]:
            param_value = param_value.lower()
            if param_value not in ["true", "false"]:
                logger.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' is not a boolean (`true` of `false`) in tool '{func_name}', degenerating to false."
                )
            return param_value == "true"
        else:
            if (
                param_type in ["object", "array", "arr"]
                or param_type.startswith("dict")
                or param_type.startswith("list")
            ):
                try:
                    param_value = json.loads(param_value)
                    return param_value
                except Exception:
                    logger.warning(
                        f"Parsed value '{param_value}' of parameter '{param_name}' cannot be parsed with json.loads in tool "
                        f"'{func_name}', will try other methods to parse it."
                    )
            try:
                param_value = safe_literal_eval(param_value)
            except Exception:
                logger.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' cannot be converted via Python `ast.literal_eval()` in tool '{func_name}', degenerating to string."
                )
            return param_value

    def _should_stream_param(
        self, param_name: str, tools: Optional[List[Tool]]
    ) -> bool:
        """Whether this parameter's value may be emitted incrementally.

        Streaming is opt-in and deliberately narrow:

        * The tool must publish a `properties` map and declare this parameter.
          Upstream PR #21829 streams UNDECLARED parameters too ("treat as
          string"), which is wrong here in two ways. It would stream the stray
          `arguments` envelope this model emits -- and once `"arguments": "` is
          on the wire, _unwrap_arguments_envelope can never repair it, because
          the streaming path cannot retract what it has already sent. It would
          also stream a parameter whose type we have no way to check.
        * The declared type must be string-like. JSON encoding of numbers,
          booleans, objects and arrays is not prefix-stable under
          re-serialization, so only strings can be cut into pieces safely.

        Anything else buffers to completion and takes the existing path.
        """
        properties = self._get_declared_properties(self.current_func_name, tools)
        if not properties or param_name not in properties:
            return False
        return self._get_param_type(properties[param_name]) in (
            "string",
            "str",
            "text",
            "varchar",
            "char",
            "enum",
        )

    def _find_safe_emit_end(self, text: str) -> int:
        """Rightmost position that can be emitted without splitting a tag.

        A chunk boundary can land inside `</parameter>`, and emitting the
        partial `</par` would put literal markup into the argument value. Stop
        at the last `<` whenever what follows it is a prefix of any tag we care
        about, and wait for the rest.
        """
        if not text:
            return 0
        last_angle = text.rfind("<")
        if last_angle == -1:
            return len(text)
        suffix = text[last_angle:]
        for tag in (
            self.parameter_end_token,
            self.parameter_prefix,
            self.function_end_token,
            self.tool_call_start_token,
            self.tool_call_end_token,
            self.tool_call_prefix,
        ):
            if tag.startswith(suffix):
                return last_angle
        return len(text)

    def _reset_streaming_param(self) -> None:
        self._streaming_param_active = False
        self._streaming_param_emitted = 0
        self._streaming_param_leading_checked = False

    def _get_declared_properties(
        self, func_name: str, tools: Optional[List[Tool]]
    ) -> Optional[dict]:
        """Return the tool's declared `properties` map, or None if it has none.

        Unlike _get_arguments_config this never falls back to the schema object
        itself. A schema that routes through $ref/$defs publishes no property
        map here, and treating the schema as one would make `type`, `required`
        and `$ref` look like argument names.

        Properties are resolved with get_schema_properties, the same helper
        _get_arguments_config uses, so this agrees with the property map that
        types parameters (including one declared only inside a top-level
        anyOf/oneOf/allOf branch) -- minus its fallback to the schema itself.
        """
        if not tools:
            return None
        for tool in tools:
            try:
                if tool.type != "function" or tool.function.name != func_name:
                    continue
                params = tool.function.parameters
            except AttributeError:
                continue
            properties = get_schema_properties(params)
            return properties or None
        return None

    def _unwrap_arguments_envelope(
        self, raw_value: str, func_name: str, tools: Optional[List[Tool]]
    ) -> Optional[dict]:
        """Unwrap a call whose real arguments arrived inside a bogus `arguments`
        parameter.

        Qwen3.8-Flash-Next drifts back to its native JSON tool-call payload while
        still emitting the XML envelope, producing

            <function=script><parameter=arguments>{"body": "..."}</parameter>

        instead of <parameter=body>.... The call then reaches the client as
        {"arguments": "{\\"body\\": ...}"} and any strict client rejects it with
        `missing field body`. Measured at 353 calls in one agent session
        (2026-09-14), i.e. the majority of that session's tool calls.

        Unwrap only when the evidence is unambiguous: the tool publishes a
        `properties` map, does not itself declare `arguments`, and the value is
        a JSON object whose keys are ALL declared. Anything else returns None
        and keeps today's behaviour.

        Values are passed through as-is: the model already typed them in JSON,
        and re-running _convert_param_value would, among other things, turn the
        literal string "null" into None.
        """
        properties = self._get_declared_properties(func_name, tools)
        if not properties or "arguments" in properties:
            return None
        try:
            payload = _STRICT_JSON.decode(raw_value)
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        if not all(key in properties for key in payload):
            return None
        # WARNING, not INFO: the rate of this is the only visible measure of the
        # model losing tool-call format, and it climbed from 1.4% to 40% of
        # requests over four hours on 2026-09-14. Repairing it silently would
        # retire the signal along with the symptom.
        logger.warning(
            "Unwrapped a stray `arguments` envelope for tool '%s' "
            "(%d bytes, keys: %s).",
            func_name,
            len(raw_value),
            ", ".join(sorted(payload)) or "<none>",
        )
        return payload

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """One-shot parsing for non-streaming scenarios."""
        if self.tool_call_start_token not in text:
            return StreamingParseResult(normal_text=text)

        calls = []
        try:
            # Simple cleanup of the text to find tool calls
            # Note: This is a simplified regex approach consistent with vLLM
            raw_tool_calls = self.tool_call_regex.findall(text)
            if not raw_tool_calls:
                # Fallback: maybe the whole text is inside the tag or tags are stripped
                if self.tool_call_prefix in text:
                    raw_tool_calls = [text]

            tool_idx = 0
            for tool_content in raw_tool_calls:
                # Find function calls
                funcs = self.tool_call_function_regex.findall(tool_content)
                for func_match in funcs:
                    func_body = func_match[0] or func_match[1]
                    if ">" not in func_body:
                        continue

                    name_end = func_body.index(">")
                    func_name = func_body[:name_end]
                    params_str = func_body[name_end + 1 :]

                    param_config = self._get_arguments_config(func_name, tools)
                    parsed_params = {}

                    for p_match in self.tool_call_parameter_regex.findall(params_str):
                        if ">" not in p_match:
                            continue
                        p_idx = p_match.index(">")
                        p_name = p_match[:p_idx]
                        p_val = p_match[p_idx + 1 :]
                        # Remove prefixing and trailing \n
                        if p_val.startswith("\n"):
                            p_val = p_val[1:]
                        if p_val.endswith("\n"):
                            p_val = p_val[:-1]

                        if p_name == "arguments":
                            envelope = self._unwrap_arguments_envelope(
                                p_val, func_name, tools
                            )
                            if envelope is not None:
                                # Splice the envelope's keys in where
                                # `arguments` stood, so a real parameter of the
                                # same name still wins -- which is what the
                                # streaming path produces, since it emits each
                                # parameter as it closes and a later duplicate
                                # key overrides an earlier one.
                                parsed_params.update(envelope)
                                continue

                        parsed_params[p_name] = self._convert_param_value(
                            p_val, p_name, param_config, func_name
                        )

                    calls.append(
                        ToolCallItem(
                            tool_index=tool_idx,
                            name=func_name,
                            parameters=json.dumps(parsed_params, ensure_ascii=False),
                        )
                    )
                    tool_idx += 1

            # Determine normal text (text before the first tool call)
            start_idx = text.find(self.tool_call_start_token)
            if start_idx == -1:
                start_idx = text.find(self.tool_call_prefix)
            normal_text = text[:start_idx] if start_idx > 0 else ""

            return StreamingParseResult(normal_text=normal_text, calls=calls)

        except Exception as e:
            logger.error(f"Error in detect_and_parse: {e}")
            return StreamingParseResult(normal_text=text)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Robust cursor-based streaming parser.
        """
        self._buffer += new_text

        # Guard against empty buffer
        if not self._buffer:
            return StreamingParseResult()

        calls = []
        normal_text_chunks = []

        while True:
            # Working text slice
            current_slice = self._buffer[self.parsed_pos :]

            # Optimization: If almost empty, wait for more
            if not current_slice:
                break

            # -------------------------------------------------------
            # 1. Priority detection: check if it's the start of Tool Call
            # -------------------------------------------------------
            if current_slice.startswith(self.tool_call_start_token):
                self.parsed_pos += len(self.tool_call_start_token)
                self.is_inside_tool_call = True
                continue

            # -------------------------------------------------------
            # 2. Function Name: <function=name>
            # -------------------------------------------------------
            if current_slice.startswith(self.tool_call_prefix):
                end_angle = current_slice.find(">")
                if end_angle != -1:
                    func_name = current_slice[len(self.tool_call_prefix) : end_angle]

                    self.current_tool_id += 1
                    self.current_tool_name_sent = True
                    self.current_tool_param_count = 0
                    self.json_started = False
                    self._reset_streaming_param()
                    self.current_func_name = func_name

                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=func_name,
                            parameters="",
                        )
                    )

                    self.parsed_pos += end_angle + 1
                    continue
                else:
                    # Incomplete tag
                    break

            # -------------------------------------------------------
            # 3. Parameter: <parameter=name>value...
            # -------------------------------------------------------
            if current_slice.startswith(self.parameter_prefix):
                name_end = current_slice.find(">")
                if name_end != -1:
                    value_start_idx = name_end + 1
                    rest_of_slice = current_slice[value_start_idx:]

                    # A parameter can end in multiple ways:
                    # 1. [Normal] Encounter </parameter>
                    # 2. [Abnormal] Encounter next <parameter=
                    # 3. [Abnormal] Encounter </function>
                    # So we need to find the smallest one as the parameter end position.
                    cand_end_param = rest_of_slice.find(self.parameter_end_token)
                    cand_next_param = rest_of_slice.find(self.parameter_prefix)
                    cand_end_func = rest_of_slice.find(self.function_end_token)

                    candidates = []
                    if cand_end_param != -1:
                        candidates.append(
                            (cand_end_param, len(self.parameter_end_token))
                        )
                    if cand_next_param != -1:
                        candidates.append((cand_next_param, 0))
                    if cand_end_func != -1:
                        candidates.append((cand_end_func, 0))

                    if candidates:
                        best_cand = min(candidates, key=lambda x: x[0])
                        end_pos = best_cand[0]
                        end_token_len = best_cand[1]

                        if self._streaming_param_active:
                            # This parameter was already going out in pieces:
                            # emit whatever is left and close the JSON string.
                            # The key and the opening quote are long gone, so
                            # there is nothing to reconsider here.
                            remaining = rest_of_slice[
                                self._streaming_param_emitted : end_pos
                            ]
                            if remaining.endswith("\n"):
                                remaining = remaining[:-1]
                            if remaining:
                                calls.append(
                                    ToolCallItem(
                                        tool_index=self.current_tool_id,
                                        parameters=json.dumps(
                                            remaining, ensure_ascii=False
                                        )[1:-1],
                                    )
                                )
                            calls.append(
                                ToolCallItem(
                                    tool_index=self.current_tool_id,
                                    parameters='"',
                                )
                            )
                            self.current_tool_param_count += 1
                            self._reset_streaming_param()
                            self.parsed_pos += value_start_idx + end_pos + end_token_len
                            continue

                        param_name = current_slice[
                            len(self.parameter_prefix) : name_end
                        ]
                        raw_value = rest_of_slice[:end_pos]

                        # Cleanup value
                        if raw_value.startswith("\n"):
                            raw_value = raw_value[1:]
                        if raw_value.endswith("\n"):
                            raw_value = raw_value[:-1]

                        # JSON Construction
                        if not self.json_started:
                            calls.append(
                                ToolCallItem(
                                    tool_index=self.current_tool_id, parameters="{"
                                )
                            )
                            self.json_started = True

                        # A stray <parameter=arguments> envelope has to be
                        # unwrapped here, not at the end: the fragment is
                        # emitted as soon as the parameter closes and is never
                        # revisited.
                        emitted = None
                        if param_name == "arguments":
                            emitted = self._unwrap_arguments_envelope(
                                raw_value, self.current_func_name, tools
                            )
                        if emitted is None:
                            param_config = self._get_arguments_config(
                                self.current_func_name, tools
                            )
                            emitted = {
                                param_name: self._convert_param_value(
                                    raw_value,
                                    param_name,
                                    param_config,
                                    self.current_func_name,
                                )
                            }

                        # Construct JSON fragment: "key": value
                        # Note: We must be careful with json.dumps to ensure valid JSON streaming
                        json_key_val = ", ".join(
                            f"{json.dumps(key)}: {json.dumps(value, ensure_ascii=False)}"
                            for key, value in emitted.items()
                        )

                        if self.current_tool_param_count > 0 and json_key_val:
                            fragment = f", {json_key_val}"
                        else:
                            fragment = json_key_val

                        calls.append(
                            ToolCallItem(
                                tool_index=self.current_tool_id, parameters=fragment
                            )
                        )
                        self.current_tool_param_count += len(emitted)

                        # Advance cursor
                        total_len = (name_end + 1) + end_pos + end_token_len
                        self.parsed_pos += total_len
                        continue

                    # No terminator yet. If this parameter is eligible, start
                    # (or continue) emitting its value incrementally instead of
                    # sitting on it until </parameter> arrives.
                    param_name = current_slice[len(self.parameter_prefix) : name_end]
                    if not self._streaming_param_active:
                        if not self._should_stream_param(param_name, tools):
                            break
                        self._streaming_param_active = True
                        self._streaming_param_emitted = 0
                        self._streaming_param_leading_checked = False
                        if not self.json_started:
                            calls.append(
                                ToolCallItem(
                                    tool_index=self.current_tool_id, parameters="{"
                                )
                            )
                            self.json_started = True
                        key_prefix = f'{json.dumps(param_name)}: "'
                        if self.current_tool_param_count > 0:
                            key_prefix = f", {key_prefix}"
                        calls.append(
                            ToolCallItem(
                                tool_index=self.current_tool_id, parameters=key_prefix
                            )
                        )

                    new_content = rest_of_slice[self._streaming_param_emitted :]
                    if not self._streaming_param_leading_checked and new_content:
                        # The template puts a newline after the opening tag.
                        if new_content[0] == "\n":
                            new_content = new_content[1:]
                            self._streaming_param_emitted += 1
                        self._streaming_param_leading_checked = True
                    if new_content:
                        safe_end = self._find_safe_emit_end(new_content)
                        # Hold back a trailing newline: it may be the format's
                        # own separator before </parameter>, which the buffered
                        # path strips. If it turns out to be real content, the
                        # next increment emits it.
                        if safe_end > 0 and new_content[safe_end - 1] == "\n":
                            safe_end -= 1
                        if safe_end > 0:
                            escaped = json.dumps(
                                new_content[:safe_end], ensure_ascii=False
                            )[1:-1]
                            if escaped:
                                calls.append(
                                    ToolCallItem(
                                        tool_index=self.current_tool_id,
                                        parameters=escaped,
                                    )
                                )
                            self._streaming_param_emitted += safe_end
                    break

                # Incomplete parameter tag or value
                break

            # -------------------------------------------------------
            # 4. Function End: </function>
            # -------------------------------------------------------
            if current_slice.startswith(self.function_end_token):
                if not self.json_started:
                    calls.append(
                        ToolCallItem(tool_index=self.current_tool_id, parameters="{")
                    )
                    self.json_started = True

                calls.append(
                    ToolCallItem(tool_index=self.current_tool_id, parameters="}")
                )
                self.parsed_pos += len(self.function_end_token)
                self.current_func_name = None
                continue

            # -------------------------------------------------------
            # 5. Tool Call End: </tool_call>
            # -------------------------------------------------------
            if current_slice.startswith(self.tool_call_end_token):
                self.parsed_pos += len(self.tool_call_end_token)
                self.is_inside_tool_call = False  # [FIX] Exit tool call region
                continue

            # -------------------------------------------------------
            # 6. Handling content / whitespace / normal text
            # -------------------------------------------------------
            # If current position is not the start of a tag (i.e., doesn't start with <), it might be plain text,
            # or a newline between two tags.
            # But we need to be careful not to output truncated tags like "<fun" as text.

            next_open_angle = current_slice.find("<")

            if next_open_angle == -1:
                # This entire segment is plain text
                if not self.is_inside_tool_call:
                    normal_text_chunks.append(current_slice)
                # [FIX] If inside tool call, discard this text (usually \n), don't append
                self.parsed_pos += len(current_slice)
                continue

            elif next_open_angle == 0:
                # Looks like a Tag, but doesn't match any known Tag above

                possible_tags = [
                    self.tool_call_start_token,
                    self.tool_call_end_token,
                    self.tool_call_prefix,
                    self.function_end_token,
                    self.parameter_prefix,
                    self.parameter_end_token,
                ]

                is_potential_tag = False
                for tag in possible_tags:
                    if tag.startswith(current_slice):
                        is_potential_tag = True
                        break

                if is_potential_tag:
                    break  # Wait for more
                else:
                    # Just a plain '<' symbol
                    if not self.is_inside_tool_call:
                        normal_text_chunks.append("<")
                    self.parsed_pos += 1
                    continue

            else:
                # '<' is in the middle
                text_segment = current_slice[:next_open_angle]
                if not self.is_inside_tool_call:
                    normal_text_chunks.append(text_segment)
                # [FIX] If inside tool call, discard whitespace/text before Tag
                self.parsed_pos += next_open_angle
                continue

        # Memory Cleanup: Slice the buffer
        # Keep unparsed part, discard parsed part
        if self.parsed_pos > 0:
            self._buffer = self._buffer[self.parsed_pos :]
            self.parsed_pos = 0

        normal_text = "".join(normal_text_chunks) if normal_text_chunks else ""
        return StreamingParseResult(calls=calls, normal_text=normal_text)

    def supports_structural_tag(self) -> bool:
        return True

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError

    def get_structural_tag_name(self) -> str:
        return "qwen_3_coder"
