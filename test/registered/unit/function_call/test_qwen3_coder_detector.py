"""Wrapper/name validation without losing rejected markup or later calls."""

import json

import pytest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.environ import envs
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

TOOLS = [
    Tool(
        function=Function(
            name="run_commands",
            parameters={
                "type": "object",
                "properties": {"command": {"type": "string"}},
            },
        )
    )
]


def function(name, value="echo hi"):
    return f"<function={name}><parameter=command>{value}</parameter></function>"


def wrapped(name, value="echo hi"):
    return f"<tool_call>\n{function(name, value)}\n</tool_call>"


def parse(text, width, forward=False):
    with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(forward):
        parser = FunctionCallParser(TOOLS, "qwen3_coder")
        if width is None:
            normal, calls = parser.parse_non_stream(text)
        else:
            normal, calls = "", []
            for i in range(0, len(text), width):
                content, increment = parser.parse_stream_chunk(text[i : i + width])
                normal += content
                calls.extend(increment)
            content, increment = parser.parse_stream_end()
            normal += content
            calls.extend(increment)
    reconstructed = {}
    for call in calls:
        if call.name:
            assert call.tool_index not in reconstructed
            reconstructed[call.tool_index] = {"name": call.name, "arguments": ""}
        assert call.tool_index in reconstructed, "argument fragment without a tool name"
        reconstructed[call.tool_index]["arguments"] += call.parameters
    assert list(reconstructed) == list(range(len(reconstructed)))
    return normal, [
        (call["name"], json.loads(call["arguments"])) for call in reconstructed.values()
    ]


@pytest.mark.parametrize("width", [None, 1, 2, 5, 13, 10000])
@pytest.mark.parametrize(
    "markup",
    [
        function("run_commands"),
        function("unknown"),
        wrapped("run_command"),
        "<tool_call></function></tool_call>",
    ],
)
def test_rejected_markup_remains_literal_text(markup, width):
    source = "Example:\n```xml\n" + markup + "\n```\nNot an invocation.\n"
    assert parse(source, width) == (source, [])


@pytest.mark.parametrize("width", [None, 1, 2, 5, 13, 10000])
def test_unknown_then_valid_keeps_text_and_exact_arguments(width):
    rejected = "Example: " + wrapped("run_command") + "\nThat was an example.\n"
    source = rejected + wrapped("run_commands", 'a "quote" and \\ slash')
    assert parse(source, width) == (
        rejected,
        [("run_commands", {"command": 'a "quote" and \\ slash'})],
    )


@pytest.mark.parametrize("width", [None, 1, 7, 10000])
def test_explicit_forward_unknown_policy_still_requires_wrapper(width):
    source = wrapped("run_command")
    assert parse(source, width, forward=True) == (
        "",
        [("run_command", {"command": "echo hi"})],
    )
    bare = function("run_command")
    assert parse(bare, width, forward=True) == (bare, [])


@pytest.mark.parametrize("width", [None, 1, 7, 10000])
def test_valid_parallel_and_fully_wrapped_declared_quotation_remain_calls(width):
    # This correction validates syntax/name, not Markdown or the model's intent.
    source = (
        "Example:\n```xml\n"
        + wrapped("run_commands")
        + wrapped("run_commands", "second")
    )
    assert parse(source, width) == (
        "Example:\n```xml\n",
        [
            ("run_commands", {"command": "echo hi"}),
            ("run_commands", {"command": "second"}),
        ],
    )


@pytest.mark.parametrize("width", [None, 1, 7, 10000])
def test_bare_markup_before_unclosed_wrapper_is_not_harvested(width):
    bare = function("run_commands", "bare")
    source = bare + "<tool_call><function=unknown></function>"
    assert parse(source, width) == (source, [])


@pytest.mark.parametrize("width", [None, 1, 2, 7, 10000])
def test_unknown_then_valid_in_same_wrapper_does_not_poison_json(width):
    unknown = function("unknown")
    source = "<tool_call>" + unknown + function("run_commands") + "</tool_call>"
    assert parse(source, width) == (
        "<tool_call>" + unknown + "</tool_call>",
        [("run_commands", {"command": "echo hi"})],
    )


@pytest.mark.parametrize("width", [None, 1, 3, 17, 10000])
@pytest.mark.parametrize("following", ["none", "sibling", "wrapper"])
@pytest.mark.parametrize(
    "nested",
    [
        function("run_commands", "bad"),
        function("run_commands", "bad") + function("run_commands", "also bad"),
        wrapped("run_commands", "bad"),
    ],
)
def test_rejected_body_nested_tags_stay_literal_until_real_close(
    width, following, nested
):
    rejected = function("unknown", "example: " + nested)
    literal = "<tool_call>" + rejected + "</tool_call>"
    expected_calls = []
    if following == "sibling":
        source = (
            "<tool_call>" + rejected + function("run_commands", "good") + "</tool_call>"
        )
        expected_calls = [("run_commands", {"command": "good"})]
    elif following == "wrapper":
        source = literal + wrapped("run_commands", "good")
        expected_calls = [("run_commands", {"command": "good"})]
    else:
        source = literal
    assert parse(source, width) == (literal, expected_calls)


@pytest.mark.parametrize("width", [None, 1, 3, 17, 10000])
@pytest.mark.parametrize("accepted_first", [False, True])
@pytest.mark.parametrize("separator", ["", "\n \t"])
def test_mixed_sibling_rejection_preserves_wrapper_in_either_order(
    width, accepted_first, separator
):
    accepted = "<function=run_commands>\n</function>"
    rejected = "<function=unknown></function>"
    body = (
        accepted + separator + rejected
        if accepted_first
        else rejected + separator + accepted
    )
    source = "<tool_call>\n" + body + "\n</tool_call>"
    remaining = separator + rejected if accepted_first else rejected + separator
    assert parse(source, width) == (
        "<tool_call>\n" + remaining + "\n</tool_call>",
        [("run_commands", {})],
    )


def test_nonstream_incomplete_trailing_wrapper_keeps_legacy_fallback_boundary():
    incomplete = "<tool_call><function=run_commands></function>"
    assert parse(wrapped("run_commands") + incomplete, None) == (
        incomplete,
        [("run_commands", {"command": "echo hi"})],
    )


@pytest.mark.parametrize("width", [1, 2, 7, 10000])
def test_incomplete_unknown_and_stray_parameters_do_not_emit_fragments(width):
    source = (
        "<tool_call><function=unknown><parameter=command>unfinished</tool_call>"
        + wrapped("run_commands")
    )
    normal, calls = parse(source, width)
    assert normal == source[: source.index("<tool_call>", 1)]
    assert calls == [("run_commands", {"command": "echo hi"})]


@pytest.mark.parametrize(
    "source",
    [
        "<tool_call>",
        "<tool_call>\n<function=",
        "<tool_call><parameter=command>value</parameter></tool_call>",
    ],
)
def test_stream_end_preserves_unrecognized_or_incomplete_wrapper(source):
    assert parse(source, 1) == (source, [])


@pytest.mark.parametrize("width", [None, 1, 7, 10000])
def test_reasoning_parser_excludes_thought_markup_before_tool_parser(width):
    # A full wrapper intentionally ends Qwen reasoning, even without </think>.
    # Bare markup in a closed thinking section must remain reasoning instead.
    thought = "Consider " + function("run_commands", "not a call")
    literal = "Quoted " + wrapped("unknown") + "\n"
    source = "<think>" + thought + "</think>" + literal + wrapped("run_commands")
    reasoning_parser = ReasoningParser(model_type="qwen3", stream_reasoning=True)
    tool_parser = FunctionCallParser(TOOLS, "qwen3_coder")
    if width is None:
        reasoning, content = reasoning_parser.parse_non_stream(source)
        normal, calls = tool_parser.parse_non_stream(content)
    else:
        reasoning, normal, calls = "", "", []
        for i in range(0, len(source), width):
            thought_chunk, content = reasoning_parser.parse_stream_chunk(
                source[i : i + width]
            )
            reasoning += thought_chunk or ""
            if content:
                text, increments = tool_parser.parse_stream_chunk(content)
                normal += text
                calls.extend(increments)
        thought_chunk, content = reasoning_parser.parse_stream_end()
        reasoning += thought_chunk or ""
        if content:
            text, increments = tool_parser.parse_stream_chunk(content)
            normal += text
            calls.extend(increments)
        text, increments = tool_parser.parse_stream_end()
        normal += text
        calls.extend(increments)
    assert reasoning == thought
    assert normal == literal
    assert [call.name for call in calls if call.name] == ["run_commands"]
    assert json.loads("".join(call.parameters for call in calls)) == {
        "command": "echo hi"
    }


@pytest.mark.parametrize("width", [1, 7, 10000])
def test_implicit_reasoning_close_still_validates_unknown_name(width):
    literal = wrapped("unknown")
    source = "<think>Consider " + literal
    reasoning_parser = ReasoningParser(model_type="qwen3", stream_reasoning=True)
    tool_parser = FunctionCallParser(TOOLS, "qwen3_coder")
    normal, calls = "", []
    for i in range(0, len(source), width):
        _, content = reasoning_parser.parse_stream_chunk(source[i : i + width])
        if content:
            text, increments = tool_parser.parse_stream_chunk(content)
            normal += text
            calls.extend(increments)
    text, increments = tool_parser.parse_stream_end()
    assert normal + text == literal
    assert calls + increments == []


# --- stray `arguments` envelope -------------------------------------------
# Qwen3.8-Flash-Next keeps the XML envelope but drifts back to its native JSON
# payload, emitting <parameter=arguments>{"command": "..."}</parameter>. The
# call then reaches the client as {"arguments": "{...}"} and strict clients
# reject it with `missing field command`. See _unwrap_arguments_envelope.

ENVELOPE_TOOLS = TOOLS + [
    Tool(
        function=Function(
            name="typed",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "n": {"type": "integer"},
                    "flag": {"type": "boolean"},
                },
            },
        )
    ),
    Tool(
        function=Function(
            name="declares_arguments",
            parameters={
                "type": "object",
                "properties": {"arguments": {"type": "string"}},
            },
        )
    ),
]


def envelope(name, payload):
    return (
        f"<tool_call>\n<function={name}>"
        f"<parameter=arguments>\n{payload}\n</parameter>"
        f"</function>\n</tool_call>"
    )


def parse_envelope(text, width):
    parser = FunctionCallParser(ENVELOPE_TOOLS, "qwen3_coder")
    if width is None:
        normal, calls = parser.parse_non_stream(text)
        return normal, calls
    normal, calls = "", []
    for i in range(0, len(text), width):
        chunk, increments = parser.parse_stream_chunk(text[i : i + width])
        normal += chunk
        calls.extend(increments)
    chunk, increments = parser.parse_stream_end()
    return normal + chunk, calls + increments


ENVELOPE_CASES = [
    # (tool, payload, expected arguments)
    ("run_commands", '{"command": "echo hi"}', {"command": "echo hi"}),
    # A payload full of the characters that made this show up in the wild:
    # heredocs, backticks, embedded quotes and newlines.
    (
        "run_commands",
        json.dumps({"command": "cat <<'EOF'\n`x` said \"hi\"\nEOF"}),
        {"command": "cat <<'EOF'\n`x` said \"hi\"\nEOF"},
    ),
    # Types survive the unwrap: they come from the JSON, not from the schema.
    (
        "typed",
        '{"path": "/a", "n": 3, "flag": true}',
        {"path": "/a", "n": 3, "flag": True},
    ),
    ("run_commands", "{}", {}),
    # Left alone: an undeclared key means this is not an envelope for this tool.
    (
        "run_commands",
        '{"command": "echo hi", "zzz": 1}',
        {"arguments": '{"command": "echo hi", "zzz": 1}'},
    ),
    # Left alone: not JSON, not an object, or a tool that declares `arguments`.
    ("run_commands", "just a string", {"arguments": "just a string"}),
    ("run_commands", "[1, 2, 3]", {"arguments": "[1, 2, 3]"}),
    (
        "declares_arguments",
        '{"arguments": "x"}',
        {"arguments": '{"arguments": "x"}'},
    ),
]


@pytest.mark.parametrize("width", [None, 1, 7, 10000])
@pytest.mark.parametrize("tool,payload,expected", ENVELOPE_CASES)
def test_arguments_envelope(width, tool, payload, expected):
    normal, calls = parse_envelope(envelope(tool, payload), width)
    assert normal == ""
    assert [c.name for c in calls if c.name] == [tool]
    assert json.loads("".join(c.parameters or "" for c in calls)) == expected


@pytest.mark.parametrize("width", [None, 1, 7, 10000])
def test_arguments_envelope_loses_to_a_real_parameter(width):
    """A real parameter of the same name wins, whichever path parsed it.

    The streaming path emits each parameter as it closes and cannot retract an
    unwrap, so the non-streaming path is made to agree rather than the reverse.
    """
    text = (
        "<tool_call>\n<function=run_commands>"
        '<parameter=arguments>\n{"command": "from envelope"}\n</parameter>'
        "<parameter=command>\nreal\n</parameter>"
        "</function>\n</tool_call>"
    )
    normal, calls = parse_envelope(text, width)
    assert normal == ""
    assert json.loads("".join(c.parameters or "" for c in calls)) == {
        "command": "real"
    }


# A tool whose schema routes through $ref/$defs publishes no `properties` map.
# _get_arguments_config falls back to the schema object itself, which would make
# `type`/`required` look like argument names; the unwrap must not use that.
REF_TOOLS = [
    Tool(
        function=Function(
            name="refschema",
            parameters={
                "type": "object",
                "required": ["command"],
                "$ref": "#/$defs/Args",
            },
        )
    )
]


@pytest.mark.parametrize("width", [None, 1, 7, 10000])
@pytest.mark.parametrize(
    "payload",
    [
        '{"required": ["a"]}',  # would unwrap into a nonsense argument set
        '{"command": "echo hi"}',  # the real shape, but unverifiable here
    ],
)
def test_no_properties_schema_is_never_unwrapped(width, payload):
    parser = FunctionCallParser(REF_TOOLS, "qwen3_coder")
    text = envelope("refschema", payload)
    if width is None:
        _, calls = parser.parse_non_stream(text)
    else:
        calls = []
        for i in range(0, len(text), width):
            _, increments = parser.parse_stream_chunk(text[i : i + width])
            calls.extend(increments)
        _, increments = parser.parse_stream_end()
        calls.extend(increments)
    assert json.loads("".join(c.parameters or "" for c in calls)) == {
        "arguments": payload
    }


@pytest.mark.parametrize("width", [None, 1, 7, 10000])
@pytest.mark.parametrize(
    "payload",
    [
        '{"n": NaN}',
        '{"n": Infinity}',
        '{"n": -Infinity}',
        '{"n": 1e400}',  # overflows to inf via parse_float, not parse_constant
    ],
)
def test_non_rfc_numbers_are_never_unwrapped(width, payload):
    """json.dumps would write a bare NaN/Infinity that no strict client parses.

    Leaving the payload as an opaque string keeps the call merely wrong, which
    is what it is today, instead of making the arguments document unparseable.
    """
    tools = ENVELOPE_TOOLS + [
        Tool(
            function=Function(
                name="numeric",
                parameters={
                    "type": "object",
                    "properties": {"n": {"type": "number"}},
                },
            )
        )
    ]
    parser = FunctionCallParser(tools, "qwen3_coder")
    text = envelope("numeric", payload)
    if width is None:
        _, calls = parser.parse_non_stream(text)
    else:
        calls = []
        for i in range(0, len(text), width):
            _, increments = parser.parse_stream_chunk(text[i : i + width])
            calls.extend(increments)
        _, increments = parser.parse_stream_end()
        calls.extend(increments)
    raw = "".join(c.parameters or "" for c in calls)
    assert json.loads(raw) == {"arguments": payload}
    # and the wire bytes stay RFC-8259 clean
    json.JSONDecoder(
        parse_constant=lambda c: pytest.fail(f"non-RFC literal on the wire: {c}")
    ).decode(raw)


# --- incremental tool-arg streaming ------------------------------------------
# Stock qwen3_coder withholds a <parameter> value until its closing tag, so a
# tool call that authors a file lands as one delta after seconds of dead air:
# a 7,459-char body measured 3 deltas, max 7,708 chars. With streaming: 472
# deltas, max 18. Only DECLARED string parameters stream -- see
# _should_stream_param for why undeclared ones must not.

STREAM_TOOLS = [
    Tool(function=Function(name="writer", parameters={"type": "object", "properties": {
        "body": {"type": "string"}, "count": {"type": "integer"},
        "flag": {"type": "boolean"}, "obj": {"type": "object"}}})),
    Tool(function=Function(name="noprops", parameters={"type": "object", "$ref": "#/$defs/X"})),
]

LONG_BODY = "\n".join(f"line {i}: " + "x" * 40 for i in range(60))


def param_call(fn, params):
    inner = "".join(f"<parameter={n}>\n{v}\n</parameter>\n" for n, v in params)
    return f"<tool_call>\n<function={fn}>\n{inner}</function>\n</tool_call>"


def stream_parse(text, width):
    parser = FunctionCallParser(STREAM_TOOLS, "qwen3_coder")
    if width is None:
        _, calls = parser.parse_non_stream(text)
        return [c.parameters or "" for c in calls]
    out = []
    for i in range(0, len(text), width):
        _, inc = parser.parse_stream_chunk(text[i : i + width])
        out.extend(c.parameters or "" for c in inc)
    _, inc = parser.parse_stream_end()
    out.extend(c.parameters or "" for c in inc)
    return out


STREAM_WIDTHS = [1, 3, 13, 500, 10000]


def test_declared_string_param_streams_incrementally():
    deltas = stream_parse(param_call("writer", [("body", LONG_BODY)]), 13)
    payload = [d for d in deltas if d not in ("{", "}")]
    assert len(payload) > 20, f"only {len(payload)} deltas -- not streaming"
    assert max(len(d) for d in payload) < len(LONG_BODY) / 2
    assert json.loads("".join(deltas)) == {"body": LONG_BODY}


@pytest.mark.parametrize("width", STREAM_WIDTHS)
@pytest.mark.parametrize("payload", [
    LONG_BODY, "plain", "", "a < b and c <= d",
    "literal </parameter> inside", "<parameter=nested>oops</parameter>",
    "unicode: éè中文", "trailing\n", "double\n\n\nnewlines",
    'quotes " backslash \\ tab \t',
    "cat > /tmp/o.txt <<'EOF'\n`x` is not decoration\nEOF",
])
def test_streaming_never_changes_content(width, payload):
    """Streaming changes delivery, never the value the client assembles."""
    text = param_call("writer", [("body", payload)])
    assert json.loads("".join(stream_parse(text, width))) == json.loads(
        "".join(stream_parse(text, None))
    )


@pytest.mark.parametrize("width", STREAM_WIDTHS)
def test_streaming_concatenation_is_a_growing_prefix(width):
    text = param_call("writer", [("body", LONG_BODY)])
    parser = FunctionCallParser(STREAM_TOOLS, "qwen3_coder")
    acc, seen = "", []
    for i in range(0, len(text), width):
        _, inc = parser.parse_stream_chunk(text[i : i + width])
        for c in inc:
            acc += c.parameters or ""
            seen.append(acc)
    _, inc = parser.parse_stream_end()
    for c in inc:
        acc += c.parameters or ""
        seen.append(acc)
    assert all(acc.startswith(p) for p in seen)


@pytest.mark.parametrize("width", STREAM_WIDTHS)
@pytest.mark.parametrize("name,raw,expected", [
    ("count", "42", 42), ("flag", "true", True), ("obj", '{"k": [1, 2]}', {"k": [1, 2]}),
])
def test_non_string_params_are_not_streamed(width, name, raw, expected):
    """JSON encoding of non-strings is not prefix-stable, so they buffer."""
    text = param_call("writer", [(name, raw)])
    assert json.loads("".join(stream_parse(text, width))) == {name: expected}


@pytest.mark.parametrize("width", STREAM_WIDTHS)
def test_arguments_envelope_still_unwraps_under_streaming(width):
    """The composition this patch has to preserve.

    Upstream PR #21829 streams undeclared parameters ("treat as string"). Doing
    that here would put `"arguments": "` on the wire before
    _unwrap_arguments_envelope could run, and the streaming path cannot retract
    what it has already sent -- so the repair would become impossible.
    """
    text = param_call("writer", [("arguments", json.dumps({"body": LONG_BODY}))])
    deltas = stream_parse(text, width)
    assert json.loads("".join(deltas)) == {"body": LONG_BODY}
    assert not any(d.startswith('"arguments"') for d in deltas)


@pytest.mark.parametrize("width", STREAM_WIDTHS)
def test_schema_without_properties_never_streams(width):
    text = param_call("noprops", [("whatever", "a long value " * 20)])
    assert json.loads("".join(stream_parse(text, width))) == json.loads(
        "".join(stream_parse(text, None))
    )
