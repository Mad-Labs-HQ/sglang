"""qwen3_coder: repairs for Qwen3.8-Flash-Next tool-call drift."""

import json

import pytest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser
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


def parse_envelope(text, width, tools=ENVELOPE_TOOLS):
    parser = FunctionCallParser(tools, "qwen3_coder")
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
    assert json.loads("".join(c.parameters or "" for c in calls)) == {"command": "real"}


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
    _, calls = parse_envelope(envelope("refschema", payload), width, REF_TOOLS)
    assert json.loads("".join(c.parameters or "" for c in calls)) == {
        "arguments": payload
    }


# Upstream #36626 resolves parameter types through top-level anyOf/oneOf/allOf
# branches. "Declared" follows the same view, so a key declared in a branch
# counts, and a branch that declares `arguments` itself still blocks the unwrap.
UNION_TOOLS = [
    Tool(
        function=Function(
            name="union",
            parameters={
                "anyOf": [
                    {"type": "object", "properties": {"path": {"type": "string"}}},
                    {"type": "object", "properties": {"n": {"type": "integer"}}},
                ]
            },
        )
    ),
    Tool(
        function=Function(
            name="union_declares_arguments",
            parameters={
                "oneOf": [
                    {"type": "object", "properties": {"path": {"type": "string"}}},
                    {
                        "type": "object",
                        "properties": {"arguments": {"type": "string"}},
                    },
                ]
            },
        )
    ),
]


@pytest.mark.parametrize("width", [None, 1, 7, 10000])
@pytest.mark.parametrize(
    "tool,payload,expected",
    [
        ("union", '{"path": "/a", "n": 2}', {"path": "/a", "n": 2}),
        (
            "union",
            '{"path": "/a", "zzz": 2}',
            {"arguments": '{"path": "/a", "zzz": 2}'},
        ),
        (
            "union_declares_arguments",
            '{"path": "/a"}',
            {"arguments": '{"path": "/a"}'},
        ),
    ],
)
def test_union_schema_envelope(width, tool, payload, expected):
    _, calls = parse_envelope(envelope(tool, payload), width, UNION_TOOLS)
    assert json.loads("".join(c.parameters or "" for c in calls)) == expected


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
    _, calls = parse_envelope(envelope("numeric", payload), width, tools)
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
    Tool(
        function=Function(
            name="writer",
            parameters={
                "type": "object",
                "properties": {
                    "body": {"type": "string"},
                    "count": {"type": "integer"},
                    "flag": {"type": "boolean"},
                    "obj": {"type": "object"},
                },
            },
        )
    ),
    Tool(
        function=Function(
            name="noprops", parameters={"type": "object", "$ref": "#/$defs/X"}
        )
    ),
    # Declared only inside a top-level union branch (#36626): still declared.
    Tool(
        function=Function(
            name="union_writer",
            parameters={
                "anyOf": [
                    {"type": "object", "properties": {"body": {"type": "string"}}},
                    {"type": "object", "properties": {"count": {"type": "integer"}}},
                ]
            },
        )
    ),
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


@pytest.mark.parametrize("tool", ["writer", "union_writer"])
def test_declared_string_param_streams_incrementally(tool):
    deltas = stream_parse(param_call(tool, [("body", LONG_BODY)]), 13)
    payload = [d for d in deltas if d not in ("{", "}")]
    assert len(payload) > 20, f"only {len(payload)} deltas -- not streaming"
    assert max(len(d) for d in payload) < len(LONG_BODY) / 2
    assert json.loads("".join(deltas)) == {"body": LONG_BODY}


@pytest.mark.parametrize("width", STREAM_WIDTHS)
@pytest.mark.parametrize(
    "payload",
    [
        LONG_BODY,
        "plain",
        "",
        "a < b and c <= d",
        "literal </parameter> inside",
        "<parameter=nested>oops</parameter>",
        "unicode: éè中文",
        "trailing\n",
        "double\n\n\nnewlines",
        'quotes " backslash \\ tab \t',
        "cat > /tmp/o.txt <<'EOF'\n`x` is not decoration\nEOF",
    ],
)
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
@pytest.mark.parametrize(
    "name,raw,expected",
    [
        ("count", "42", 42),
        ("flag", "true", True),
        ("obj", '{"k": [1, 2]}', {"k": [1, 2]}),
    ],
)
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
@pytest.mark.parametrize("tool", ["writer", "noprops"])
def test_undeclared_param_is_never_streamed(width, tool):
    """An undeclared parameter arrives in one piece, whatever the chunking.

    This is the semantic that differs from upstream PR #21829 on purpose.
    """
    value = "a long value " * 20
    deltas = stream_parse(param_call(tool, [("notes", value)]), width)
    assert [d for d in deltas if "notes" in d] == [f'"notes": {json.dumps(value)}']
    assert json.loads("".join(deltas)) == {"notes": value}


@pytest.mark.parametrize("width", STREAM_WIDTHS)
def test_schema_without_properties_never_streams(width):
    text = param_call("noprops", [("whatever", "a long value " * 20)])
    assert json.loads("".join(stream_parse(text, width))) == json.loads(
        "".join(stream_parse(text, None))
    )
