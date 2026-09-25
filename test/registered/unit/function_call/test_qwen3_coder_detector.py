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
