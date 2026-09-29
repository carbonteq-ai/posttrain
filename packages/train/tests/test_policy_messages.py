"""Rejected tool attempts must survive both trainers' message projection."""

from types import SimpleNamespace

import pytest
from posttrain.train.policy_messages import parsed_policy_message


class Tokenizer:
    def decode(self, ids, *, skip_special_tokens):
        assert skip_special_tokens is False
        return "".join(chr(value) for value in ids)


@pytest.mark.parametrize("status", ["invalid_json", "unclosed_block", "malformed_structure"])
def test_rejected_calls_preserve_exact_text_and_reasoning(status):
    raw = "<tool_call>\n<function=asana_get_task>\n<parameter=gid>\nasana_123\n</parameter>\n</function>\n</tool_call>"
    ids = list(map(ord, raw))
    parsed = SimpleNamespace(
        content="",
        reasoning_content="Check the created task.",
        tool_calls=[
            SimpleNamespace(name="asana_get_task", status=SimpleNamespace(value=status), token_span=(0, len(ids)))
        ],
    )
    message = parsed_policy_message(parsed, ids, Tokenizer())
    assert message == {
        "role": "assistant",
        "content": raw,
        "reasoning_content": "Check the created task.",
        "provider_state": [
            {
                "type": "posttrain.rejected_tool_call",
                "status": status,
                "token_span": [0, len(ids)],
                "raw": raw,
            }
        ],
    }
    assert ids == list(map(ord, raw))


def test_mixed_calls_keep_accepted_action_and_rejected_evidence_separate():
    parsed = SimpleNamespace(
        content="Explanation",
        reasoning_content=None,
        tool_calls=[
            SimpleNamespace(
                id=None, name="create_task", arguments={"name": "review"}, status=SimpleNamespace(value="ok")
            ),
            SimpleNamespace(name=None, status=SimpleNamespace(value="unclosed_block"), token_span=(0, 3)),
        ],
    )
    message = parsed_policy_message(parsed, list(map(ord, "bad")), Tokenizer())
    assert message["content"] == "Explanation\nbad"
    assert message["tool_calls"] == [{"id": "call_0", "name": "create_task", "arguments": '{"name":"review"}'}]
    assert message["provider_state"][0]["status"] == "unclosed_block"


def test_lfm_pythonic_content_is_recovered_as_structured_tool_calls():
    raw = (
        "<|tool_call_start|>[slack_send(channel='marketing', text='Launch'), "
        "asana_create(workspace='ws_marketing', priority=2)]<|tool_call_end|>"
    )
    parsed = SimpleNamespace(content=raw, reasoning_content="Execute both actions.", tool_calls=[])
    protocol = SimpleNamespace(
        id="lfm2_pythonic",
        start_token="<|tool_call_start|>",
        end_token="<|tool_call_end|>",
    )
    tools = [
        {"type": "function", "function": {"name": "slack_send"}},
        {"type": "function", "function": {"name": "asana_create"}},
    ]

    message = parsed_policy_message(
        parsed,
        list(map(ord, raw)),
        Tokenizer(),
        tool_call_protocol=protocol,
        tools=tools,
    )

    assert message == {
        "role": "assistant",
        "content": None,
        "reasoning_content": "Execute both actions.",
        "tool_calls": [
            {"id": "call_0", "name": "slack_send", "arguments": '{"channel":"marketing","text":"Launch"}'},
            {"id": "call_1", "name": "asana_create", "arguments": '{"workspace":"ws_marketing","priority":2}'},
        ],
    }


def test_lfm_pythonic_content_accepts_flat_verifiers_tool_records():
    raw = "<|tool_call_start|>[salesforce_task_create(subject='Proposal')]<|tool_call_end|>"
    parsed = SimpleNamespace(content=raw, reasoning_content=None, tool_calls=[])
    protocol = SimpleNamespace(
        id="lfm2_pythonic",
        start_token="<|tool_call_start|>",
        end_token="<|tool_call_end|>",
    )

    message = parsed_policy_message(
        parsed,
        list(map(ord, raw)),
        Tokenizer(),
        tool_call_protocol=protocol,
        tools=[{"name": "salesforce_task_create", "description": "", "parameters": {}}],
    )

    assert message["content"] is None
    assert message["tool_calls"] == [
        {"id": "call_0", "name": "salesforce_task_create", "arguments": '{"subject":"Proposal"}'}
    ]


def test_k2_ifm_xml_completion_preserves_reasoning_and_parses_actions():
    raw = (
        "Check the current task.</ifm|think>"
        "<ifm|tool_calls>\n<ifm|tool_call>asana_get_task\n"
        "<ifm|arg_key>gid</ifm|arg_key>\n<ifm|arg_value>asana_123</ifm|arg_value>\n"
        "<ifm|arg_key>limit</ifm|arg_key>\n<ifm|arg_value>2</ifm|arg_value>\n"
        "</ifm|tool_call>\n</ifm|tool_calls><|ifm|im_end|>"
    )
    message = parsed_policy_message(
        SimpleNamespace(content=raw, reasoning_content=None, tool_calls=[]),
        list(map(ord, raw)),
        Tokenizer(),
        tool_call_protocol=SimpleNamespace(id="k2_ifm_xml"),
        tools=[{"type": "function", "function": {"name": "asana_get_task"}}],
    )

    assert message == {
        "role": "assistant",
        "content": None,
        "reasoning_content": "Check the current task.",
        "tool_calls": [{"id": "call_0", "name": "asana_get_task", "arguments": '{"gid":"asana_123","limit":2}'}],
    }


@pytest.mark.parametrize(
    "body",
    [
        "[unknown_tool(value='x')]",
        "[allowed_tool('x')]",
        "[allowed_tool(value=get_secret())]",
        "[allowed_tool(**{'value': 'x'})]",
    ],
)
def test_lfm_pythonic_recovery_rejects_unknown_or_nonliteral_calls(body):
    raw = f"<|tool_call_start|>{body}<|tool_call_end|>"
    parsed = SimpleNamespace(content=raw, reasoning_content=None, tool_calls=[])
    protocol = SimpleNamespace(
        id="lfm2_pythonic",
        start_token="<|tool_call_start|>",
        end_token="<|tool_call_end|>",
    )

    message = parsed_policy_message(
        parsed,
        list(map(ord, raw)),
        Tokenizer(),
        tool_call_protocol=protocol,
        tools=[{"type": "function", "function": {"name": "allowed_tool"}}],
    )

    assert message == {"role": "assistant", "content": raw}


@pytest.mark.parametrize("span", [None, (-1, 2), (0, 9), (1, 1)])
def test_missing_or_invalid_provenance_fails_instead_of_creating_empty_turn(span):
    parsed = SimpleNamespace(
        content="",
        reasoning_content=None,
        tool_calls=[
            SimpleNamespace(name=None, status=SimpleNamespace(value="invalid_json"), token_span=span),
        ],
    )
    with pytest.raises(ValueError, match="exact sampled-token span"):
        parsed_policy_message(parsed, [1, 2], Tokenizer())


def _train_client_parse(renderers):
    status = renderers.ToolCallParseStatus
    # Qwen3.5 XML call whose array parameter was sampled as a bare word
    # (verl-vortex-p6-qwen08b-verl-vortex-ws-r3, update 1): the renderer keeps
    # the value as text and marks the call invalid_json.
    raw = "<tool_call>asana_create_task projects=proj_eng</tool_call>"
    calls = [
        renderers.ParsedToolCall(
            raw="asana",
            name="asana_create_task",
            arguments={"projects": "proj_eng"},
            token_span=(0, 10),
            status=status.INVALID_JSON,
        ),
        renderers.ParsedToolCall(raw="ok", name="asana_get_task", arguments={"gid": "a", "n": 2}, token_span=(10, 20)),
        renderers.ParsedToolCall(
            raw="x", name="jira_nope", arguments={}, token_span=(20, 30), status=status.UNKNOWN_TOOL
        ),
        renderers.ParsedToolCall(raw="y", token_span=(30, len(raw)), status=status.UNCLOSED_BLOCK),
    ]
    return raw, renderers.ParsedResponse(content="Creating it.", reasoning_content="plan", tool_calls=calls)


def test_train_client_admission_acts_on_the_calls_verifiers_train_client_runs():
    """veRL takes TRL's (Verifiers train client) actions: named calls run, unknown or nameless ones drop."""

    renderers = pytest.importorskip("renderers")
    train = pytest.importorskip("verifiers.v1.clients.train")
    raw, parsed = _train_client_parse(renderers)
    ids = list(map(ord, raw))

    message = parsed_policy_message(parsed, ids, Tokenizer(), admission="verifiers-train-client")
    reference = train.response_from_generate(
        {"content": parsed.content, "reasoning_content": parsed.reasoning_content, "tool_calls": parsed.tool_calls},
        "policy",
    ).message

    assert message["content"] == reference.content == "Creating it."
    assert message["reasoning_content"] == reference.reasoning_content
    assert message["tool_calls"] == [
        call.model_dump(include={"id", "name", "arguments"}) for call in reference.tool_calls
    ]
    assert [call["name"] for call in message["tool_calls"]] == ["asana_create_task", "asana_get_task"]
    assert message["tool_calls"][0]["arguments"] == '{"projects": "proj_eng"}'
    assert [(item["type"], item["status"], item["raw"]) for item in message["provider_state"]] == [
        ("posttrain.nonconforming_tool_call", "invalid_json", raw[0:10]),
        ("posttrain.rejected_tool_call", "unknown_tool", raw[20:30]),
        ("posttrain.rejected_tool_call", "unclosed_block", raw[30:]),
    ]


def test_strict_admission_still_keeps_nonconforming_calls_as_text():
    renderers = pytest.importorskip("renderers")
    raw, parsed = _train_client_parse(renderers)

    message = parsed_policy_message(parsed, list(map(ord, raw)), Tokenizer())

    assert [call["name"] for call in message["tool_calls"]] == ["asana_get_task"]
    assert message["content"] == "\n".join(["Creating it.", raw[0:10], raw[20:30], raw[30:]])
