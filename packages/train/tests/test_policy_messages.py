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
