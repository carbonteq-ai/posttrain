"""The native tool gate must reconstruct fragments and reject invalid calls."""

import pytest
from posttrain_lab.lfm_moe_smoke import _tool_call


def test_tool_gate_reconstructs_streamed_arguments():
    call = _tool_call(
        (
            {"index": 0, "id": "call_1", "function": {"name": "lookup_", "arguments": '{"key":'}},
            {"index": 0, "function": {"name": "code", "arguments": '"readiness"}'}},
        )
    )
    assert call["id"] == "call_1"
    assert call["function"]["name"] == "lookup_code"


def test_tool_gate_rejects_wrong_arguments_and_no_call():
    with pytest.raises(RuntimeError, match="incorrect function arguments"):
        _tool_call(({"id": "c", "function": {"name": "lookup_code", "arguments": '{"key":"other"}'}},))
    with pytest.raises(RuntimeError, match="exactly one"):
        _tool_call(())
