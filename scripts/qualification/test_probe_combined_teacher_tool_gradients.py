import pytest
from probe_combined_teacher_tool_gradients import forbidden_positions, tool_positions


def test_reasoning_is_excluded_from_tool_gradient_support():
    assert tool_positions([1, 2, 100, 3, 4, 101, 5, 100, 6, 101], 100, 101) == [2, 3, 4, 5, 7, 8, 9]
    assert tool_positions([1, 2, 3], 100, 101) == []


def test_incomplete_and_nested_tool_spans_are_rejected():
    for ids in [[100, 2], [101], [100, 100, 101]]:
        with pytest.raises(ValueError):
            tool_positions(ids, 100, 101)


def test_forbidden_text_is_only_measured_inside_tool_call_and_when_decided():
    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return list(map(ord, text))

    text = "BAD reasoning / tool BAD SAFE"
    checks = [{"type": "slack_message_not_in_channel", "passed": False, "excluded": False,
               "params": {"text_contains": "BAD"}},
              {"type": "slack_message_not_in_channel", "passed": False, "excluded": True,
               "params": {"text_contains": "SAFE"}}]
    positions, terms = forbidden_positions(list(map(ord, text)), Tokenizer(), checks, range(16, len(text)))
    assert terms == ["BAD"] and positions and min(positions) >= 16
