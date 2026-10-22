import json

import pytest
from assemble_teacher_context_review import GENERATION_PREFIX, compose, parse_feedback


def test_composition_preserves_history_and_actual_facts():
    facts = {"terminal_native_assertions": [{"passed": False, "params": {"x": "ACTUAL"}}],
             "native_manifest_findings": {"missing_signal_ids": ["unknown"]},
             "current_action_observations": [{"response": "REAL_TOOL_RESULT"}]}
    old = "Factual hindsight from this student episode:\n" + json.dumps(facts) + "\nold instruction"
    historical = "REAL HISTORY\n<|im_start|>user\n" + old + "<|im_end|>\n" + GENERATION_PREFIX
    context, supplement, before, after = compose(historical, old, {"requirements": ["New guide"]}, facts)
    assert context == before + supplement + after
    assert context.startswith("REAL HISTORY\n")
    assert parse_feedback(supplement) == facts
    assert "ACTUAL" in context and "REAL_TOOL_RESULT" in context
    assert "A failed episode is not proof that this selected action caused the failure" in context


def test_missing_or_ambiguous_historical_feedback_fails_closed():
    with pytest.raises(ValueError, match="uniquely"):
        compose("history", "not present", {}, {})
    with pytest.raises(ValueError, match="uniquely"):
        compose("duplicate duplicate", "duplicate", {}, {})
