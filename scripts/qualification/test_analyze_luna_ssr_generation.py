import json

from analyze_luna_ssr_generation import inspect_rationale


def row(prefix, rationale):
    return {
        "task": "simple.test",
        "action_index": 0,
        "request": {
            "prefix_evidence": [{"id": "task", "value": prefix}],
            "target_action": {"name": "send", "arguments": {"to": "hidden@example.com"}},
        },
        "text": json.dumps({"supported": True, "rationale": rationale, "evidence_refs": ["task"]}),
    }


def test_known_reference_value_is_flagged_when_missing_from_acting_prefix():
    result = inspect_rationale(row("Notify the owner", "Send to hidden@example.com."))
    assert result["review_flags"] == ["target_literal_absent_from_prefix"]
    assert not result["sft_qualified"]


def test_value_from_public_request_does_not_raise_literal_review_flag():
    result = inspect_rationale(row("Notify hidden@example.com", "Send to hidden@example.com."))
    assert not result["review_flags"]
    assert result["reference_scope_valid"]
    assert not result["semantic_grounding_verified"]
