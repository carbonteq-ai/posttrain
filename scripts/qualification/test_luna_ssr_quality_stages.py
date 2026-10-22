import pytest
from luna_ssr_quality_stages import evidence_request, validate_ledger, validate_workflow


def request():
    return {
        "action_index": 0,
        "prefix_evidence": [{"id": "task", "value": "Read the request."}],
        "target_action": {"name": "search", "arguments": {"id": "privileged-secret"}},
        "teacher_only_guidance": {"answer": "future-secret"},
    }


def test_evidence_stage_has_no_gold_arguments_or_privileged_goals():
    result = evidence_request(request())
    assert "privileged-secret" not in str(result)
    assert "future-secret" not in str(result)
    assert result["next_tool_family_hint"] == "search"


def test_ledger_requires_exact_quote_from_the_cited_prefix():
    result = validate_ledger(
        request(),
        {"facts": [{"evidence_ref": "task", "quote": "Read the request"}], "missing_information": ["record ID"]},
    )
    assert result["exact_quote_scope_verified"]
    assert not result["factual_interpretation_verified"]


@pytest.mark.parametrize("ref,quote", [("future:1", "Read"), ("task", "privileged-secret"), ("task", "")])
def test_ledger_rejects_future_refs_and_invented_quotes(ref, quote):
    with pytest.raises(ValueError):
        validate_ledger(request(), {"facts": [{"evidence_ref": ref, "quote": quote}], "missing_information": []})


def test_workflow_phases_separate_read_decide_act_and_verify():
    value = {
        "workflow_plan": [
            {"phase": phase, "operation": "Check the necessary evidence."}
            for phase in ("READ", "DECIDE", "ACT", "VERIFY")
        ],
        "reference_issues": [],
    }
    validate_workflow(value)
    value["workflow_plan"][2]["phase"] = "CREATE"
    with pytest.raises(ValueError):
        validate_workflow(value)
