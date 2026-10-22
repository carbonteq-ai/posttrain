import asyncio

import pytest
from sdpo_checkwise_preparation import (
    abstract_workflow_requirements,
    abstraction_text,
    normalize_instruction_wording,
    preparation_rules,
    prepare,
    reference_scope,
    schema,
    source_provenance,
    validate_instruction_text,
)


def test_cosmetic_normalization_only_changes_instruction_noun():
    value = {"instruction": "Read the entire dataset then compare rows.",
             "check_id": "dataset-check", "source": "dataset", "steps": [
                 {"instruction": "Use datasets.", "tool": "read_dataset"}]}
    normalized = normalize_instruction_wording(value)
    assert normalized["instruction"] == "Read the entire records then compare rows."
    assert normalized["steps"][0]["instruction"] == "Use records."
    assert normalized["check_id"] == "dataset-check" and normalized["source"] == "dataset"
    assert normalized["steps"][0]["tool"] == "read_dataset"
    assert value["instruction"].startswith("Read the entire dataset")


def test_abstraction_hides_unlisted_private_ids_but_preserves_public_ids():
    text = "Read ws_policy then send to private@example.net using public ws_public."
    result = abstraction_text(text, [], "Use ws_public.")
    assert "ws_policy" not in result and "private@example.net" not in result
    assert "ws_public" in result


def test_procedure_synthesis_cannot_copy_observed_private_numeric_policy_examples():
    value = material()
    value["user_request"] += " Today is Feb 9, 2026."
    value["reference_actions"][0]["result"]["observed_result"]["threshold"] = 36.5
    original = [{"check_id": "a", "instruction": "Compare to 36.5h hours on Feb 9, 2026."}]
    result = abstract_workflow_requirements(original, value, [])
    assert "36.5" not in result[0]["instruction"] and "Feb 9, 2026" in result[0]["instruction"]
    assert "36.5" in original[0]["instruction"]


def test_source_provenance_keeps_the_excluded_entity_reason_and_policy_location():
    import copy

    value = material()
    value["native_data"]["initial_state"] = {
        "rows": [{"Employee": "Diana Parks", "Notes": "payroll audit hold"}],
        "messages": [{"body": "Audit note title: Phone Update Processed"}]}
    before = copy.deepcopy(value)
    checks = [{"effect_match": {"kind": "literal", "value": "Diana Parks"}},
              {"effect_match": {"kind": "literal", "value": "Phone Update Processed"}}]
    result = source_provenance(value, checks)
    assert result["records"][0]["record"]["Notes"] == "payroll audit hold"
    assert result["records"][1]["source_path"] == ["messages", 0]
    assert value == before


def test_observed_asana_parameter_lesson_is_not_injected_into_unrelated_tasks():
    generic = preparation_rules({"tool_definitions": [{"function": {"name": "gmail_find_email"}}]})
    specific = preparation_rules({"tool_definitions": [{"function": {"name": "asana_create_task"}}]})
    assert "workspace_info" not in generic and "workspace_info" in specific
    assert "success:true response is operation status" in generic


def test_execution_instruction_cannot_reference_an_invisible_binding():
    with pytest.raises(ValueError, match="hidden source bookkeeping"):
        validate_instruction_text(material(), ["Use the mailroom-summary binding to obtain the channel."], [])


def material():
    return {"task": "simple.example", "user_request": "Read then create one contact.",
        "native_data": {"initial_state": {}},
        "tool_definitions": [{"type": "function", "function": {"name": "create"}}],
        "goal_manifest": {"sources": {}, "checks": [{"check_id": "required-create",
            "role": "goal", "operator": "effects.required_when@1"}]},
        "reference_actions": [{"call": {"name": "create", "arguments": {}},
                               "result": {"observed_result": {"success": True}}}]}


def test_private_literal_repair_identifies_copied_value_without_exporting_it():
    value = material()
    value["goal_manifest"]["checks"][0]["effect_match"] = {"kind": "literal", "value": "private-123"}

    async def generate(stage, prompt, data):
        if stage == "check_brief":
            return {"check_id": "required-create", "role": "goal", "instruction": "Create private-123."}
        if stage == "check_brief_repair":
            assert "private-123" in data["private_source_literals_to_remove"]
            assert "Remove EVERY" in prompt
            return {"check_id": "required-create", "role": "goal", "instruction": "Create the contact identified by the retrieved request."}
        if stage.endswith("review"):
            return {"approved": True, "issues": [], "reason": "Source preserved."}
        return {"steps": [{"tool": "create", "instruction": "Create using observed values."}]}

    result = asyncio.run(prepare(value, generate, lambda *_: None, private_preparation=True))
    assert "private-123" not in str(result["instructions"])


def test_abstraction_and_workflow_cannot_reanchor_on_raw_contract_or_reference():
    value = material()
    seen = []

    async def generate(stage, prompt, data):
        seen.append(stage)
        if stage == "check_brief":
            assert "private_preparation_evidence" in data
            return {"check_id": "required-create", "role": "goal", "instruction": "Create one contact."}
        if stage == "check_abstraction":
            assert "private_preparation_evidence" not in data and "source_bindings" not in data
            assert "private_interpretation" in data
            return {"check_id": "required-create", "role": "goal", "instruction": "Create the requested contact using retrieved details."}
        if stage == "workflow_brief":
            assert "private_preparation_evidence" not in data and "historical_procedure" not in data
        if stage.endswith("review"):
            if stage == "bundle_review":
                assert data["source_checks"] == value["goal_manifest"]["checks"]
            return {"approved": True, "issues": [], "reason": "Source preserved."}
        return {"steps": [{"tool": "create", "instruction": "Create using observed values."}]}

    result = asyncio.run(prepare(value, generate, lambda *_: None,
        private_preparation=True, abstract_private_briefs=True))
    assert "check_abstraction" in seen and result["review_approved"]


@pytest.mark.parametrize("private", [False, True])
def test_failed_reference_findings_are_private_preparation_evidence_only(private):
    value = material()
    quality = {"partial_credit": .95, "success": 0, "excluded_assertions": 2,
               "failed_assertions": [{"params": {"to": "privateqa@example.net"}, "passed": False}]}
    value["historical_reference_quality"] = quality
    captured = []

    class Captured(Exception):
        pass

    async def generate(stage, prompt, data):
        captured.append(data["historical_reference_quality"])
        raise Captured()

    with pytest.raises(Captured):
        asyncio.run(prepare(value, generate, lambda *_: None, private_preparation=private))
    if private:
        assert captured == [quality]
    else:
        assert captured[0]["known_failed_assertion_count"] == 1
        assert "failed_assertions" not in captured[0]


def test_single_check_and_workflow_are_reviewed_without_action_id_graph():
    records = []

    async def generate(stage, prompt, data):
        assert "NOT additional API parameters" in prompt
        schema(stage, data)
        if stage == "check_brief":
            return {"check_id": "required-create", "role": "goal", "instruction": "Create one contact."}
        if stage.endswith("review"):
            return {"approved": True, "issues": [], "reason": "The source is preserved."}
        return {"steps": [{"tool": "create", "instruction": "Create using current observed values."}]}

    result = asyncio.run(prepare(material(), generate, lambda n, _: records.append(n)))
    assert result["review_approved"] is True
    assert records == ["checkwise_source_separation", "historical_reference_binding",
                       "check_brief", "check_review", "workflow_brief", "workflow_review"]
    assert result["requirements"][0]["check_id"] == "required-create"


def test_reference_scope_preserves_false_unknown_and_current_source_binding():
    from sdpo_guidance_context import digest

    source = material()
    binding = {"current_native_state_sha256": digest(source["native_data"]["initial_state"]),
               "current_manifest_sha256": digest(source["goal_manifest"]),
               "historical_state_equal": False, "historical_assertions_equal": True}
    scope = reference_scope(source, binding)
    assert scope["source_binding_verified"] and scope["state_matches_current_task"] is False
    assert "not a current observation" in scope["authority"]
    assert reference_scope(source, None)["state_matches_current_task"] is None
    binding["current_manifest_sha256"] = "stale"
    with pytest.raises(ValueError, match="binding differs"):
        reference_scope(source, binding)


def test_every_generation_stage_receives_the_actual_reference_scope():
    from sdpo_guidance_context import digest

    source = material()
    binding = {"current_native_state_sha256": digest(source["native_data"]["initial_state"]),
               "current_manifest_sha256": digest(source["goal_manifest"]),
               "historical_state_equal": False, "historical_assertions_equal": True}
    stages = []

    async def generate(stage, prompt, data):
        stages.append(stage)
        assert data["historical_reference_scope"]["state_matches_current_task"] is False
        if stage == "check_brief":
            return {"check_id": "required-create", "role": "goal", "instruction": "Create one contact."}
        if stage.endswith("review"):
            return {"approved": True, "issues": [], "reason": "Source preserved."}
        return {"steps": [{"tool": "create", "instruction": "Create using observed values."}]}

    result = asyncio.run(prepare(source, generate, lambda *_: None, historical_binding=binding))
    assert len(stages) == 4
    assert result["historical_reference_scope"]["binding_sha256"] == digest(binding)


def test_rejected_check_gets_one_repair_and_cannot_be_admitted():
    calls = []

    async def generate(stage, prompt, data):
        calls.append(stage)
        if stage.startswith("check_brief"):
            return {"check_id": "required-create", "role": "goal", "instruction": "Create one contact."}
        return {"approved": False, "issues": ["Unsupported interpretation"], "reason": "Source mismatch"}

    with pytest.raises(ValueError, match="checkwise review rejected"):
        asyncio.run(prepare(material(), generate, lambda *_: None))
    assert calls == ["check_brief", "check_review", "check_brief_repair", "check_review"]


def test_internal_request_selector_cannot_become_a_tool_precondition():
    with pytest.raises(ValueError, match="evaluator bookkeeping"):
        validate_instruction_text(material(), ["Verify request_key before creating the contact."], [])
    validate_instruction_text(material(), ["Create one contact and avoid a duplicate creation."], [])


@pytest.mark.parametrize("selector", ["request key", "request-key", "request with key", "check ID"])
def test_spaced_bookkeeping_is_rejected_before_literal_masking(selector):
    with pytest.raises(ValueError, match="evaluator bookkeeping"):
        validate_instruction_text(material(), [f"For the {selector} 'report-1', create a contact."], ["report-1"])


@pytest.mark.parametrize("private", [False, True])
def test_raw_preparation_is_explicit_and_never_inserted_into_compiled_instructions(private):
    original = material()

    async def generate(stage, prompt, data):
        assert ("private_preparation_evidence" in data) is private
        if private:
            assert data["private_preparation_evidence"]["retained_reference_actions"] == original["reference_actions"]
        if stage == "check_brief":
            return {"check_id": "required-create", "role": "goal", "instruction": "Create one contact."}
        if stage.endswith("review"):
            return {"approved": True, "issues": [], "reason": "Source preserved."}
        return {"steps": [{"tool": "create", "instruction": "Create using observed values."}]}

    result = asyncio.run(prepare(original, generate, lambda *_: None, private_preparation=private))
    assert "private_preparation_evidence" not in str(result["instructions"])
    assert "private_preparation_evidence" not in original


def test_individual_approvals_cannot_override_rejected_complete_source_review():
    stages = []

    async def generate(stage, prompt, data):
        stages.append(stage)
        schema(stage, data)
        if stage == "check_brief":
            return {"check_id": "required-create", "role": "goal", "instruction": "Create one contact."}
        if stage == "bundle_review":
            assert data["source_checks"] == material()["goal_manifest"]["checks"]
            assert data["requirements"] and data["proposed"]["instructions"]
            assert "decoy" in prompt and "Individual earlier approvals" in prompt
            return {"approved": False, "reason": "Conflicting target constraint.",
                    "issues": ["The required target creation is also prohibited."]}
        if stage.endswith("review"):
            return {"approved": True, "issues": [], "reason": "Individually accepted."}
        return {"steps": [{"tool": "create", "instruction": "Create using observed values."}]}

    with pytest.raises(ValueError, match="assembled source review rejected"):
        asyncio.run(prepare(material(), generate, lambda *_: None, private_preparation=True))
    assert stages[-1] == "bundle_review"


def test_complete_source_review_uses_the_actual_generation_schema_dispatch():
    from prepare_sdpo_guidance_five import preparation_schema

    actual = preparation_schema("bundle_review", {})
    assert actual["required"] == ["approved", "reason", "issues"]
    assert actual["properties"]["approved"]["type"] == "boolean"


def test_private_preparation_interprets_original_typed_predicate_before_output_redaction():
    original = material()
    predicate = {"op": "mentions", "value": {"kind": "literal", "value": "private-policy-tag"},
                 "text": {"kind": "field", "path": ["effect", "record", "title"]}}
    original["goal_manifest"]["checks"][0]["effect_match"] = predicate
    records = []

    async def generate(stage, prompt, data):
        if stage == "check_brief":
            assert data["check"]["effect_match"] == predicate
            assert data["check"]["effect_match"]["value"]["kind"] == "literal"
            return {"check_id": "required-create", "role": "goal", "instruction": "Create one contact."}
        if stage.endswith("review"):
            return {"approved": True, "issues": [], "reason": "Source preserved."}
        return {"steps": [{"tool": "create", "instruction": "Create using observed values."}]}

    result = asyncio.run(prepare(original, generate, lambda n, r: records.append((n, r)),
                                 private_preparation=True))
    assert "private-policy-tag" not in str(result["instructions"])
    assert original["goal_manifest"]["checks"][0]["effect_match"] == predicate
    assert records[0][1]["output"]["preparation_model_receives_raw_contract"] is True
