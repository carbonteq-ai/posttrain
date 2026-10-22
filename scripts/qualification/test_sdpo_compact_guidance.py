import copy
import json
from pathlib import Path

import pytest
from sdpo_compact_guidance import compile_guidance, instructions, render

EVIDENCE = Path(__file__).resolve().parents[2] / "docs/research/evidence/sdpo-guidance-five-20261019"
TASKS = ["sales.soft_matching", "hr.twilio_emergency_notification",
         "marketing.email_campaign_analytics", "hr.recruitment_dedup_alert"]


def materials():
    return {r["task"]: r["material"] for r in json.loads((EVIDENCE / "balanced-task-source-audit.json").read_text())["eligible"]}


@pytest.mark.parametrize("task", TASKS)
def test_real_four_tasks_are_deterministic_complete_and_source_immutable(task):
    material = materials()[task]
    before = copy.deepcopy(material)
    compiled = compile_guidance(material)
    assert compiled == compile_guidance(material) and material == before
    checks = [c for c in material["goal_manifest"]["checks"] if c["role"] in ("goal", "harm")]
    assert compiled["audit"]["check_coverage"] == [c["check_id"] for c in checks]
    assert [c["raw"] for c in compiled["audit"]["checks"]] == checks
    assert len(compiled["sheet"]["constraints"]) == len(checks)
    text = instructions(compiled)[0]
    for value in compiled["audit"]["private_values"].values():
        for item in value if isinstance(value, list) else [value]:
            if isinstance(item, str) and len(item) > 4:
                assert item not in text


def test_contact_identity_and_harm_polarity_share_the_same_private_binding():
    compiled = compile_guidance(materials()[TASKS[0]])
    rows = compiled["sheet"]["constraints"]
    assert rows[0]["when"] == "row.id = V1"
    assert rows[1]["when"] == "row.id != V1" and rows[1]["mode"] == "FORBID"
    assert rows[-1]["when"] == "row.id != V1"
    assert rows[-1]["match"] == "effect.record.parent_id = row.id"
    assert "request ID" not in rows[-1]["match"]
    assert "Phone Update Processed" not in instructions(compiled)[0]


def test_emergency_audiences_and_exact_headers_and_prior_read_join_stay_distinct():
    compiled = compile_guidance(materials()[TASKS[1]])
    rows = compiled["sheet"]["constraints"]
    assert 'row."SMS Opt-In"' in rows[0]["when"]
    assert rows[1]["when"] == "NOT(" + rows[0]["when"] + ")"
    assert rows[2]["each_row_from"] != rows[0]["each_row_from"]
    assert 'row."Facilities Email" IN effect.recipients' in rows[2]["match"]
    assert rows[0]["constraints"]["effect_joins"][0]["timing"] == "before"
    assert "all_rows_returned = true" in rows[0]["constraints"]["effect_joins"][0]["where"]


def test_public_request_binding_is_resolved_not_presented_as_invisible_actor_field():
    compiled = compile_guidance(materials()[TASKS[2]])
    row = compiled["sheet"]["constraints"][0]
    assert row["when"] == "true"
    assert "row.target" not in row["match"] and "effect.channel_id = V" in row["match"]
    assert "gmail_find_email" in compiled["sheet"]["historical_read_tools"]
    assert "SENT" not in instructions(compiled)[0]
    assert len(compiled["sheet"]["constraints"]) == 1  # Manifest assesses delivery only.


def test_duplicate_alert_keeps_line_cooccurrence_and_no_added_timestamp_write():
    compiled = compile_guidance(materials()[TASKS[3]])
    assert "CONTAINS_TOGETHER[line]" in compiled["sheet"]["constraints"][1]["match"]
    assert '"merge"' in compiled["sheet"]["constraints"][1]["match"]
    assert "Last Notified" not in instructions(compiled)[0]


def test_unknown_operator_blocks_instead_of_silently_dropping_requirement():
    material = copy.deepcopy(materials()[TASKS[0]])
    material["goal_manifest"]["checks"][0]["operator"] = "new.operator@1"
    with pytest.raises(ValueError, match="unsupported operator"):
        compile_guidance(material)


@pytest.mark.parametrize("operator", ["sheets.retained_when@1", "records.retained_when@1",
                                      "objects.created_and_retained@1", "record.fields_equal@1"])
def test_retained_and_field_checks_keep_operator_semantics_and_private_values(operator):
    material = copy.deepcopy(materials()[TASKS[0]])
    check = material["goal_manifest"]["checks"][0]
    check["operator"] = operator
    check["retained_when"] = {"op": "eq", "left": {"kind": "field", "path": ["retained", "id"]},
                              "right": {"kind": "literal", "value": "SECRET-ONLY-EXPECTED"}}
    if operator == "record.fields_equal@1":
        check["expected"] = [{"field": "amount", "value": 175987654, "comparison": "number"}]
    compiled = compile_guidance(material)
    row = compiled["sheet"]["constraints"][0]
    assert row["operator"] == operator and row["semantics"]
    assert compiled["audit"]["checks"][0]["raw"] == check
    assert "SECRET-ONLY-EXPECTED" not in instructions(compiled)[0]
    assert "175987654" not in instructions(compiled)[0]


def test_exclude_none_literal_null_remains_null_not_a_missing_binding():
    material = copy.deepcopy(materials()[TASKS[0]])
    check = material["goal_manifest"]["checks"][0]
    check["effect_match"] = {"op": "eq", "left": {"kind": "literal"},
                             "right": {"kind": "literal", "value": None}}
    assert compile_guidance(material)["sheet"]["constraints"][0]["match"] == "true"


def test_native_copy_from_with_null_default_is_not_compiled_as_null():
    material = copy.deepcopy(materials()[TASKS[2]])
    source = next(s for s in material["goal_manifest"]["sources"].values()
                  if s["adapter"] == "public.request@1")
    source["fields"]["target"] = {"value": None, "copy_from": ["task_evidence", "initial", "selected_target"]}
    material["native_data"]["initial_state"]["selected_target"] = "private_runtime_channel"
    compiled = compile_guidance(material)
    text = instructions(compiled)[0]
    assert "effect.channel_id = V" in text
    assert "effect.channel_id = null" not in text and "private_runtime_channel" not in text
    assert "private_runtime_channel" in compiled["audit"]["private_values"].values()
    del material["native_data"]["initial_state"]["selected_target"]
    with pytest.raises(ValueError, match="unresolved native copy_from"):
        compile_guidance(material)


def test_nested_boolean_and_mentions_modes_are_not_flattened():
    assert render({"op": "not", "arg": {"op": "any", "args": [
        {"op": "eq", "left": {"field": ["row", "id"]}, "right": {"observed_value": "V1"}},
        {"op": "ne", "left": {"field": ["row", "id"]}, "right": {"observed_value": "V2"}}]}}) == "NOT((row.id = V1 OR row.id != V2))"


def test_teacher_wrapper_preserves_original_context_and_excludes_private_sidecar():
    from qualify_sdpo_compact_four import guided_messages

    material = materials()[TASKS[0]]
    rows = [{"task": TASKS[0], "material": material}]
    compiled = {TASKS[0]: compile_guidance(material)}
    messages = [{"role": "user", "content": material["user_request"]}]
    before = copy.deepcopy(messages)
    output, task = guided_messages(messages, rows, compiled)
    assert task == TASKS[0] and output[:-1] == messages == before
    assert "private_values" not in output[-1]["content"]
    assert "003xx000004KOC1" not in output[-1]["content"]
    with pytest.raises(ValueError, match="exact unique"):
        guided_messages([], rows, compiled)


def test_phone_format_alternative_resolves_to_same_observation_location():
    compiled = compile_guidance(materials()[TASKS[0]])
    assert "gmail.messages.*.body_plain" in compiled["sheet"]["retrieve_values_from"]["V3"]


def test_retrieval_routes_use_only_declared_capabilities_and_public_channel_name():
    hr = compile_guidance(materials()[TASKS[1]])["sheet"]
    assert any("first discover" in line and "returned bare id" in line for line in hr["retrieval_prerequisites"])
    assert any('"#company-announcements"' in line for line in hr["retrieval_prerequisites"])
    sales = compile_guidance(materials()[TASKS[0]])["sheet"]
    assert not any("google_drive_find_multiple_files" in line for line in sales["retrieval_prerequisites"])
    marketing = compile_guidance(materials()[TASKS[2]])["sheet"]
    assert "request_key" not in json.dumps(marketing)
    assert "tracking_code" not in json.dumps(marketing)
