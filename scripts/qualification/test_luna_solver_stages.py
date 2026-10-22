import asyncio
import copy
import json
from pathlib import Path

import pytest
from luna_solver_stages import prepare_instructions


def fixture():
    material = {
        "user_request": "Read the request and update the contact mobile number.",
        "tool_definitions": [],
        "goal_manifest": {
            "checks": [
                {"check_id": "read-before-write", "role": "goal", "order": "returned_before_dispatch"},
                {"check_id": "other-records", "role": "harm", "operator": "effects.prohibited_when@1"},
            ]
        },
        "reference_actions": [
            {"call": {"name": "read", "arguments": {}}, "result": {"observed_result": {"success": True}}},
            {
                "call": {"name": "update", "arguments": {"id": "hidden_1234"}},
                "result": {"observed_result": {"success": True}},
            },
        ],
    }
    replies = {
        "requirements": {
            "requirements": [
                {
                    "check_id": "read-before-write",
                    "role": "goal",
                    "instruction": "Read before update.",
                    "evidence": ["/checks/0"],
                },
                {
                    "check_id": "other-records",
                    "role": "harm",
                    "instruction": "Leave other records unchanged.",
                    "evidence": ["/checks/1"],
                },
            ]
        },
        "workflow": {
            "actions": [
                {
                    "action_id": "a1",
                    "tool": "read",
                    "disposition": "reuse",
                    "instruction": "Retrieve the request.",
                    "depends_on": [],
                },
                {
                    "action_id": "a2",
                    "tool": "update",
                    "disposition": "reuse",
                    "instruction": "Update the retrieved contact.",
                    "depends_on": ["a1"],
                },
            ]
        },
        "assembly": {
            "steps": [
                {
                    "step_id": "s1",
                    "instruction": "Read the request.",
                    "depends_on": [],
                    "check_ids": [],
                    "action_ids": ["a1"],
                },
                {
                    "step_id": "s2",
                    "instruction": "Update only the matching contact.",
                    "depends_on": ["s1"],
                    "check_ids": ["read-before-write", "other-records"],
                    "action_ids": ["a2"],
                },
            ]
        },
        "review": {
            "checks": [
                {"check_id": c, "covered": True, "reason": "The instruction preserves the condition."}
                for c in ("read-before-write", "other-records")
            ],
            "workflow_faithful": True,
            "no_private_literals": True,
            "approved": True,
            "issues": [],
        },
    }
    return material, replies


def run(material, replies):
    records, inputs = [], {}

    async def generate(stage, prompt, data):
        inputs[stage] = copy.deepcopy(data)
        return replies[stage]

    lines = asyncio.run(prepare_instructions(material, generate, lambda n, v: records.append((n, v))))
    return lines, records, inputs


def test_all_same_model_stages_are_called_and_original_evidence_survives():
    material, replies = fixture()
    lines, records, inputs = run(material, replies)
    assert [n for n, _ in records] == ["requirements", "workflow", "assembly", "review"]
    assert inputs["requirements"]["manifest"]["checks"][0]["order"] == "returned_before_dispatch"
    assert inputs["workflow"]["actions"][1]["call"]["arguments"]["id"] == "hidden_1234"
    assert "hidden_1234" not in str(lines)
    assert "Wait for completed steps s1 before dispatch" in lines[1]
    assert material["goal_manifest"] == inputs["review"]["manifest"]


def test_optin_validation_repair_keeps_dependency_gate_strict():
    material, replies = fixture()
    material["task"] = "test.example"
    good = copy.deepcopy(replies["assembly"])
    replies["assembly"]["steps"][1]["depends_on"] = ["a1"]
    seen = []

    async def generate(stage, prompt, data):
        seen.append(stage)
        return copy.deepcopy(good if stage == "assembly_repair" else replies[stage])

    lines = asyncio.run(prepare_instructions(material, generate, lambda *_: None,
                                             repair_invalid_assembly=True))
    assert seen == ["requirements", "workflow", "assembly", "assembly_repair", "review"]
    assert "Wait for completed steps s1" in lines[-1]


@pytest.mark.parametrize(
    "fault", ["lost_harm", "wrong_evidence", "forward_dependency", "leaked_id", "review_rejected", "empty_reused"]
)
def test_pipeline_stops_before_solver_for_invalid_instruction_stage(fault):
    material, replies = fixture()
    if fault == "lost_harm":
        replies["requirements"]["requirements"].pop()
    elif fault == "wrong_evidence":
        replies["requirements"]["requirements"][0]["evidence"] = ["/checks/1"]
    elif fault == "forward_dependency":
        replies["assembly"]["steps"][0]["depends_on"] = ["s2"]
    elif fault == "leaked_id":
        replies["assembly"]["steps"][1]["instruction"] = "Update hidden_1234."
    elif fault == "empty_reused":
        material["reference_actions"][0]["result"]["observed_result"]["result_count"] = 0
    else:
        replies["review"]["approved"] = False
    with pytest.raises(ValueError):
        run(material, replies)


def test_actual_two_task_sources_retain_predicates_and_read_order_for_llm():
    path = (
        Path(__file__).resolve().parents[2]
        / "docs/research/evidence/luna-ssr-sft-20261018/solver-pilot2-r2/4b/materials.json"
    )
    materials = json.loads(path.read_text())
    captured = []

    async def stop_after_input(stage, prompt, data):
        captured.append(data)
        raise RuntimeError("input captured without GPU generation")

    for material in materials.values():
        with pytest.raises(RuntimeError, match="input captured"):
            asyncio.run(prepare_instructions(material, stop_after_input, lambda *_: None))
    assert len(captured) == 2
    for data in captured:
        serialized = json.dumps(data["manifest"])
        assert "returned_before_dispatch" in serialized
        assert "effect_match" in serialized
        assert any(c["role"] == "harm" for c in data["manifest"]["checks"])


def test_advisory_critic_is_explicitly_diagnostic_and_does_not_bypass_literal_gate():
    material, replies = fixture()
    material["task"] = "test"
    replies["review"]["approved"] = False
    records = []

    async def generate(stage, prompt, data):
        return replies[stage]

    lines = asyncio.run(
        prepare_instructions(material, generate, lambda n, v: records.append((n, v)), "structured-provenance-v4")
    )
    assert lines
    assert records[-1][0] == "review_rejected_diagnostic_only"
    assert records[-1][1]["output"]["sft_qualified"] is False
    replies["assembly"]["steps"][1]["instruction"] = "Update hidden_1234."
    with pytest.raises(ValueError, match="nonpublic literal"):
        asyncio.run(prepare_instructions(material, generate, lambda *_: None, "structured-provenance-v4"))


def test_retained_requirements_prevent_assembly_dropping_a_guard():
    material, replies = fixture()
    material["task"] = "test"
    replies["assembly"]["steps"][1]["check_ids"] = ["read-before-write"]
    records = []

    async def generate(stage, prompt, data):
        return replies[stage]

    lines = asyncio.run(
        prepare_instructions(material, generate, lambda n, v: records.append((n, v)), "structured-provenance-v4")
    )
    assert "harm requirement: Leave other records unchanged." in lines
    retained = next(v for n, v in records if n == "preserved_requirements")
    assert retained["output"]["plan_uncited_check_ids"] == ["other-records"]
