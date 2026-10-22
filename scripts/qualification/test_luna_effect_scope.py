import asyncio
import copy
import json
from pathlib import Path

import pytest
from luna_effect_scope import SCOPE_RULES, effect_scopes, private_email_domains
from luna_solver_prompts import check_hint_literals
from luna_solver_schema import instruction_schema
from luna_solver_separation import observation_bound_sources
from luna_solver_stages import prepare_instructions


def test_real_creation_and_overwrite_scopes_remain_distinct_even_with_same_event_kind():
    path = (Path(__file__).resolve().parents[2] /
            "docs/research/evidence/luna-ssr-sft-20261018/sft-batch01-20261019-r2/9b/materials.json")
    m = next(m for m in json.loads(path.read_text()).values() if m["task"] == "simple.airtable_find_update")
    scopes = effect_scopes(m["goal_manifest"])
    assert scopes[0]["role"] == "goal" and scopes[0]["effect_collection"] == ["actions", "createRecord"]
    assert scopes[1]["role"] == "harm" and scopes[1]["effect_collection"] == ["actions", "updateRecord"]
    assert scopes[0]["event_kind"] == scopes[1]["event_kind"] == "create"
    assert scopes[1]["predicate"] == m["goal_manifest"]["checks"][1]["prohibited_when"]
    assert "does NOT prohibit a requested creation" in SCOPE_RULES


def test_schema_excludes_diagnostics_without_shifting_original_evidence_pointers():
    checks = [{"check_id": "g", "role": "goal"}, {"check_id": "coverage", "role": "diagnostic"},
              {"check_id": "h", "role": "harm"}]
    schema = instruction_schema("requirements", {"manifest": {"checks": checks}})
    rows = schema["properties"]["requirements"]
    assert rows["minItems"] == rows["maxItems"] == 2
    assert rows["items"]["properties"]["check_id"]["enum"] == ["g", "h"]
    assert rows["items"]["properties"]["evidence"]["items"]["enum"] == ["/checks/0", "/checks/2"]
    review = instruction_schema("review", {"manifest": {"checks": checks}})
    assert review["properties"]["checks"]["minItems"] == 2


@pytest.mark.parametrize("repair_approved", [True, False])
@pytest.mark.parametrize("variant", ["scope-repair-provenance-v5", "scope-repair-natural-provenance-v5"])
def test_actual_bad_plan_receives_only_one_repair_and_cannot_bypass_rejection(repair_approved, variant):
    root = (Path(__file__).resolve().parents[2] /
            "docs/research/evidence/luna-ssr-sft-20261018/adaptive-next-five/experiment02/9b")
    material = next(m for m in json.loads((root / "materials.json").read_text()).values()
                    if m["task"] == "simple.airtable_find_update")
    replies = {s["stage"]: s["output"] for s in map(json.loads, (root / "instruction-stages.jsonl").read_text().splitlines())
               if s["task"] == material["task"] and s["stage"] in ("requirements", "workflow", "assembly", "review")}
    assert not replies["review"]["approved"]
    repaired = copy.deepcopy(replies["assembly"])
    repaired["steps"][0]["instruction"] = "Find the contact requested by the user, then create a separate VIP record without updating the old one."
    replies["assembly_repair"] = repaired
    review = copy.deepcopy(replies["review"])
    review.update(approved=repair_approved, workflow_faithful=repair_approved,
                  issues=[] if repair_approved else ["Unresolved invention"])
    for check in review["checks"]:
        check["covered"] = repair_approved
        check["reason"] = "Repaired scope reviewed." if repair_approved else "Still incorrect."
    replies["review_repair"] = review
    calls = []

    async def generate(stage, prompt, data):
        calls.append(stage)
        return copy.deepcopy(replies[stage])

    run = prepare_instructions(material, generate, lambda *_: None, variant)
    if repair_approved:
        lines = asyncio.run(run)
        assert "verify the absence" not in "\n".join(lines)
        if "natural" in variant:
            assert not any("requirement:" in line or line.startswith("s1:") for line in lines)
    else:
        with pytest.raises(ValueError, match="bounded assembly repair rejected"):
            asyncio.run(run)
    assert calls == ["requirements", "workflow", "assembly", "review", "assembly_repair", "review_repair"]


def test_sales_domain_extraction_is_typed_without_revealing_the_answer():
    path = (Path(__file__).resolve().parents[2] /
            "docs/research/evidence/luna-ssr-sft-20261018/adaptive-next-five/experiment03/9b/materials.json")
    m = next(m for m in json.loads(path.read_text()).values() if m["task"] == "sales.soft_matching")
    contract, actions, _ = observation_bound_sources(m)
    text = json.dumps([contract, actions])
    assert "extract the email sender domain (part after @), not the sender display name" in text
    assert "nexgensol.example.com" not in text
    private = private_email_domains(m, m["user_request"] + json.dumps(m["tool_definitions"]))
    assert "nexgensol.example.com" in private
    with pytest.raises(ValueError, match="nonpublic literal"):
        check_hint_literals(["Use nexgensol.example.com"], m["user_request"], private)
