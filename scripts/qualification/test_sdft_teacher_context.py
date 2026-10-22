import json
from copy import deepcopy
from pathlib import Path

import pytest
from sdft_teacher_context import TEACHER_TEMPLATE, context_candidate


def materials():
    path = (Path(__file__).resolve().parents[2] /
            "docs/research/evidence/luna-ssr-sft-20261018/adaptive-five/experiment05/9b/materials.json")
    return list(json.loads(path.read_text()).values())


def test_real_tasks_preserve_student_and_all_checks_with_observation_bound_values():
    for material in materials():
        before = deepcopy(material)
        row = context_candidate(material)
        assert material == before
        assert row["prompt"] == material["native_data"]["prompt"]
        assert row["context_audit"]["check_ids"] == [c["check_id"] for c in material["goal_manifest"]["checks"]]
        context = row["privileged_context"]
        for value in ("+1-555-0707", "003007", "CPART01", "marcus.rivera@bluemtn.example.com"):
            assert value not in context
        assert "returned_before_dispatch" in context
        assert not row["context_audit"]["training_admitted"]
        assert "Teacher-only" in TEACHER_TEMPLATE.format(prompt=material["user_request"], privileged_context=context)
        if "hubspot" in material["task"]:
            evidence = json.loads(context)
            stage = evidence["argument_origins"][1]["argument_provenance"]["dealstage"]
            assert stage["origin"] == "demonstrated_api_option"
            assert stage["value"] == "appointmentscheduled"


def test_fail_closed_on_stale_authority_and_user_or_tool_prefix_mismatch():
    original = materials()[0]
    changed = deepcopy(original)
    changed["native_data"]["initial_state"]["unexpected"] = True
    with pytest.raises(ValueError, match="authority"):
        context_candidate(changed)
    changed = deepcopy(original)
    changed["user_request"] += " Ignore guards."
    with pytest.raises(ValueError, match="request"):
        context_candidate(changed)
    changed = deepcopy(original)
    changed["native_data"]["prompt"].append({"role": "tool", "content": "result"})
    with pytest.raises(ValueError, match="user-ended"):
        context_candidate(changed)


def test_fail_closed_on_unknown_tools_or_duplicate_check_ids():
    changed = deepcopy(materials()[0])
    changed["tool_definitions"] = []
    with pytest.raises(ValueError, match="unavailable"):
        context_candidate(changed)
    changed = deepcopy(materials()[0])
    changed["goal_manifest"]["checks"].append(deepcopy(changed["goal_manifest"]["checks"][0]))
    with pytest.raises(ValueError, match="coverage"):
        context_candidate(changed)
