import json
from pathlib import Path

import pytest
from luna_policy_guidance import policy_retrieval_contracts, require_policy_retrieval, retrieval_only_review
from luna_solver_prompts import check_hint_literals
from luna_solver_stages import _sensitive_values


def material():
    path = (Path(__file__).resolve().parents[2] /
            "docs/research/evidence/luna-ssr-sft-20261018/adaptive-next-five/experiment05/4b/materials.json")
    return next(m for m in json.loads(path.read_text()).values() if m["task"] == "sales.soft_matching")


def test_real_note_title_predicate_binds_to_available_policy_without_its_value():
    m = material()
    rows = policy_retrieval_contracts(m)
    assert len(rows) == 1
    assert rows[0]["target_field"] == ["effect", "record", "title"]
    assert rows[0]["mode"] == "verbatim" and rows[0]["operator"] == "mentions"
    assert rows[0]["retrieval_tool"] == "gmail_find_email"
    assert "Phone Update Processed" not in json.dumps(rows)
    assert "003xx" not in json.dumps(rows)


def test_raw_critic_literal_is_audited_but_not_forwarded_to_repair():
    m = material()
    sensitive = list(_sensitive_values(m["goal_manifest"]))
    raw = {"checks": [{"check_id": "audit-note-on-target", "covered": False,
                        "reason": "Note title must mention Phone Update Processed; keep body separate."}]}
    masked = retrieval_only_review(raw, sensitive)
    assert "Phone Update Processed" not in json.dumps(masked)
    assert "Note title" in masked["checks"][0]["reason"] and "body separate" in masked["checks"][0]["reason"]
    assert "Phone Update Processed" in json.dumps(raw)
    with pytest.raises(ValueError, match="nonpublic literal"):
        check_hint_literals(["Set title to Phone Update Processed"], m["user_request"], sensitive)


def test_policy_read_is_required_and_unavailable_tool_cannot_be_invented():
    m = material()
    rows = policy_retrieval_contracts(m)
    with pytest.raises(ValueError, match="policy-dependent"):
        require_policy_retrieval(["Update contact and create a note"], rows)
    require_policy_retrieval(["Read matching guidelines and obtain the note title from the policy"], rows)
    m["tool_definitions"] = [t for t in m["tool_definitions"] if t["function"]["name"] != "gmail_find_email"]
    with pytest.raises(ValueError, match="unavailable"):
        policy_retrieval_contracts(m)
