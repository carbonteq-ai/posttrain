import pytest
from audit_sdpo_guidance_findings import audit


def fixture():
    return ({"id": "one", "task": {"data": {"name": "task", "initial_state": {},
        "prompt": [{"role": "user", "content": "Act."}]}},
        "traces": [{"assessment_finalization_state": "completed", "assessment_batches": []}]},
        {"rows": [{"task": "task", "material": {"native_data": {"initial_state": {}},
            "user_request": "Act.", "goal_manifest": {"checks": [
                {"check_id": "g", "signal_id": "goal", "role": "goal"},
                {"check_id": "h", "signal_id": "harm", "role": "harm"}]}}}]})


def test_absent_findings_never_become_harm_free():
    episode, material = fixture()
    result = audit([episode], material)
    assert result["missing_signal_episode_count"] == 1
    assert result["episodes"][0]["harm_free"] is None
    assert result["episodes"][0]["missing_signal_ids"] == ["goal", "harm"]


def test_wrong_state_cannot_be_audited_with_another_manifest():
    episode, material = fixture()
    episode["task"]["data"]["initial_state"] = {"different": True}
    with pytest.raises(ValueError, match="initial-state"):
        audit([episode], material)
