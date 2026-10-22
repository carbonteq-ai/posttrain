"""Protect teacher/student separation and exact current-state source binding."""

import copy

import pytest
from sdpo_guidance_context import digest, native_findings, teacher_messages


def fixture():
    call = {"messages": [{"role": "user", "content": "Read then update the record"}],
            "prefix_ids": [1, 2], "action_ids": [3, 4]}
    task = {"name": "sales.example", "initial_state": {"records": ["current"]},
            "prompt": copy.deepcopy(call["messages"])}
    bundle = {"task": task["name"], "current_state_sha256": digest(task["initial_state"]),
        "user_request_sha256": digest(call["messages"][0]["content"]),
        "review_approved": True, "prepared_by_model": "Qwen/Qwen3.5-4B",
        "instructions": ["Retrieve policy before using it"], "check_ids": ["goal1", "guard1"],
        "check_roles": {"goal1": "goal", "guard1": "harm"},
        "check_signals": {"goal1": "native_goal_signal", "guard1": "native_harm_signal"}}
    return call, task, bundle


def test_only_teacher_gains_context_and_no_action_is_replaced():
    call, task, bundle = fixture()
    old = copy.deepcopy(call)
    messages, audit = teacher_messages(call, task, bundle, [{"passed": False}], [{"results": []}])
    assert call == old
    assert messages[:-1] == old["messages"]
    assert "Retrieve policy" in messages[-1]["content"]
    assert audit["student_receives_guidance"] is False
    assert audit["terminal_findings_are_per_action_credit"] is False


def test_matched_feedback_only_control_removes_guide_without_changing_factual_feedback():
    call, task, bundle = fixture()
    full, _ = teacher_messages(call, task, bundle, [{"passed": False}], [{"results": []}])
    control, audit = teacher_messages(call, task, bundle, [{"passed": False}], [{"results": []}],
                                     context_mode="feedback-only")
    assert "Retrieve policy" not in control[-1]["content"]
    assert '"guidance": null' in control[-1]["content"]
    assert full[-1]["content"].split("Factual hindsight", 1)[1] == control[-1]["content"].split("Factual hindsight", 1)[1]
    assert full[:-1] == control[:-1] == call["messages"]
    assert audit["teacher_receives_guidance"] is False


def test_status_projection_keeps_original_student_and_native_missingness():
    call, task, bundle = fixture()
    old = copy.deepcopy(call)
    findings = {"summary": [], "missing_signal_ids": ["unknown_goal"], "per_action_credit": False}
    messages, audit = teacher_messages(call, task, bundle,
        [{"type": "phone_equals", "passed": False, "params": {"phone": "PRIVATE-PHONE"}}],
        [{"tool": "search", "response": '{"success":true,"messages":[],"result_count":0}'}],
        findings, context_mode="feedback-status")
    assert call == old and messages[:-1] == old["messages"]
    text = messages[-1]["content"]
    assert "PRIVATE-PHONE" not in text and "Retrieve policy" not in text
    assert '"empty_result": true' in text and "unknown_goal" in text
    assert audit["hindsight_projection"] == "status-only-candidate-v1"
    assert audit["student_receives_guidance"] is False
    assert audit["teacher_receives_guidance"] is False


@pytest.mark.parametrize("field,value", [("review_approved", False),
    ("prepared_by_model", "Qwen/Qwen3.5-27B"), ("current_state_sha256", "stale"),
    ("instructions", []), ("check_ids", ["same", "same"]), ("user_request_sha256", "wrong")])
def test_reject_unbound_or_unreviewed_context(field, value):
    call, task, bundle = fixture()
    bundle[field] = value
    with pytest.raises(ValueError):
        teacher_messages(call, task, bundle, [], [])


def test_missing_assessment_is_unknown_and_deduplicated_goal_failure_survives():
    call, task, bundle = fixture()
    item = {"assessment_id": "a1", "signal": {"signal_id": "native_goal_signal"},
            "status": "valid", "value": 0, "reason": "goal_not_met"}
    batch = {"run": {"status": "complete"}, "assessments": [item]}
    episode = {"traces": [{"assessment_batches": [batch, copy.deepcopy(batch)]}]}
    result = native_findings(episode, bundle)
    assert result["observed_obligation_failure"] is True
    assert len(result["findings"]) == 1
    assert result["missing_signal_ids"] == ["native_harm_signal"]
    assert result["per_action_credit"] is False
    messages, _ = teacher_messages(call, task, bundle, [], [], result)
    assert "native_goal_signal" in messages[-1]["content"]
    assert "assessment_id" not in messages[-1]["content"]
    assert result["summary"][0]["zero_values"] == 1


def test_frontier_sees_native_goal_failure_even_when_legacy_success_passes():
    from sdpo_agent_config import balanced_rows

    common = {"task": "sales.example", "episode_id": "e1", "native_success": 1,
              "executed_tool_responses": 1, "terminal_text": False}
    good = {**common, "generation_index": 0}
    bad = {**common, "generation_index": 3, "observed_obligation_failure": True}
    chosen = balanced_rows([good, bad], batch=1, tasks=["sales.example"], mode="first-error-frontier")
    assert chosen == [bad]
