import copy
import json

import pytest
from qualify_combined_teacher_contexts import (
    arm_batch,
    bind_generated_threads,
    native_messages,
    retained_history_arguments,
    serving_identity,
)


def test_context_treatment_preserves_student_actions_and_other_rows():
    original = [{"episode_id": str(i), "generation_index": i, "student_prefix_ids": [100, i],
                 "action_ids": [200, i], "teacher_prefix_ids": [300, i]} for i in range(32)]
    before = copy.deepcopy(original)
    alternatives = {(str(i), i): [400, i] for i in range(4)}
    treatment, changed = arm_batch(original, alternatives, "combined_teacher")
    control, control_changed = arm_batch(original, alternatives, "original_teacher")
    assert original == before == control and control_changed == 0 and changed == 4
    for i, row in enumerate(treatment):
        assert row["student_prefix_ids"] == before[i]["student_prefix_ids"]
        assert row["action_ids"] == before[i]["action_ids"]
        assert row["teacher_prefix_ids"] == ([400, i] if i < 4 else before[i]["teacher_prefix_ids"])


def test_partial_treatment_cannot_silently_run():
    with pytest.raises(ValueError, match="exactly four"):
        arm_batch([{"episode_id": "one", "generation_index": 1}], {}, "combined_teacher")


def test_missing_generated_threads_bind_to_real_observations_only():
    initial = {"gmail": {"messages": [{"id": "m1", "body_plain": "original"}, {"id": "unobserved"}]}}
    messages = [{"role": "tool", "content": json.dumps({"messages": [{"id": "m1", "thread_id": "real_thread"}]})}]
    binding = bind_generated_threads(initial, messages)
    assert len(binding) == 1
    assert initial["gmail"]["messages"] == [{"id": "m1", "body_plain": "original", "thread_id": "real_thread"}, {"id": "unobserved"}]
    initial["gmail"]["messages"][0]["thread_id"] = "explicit_other"
    with pytest.raises(ValueError, match="explicit"):
        bind_generated_threads(initial, messages)


def test_reasoning_alias_preserves_original_and_rejects_conflict():
    messages = [{"role": "assistant", "reasoning": "original reasoning", "tool_calls": []}]
    restored = native_messages(messages)
    assert restored[0]["reasoning_content"] == "original reasoning"
    assert "reasoning_content" not in messages[0]
    with pytest.raises(ValueError, match="conflicting"):
        native_messages([{**messages[0], "reasoning_content": "different"}])


def test_reference_identity_only_ignores_session_timestamp():
    assert serving_identity("slots=40, instance_id='123'") == serving_identity("slots=40, instance_id='456'")
    assert serving_identity("slots=40, instance_id='123'") != serving_identity("slots=8, instance_id='456'")
    with pytest.raises(ValueError):
        serving_identity("slots=40")


def test_historical_replay_uses_actual_captured_arguments_without_inventing_aliases():
    observed = {"success": True, "spreadsheet": {"id": "sheet"}}
    messages = [{"role": "assistant", "tool_calls": [{"id": "c1", "function": {
        "name": "read", "arguments": '{"include_grid_data":"True"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": json.dumps(observed)}]
    event = {"tool_name": "read", "status": "returned", "arguments_json": '{"includeGridData":false}',
             "result_json": json.dumps(json.dumps(observed)), "occurrence_index": 3}
    case = {"call": {"messages": messages}, "episode": {"traces": [{"info": {
        "automationbench_capture": {"events": [event]}}}]}}
    arguments, audit = retained_history_arguments(case)
    assert arguments == {"c1": {"includeGridData": False}}
    assert audit[0]["model_arguments"] == {"include_grid_data": "True"}
    assert audit[0]["retained_occurrence_index"] == 3
    case["episode"]["traces"][0]["info"]["automationbench_capture"]["events"].append(
        {**event, "arguments_json": '{"includeGridData":true}'})
    with pytest.raises(ValueError, match="ambiguous retained"):
        retained_history_arguments(case)
