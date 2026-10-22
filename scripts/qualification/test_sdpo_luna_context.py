"""Offline reference boundaries and faithful observation rendering."""

import copy

import pytest
from sdpo_luna_context import (
    digest,
    native_task_bindings,
    reference_for_task,
    teacher_text,
    unavailable_reference_tools,
)
from sdpo_luna_preflight import render_events


def fixture():
    data = {"name": "training.task", "initial_state": {"records": []},
            "assertions": [{"type": "created"}], "prompt": [{"role": "user", "content": "Create record"}]}
    reference = {"current_native_checks_pass": True, "observations": "observed calls",
                 "initial_state_sha256": digest(data["initial_state"]),
                 "assertions_sha256": digest(data["assertions"]),
                 "user_messages_sha256": digest(["Create record"])}
    return data, reference


def test_reference_binds_state_goal_and_user_instruction_without_mutating_student():
    data, reference = fixture()
    original = copy.deepcopy(data)
    assert reference_for_task(data, {data["name"]: reference}) is reference
    for key, value in [("initial_state", {"records": ["different"]}),
                       ("assertions", [{"type": "different_goal"}]),
                       ("prompt", [{"role": "user", "content": "Different task"}])]:
        altered = {**data, key: value}
        with pytest.raises(ValueError, match="live task binding"):
            reference_for_task(altered, {data["name"]: reference})
    assert data == original
    assert reference_for_task(data, {}) is None
    with pytest.raises(ValueError):
        reference_for_task(data, {data["name"]: {**reference, "current_native_checks_pass": False}})


def test_teacher_reference_is_historical_observation_and_does_not_claim_gold_thinking():
    _, reference = fixture()
    text = teacher_text(reference)
    assert reference["observations"] in text
    assert "intermediate mistakes" in text
    assert "hidden thinking" in text and "unavailable" in text
    assert "not actions already executed" in text


def test_render_keeps_failed_calls_results_and_exact_underlying_arguments():
    import json

    event = {"tool_name": "execute_tool", "arguments_json":
             '{"tool_name":"update_record","arguments":"{\\"id\\":\\"0002\\"}"}',
             "result_json": '{"success":false,"error":"unknown id"}', "error_json": None,
             "status": "returned", "occurrence_index": 3}
    rendered = json.loads(render_events([event]))
    assert len(rendered) == 1
    assert rendered[0]["underlying_arguments"] == {"id": "0002"}
    assert rendered[0]["observed_result"] == json.loads(event["result_json"])
    assert rendered[0]["source_occurrence_index"] == 3


def test_discovery_projection_keeps_all_domain_results_without_truncating_or_changing_ids():
    import json

    payload = {"rows": [{"id": "0002", "content": "full payload" * 1000}]}
    events = [{"tool_name": "search_tools"},
              {"tool_name": "execute_tool", "arguments_json":
               '{"tool_name":"get_record","arguments":"{\\"id\\":\\"0002\\"}"}',
               "result_json": json.dumps(json.dumps(payload)), "error_json": None,
               "status": "returned", "occurrence_index": 2}]
    rendered = json.loads(render_events(events))
    assert len(rendered) == 1
    assert rendered[0]["observed_result"] == payload
    assert rendered[0]["underlying_arguments"] == {"id": "0002"}
    assert rendered[0]["source_occurrence_index"] == 2


def test_wire_task_extensions_are_recovered_only_by_matching_native_identity():
    data, _ = fixture()
    wire = {"id": "native1", "task": {"data": {k: data[k] for k in ("name", "prompt")}}}
    retained = {"id": "native1", "task": {"data": data}}
    assert native_task_bindings([wire], [retained]) == {"native1": data}
    with pytest.raises(ValueError, match="identity"):
        native_task_bindings([wire], [{**retained, "id": "different"}])
    with pytest.raises(ValueError, match="identity"):
        native_task_bindings([wire], [retained, retained])
    altered = copy.deepcopy(retained)
    altered["task"]["data"]["prompt"][0]["content"] = "Different task"
    with pytest.raises(ValueError, match="instruction"):
        native_task_bindings([wire], [altered])


def test_successful_source_does_not_authorize_an_unavailable_tool():
    import json

    reference = {"observations": json.dumps([{"source_tool": "execute_tool", "underlying_tool": "lookup_row"}])}
    live = [{"function": {"name": "read_sheet"}}]
    assert unavailable_reference_tools(reference, live) == ["lookup_row"]
    assert unavailable_reference_tools(reference, live + [{"function": {"name": "lookup_row"}}]) == []
