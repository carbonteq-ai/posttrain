import json

from sdpo_hindsight_projection import project


def test_private_answers_and_future_entities_are_absent():
    result = project([
        {"type": "phone_equals", "passed": False, "params": {"phone": "SECRET-PHONE"}},
        {"type": "decoy_unchanged", "passed": True, "excluded": True,
         "params": {"id": "SECRET-DECOY"}},
    ], [{"tool": "create_task", "response": json.dumps({"success": True,
        "results": [{"id": "FUTURE-TASK-ID"}], "count": 1})}])
    text = json.dumps(result)
    assert all(value not in text for value in ("SECRET-PHONE", "SECRET-DECOY", "FUTURE-TASK-ID"))
    assert result["excluded_assertion_count"] == 1
    assert result["terminal_native_assertions"] == [{"type": "phone_equals", "passed": False}]
    assert result["current_action_observations"][0]["per_action_correctness"] is None


def test_empty_success_is_not_a_negative_action_label():
    result = project([], [{"tool": "search", "response": '{"success":true,"messages":[],"result_count":0}'}])
    row = result["current_action_observations"][0]
    assert row["response_status"] == "reported_success"
    assert row["empty_result"] is True
    assert row["per_action_correctness"] is None


def test_unknown_transport_and_nested_emptiness_stay_unknown():
    result = project([], [{"tool": "get", "response": "unstructured error-looking text"},
                          {"tool": "get", "response": {"success": True, "contact": {"attachments": []}}}])
    assert result["current_action_observations"][0]["response_status"] == "unknown"
    assert all(r["empty_result"] is None for r in result["current_action_observations"])


def test_reported_error_has_no_invented_repair():
    result = project([], [{"tool": "query", "response": {"error": "private query text"}}])
    assert result["current_action_observations"][0]["response_status"] == "reported_error"
    assert "private query text" not in json.dumps(result)
