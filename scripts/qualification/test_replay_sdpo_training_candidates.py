import copy
import json

import pytest
from replay_sdpo_training_candidates import replay, response, succeeded, verify_capture


def test_pinned_salesforce_read_contract_preserves_raw_payload_and_error():
    for name in ("salesforce_query", "salesforce_find_records"):
        value = {"results": [], "count": 0}
        assert response(value, name) is value and succeeded(value)
        error = {"error": "Invalid SOQL WHERE clause"}
        assert response(error, name) is error and not succeeded(error)
        with pytest.raises(ValueError):
            response({"results": [], "count": 1}, name)
        with pytest.raises(ValueError):
            response({}, name)
    with pytest.raises(ValueError):
        response({"results": [], "count": 0}, "unknown_tool")


def fixture():
    native = {"name": "simple.test", "prompt": [{"role": "user", "content": "Find then reply"}],
              "initial_state": {"sent": []}, "zapier_tools": ["search", "reply"], "assertions": []}
    actions = [{"call": {"name": tool, "arguments": {}},
                "result": {"observed_result": result, "observed_error": None}}
               for tool, result in [("search", {"success": False, "error": "unsupported query"}),
                                    ("search", {"success": True}), ("reply", {"success": True})]]
    return {"task": "simple.test", "material": {"native_data": copy.deepcopy(native)},
            "reference_actions": actions}, native


def invoker(responses, mutate=True):
    remaining = iter(responses)

    def invoke(owner, name, **kwargs):
        if name == "reply" and mutate:
            owner.state.world["sent"].append("reply")
        return next(remaining)

    return invoke


def test_expected_bad_search_recovery_and_persisted_reply():
    row, native = fixture()
    before = copy.deepcopy(native)
    result = replay(row, native, invoker([{"success": False, "error": "unsupported query"},
                                         {"success": True}, {"success": True}]))
    assert result["expected_failed_calls"] == 1 and result["successful_calls"] == 2
    assert result["qualified"] and native == before


@pytest.mark.parametrize("responses,mutate,match", [
    ([{"success": False, "error": "different error"}], True, "expected error changed"),
    ([{"success": False, "error": "unsupported query"}, {"success": True}, {"success": False}],
     True, "status changed"),
    ([{"success": False, "error": "unsupported query"}, {"success": True}, {"success": True}],
     False, "did not persist"),
    ([{}], True, "explicit success"),
])
def test_incompatible_native_path_rejected(responses, mutate, match):
    row, native = fixture()
    with pytest.raises(ValueError, match=match):
        replay(row, native, invoker(responses, mutate))


def test_changed_native_source_rejected_before_any_call():
    row, native = fixture()
    native["initial_state"] = {"different": True}
    with pytest.raises(ValueError, match="initial_state differs"):
        replay(row, native, invoker([]))


def test_native_capture_must_match_actual_call_and_result():
    result = {"calls": [{"index": 0, "tool": "reply", "observed": {"success": True}}]}
    events = [{"action": {"occurrence_index": 0, "tool_name": "reply", "status": "returned",
                           "result_json": json.dumps(json.dumps({"success": True}))}}]
    assert verify_capture(result, events)["calls"] == 1
    with pytest.raises(ValueError, match="every replayed call"):
        verify_capture(result, [])
    events[0]["action"]["result_json"] = json.dumps({"success": False})
    with pytest.raises(ValueError, match="differs from replay"):
        verify_capture(result, events)


def test_execution_success_does_not_hide_failed_native_goal_scoring():
    row, native = fixture()

    def score(**kwargs):
        assert kwargs["world"]["sent"] == ["reply"]
        return {"partial_credit": .9, "task_completed_correctly": 0, "assertions_excluded": 2}

    result = replay(row, native, invoker([{"success": False, "error": "unsupported query"},
        {"success": True}, {"success": True}]), scorer=score)
    assert result["qualified"]  # Invoke/capture compatibility, explicitly not semantic completion.
    assert result["native_scoring"]["task_completed_correctly"] == 0
    assert result["native_scoring"]["assertions_excluded"] == 2
