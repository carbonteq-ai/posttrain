import json

from analyze_self_repair_run import native_tool_observations


def test_error_only_soql_response_is_counted_without_relabelling_successful_empty_search():
    responses = [{"error": "Invalid SOQL WHERE clause"},
        {"success": False, "error": "jira_project_not_found", "results": [], "count": 0},
        {"success": True, "results": [], "count": 0}, {"success": True, "error": None}]
    nodes = [{"message": {"role": "tool", "tool_call_id": str(i), "content": json.dumps(r)}}
             for i, r in enumerate(responses)]
    episode = {"task": {"data": {"name": "task"}}, "traces": [{"nodes": nodes}]}
    totals = native_tool_observations([episode])["totals"]
    assert totals["explicit_tool_error_responses"] == 1
    assert totals["structured_error_field_responses"] == 2
    assert totals["observed_tool_error_responses"] == 2


def test_text_exception_and_null_tool_calls_are_retained():
    episode = {"task": {"data": {"name": "task"}}, "traces": [{"nodes": [
        {"message": {"role": "assistant", "tool_calls": None}},
        {"message": {"role": "tool", "content": "Error executing tool x: failed"}}]}]}
    totals = native_tool_observations([episode])["totals"]
    assert totals["observed_tool_error_responses"] == 1
    assert totals["textual_tool_error_responses"] == 1
