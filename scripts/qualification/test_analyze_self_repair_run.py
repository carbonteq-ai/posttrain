import json

from analyze_self_repair_run import native_tool_observations


def test_repeated_empty_responses_are_linked_and_terminal_attempt_is_not_a_response():
    nodes = []
    for i in range(3):
        nodes.append({"message": {"role": "assistant", "tool_calls": [{"id": str(i),
            "name": "gmail_find_email", "arguments": '{"query":"from:invented.example"}'}]}})
        if i < 2:
            nodes.append({"message": {"role": "tool", "tool_call_id": str(i),
                "content": json.dumps({"success": True, "messages": [], "result_count": 0})}})
    result = native_tool_observations([{"task": {"data": {"name": "task"}}, "traces": [{"nodes": nodes}]}])["totals"]
    assert result["attempted_tool_calls"] == 3
    assert result["repeated_exact_tool_calls"] == 2
    assert result["factual_empty_retrieval_responses"] == 2
    assert result["repeated_identical_empty_responses"] == 1
    assert result["explicit_tool_error_responses"] == 0
    assert result["unmatched_tool_responses"] == 0


def test_successful_nonempty_results_are_not_empty_and_errors_remain_separate():
    nodes = [{"message": {"role": "tool", "tool_call_id": "missing",
             "content": json.dumps({"success": False, "error": "resource not found"})}},
             {"message": {"role": "tool", "tool_call_id": "missing2",
             "content": json.dumps({"success": True, "messages": [{"id": "found"}], "result_count": 1})}}]
    result = native_tool_observations([{"task": {"data": {"name": "task"}}, "traces": [{"nodes": nodes}]}])["totals"]
    assert result.get("factual_empty_retrieval_responses", 0) == 0
    assert result["explicit_tool_error_responses"] == 1
    assert result["unmatched_tool_responses"] == 2
