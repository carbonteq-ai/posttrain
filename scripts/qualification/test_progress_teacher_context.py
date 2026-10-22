import json

import pytest
from progress_teacher_context import (
    balanced_batch,
    compose,
    compose_brief,
    observation_ledger,
    observed_policy_candidates,
    semantic_selection,
)


def test_system_rules_and_progress_preserved_but_hidden_expected_values_not_injected():
    messages = [{"role": "system", "content": "Never name excluded items."},
                {"role": "user", "content": "Send the report."},
                {"role": "tool", "name": "read_rows", "content": '{"rows": [], "success": true}'}]
    text = compose(messages, {"constraints": []}, [{"type": "not_sent", "passed": False,
                    "excluded": False, "params": {"expected": "SECRET_POLICY_CODE"}}], [])
    assert messages[0]["content"] in text and "SECRET_POLICY_CODE" not in text
    assert "do not restart" in text.lower() and "Excluded section" in text
    assert observation_ledger(messages)[0]["status"] == "returned_empty"
    assert json.loads(text.split("\n", 1)[1].split("\nProduce", 1)[0])["system_restrictions_verbatim"] == [messages[0]["content"]]


def test_error_followed_by_read_is_recorded_as_distinct_evidence_not_goal_success():
    ledger = observation_ledger([{"role": "tool", "content": '{"success": false}'},
                                 {"role": "tool", "content": '{"rows": [{"id": "a"}]}'}])
    assert [r["status"] for r in ledger] == ["explicit_error", "returned"]


def test_semantic_selection_targets_forbidden_write_instead_of_working_read():
    calls = [{"prefix_ids": [i], "action_ids": [i+2], "message": {"tool_calls": [
        {"id": str(i), "function": {"name": tool, "arguments": arg}}]}} for i, (tool, arg) in enumerate(
            [("read_rows", "{}"), ("slack_send", '{"text":"forbidden name"}')])]
    rows = [{"task": "t", "episode_id": "e", "generation_index": i,
             "student_prefix_ids": c["prefix_ids"], "action_ids": c["action_ids"], "native_success": 0}
            for i, c in enumerate(calls)]
    episodes = [{"id": "e", "traces": [{"nodes": [], "info": {"automationbench": {"assertions": [
        {"type": "slack_message_not_in_channel", "passed": False, "excluded": False,
         "params": {"text_contains": "forbidden name"}}]}}}]}]
    assert semantic_selection(rows, calls, episodes, ["t"])[0]["generation_index"] == 1
    with pytest.raises(ValueError, match="insufficient"):
        balanced_batch(rows, rows[:1], ["t"], 3)


def test_business_brief_preserves_relationships_and_does_not_replay_future_success_payload():
    text = compose_brief("finance.intercompany_balance_recon", [{"role": "system", "content": "Silent exclusions"}],
                         [{"type": "missing", "passed": False, "params": {"secret": "EXPECTED_PRIVATE"}}],
                         [{"tool": "send", "response": '{"success":true,"body":"BAD_FUTURE_BODY"}'}])
    assert "HQ receivable against that division's payable" in text
    assert "EXPECTED_PRIVATE" not in text and "BAD_FUTURE_BODY" not in text
    assert "NONE of these later API results" in text
    with pytest.raises(ValueError, match="no reviewed business brief"):
        compose_brief("unknown", [], [], [])


def test_plain_native_rejection_is_preserved_in_feedback_and_observation_ledger():
    error = "Unsupported arguments for gmail_send_email: bad_key. Use the declared tool parameter names. No action was executed."
    messages = [{"role": "system", "content": "Public rules"},
                {"role": "tool", "name": "gmail_send_email", "content": error}]
    text = compose_brief("marketing.guest_post_outreach", messages, [],
                         [{"tool": "gmail_send_email", "response": error}])
    assert error in text
    assert observation_ledger(messages)[0]["status"] == "explicit_error"
    assert '"recorded_api_success": false' in text


def test_native_mcp_error_is_not_mislabeled_as_successful_return():
    messages = [{"role": "tool", "content": {"isError": True, "content": [{"text": "validation failed"}]}}]
    assert observation_ledger(messages)[0]["status"] == "explicit_error"
    assert observation_ledger([{"role": "tool", "content": "an ordinary string"}])[0]["status"] == "returned"


def test_plain_rejection_selects_actual_failed_call_instead_of_working_read():
    calls = [{"prefix_ids": [i], "action_ids": [i+2], "message": {"tool_calls": [
        {"id": str(i), "function": {"name": "read", "arguments": "{}"}}]}} for i in range(2)]
    rows = [{"task": "t", "episode_id": "e", "generation_index": i,
             "student_prefix_ids": c["prefix_ids"], "action_ids": c["action_ids"], "native_success": 0}
            for i, c in enumerate(calls)]
    episodes = [{"id": "e", "traces": [{"nodes": [{"message": {
        "role": "tool", "tool_call_id": "1", "content": "Unsupported arguments for read: bad. No action was executed."}}]}]}]
    assert semantic_selection(rows, calls, episodes, ["t"])[0]["generation_index"] == 1


def test_policy_quotes_and_progress_arguments_come_only_from_prior_tool_history():
    messages = [{"role": "system", "content": "Policy not an observation"},
                {"role": "assistant", "tool_calls": [{"id": "r", "function": {
                    "name": "read", "arguments": '{"worksheet":"observed-policy"}'}}]},
                {"role": "tool", "tool_call_id": "r", "content": json.dumps({"rows": [
                    {"Policy": "Termination", "Steps": "Revoke immediately."}]})},
                {"role": "tool", "content": json.dumps({"messages": [
                    {"subject": "Current Guidelines", "body_plain": "Use OBSERVED-CODE", "id": "seen"}]})}]
    policy = observed_policy_candidates(messages)
    assert len(policy) == 2 and policy[0]["source"]["Steps"] == "Revoke immediately."
    assert policy[1]["source"]["body_plain"] == "Use OBSERVED-CODE"
    ledger = observation_ledger(messages)
    assert ledger[0]["observed_request"]["arguments"] == {"worksheet": "observed-policy"}
    text = compose_brief("finance.intercompany_balance_recon", messages, [], [])
    assert "Report only discrepant entities" in text
    assert "Report matches and mismatches" not in text


def test_corpus_context_always_preserves_constraints_without_future_success_body():
    from progress_teacher_context import compose_corpus

    text = compose_corpus("unreviewed.task", [{"role": "system", "content": "Do not name exclusions"},
                           {"role": "user", "content": "Public request"}],
                          {"constraints": [{"mode": "FORBID", "match": "test"}]}, [],
                          [{"tool": "read", "response": '{"success":true,"body":"FUTURE-HIDDEN"}'}])
    assert "FUTURE-HIDDEN" not in text
    assert '"mode": "FORBID"' in text
    assert '"recorded_api_success": true' in text
    assert "Equality never satisfies strictly greater" in text


def test_actual_native_negative_assertion_selects_forbidden_write_before_earlier_read_error():
    calls = [{"prefix_ids": [i], "action_ids": [i+2], "message": {"tool_calls": [{
        "id": str(i), "function": {"name": "read" if i == 0 else "gmail_send_email",
                                    "arguments": '{}' if i == 0 else '{"body":"East is fully matched"}'}}]}}
        for i in range(2)]
    rows = [{"task": "finance.t", "episode_id": "e", "generation_index": i,
             "student_prefix_ids": c["prefix_ids"], "action_ids": c["action_ids"], "native_success": 0}
            for i, c in enumerate(calls)]
    episodes = [{"id": "e", "traces": [{"nodes": [{"message": {"role": "tool", "tool_call_id": "0",
        "content": "Unsupported arguments for read: typo. No action was executed."}}],
        "info": {"automationbench": {"assertions": [{"type": "gmail_message_sent_to_with_body_not_contains",
            "passed": False, "excluded": False, "params": {"body_not_contains": "East"}}]}}}]}]
    assert semantic_selection(rows, calls, episodes, ["finance.t"])[0]["generation_index"] == 1


def test_negative_assertion_and_negative_parameter_polarity_are_both_preserved():
    from progress_teacher_context import forbidden_terms

    assert forbidden_terms({"type": "slack_message_not_exists", "params": {"text_contains": "Excluded"}}) == ["Excluded"]
    assert forbidden_terms({"type": "gmail_body_not_contains", "params": {"body_not_contains": ["A", "B"]}}) == ["A", "B"]
    assert forbidden_terms({"type": "gmail_body_contains", "params": {"body_contains": "Required"}}) == []


def test_empty_successful_read_is_factual_feedback_and_selected_over_clean_first_read():
    from progress_teacher_context import compose_corpus, retrieval_summary

    assert retrieval_summary("google_drive_find_multiple_files", '{"success":true,"results":[]}')["empty_returned_collection"]
    assert not retrieval_summary("read", '{"success":true,"headers":[]}')["empty_returned_collection"]
    calls = [{"prefix_ids": [i], "action_ids": [i+2], "messages": [], "message": {"tool_calls": [{
        "id": str(i), "function": {"name": "google_drive_find_multiple_files", "arguments": json.dumps({"title": str(i)})}}]}}
        for i in range(2)]
    rows = [{"task": "t", "episode_id": "e", "generation_index": i,
             "student_prefix_ids": c["prefix_ids"], "action_ids": c["action_ids"], "native_success": 0}
            for i, c in enumerate(calls)]
    episodes = [{"id": "e", "traces": [{"nodes": [{"message": {"role": "tool", "tool_call_id": str(i),
        "content": json.dumps({"success": True, "results": [{"id": "found"}] if i == 0 else []})}}
        for i in range(2)]}]}]
    assert semantic_selection(rows, calls, episodes, ["t"])[0]["generation_index"] == 1
    text = compose_corpus("t", [{"role": "system", "content": "Rules"}], {}, [], [{
        "tool": "google_drive_find_multiple_files", "response": '{"success":true,"results":[]}',
        "attempted_arguments": '{"title":"failed query"}'}])
    assert '"empty_returned_collection": true' in text and '"recorded_api_success": true' in text
    assert "failed query" in text and "untrusted proposals" in text


def test_repeated_successful_read_detection_respects_intervening_writes():
    from progress_teacher_context import repeated_successful_read

    tool = {"id": "c", "function": {"name": "gmail_find_email", "arguments": '{"query":"safety"}'}}
    call = {"messages": [{"role": "assistant", "tool_calls": [tool]},
                          {"role": "tool", "tool_call_id": "c", "content": '{"success":true,"messages":[{"id":"seen"}]}'}]}
    assert repeated_successful_read(call, tool)
    call["messages"].append({"role": "assistant", "tool_calls": [{"id": "write", "function": {
        "name": "gmail_send_email", "arguments": '{}'}}]})
    assert not repeated_successful_read(call, tool)


def test_attempt_arguments_bound_to_actual_call_are_not_retrieved_fact():
    from progress_teacher_context import selected_observations

    call = {"message": {"tool_calls": [{"id": "c", "function": {"name": "gmail_find_email", "arguments": '{"query":"q"}'}}]}}
    result = selected_observations(call, [{"role": "tool", "tool_call_id": "c", "content": '{"success":true,"messages":[]}'},
                                        {"role": "tool", "tool_call_id": "other", "content": "unrelated"}])
    assert len(result) == 1 and result[0]["attempted_arguments"] == '{"query":"q"}'
