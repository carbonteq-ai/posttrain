from audit_progress_corpus_proposals import review_flags


def test_schema_parameter_keys_are_not_hallucinated_identifier_values():
    case = {"call": {"messages": [{"role": "user", "content": "Add to list_002"}]},
            "episode": {"traces": [{"info": {}}]}}
    proposal = {"complete": True, "message": {"tool_calls": [{"function": {
        "name": "mailchimp_add_subscriber", "arguments": {"list_id": "list_002"}}}]}}
    assert review_flags(case, proposal) == []
    proposal["message"]["tool_calls"][0]["function"]["arguments"]["list_id"] = "list_invented"
    assert review_flags(case, proposal)[0]["values"] == ["list_invented"]


def test_prior_model_assertion_does_not_ground_a_new_write_address():
    case = {"call": {"messages": [{"role": "assistant", "content": "Send to madeup@example.com"}]},
            "episode": {"traces": [{"info": {}}]}}
    proposal = {"complete": True, "message": {"tool_calls": [{"function": {
        "name": "gmail_send_email", "arguments": {"to": "madeup@example.com"}}}]}}
    assert review_flags(case, proposal)[0]["values"] == ["madeup@example.com"]


def test_incomplete_proposal_is_missing_evidence_not_zero_reward():
    assert review_flags({"call": {"messages": []}}, {"complete": False}) == [
        {"kind": "incomplete_proposal", "missing_not_zero": True}]
