import pytest
from luna_solver_prompts import check_hint_literals, compact_reference, guided_messages, user_key, validate_hints


def test_reference_summary_input_removes_historical_values_and_results():
    actions = [
        {
            "call": {"name": "send", "arguments": {"to": "hidden@example.com"}},
            "result": {"observed_result": {"success": True, "id": "secret-id"}, "observed_error": None},
        }
    ]
    value = compact_reference(actions)
    assert value == [{"tool": "send", "argument_fields": ["to"], "observed_success": True, "had_error": False}]
    assert "hidden@example.com" not in str(value) and "secret-id" not in str(value)


def test_clean_student_input_is_preserved_separately_from_actor_hints():
    clean = [{"role": "system", "content": "Solve the task."}, {"role": "user", "content": "Notify the owner."}]
    actual = guided_messages(clean, ["Retrieve the owner before sending."])
    assert "Operational checklist" in actual[0]["content"]
    assert clean[0]["content"] == "Solve the task."
    assert user_key(actual) == user_key(clean)


def test_nonpublic_values_and_generation_bookkeeping_are_rejected():
    with pytest.raises(ValueError):
        check_hint_literals(["Email hidden@example.com"], "Notify the owner", {"to": "hidden@example.com"})
    with pytest.raises(ValueError):
        check_hint_literals(["Follow the Luna annotation"], "Notify the owner", {})
    check_hint_literals(["Email public@example.com"], "Email public@example.com", {})


def test_summary_caps_are_actual_validation_not_silent_clipping():
    assert validate_hints({"checks": ["Verify the selected row."]}, ["checks"], [6], 100)
    with pytest.raises(ValueError):
        validate_hints({"checks": ["long " * 101]}, ["checks"], [6], 100)


def test_novel_bare_domain_examples_are_not_allowed_to_become_actor_facts():
    with pytest.raises(ValueError, match="nonpublic literal"):
        check_hint_literals(["Use the sender domain, e.g. nexgen.com"], "Find the HR email from NexGen", {})
    check_hint_literals(["Use public.example.com"], "Check public.example.com", {})
    check_hint_literals(["Use Account.Name and Contact.Title fields"], "Query Salesforce", {})


def test_public_search_operator_is_not_private_world_value():

    public = "HR at NexGen sent over a contact update request. Find it in the inbox."
    query = "in:inbox NexGen HR"
    check_hint_literals([f"Search for {query}"], public, {"query": query})
    for private_query in ("in:inbox PrivateSender HR", "in:inbox NexGen 2047", "in:inbox hr@private.com HR"):
        with pytest.raises(ValueError, match="nonpublic literal"):
            check_hint_literals([private_query], public, {"query": private_query})
