import json

from author_deepseek_task_context import private_copy_flags, task_input, validate


def test_input_never_serializes_luna_or_world_state():
    material = {"task": "example", "user_request": "Send a report", "goal_manifest": {
        "sources": {}, "checks": []}, "tool_definitions": [],
        "reference_actions": "SECRET_LUNA_EXECUTION", "native_data": "PRIVATE_WORLD",
        "source": "LUNA_SOURCE_PATH"}
    result = json.dumps(task_input(material))
    assert "SECRET_LUNA_EXECUTION" not in result
    assert "PRIVATE_WORLD" not in result
    assert "LUNA_SOURCE_PATH" not in result


def test_duplicate_check_and_changed_role_are_rejected():
    inp = {"goal_harm_contract": {"checks": [{"check_id": "c1", "role": "harm"}]}}
    row = dict(check_id="c1", role="goal", requirement="No extra writes",
               retrieval_needed="Inspect policy", verification="Read effects", uncertainty="None")
    context = dict(task_requirements=[], check_guidance=[row, row], workflow=[],
                   constraints=[], unknowns=[], coverage_limits=[])
    errors = validate(context, inp)
    assert "check coverage missing/duplicated/extra" in errors
    assert "check polarity changed" in errors


def test_public_system_restrictions_are_supplied_without_private_world_or_assistant_trace():
    material = {"task": "example", "user_request": "Send a report", "goal_manifest": {
        "sources": {}, "checks": []}, "tool_definitions": [], "native_data": {
            "initial_state": {"secret": "PRIVATE_WORLD"}, "prompt": [
                {"role": "system", "content": "Do not name excluded items."},
                {"role": "assistant", "content": "PRIVATE_TRACE"}]}}
    result = task_input(material)
    assert result["public_system_instructions"] == ["Do not name excluded items."]
    assert "PRIVATE_WORLD" not in json.dumps(result) and "PRIVATE_TRACE" not in json.dumps(result)


def test_private_email_flag_does_not_flag_public_address():
    inp = {"public_request": "Send to public@example.com", "goal_harm_contract": {
        "checks": [{"value": "private@example.com"}, {"value": "public@example.com"}]}}
    assert private_copy_flags({"text": "private@example.com public@example.com"}, inp) == ["private@example.com"]
