from sdpo_action_risks import action_risks


def call(arguments, additional=True):
    return {"tools": [{"function": {"name": "create_task", "parameters": {
                "properties": {"name": {}, "dueDate": {}, "due_on": {}},
                "additionalProperties": additional}}}],
            "message": {"tool_calls": [{"function": {"name": "create_task", "arguments": arguments}}]}}


def test_silent_undeclared_field_is_a_risk_without_inventing_an_api_error():
    risks = action_risks(call('{"name":"Task","due_date":"2026-03-14"}'),
                         [{"content": '{"success":true,"results":[{"name":"Task"}]}'}])
    assert risks[0]["fields"] == ["due_date"]
    assert risks[0]["forbidden_by_schema"] is False
    assert len(risks) == 1
    assert action_risks(call({"name": "Task", "dueDate": "2026-03-14"}), []) == []
    assert action_risks(call({"name": "Task", "due_on": "2026-03-14"}), []) == []


def test_schema_forbidden_field_and_error_only_response_are_distinguished():
    risks = action_risks(call({"due_date": "2026-03-14"}, False),
                         [{"content": '{"error":"Unsupported SOQL syntax"}'}])
    assert risks[0]["forbidden_by_schema"] is True
    assert risks[1]["kind"] == "structured_error_response"


def test_empty_successful_search_is_not_automatically_a_failure():
    assert action_risks(call({}), [{"content": '{"success":true,"results":[]}'}]) == []
    assert action_risks(call({}), [{"content": '{"error":null}'}]) == []


def test_unconstrained_object_does_not_make_every_argument_a_sampling_risk():
    value = call({"any_field": 7})
    value["tools"][0]["function"]["parameters"] = {"type": "object", "additionalProperties": True}
    assert action_risks(value, []) == []
