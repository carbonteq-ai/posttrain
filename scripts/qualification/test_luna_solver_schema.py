import pytest
from luna_solver_schema import instruction_schema

jsonschema = pytest.importorskip("jsonschema")


def test_workflow_rejects_observed_string_dependency_instead_of_array():
    data = {"actions": [{"action_id": "a1", "call": {"name": "read"}}]}
    schema = instruction_schema("workflow", data)
    jsonschema.Draft202012Validator.check_schema(schema)
    value = {
        "actions": [
            {"action_id": "a1", "tool": "read", "disposition": "reuse", "instruction": "Read.", "depends_on": "a1"}
        ]
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(value, schema)


def test_requirement_rejects_invented_manifest_root():
    schema = instruction_schema("requirements", {"manifest": {"checks": [{"check_id": "g1", "role": "goal"}]}})
    value = {
        "requirements": [{"check_id": "g1", "role": "goal", "instruction": "Read.", "evidence": ["/manifest/checks/0"]}]
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(value, schema)
    value["requirements"][0]["evidence"] = ["/checks/0"]
    jsonschema.validate(value, schema)
