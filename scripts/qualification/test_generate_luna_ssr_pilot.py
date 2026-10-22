import pytest
from generate_luna_ssr_pilot import parsed_json, validate_skeleton


def test_native_thinking_is_separate_from_final_annotation():
    assert parsed_json('private draft {not json}</think>\n```json\n{"supported":true}\n```') == {"supported": True}


def test_incomplete_or_invalid_final_json_is_rejected():
    with pytest.raises(ValueError):
        parsed_json('draft</think>{"supported":')


def test_skeleton_requires_real_tagged_operations():
    validate_skeleton({"structural_skeleton": [{"tag": "RETR", "summary": "Retrieve the requested source record."}]})
    with pytest.raises(ValueError):
        validate_skeleton({"structural_skeleton": [{"tag": "MADE_UP", "summary": "Claim success."}]})
