import json

from sdpo_schema_parser import guarded_coerce


def buggy_container_coerce(raw, types):
    # Exact failure mode of the pinned implementation, without importing vLLM.
    if "object" in types or "array" in types:
        try:
            return json.loads(raw)
        except ValueError:
            pass
    return raw


def test_undeclared_scalars_do_not_enter_tools_through_container_union():
    for raw in ("1", "2", "500001", "true", "false", "1.5"):
        assert guarded_coerce(raw, ["string", "object", "array", "null"], buggy_container_coerce) == raw


def test_valid_containers_and_null_keep_native_meaning():
    types = ["string", "object", "array", "null"]
    for raw in ('{"id":"2"}', '["2"]', "null"):
        assert guarded_coerce(raw, types, buggy_container_coerce) == json.loads(raw)


def test_guard_does_not_invent_missing_id_or_container_for_invalid_scalar():
    for raw in ("2", "true", "unknown-id"):
        assert guarded_coerce(raw, ["object"], buggy_container_coerce) == raw


def test_declared_numeric_boolean_and_number_accept_native_value():
    assert guarded_coerce("2", ["integer"], lambda r, t: int(r)) == 2
    assert guarded_coerce("2", ["number"], lambda r, t: int(r)) == 2
    assert guarded_coerce("true", ["boolean"], lambda r, t: True) is True
