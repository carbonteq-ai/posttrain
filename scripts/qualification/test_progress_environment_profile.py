import pytest
from progress_environment_profile import (
    ACCESS,
    DISCOVERY,
    FROZEN,
    REPAIRED,
    task_options,
    taskset_options,
    validate_native_source,
    validate_reference,
    verified_rejection,
)


def test_frozen_default_and_explicit_repaired_selection():
    assert task_options() == task_options(FROZEN) == {}
    assert task_options(REPAIRED) == {
        "world_time_context_if_declared": True, "strict_tool_arguments": True}
    with pytest.raises(ValueError, match="unknown environment"):
        task_options("typo")


def test_repaired_controls_cannot_reuse_old_populations():
    validate_reference(REPAIRED, None)
    validate_reference(FROZEN, "old")
    with pytest.raises(ValueError, match="fresh controls"):
        validate_reference(REPAIRED, "old")


def test_only_exact_strict_middleware_error_is_a_verified_noop():
    text = "Unsupported arguments for read: wrong. Use the declared tool parameter names. No action was executed."
    assert verified_rejection("read", {"wrong": 1}, {"correct"}, text)
    assert verified_rejection("read", {"wrong": 1}, {"correct"},
                              {"isError": True, "content": [{"text": text}]})
    assert not verified_rejection("read", {"correct": 1}, {"correct"}, text)
    assert not verified_rejection("read", {"wrong": 1}, {"correct"}, text + " different")


def test_native_only_reuse_requires_completed_identical_repaired_source():
    source = {"environment_profile": REPAIRED, "environment_base_revision": "abc", "environment_patch_sha256": "def"}
    receipt = {"environment_profile": REPAIRED, "runtime_completed": True}
    validate_native_source(source, dict(source), receipt)
    for changed in [{**source, "environment_profile": FROZEN}, {**source, "environment_patch_sha256": "other"}]:
        with pytest.raises(ValueError, match="identical repaired"):
            validate_native_source(source, changed, receipt)
    with pytest.raises(ValueError, match="completed"):
        validate_native_source(source, source, {**receipt, "runtime_completed": False})


def test_read_only_calendly_discovery_is_task_scoped_and_does_not_leak_into_other_batches():
    name = "operations.calendly_equipment_inspection"
    assert task_options(ACCESS) == task_options(REPAIRED)
    assert taskset_options(ACCESS, [name]) == {
        "additional_tools_by_task": {name: ("calendly_list_event_types",)}}
    assert taskset_options(ACCESS, ["finance.other"]) == {"additional_tools_by_task": {}}
    assert taskset_options(REPAIRED, [name]) == {}
    source = {"environment_profile": ACCESS, "environment_base_revision": "abc", "environment_patch_sha256": "def"}
    with pytest.raises(ValueError, match="identical"):
        validate_native_source(source, {**source, "environment_profile": REPAIRED},
                               {"environment_profile": REPAIRED, "runtime_completed": True})


def test_jira_discovery_is_additive_read_only_and_task_scoped():
    names = ["support.hiver_escalation_patterns", "operations.calendly_equipment_inspection", "finance.other"]
    assert taskset_options(DISCOVERY, names) == {"additional_tools_by_task": {
        names[0]: ("jira_project",), names[1]: ("calendly_list_event_types",)}}
    assert taskset_options(ACCESS, [names[0]]) == {"additional_tools_by_task": {}}
    assert taskset_options(DISCOVERY, ["finance.other"]) == {"additional_tools_by_task": {}}
