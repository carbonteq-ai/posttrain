"""Reject corrupted or disconnected offline state evidence before rubric admission."""

import hashlib
import json

import pytest
from pydantic import BaseModel, Field
from review_sdpo_luna_pool124 import (
    bind_generated_ids,
    capture_states,
    load_native_initialization,
    reference_bundle,
)


def fixture():
    initial, final = {"value": 1}, {"value": 2}
    snapshots = {}
    for state in [initial, final]:
        raw = json.dumps(state)
        snapshots[hashlib.sha256(raw.encode()).hexdigest()] = raw
    before, after = snapshots
    return {"task": {"data": {"initial_state": initial}}, "info": {
        "automationbench": {"end_state": final}, "automationbench_capture": {
            "initial_digest": before, "snapshots": snapshots,
            "events": [{"before_digest": before, "after_digest": after}]}}}


def test_hashed_endpoint_must_match_bound_initial_and_recorded_final():
    trace = fixture()
    assert capture_states(trace, lambda x: x) == {"value": 2}
    trace["task"]["data"]["initial_state"] = {"value": 0}
    with pytest.raises(ValueError, match="bound task state"):
        capture_states(trace, lambda x: x)
    trace = fixture()
    trace["info"]["automationbench"]["end_state"] = {"value": 3}
    with pytest.raises(ValueError, match="recorded endpoint"):
        capture_states(trace, lambda x: x)


def test_corrupted_snapshot_or_discontinuous_chain_is_rejected():
    trace = fixture()
    capture = trace["info"]["automationbench_capture"]
    capture["snapshots"][capture["initial_digest"]] = '{}'
    with pytest.raises(ValueError, match="bytes"):
        capture_states(trace, lambda x: x)
    trace = fixture()
    events = trace["info"]["automationbench_capture"]["events"]
    events.append(dict(events[0]))
    with pytest.raises(ValueError, match="discontinuous"):
        capture_states(trace, lambda x: x)


def test_serialized_snapshot_map_retains_hash_validation():
    trace = fixture()
    capture = trace["info"]["automationbench_capture"]
    capture["snapshots"] = json.dumps(capture["snapshots"])
    assert capture_states(trace, lambda x: x) == {"value": 2}


def test_generated_id_binding_never_changes_seeded_ids_or_record_values():
    class Row(BaseModel):
        id: str = Field(default_factory=lambda: "generated")
        value: int = 1

    generated = Row()
    assert bind_generated_ids(generated, {"id": "archived", "value": 2}) == [
        {"path": ".id", "captured_id": "archived"}]
    assert generated.id == "archived" and generated.value == 1
    seeded = Row(id="seeded")
    assert bind_generated_ids(seeded, {"id": "changed", "value": 2}) == []
    assert seeded.id == "seeded" and seeded.value == 1


def test_native_helper_loading_does_not_execute_legacy_runtime_imports(tmp_path):
    source = tmp_path / "runner.py"
    source.write_text('''
raise RuntimeError("must not import legacy runner")
_SERVICE_FIELDS = sorted(WorldState.model_fields)
def strip_none_values(value):
    return value
def _service_for_name(value):
    return value if value in _SERVICE_FIELDS else None
def compute_allowed_services(initial, assertions, tools):
    return sorted(name for name in initial if _service_for_name(name))
''')
    class World:
        model_fields = {"gmail": None}

    helpers, sha = load_native_initialization(World, source)
    assert len(sha) == 64
    assert helpers["compute_allowed_services"]({"gmail": {}, "other": {}}, [], []) == ["gmail"]


def test_reference_bundle_cannot_promote_partial_or_ambiguous_source_audits():
    reference = {"source_episode_sha256": "sha", "current_native_checks_pass": True}
    row = {"task": "train", "source_episode_sha256": "sha", "qualified": True,
           "errors": [], "checks": [{"final_passed": True}]}
    audit = {"references": {"train": reference}, "audits": [row], "training_tasks": 124,
             "tasks_without_admitted_reference": ["other"], "scope": "test",
             "initialization_policy": "strict", "pinned_runner_source_sha256": "source"}
    bundle = reference_bundle(audit)
    assert bundle["current_rubric_state_checks_complete"]
    assert bundle["teacher_context_length_and_live_binding_qualified"] is False
    with pytest.raises(ValueError, match="passing source"):
        reference_bundle(audit | {"audits": [row | {"checks": [{"final_passed": False}]}]})
    with pytest.raises(ValueError, match="passing source"):
        reference_bundle(audit | {"audits": [row, row]})
