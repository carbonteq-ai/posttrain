"""Complete update seals reject partial, mismatched and corrupted recovery."""

import json
from dataclasses import replace

import pytest
from posttrain.train.update_records import InvalidPolicyUpdate
from posttrain.train.update_recovery import (
    FILENAME,
    UpdateRecoveryIdentity,
    UpdateRecoveryState,
    checkpoint_component,
    commit_update_recovery,
    load_update_recovery,
)


def boundary(path):
    identity = UpdateRecoveryIdentity(
        "population",
        "credit",
        "objective",
        ("update-0", "update-1"),
        "execution",
        "correction",
        "runtime",
        "score@1",
        0.7,
        1,
    )
    components = []
    for filename, role in (
        ("native-state.bin", "native-checkpoint"),
        ("scores.bin", "frozen-policy-scores"),
        ("population.json", "resolved-population"),
    ):
        (path / filename).write_bytes(filename.encode())
        components.append(checkpoint_component(path, filename, role=role))
    return UpdateRecoveryState(identity, 1, 1, 2, 1, "policy/applied-1", tuple(components))


@pytest.mark.parametrize("role", ["native-checkpoint", "frozen-policy-scores", "resolved-population"])
def test_each_recovery_evidence_role_is_required(tmp_path, role):
    state = boundary(tmp_path)
    with pytest.raises(InvalidPolicyUpdate, match="must bind"):
        replace(state, components=tuple(item for item in state.components if item.role != role))


def test_resolved_population_corruption_is_rejected(tmp_path):
    state = boundary(tmp_path)
    commit_update_recovery(tmp_path, state)
    (tmp_path / "population.json").write_bytes(b"changed masks or credit")
    with pytest.raises(InvalidPolicyUpdate, match="bytes changed"):
        load_update_recovery(tmp_path, state.identity)


def test_sealed_boundary_roundtrip_idempotence_and_component_corruption(tmp_path):
    state = boundary(tmp_path)
    with pytest.raises(InvalidPolicyUpdate, match="complete resolved"):
        load_update_recovery(tmp_path, state.identity)
    commit_update_recovery(tmp_path, state)
    assert load_update_recovery(tmp_path, state.identity) == state
    commit_update_recovery(tmp_path, state)
    with pytest.raises(InvalidPolicyUpdate, match="already committed"):
        commit_update_recovery(tmp_path, replace(state, attempts=3))
    (tmp_path / "native-state.bin").write_bytes(b"same filename, different optimizer state")
    with pytest.raises(InvalidPolicyUpdate, match="bytes changed"):
        load_update_recovery(tmp_path, state.identity)


@pytest.mark.parametrize(
    "field,value",
    [
        ("runtime_identity", "different"),
        ("world_size", 2),
        ("credit_digest", "different"),
        ("update_digests", ("update-1", "update-0")),
        ("sampler_correction_digest", "different"),
    ],
)
def test_changed_recovery_contract_is_rejected(tmp_path, field, value):
    state = boundary(tmp_path)
    commit_update_recovery(tmp_path, state)
    with pytest.raises(InvalidPolicyUpdate, match="identity changed"):
        load_update_recovery(tmp_path, replace(state.identity, **{field: value}))


def test_interrupted_seal_never_becomes_a_resume_boundary(tmp_path, monkeypatch):
    state = boundary(tmp_path)

    def interrupted(*args):
        raise OSError("interrupted before atomic metadata publication")

    with monkeypatch.context() as context:
        context.setattr("posttrain.train.update_recovery.os.replace", interrupted)
        with pytest.raises(OSError, match="interrupted"):
            commit_update_recovery(tmp_path, state)
    assert not (tmp_path / FILENAME).exists()
    assert not list(tmp_path.glob(".resolved-update-*"))
    commit_update_recovery(tmp_path, state)
    assert load_update_recovery(tmp_path, state.identity) == state


@pytest.mark.parametrize(
    "changes",
    [
        {"native_applied_updates": 2},
        {"next_update": 2},
        {"attempts": 0},
        {"applied_updates": True},
        {"next_update": 3, "applied_updates": 3, "native_applied_updates": 3, "attempts": 3},
    ],
)
def test_incoherent_native_or_resolved_cursor_is_rejected(tmp_path, changes):
    state = boundary(tmp_path)
    with pytest.raises(InvalidPolicyUpdate):
        replace(state, **changes)


def test_population_local_cursor_is_bound_to_prior_run_work(tmp_path):
    state = boundary(tmp_path)
    identity = replace(state.identity, applied_update_offset=8, attempt_offset=11)
    state = replace(state, identity=identity, native_applied_updates=9)
    commit_update_recovery(tmp_path, state)
    assert load_update_recovery(tmp_path, identity) == state
    assert state.next_update == state.applied_updates == 1
    with pytest.raises(InvalidPolicyUpdate, match="identity changed"):
        load_update_recovery(tmp_path, replace(identity, attempt_offset=12))
    with pytest.raises(InvalidPolicyUpdate, match="boundaries disagree"):
        replace(state, native_applied_updates=1)


def test_legacy_zero_offset_seal_is_readable_but_cannot_hide_prior_work(tmp_path):
    state = boundary(tmp_path)
    commit_update_recovery(tmp_path, state)
    target = tmp_path / FILENAME
    payload = json.loads(target.read_text())
    payload["schema"] = "posttrain.resolved-update.v1"
    for field in ("applied_update_offset", "attempt_offset"):
        del payload["state"]["identity"][field]
    target.write_text(json.dumps(payload))
    assert load_update_recovery(tmp_path, state.identity) == state
    payload["state"]["identity"].update(applied_update_offset=8, attempt_offset=11)
    payload["state"]["native_applied_updates"] = 9
    target.write_text(json.dumps(payload))
    with pytest.raises(InvalidPolicyUpdate, match="valid complete"):
        load_update_recovery(tmp_path, replace(state.identity, applied_update_offset=8, attempt_offset=11))


@pytest.mark.parametrize(
    "changes",
    [{"applied_update_offset": True}, {"attempt_offset": -1}, {"applied_update_offset": 4, "attempt_offset": 3}],
)
def test_invalid_prior_population_counters_are_rejected(tmp_path, changes):
    with pytest.raises(InvalidPolicyUpdate, match="offsets|prior attempts"):
        replace(boundary(tmp_path).identity, **changes)
