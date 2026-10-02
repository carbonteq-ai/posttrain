"""Frozen old/reference score transport does not re-score a resumed population."""

from collections.abc import MutableMapping
from dataclasses import replace
from types import SimpleNamespace
from typing import TypedDict, cast

import pytest
from posttrain.train.backends.trl.policy_updates import ResolvedTRLPopulation
from posttrain.train.update_plan import PolicyExecutionBudget, PolicyUpdateSchedule, resolve_updates
from posttrain.train.update_records import ActionRef, InvalidPolicyUpdate

torch = pytest.importorskip("torch")

from posttrain.train.backends.policy_update_recovery import (  # noqa: E402
    load_retained_population,
    load_sampler_correction,
    population_recovery_identity,
    restore_population_recovery,
    save_population_recovery,
)

from .test_update_execution import resolved  # noqa: E402
from .test_update_scoring import CausalModel  # noqa: E402


class RecoveryBoundary(TypedDict):
    runtime_identity: str
    world_size: int
    native_applied_updates: int


def test_recovery_reconstructs_records_before_loading_frozen_scores(tmp_path):
    original = population(applied_update_offset=8, attempt_offset=11)
    model = CausalModel()
    original.loss(model, 0, torch.device("cpu"))
    optimizer = SimpleNamespace(step_was_skipped=False)
    original.before_step(optimizer)
    original.complete_step(optimizer)
    (tmp_path / "native.bin").write_bytes(b"native checkpoint fixture")
    kwargs = RecoveryBoundary(runtime_identity="typed-fixture@1", world_size=1, native_applied_updates=9)
    save_population_recovery(original, tmp_path, native_components=("native.bin",), **kwargs)
    identity = population_recovery_identity(original, runtime_identity=kwargs["runtime_identity"], world_size=1)
    retained = load_retained_population(tmp_path, identity, sampler_correction=None)
    rebuilt = ResolvedTRLPopulation.from_resolved(
        retained.resolved,
        read_input=original.read_input,
        score_temperature=original.score_temperature,
        score_contract=original.score_contract,
        sampler_correction=None,
        max_overflow_retries=retained.max_overflow_retries,
        applied_update_offset=retained.applied_update_offset,
        attempt_offset=retained.attempt_offset,
    )
    assert rebuilt.updates == original.updates and rebuilt.credit == original.credit
    assert rebuilt.old is None and rebuilt.next_update == 0
    restore_population_recovery(rebuilt, tmp_path, device=torch.device("cpu"), **kwargs)
    assert rebuilt.global_applied_updates == 9 and rebuilt.global_attempts == 12
    assert rebuilt.old is not None and original.old is not None
    assert all(torch.equal(rebuilt.old.values[action], value) for action, value in original.old.values.items())
    (tmp_path / "posttrain-update-population.json").write_text("{}")
    with pytest.raises(InvalidPolicyUpdate, match="changed|component|digest|differs"):
        load_retained_population(tmp_path, identity, sampler_correction=None)


def population(**kwargs):
    snapshot, source, credit, spec, update, capabilities = resolved("sampo@1")
    updates = resolve_updates(snapshot, PolicyUpdateSchedule("episode", 2, epochs=2), update.objective)
    return ResolvedTRLPopulation(
        updates,
        credit,
        spec,
        PolicyExecutionBudget(1, 100, 10000),
        capabilities,
        lambda view: source,
        0.7,
        "score@1",
        None,
        **kwargs,
    )


def test_detached_correction_is_sealed_and_restored_without_rescoring(tmp_path):
    original = population()
    original.sampler_correction = {
        record.action: (index + 1) / 4 for index, record in enumerate(original.updates[0].population.actions)
    }
    original.loss(CausalModel(), 0, torch.device("cpu"))
    optimizer = SimpleNamespace(step_was_skipped=False)
    original.before_step(optimizer)
    original.complete_step(optimizer)
    (tmp_path / "native.bin").write_bytes(b"correction checkpoint fixture")
    kwargs = RecoveryBoundary(runtime_identity="correction-fixture@1", world_size=1, native_applied_updates=1)
    saved = save_population_recovery(original, tmp_path, native_components=("native.bin",), **kwargs)
    correction = load_sampler_correction(tmp_path, saved.identity)
    assert correction is not None
    assert correction == original.sampler_correction
    assert sum(component.role == "sampler-correction" for component in saved.components) == 1
    with pytest.raises(TypeError):
        cast(MutableMapping[ActionRef, float], correction)[next(iter(correction))] = 1
    resumed = population()
    resumed.sampler_correction = correction
    restore_population_recovery(resumed, tmp_path, device=torch.device("cpu"), **kwargs)
    assert resumed.old is not None
    assert resumed.sampler_correction == original.sampler_correction
    # Subsequent objective/adjoints use the same frozen correction and old scores.
    a, b = CausalModel(), CausalModel()
    b.load_state_dict(a.state_dict())
    losses = [p.loss(m, 1, torch.device("cpu")) for p, m in ((original, a), (resumed, b))]
    torch.testing.assert_close(losses[0], losses[1], rtol=0, atol=0)
    for loss in losses:
        loss.backward()
    for left, right in zip(a.parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
    (tmp_path / "posttrain-sampler-correction.json").write_text("{}")
    with pytest.raises(InvalidPolicyUpdate, match="bytes changed"):
        load_sampler_correction(tmp_path, saved.identity)


def test_uncorrected_checkpoint_keeps_existing_identity_and_no_correction_file(tmp_path):
    original = population()
    original.loss(CausalModel(), 0, torch.device("cpu"))
    optimizer = SimpleNamespace(step_was_skipped=False)
    original.before_step(optimizer)
    original.complete_step(optimizer)
    (tmp_path / "native.bin").write_bytes(b"uncorrected fixture")
    saved = save_population_recovery(
        original,
        tmp_path,
        native_components=("native.bin",),
        runtime_identity="uncorrected@1",
        world_size=1,
        native_applied_updates=1,
    )
    assert load_sampler_correction(tmp_path, saved.identity) is None
    assert not (tmp_path / "posttrain-sampler-correction.json").exists()


def test_changed_physical_layout_invalidates_recovery_identity():
    original = population()
    identity = population_recovery_identity(original, runtime_identity="layout@1", world_size=1)
    for layout in ("dense-pack", "dense-population"):
        changed = replace(original, capabilities=replace(original.capabilities, context_layout=layout))
        candidate = population_recovery_identity(changed, runtime_identity="layout@1", world_size=1)
        assert candidate.execution_digest != identity.execution_digest
        assert candidate.population_digest == identity.population_digest
        assert candidate.objective_digest == identity.objective_digest


@pytest.mark.parametrize("schema", ["posttrain.resolved-population.v1", "posttrain.resolved-population.v2"])
def test_sealed_legacy_checkpoint_restores_without_reinterpreting_layout(tmp_path, monkeypatch, schema):
    import json

    import posttrain.train.backends.policy_update_recovery as recovery

    original = population()
    original.loss(CausalModel(), 0, torch.device("cpu"))
    optimizer = SimpleNamespace(step_was_skipped=False)
    original.before_step(optimizer)
    original.complete_step(optimizer)
    writer = recovery._population_payload

    def legacy_writer(population):
        payload = writer(population)
        payload["schema"] = schema
        payload["capabilities"].pop("context_layout")
        if schema.endswith("v1"):
            payload.pop("applied_update_offset")
            payload.pop("attempt_offset")
        return payload

    (tmp_path / "native.bin").write_bytes(b"legacy native fixture")
    kwargs = RecoveryBoundary(runtime_identity="legacy-layout@1", world_size=1, native_applied_updates=1)
    with monkeypatch.context() as patch:
        patch.setattr(recovery, "_population_payload", legacy_writer)
        save_population_recovery(original, tmp_path, native_components=("native.bin",), **kwargs)
    assert json.loads((tmp_path / "posttrain-update-population.json").read_text())["schema"] == schema
    identity = population_recovery_identity(original, runtime_identity=kwargs["runtime_identity"], world_size=1)
    retained = load_retained_population(tmp_path, identity, sampler_correction=None)
    assert retained.resolved.capabilities.context_layout == "ragged"
    resumed = population()
    restore_population_recovery(resumed, tmp_path, device=torch.device("cpu"), **kwargs)
    assert resumed.next_update == resumed.applied_updates == 1
    assert all(torch.equal(value, resumed.old.values[action]) for action, value in original.old.values.items())


def test_population_cursor_and_frozen_scores_roundtrip_without_mutating_on_failure(tmp_path):
    original = population()
    original.loss(CausalModel(), 0, torch.device("cpu"))
    (tmp_path / "native.bin").write_bytes(b"fixture native state; not a model qualification")
    kwargs = RecoveryBoundary(runtime_identity="fixture@1", world_size=1, native_applied_updates=1)
    with pytest.raises(InvalidPolicyUpdate, match="incomplete"):
        save_population_recovery(original, tmp_path, native_components=("native.bin",), **kwargs)
    optimizer = SimpleNamespace(step_was_skipped=False)
    original.before_step(optimizer)
    original.complete_step(optimizer)
    save_population_recovery(original, tmp_path, native_components=("native.bin",), **kwargs)
    resumed = population()
    with pytest.raises(InvalidPolicyUpdate, match="identity changed"):
        restore_population_recovery(
            resumed, tmp_path, device=torch.device("cpu"), **{**kwargs, "runtime_identity": "different"}
        )
    assert resumed.old is None and resumed.applied_updates == resumed.next_update == 0
    resumed.max_overflow_retries = 1
    with pytest.raises(InvalidPolicyUpdate, match="identity changed"):
        restore_population_recovery(resumed, tmp_path, device=torch.device("cpu"), **kwargs)
    assert resumed.old is None and resumed.applied_updates == resumed.next_update == 0
    resumed.max_overflow_retries = 0
    restore_population_recovery(resumed, tmp_path, device=torch.device("cpu"), **kwargs)
    assert resumed.applied_updates == resumed.next_update == resumed.attempts == 1
    assert original.old is not None and resumed.old is not None
    for action in original.old.values:
        torch.testing.assert_close(resumed.old.values[action], original.old.values[action], rtol=0, atol=0)
    resumed.loss(CausalModel(), 1, torch.device("cpu"))
    assert resumed.last_evaluation.parameter_version.endswith("applied-1")


def test_nonzero_run_offset_restores_local_scores_and_cursor(tmp_path):
    original = population(applied_update_offset=8, attempt_offset=11)
    original.loss(CausalModel(), 0, torch.device("cpu"))
    optimizer = SimpleNamespace(step_was_skipped=False)
    original.before_step(optimizer)
    original.complete_step(optimizer)
    (tmp_path / "native.bin").write_bytes(b"native global step 9 fixture")
    kwargs = RecoveryBoundary(runtime_identity="offset-fixture@1", world_size=1, native_applied_updates=9)
    save_population_recovery(original, tmp_path, native_components=("native.bin",), **kwargs)
    resumed = population(applied_update_offset=8, attempt_offset=11)
    restore_population_recovery(resumed, tmp_path, device=torch.device("cpu"), **kwargs)
    assert resumed.next_update == resumed.applied_updates == 1
    assert resumed.global_applied_updates == 9 and resumed.global_attempts == 12
    assert original.old is not None and resumed.old is not None
    for action in original.old.values:
        torch.testing.assert_close(resumed.old.values[action], original.old.values[action], rtol=0, atol=0)
    wrong = population(applied_update_offset=8, attempt_offset=12)
    with pytest.raises(InvalidPolicyUpdate, match="identity changed"):
        restore_population_recovery(wrong, tmp_path, device=torch.device("cpu"), **kwargs)
    assert wrong.old is None and wrong.next_update == 0
