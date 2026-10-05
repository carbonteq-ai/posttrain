"""Resolved job composition preserves credit and refuses changed recovery meaning."""

import math
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, cast

import pytest
from posttrain.train.backends.policy_update_admission import AdmittedNativePopulation
from posttrain.train.backends.trl.policy_job import ResolvedTRLJob
from posttrain.train.online_rl import BehaviorPolicySpan
from posttrain.train.profiles import GRPOSettings, TrainingLoop
from posttrain.train.requests import GRPORequest
from posttrain.train.update_plan import PolicyExecutionBudget, PolicyUpdateSchedule, PolicyUpdateSettings
from posttrain.train.update_records import InvalidPolicyUpdate

from .test_update_admission import source


def test_runtime_identity_rejects_changed_numerical_execution(tmp_path, monkeypatch):
    from posttrain.train.backends.trl.policy_job import job_identity

    from .test_update_scoring import CausalModel

    @dataclass(frozen=True)
    class TrainingIdentity:
        renderer: str = "renderer@1"

    request = SimpleNamespace(
        policy="policy@1",
        settings=GRPOSettings(
            "test", TrainingLoop(max_steps=2, gradient_accumulation_steps=2), num_prompts_per_step=1, num_generations=2
        ),
        training=TrainingIdentity(),
        inference="transformers@1",
        environment="native@1",
    )
    monkeypatch.setattr("importlib.metadata.version", lambda name: "qualification")
    controls = {"deterministic_algorithms": False, "environment": {"CUBLAS_WORKSPACE_CONFIG": None}}
    monkeypatch.setattr("posttrain.train.backends.trl.policy_job.numerical_execution_identity", lambda: controls)
    original, template = job_identity(request, SimpleNamespace(chat_template="template"), CausalModel)
    controls = {"deterministic_algorithms": True, "environment": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}}
    changed, changed_template = job_identity(request, SimpleNamespace(chat_template="template"), CausalModel)
    assert changed != original
    assert changed_template == template


def test_runtime_identity_rejects_changed_inherited_checkpoint_loader(tmp_path, monkeypatch):
    import inspect

    from posttrain.train.backends.trl.policy_job import job_identity

    class NativeBase:
        pass

    class NativeTrainer(NativeBase):
        pass

    @dataclass(frozen=True)
    class TrainingIdentity:
        renderer: str = "renderer@1"

    source = tmp_path / "native_base.py"
    source.write_text("root adapter skipped")
    original_sourcefile = inspect.getsourcefile
    monkeypatch.setattr(
        inspect, "getsourcefile", lambda cls: str(source) if cls is NativeBase else original_sourcefile(cls)
    )
    monkeypatch.setattr("importlib.metadata.version", lambda name: "qualification")
    monkeypatch.setattr("posttrain.train.backends.trl.policy_job.numerical_execution_identity", lambda: {})
    request = SimpleNamespace(
        policy="policy@1",
        settings="settings@1",
        training=TrainingIdentity(),
        inference="transformers@1",
        environment="native@1",
    )
    original, template = job_identity(request, SimpleNamespace(chat_template="template"), NativeTrainer)
    source.write_text("root adapter restored")
    changed, changed_template = job_identity(request, SimpleNamespace(chat_template="template"), NativeTrainer)
    assert original != changed
    assert template == changed_template


def test_determinism_uses_native_controls_before_model_loading(monkeypatch):
    utilities = pytest.importorskip("transformers.trainer_utils")
    from posttrain.train.backends.trl.policy_config import _configure_training_determinism, _full_determinism

    calls = []
    monkeypatch.setattr(utilities, "enable_full_determinism", calls.append)
    request = SimpleNamespace(
        training=SimpleNamespace(backend_options={}), settings=SimpleNamespace(loop=SimpleNamespace(seed=71))
    )
    _configure_training_determinism(cast(GRPORequest, request))
    assert calls == []
    request.training.backend_options = {"full_determinism": True}
    _configure_training_determinism(cast(GRPORequest, request))
    assert calls == [71]
    for invalid in (1, "true", None):
        with pytest.raises(ValueError, match="boolean"):
            _full_determinism({"full_determinism": invalid})


def job(tmp_path, **changes):
    selected = GRPOSettings(
        "job-test",
        TrainingLoop(max_steps=5),
        num_prompts_per_step=1,
        num_generations=2,
        max_prompt_length=64,
        max_completion_length=64,
        beta=0,
        policy_updates=PolicyUpdateSettings(PolicyUpdateSchedule("episode", 1), PolicyExecutionBudget(2, 100, 1000)),
    )
    request = SimpleNamespace(
        settings=selected,
        bridge=SimpleNamespace(policy_update_context_contract="causal-text@1", retain_population=lambda values: None),
        training=SimpleNamespace(runtime=SimpleNamespace(nodes=1, devices_per_node=1), backend_options={}),
        inference=SimpleNamespace(backend="transformers@1"),
    )
    context = SimpleNamespace(run_id="run")
    values: dict[str, Any] = dict(
        context=context,
        request=request,
        tokenizer=object(),
        rows=[{"example_id": "task", "prompt": []}],
        runtime_identity="runtime@1",
        template_revision="template@1",
        score_temperature=0.8,
        totals=SimpleNamespace(),
    )
    values.update(changes)
    return ResolvedTRLJob(**values)


def admitted(candidate, applied=3, attempts=5, sampler_version=None):
    rollouts, evidence, decode = source()
    rollouts = tuple(replace(rollout, behavior_policy=BehaviorPolicySpan(applied, applied)) for rollout in rollouts)
    return AdmittedNativePopulation.from_rollouts(
        rollouts,
        candidate.request.settings,
        candidate.capabilities,
        population_id=f"population-at-{applied}",
        native_evidence_ref="artifact:episodes",
        read_evidence=lambda _: evidence,
        decode=decode,
        template_revision=candidate.template_revision,
        versions=(
            candidate.versions(applied)
            if sampler_version is None
            else replace(candidate.versions(applied), sampler=sampler_version)
        ),
        sampler_step=applied,
        selector_digest="selection@1",
        applied_update_offset=applied,
        attempt_offset=attempts,
    ), decode


def test_native_job_slot_collects_complete_rows_once_for_multiple_updates(tmp_path, monkeypatch):
    candidate = job(tmp_path)
    prepared, _ = admitted(candidate, 0, 0)
    calls = []

    def collect(context, request, tokenizer, trainer, rows, capabilities, **kwargs):
        calls.append((rows, kwargs))
        return prepared

    monkeypatch.setattr("posttrain.train.backends.trl.policy_job.collect_resolved_population", collect)
    candidate.trainer = SimpleNamespace(state=SimpleNamespace(global_step=0))
    population, index = candidate.run.occurrence(0, 0)
    assert population.applied_update_offset == 0 and population.attempt_offset == 0
    assert index == 0
    assert calls[0][0] == [{"example_id": "task", "prompt": []}] * 2
    assert calls[0][1]["versions"] == candidate.versions(0)
    population.next_update = population.applied_updates = population.attempts = 1
    candidate.trainer.state.global_step = 1
    reused, index = candidate.run.occurrence(1, 1)
    assert reused is population and index == 1
    assert len(calls) == 1


@pytest.mark.parametrize(
    "sampling", [{"top_p": 0.9}, {"top_k": 20}, {"min_p": 0.1}, {"repetition_penalty": 1.1}, {"presence_penalty": 0.2}]
)
def test_job_rejects_unqualified_sampler_correction(tmp_path, sampling):
    from posttrain.train.backends.trl.policy_job import validate_resolved_job

    candidate = job(tmp_path)
    object.__setattr__(candidate.request.inference, "sampling", sampling)
    with pytest.raises(InvalidPolicyUpdate, match="warped sampling distribution"):
        validate_resolved_job(candidate.request)


@pytest.mark.parametrize("sampler_version", [None, "original-sampler@1"])
def test_job_restore_uses_sealed_native_evidence_and_frozen_credit(tmp_path, monkeypatch, sampler_version):
    torch = pytest.importorskip("torch")
    from posttrain.train.backends.policy_update_recovery import save_population_recovery
    from posttrain.train.backends.trl.policy_updates import ResolvedTRLPopulation

    from .test_update_scoring import CausalModel

    candidate = job(tmp_path)
    prepared, decode = admitted(candidate, sampler_version=sampler_version)
    population = ResolvedTRLPopulation.from_admitted(
        prepared,
        score_temperature=0.8,
        score_contract="posttrain.causal-text-tempered-logsoftmax-fp32@1",
        sampler_correction=None,
    )
    from posttrain.train.update_sampler_correction import recipe_sampler_correction_weights

    sampled = prepared.read_input.sampling_log_scores(prepared.resolved.snapshot)
    population.prepare_sampler_correction = lambda old: recipe_sampler_correction_weights(
        candidate.request.settings, prepared.resolved.snapshot, old, sampled
    )
    population.loss(CausalModel(), 0, torch.device("cpu"))
    optimizer = SimpleNamespace(step_was_skipped=False)
    population.before_step(optimizer)
    population.complete_step(optimizer)
    (tmp_path / "native.bin").write_bytes(b"native fixture")
    save_population_recovery(
        population,
        tmp_path,
        runtime_identity="runtime@1",
        world_size=1,
        native_applied_updates=4,
        native_components=("native.bin",),
    )
    monkeypatch.setattr("posttrain.train.backends.trl.policy_job._decode_native", decode)
    restored = candidate.restore(tmp_path)
    assert restored.credit == population.credit
    assert restored.updates == population.updates
    assert restored.sampler_correction == population.sampler_correction
    assert restored.old is None and restored.next_update == 0  # native state loading is a later boundary
    original_versions = candidate.versions(3)
    for field_name in ("old_score", "current", "reference"):
        monkeypatch.setattr(
            candidate,
            "versions",
            lambda applied, name=field_name: replace(original_versions, **{name: "changed-executor-version@1"}),
        )
        with pytest.raises(InvalidPolicyUpdate, match="selected population contracts"):
            candidate.restore(tmp_path)
    altered = job(tmp_path, template_revision="different-template")
    with pytest.raises(InvalidPolicyUpdate, match="selected population contracts"):
        altered.restore(tmp_path)
    altered.request = cast(GRPORequest, SimpleNamespace(**vars(candidate.request)))
    altered.template_revision = candidate.template_revision
    object.__setattr__(altered.request, "settings", replace(candidate.request.settings, clip_epsilon_low=0.05))
    with pytest.raises(InvalidPolicyUpdate, match="selected population contracts"):
        altered.restore(tmp_path)


def test_model_export_excludes_resolved_checkpoint_state(tmp_path):
    from posttrain.train.backends.trl.common import _project_checkpoint_model_view

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    for name in (
        "adapter_config.json",
        "adapter_model.safetensors",
        "tokenizer.json",
        "posttrain-resolved-update.json",
        "posttrain-frozen-policy-scores.pt",
        "posttrain-update-population.json",
        "posttrain-native-population.bin",
        "posttrain-sampler-correction.json",
    ):
        (checkpoint / name).write_text("{}")
    destination = _project_checkpoint_model_view(checkpoint, tmp_path / "model")
    assert {path.name for path in destination.iterdir()} == {
        "adapter_config.json",
        "adapter_model.safetensors",
        "tokenizer.json",
    }


@pytest.mark.parametrize("beta", [0, 0.04])
def test_applied_observation_separates_total_policy_and_weighted_kl(tmp_path, beta):
    torch = pytest.importorskip("torch")
    from posttrain.train.backends.trl.policy_updates import ResolvedTRLPopulation
    from posttrain.train.update_objectives import resolve_objective_term

    from .test_update_scoring import CausalModel

    metrics, events = [], []
    context = SimpleNamespace(
        run_id="run",
        metrics=lambda values, **kw: metrics.append((values, kw)),
        event=lambda name, values: events.append((name, values)),
    )
    candidate = job(tmp_path, context=context)
    object.__setattr__(candidate.request, "settings", replace(candidate.request.settings, beta=beta))
    prepared, _ = admitted(candidate, 0, 0)
    population = ResolvedTRLPopulation.from_admitted(
        prepared,
        score_temperature=0.8,
        score_contract="posttrain.causal-text-tempered-logsoftmax-fp32@1",
        sampler_correction=None,
    )
    model = CausalModel()
    if beta:
        from posttrain.train.backends.policy_update_scoring import freeze_population_scores

        assert population.updates[0].population.versions.reference is not None
        population.reference = freeze_population_scores(
            model,
            population.updates[0].population,
            read_input=population.read_input,
            device=torch.device("cpu"),
            policy_version=population.updates[0].population.versions.reference,
            score_contract=population.score_contract,
            score_temperature=0.8,
        )
    population.loss(model, 0, torch.device("cpu"))
    optimizer = SimpleNamespace(step_was_skipped=False)
    population.before_step(optimizer)
    population.complete_step(optimizer)
    term = resolve_objective_term(
        population.updates[0],
        population.spec,
        population.credit,
        parameter_version=population.last_evaluation.parameter_version,
    )
    # A known evaluator result isolates reporting from objective implementation.
    population.last_evaluation = replace(
        population.last_evaluation,
        loss=torch.tensor(3.0),
        policy_loss=torch.tensor(2.0),
        kl_loss=torch.tensor(1.0 if beta else 0.0),
        term_digest=term.digest,
    )
    candidate.run.current = population
    candidate.observe_applied_update(1)
    candidate.observe_applied_update(1)
    assert len(metrics) == len(events) == 1
    values, metadata = metrics[0]
    assert metadata["step"] == 1
    assert values["train/rl/loss"] == 3
    assert values["train/rl/policy_loss"] == 2
    assert values["train/rl/kl_loss"] == (1 if beta else 0)
    if beta:
        assert values["train/rl/kl"] == 25
    else:
        assert "train/rl/kl" not in values  # zero beta does not measure divergence
    credit = {value.action: value.advantage for value in population.credit.values}
    selected = [credit[weight.action] for weight in term.policy_weights]
    assert values["train/rl/advantage_abs_mean"] == pytest.approx(sum(map(abs, selected)) / len(selected))
    with pytest.raises(InvalidPolicyUpdate, match="committed applied boundary"):
        candidate.observe_applied_update(2)


def test_applied_observation_reports_clipping_correction_and_entropy(tmp_path):
    """Two declared updates share one collection; the second runs at changed parameters."""
    torch = pytest.importorskip("torch")
    from posttrain.train.backends.trl.policy_updates import ResolvedTRLPopulation
    from posttrain.train.update_objectives import resolve_objective_term
    from posttrain.train.update_sampler_correction import recipe_sampler_correction_weights

    from .test_update_scoring import CausalModel

    metrics = []
    context = SimpleNamespace(
        run_id="run", metrics=lambda values, **kw: metrics.append(values), event=lambda name, values: None
    )
    candidate = job(tmp_path, context=context)
    # Token truncation keeps each action's correction weight separately readable.
    settings = replace(candidate.request.settings, importance_sampling_mode="token_truncate")
    object.__setattr__(candidate.request, "settings", settings)
    prepared, _ = admitted(candidate, 0, 0)
    population = ResolvedTRLPopulation.from_admitted(
        prepared,
        score_temperature=0.8,
        score_contract="posttrain.causal-text-tempered-logsoftmax-fp32@1",
        sampler_correction=None,
    )
    snapshot = prepared.resolved.snapshot
    actions = [record.action for record in snapshot.actions]
    # log(old) - log(sampled): above the 3.0 cap, inside, and below the 0.1 floor.
    deltas = dict(zip(actions, (2.0, 0.25, -3.0, 0.0), strict=True))
    population.prepare_sampler_correction = lambda old: recipe_sampler_correction_weights(
        settings, snapshot, old, {action: old[action] - deltas[action] for action in actions}
    )
    model = CausalModel()
    optimizer = SimpleNamespace(step_was_skipped=False)
    candidate.run.current = population

    def entropy(action):
        view = {view.id: view for view in snapshot.conditioning}[
            {record.action: record for record in snapshot.actions}[action].conditioning_id
        ]
        inputs = population.read_input(view)
        tokens = torch.tensor([inputs.token_ids])
        with torch.no_grad():
            logits = model(tokens, torch.ones_like(tokens), torch.arange(tokens.shape[1]).unsqueeze(0), False).logits
        position = dict(inputs.action_positions)[action.token_index]
        probabilities = (logits[0, position - 1] / 0.8).softmax(-1)
        return float(-(probabilities * probabilities.log()).sum())

    def apply(index):
        population.loss(model, index, torch.device("cpu"))
        population.before_step(optimizer)
        population.complete_step(optimizer)
        term = resolve_objective_term(
            population.updates[index],
            population.spec,
            population.credit,
            parameter_version=population.last_evaluation.parameter_version,
        )
        candidate.observe_applied_update(index + 1)
        return [weighted.action for weighted in term.policy_weights], metrics[-1]

    selected, first = apply(0)
    weights = {action: min(max(math.exp(delta), 0.1), 3.0) for action, delta in deltas.items()}
    chosen = [weights[action] for action in selected]
    # The first update evaluates at the sampling parameters: ratios are one.
    assert first["train/rl/clip_fraction"] == 0.0
    assert first["train/rl/importance_sampling_ratio_mean"] == pytest.approx(sum(chosen) / len(chosen))
    assert first["train/rl/importance_sampling_ratio_min"] == pytest.approx(min(chosen))
    assert first["train/rl/importance_sampling_ratio_max"] == pytest.approx(max(chosen))
    clamped = sum(abs(deltas[action]) > 1 for action in selected) / len(selected)
    assert first["train/rl/importance_sampling_ratio_clamped_fraction"] == pytest.approx(clamped)
    assert first["train/rl/entropy"] == pytest.approx(sum(map(entropy, selected)) / len(selected), rel=1e-5)

    # Move the second update's sampled tokens in its advantage direction far
    # enough that every PPO ratio leaves the clip interval.
    credit = {value.action: value.advantage for value in population.credit.values}
    second = population.updates[1]
    term = resolve_objective_term(second, population.spec, population.credit)
    sign = 1.0 if credit[term.policy_weights[0].action] > 0 else -1.0
    with torch.no_grad():
        model.head.bias[4] += 6 * sign
        model.head.bias[5] += 6 * sign
    selected, last = apply(1)
    ratios = {action: float(population.last_evaluation.ratios[action].detach()) for action in selected}
    expected = [
        (credit[action] > 0 and ratios[action] > 1 + population.spec.clip_high)
        or (credit[action] < 0 and ratios[action] < 1 - population.spec.clip_low)
        for action in selected
    ]
    assert all(expected)
    assert last["train/rl/clip_fraction"] == 1.0
    assert last["train/rl/clip_fraction_high" if sign > 0 else "train/rl/clip_fraction_low"] == 1.0
    assert last["train/rl/entropy"] == pytest.approx(sum(map(entropy, selected)) / len(selected), rel=1e-5)
