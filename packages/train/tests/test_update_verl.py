"""veRL transport and native BaseEngine protocol checks; not FSDP qualification."""

from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import pytest
from posttrain.common import RunContext
from posttrain.train.backends.verl.policy_updates import (
    ResolvedVeRLPopulation,
    ResolvedVeRLRun,
    selected_native_outputs,
)
from posttrain.train.profiles import SAMPOSettings, TrainingLoop
from posttrain.train.update_plan import PolicyExecutionBudget, PolicyUpdateSchedule, PolicyUpdateSettings
from posttrain.train.update_records import InvalidPolicyUpdate
from posttrain.train.update_resolution import resolve_policy_population

torch = pytest.importorskip("torch")
pytest.importorskip("verl")

from verl.workers.engine.base import BaseEngine  # noqa: E402

from .test_update_execution import resolved  # noqa: E402
from .test_update_scoring import CausalModel  # noqa: E402


def population(population_id=None, **kwargs):
    snapshot, source, credit, spec, update, capabilities = resolved("sampo@1")
    if population_id is not None:
        snapshot = replace(snapshot, id=population_id, versions=replace(snapshot.versions,
            old_score=f"old-{population_id}", current=f"policy-{population_id}", sampler=f"sampler-{population_id}"))
        credit = replace(credit, population_digest=snapshot.digest)
    selection = SAMPOSettings(id="native-resolved", loop=TrainingLoop(max_steps=1, per_device_batch_size=1),
                              clip_epsilon_low=spec.clip_low, clip_epsilon_high=spec.clip_high,
                              policy_updates=PolicyUpdateSettings(PolicyUpdateSchedule("episode", 2),
                                                                  PolicyExecutionBudget(1, 100, 10000)))
    prepared = resolve_policy_population(snapshot, credit, selection, capabilities)
    if population_id is None:
        assert prepared.updates == (update,)
    runtime = ResolvedVeRLPopulation.from_resolved(prepared, read_input=lambda view: source,
                                                 score_temperature=1.0, score_contract="native-fixture@1",
                                                 sampler_correction=None, **kwargs)
    assert runtime.credit is prepared.credit
    return runtime


def test_manifest_actor_factory_requires_real_identity_and_publishing_observer(tmp_path):
    from posttrain.common import NullObserver, RunContext
    from posttrain.train.backends.verl.contracts import VerlLaunchManifest, VerlRunContext
    from posttrain.train.backends.verl.launcher import build_grpo_launch_plan
    from posttrain.train.backends.verl.policy_observer import ResolvedWorkerObserver
    from posttrain.train.backends.verl.policy_session import actor_session_from_manifest

    from .test_update_resolution import capabilities
    from .test_verl_backend import _grpo_request

    request = _grpo_request()
    selection = PolicyUpdateSettings(PolicyUpdateSchedule("episode", 1), PolicyExecutionBudget(1, 100, 1000))
    settings = replace(request.settings, loop=replace(request.settings.loop, gradient_accumulation_steps=1),
                       beta=0., policy_updates=selection)
    payload = build_grpo_launch_plan(request, tmp_path).model_dump()
    payload["payload"]["resolved_settings"] = {"kind": "grpo", "settings": settings}
    payload["payload"]["algorithm"]["policy_updates"] = selection
    payload["payload"]["algorithm"]["beta"] = 0.
    payload["payload"]["training"]["loop"]["gradient_accumulation_steps"] = 1
    manifest = VerlLaunchManifest.model_validate(payload)
    kwargs = {"capabilities": capabilities(), "runtime_identity": "runtime@1", "template_revision": "template@1",
              "score_temperature": 1., "score_contract": "native-fixture@1"}
    with pytest.raises(InvalidPolicyUpdate, match="actual host context"):
        actor_session_from_manifest(manifest, NativeOperatorEngine(), NullObserver(), **kwargs)
    context = RunContext("project", "work", "real-run", "train.grpo", "job@1", tmp_path)
    manifest = manifest.model_copy(update={"run_context": VerlRunContext.model_validate({
        **context.identity_attributes, "workspace": context.workspace})})
    manifest = VerlLaunchManifest.model_validate_json(manifest.model_dump_json())
    with pytest.raises(InvalidPolicyUpdate, match="publishing observer"):
        actor_session_from_manifest(manifest, NativeOperatorEngine(), NullObserver(), **kwargs)
    assert manifest.run_context is not None
    observer = ResolvedWorkerObserver(tmp_path / "journal.jsonl", manifest.run_context)
    session = actor_session_from_manifest(manifest, NativeOperatorEngine(), observer, **kwargs)
    assert session.context.run_id == "real-run"
    assert session.context.observer is observer
    assert session.settings == settings and session.state()["applied"] == 0


def test_native_jagged_transport_preserves_original_score_positions():
    runtime = population()
    rows = runtime._rows(runtime.updates[0].dependencies)
    data = runtime._batch(rows)
    assert data["input_ids"].is_nested
    assert data["input_ids"].unbind()[0].tolist() == [1, 2, 3, 4, 5]
    assert data["position_ids"].unbind()[0].tolist() == list(range(5))
    assert data["loss_mask"].tolist() == [[0, 0, 1, 1], [0, 0, 1, 1]]
    output = {"log_probs": torch.nested.as_nested_tensor([torch.arange(5.), torch.arange(10., 15.)],
                                                        layout=torch.jagged)}
    assert [value.item() for value in runtime._scores(output, data, rows).values()] == [2., 3., 12., 13.]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_selected_native_projection_matches_full_reference_values_and_gradients(dtype):
    runtime = population()
    rows = runtime._rows(runtime.updates[0].dependencies)
    data = runtime._batch(rows)
    torch.manual_seed(19)
    logits = torch.randn(2, 5, 8, dtype=dtype, requires_grad=True)
    output = selected_native_outputs(SimpleNamespace(logits=logits), {"temperature": torch.tensor([0.7, 0.7])}, data)
    actual = torch.stack(tuple(runtime._scores(output, data, rows).values()))
    full = (logits.float() / 0.7).log_softmax(-1)
    expected = torch.stack([full[row, position, data["input_ids"].unbind()[row][position + 1]]
                            for row in range(2) for position in (2, 3)])
    torch.testing.assert_close(actual, expected)
    left = torch.autograd.grad(actual.sum(), logits, retain_graph=True)[0]
    right = torch.autograd.grad(expected.sum(), logits)[0]
    torch.testing.assert_close(left, right)
    assert not bool(left[:, :2].any())
    with pytest.raises(InvalidPolicyUpdate, match="truncate"):
        selected_native_outputs(SimpleNamespace(logits=logits[:, :4]), {"temperature": torch.tensor([0.7, 0.7])}, data)


class NativeOperatorEngine(BaseEngine):
    """Real BaseEngine lifecycle with an explicit small causal-model fixture."""

    def __init__(self, skip_first=False):
        self.model = CausalModel()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: 1.0)
        self.last_loss_scale_metrics = {}
        self.steps = 0
        self.lr_steps = 0
        self.skip_first = skip_first

    def get_data_parallel_size(self):
        return 1

    def eval_mode(self):
        return nullcontext()

    def train_mode(self):
        return nullcontext()

    def is_mp_src_rank_with_outputs(self):
        return True

    def optimizer_zero_grad(self):
        self.optimizer.zero_grad()

    def optimizer_step(self):
        self.steps += 1
        norm = sum(parameter.grad.square().sum().item() for parameter in self.model.parameters()) ** 0.5
        skipped = self.skip_first and self.steps == 1
        self.last_loss_scale_metrics = {"optimizer_step_skipped": float(skipped)}
        if not skipped:
            self.optimizer.step()
        return norm

    def lr_scheduler_step(self):
        self.lr_steps += 1
        self.lr_scheduler.step()

    def forward_backward_batch(self, data, loss_function, forward_only=False):
        from verl.workers.engine.utils import prepare_micro_batches

        outputs = []
        micro_batches, _ = prepare_micro_batches(data)
        for micro_batch in micro_batches:
            probabilities = []
            for ids in micro_batch["input_ids"].unbind():
                positions = torch.arange(len(ids)).unsqueeze(0)
                raw = self.model(ids.unsqueeze(0), torch.ones_like(ids).unsqueeze(0), positions, False).logits[0]
                probabilities.append(raw.log_softmax(-1).gather(1, ids.roll(-1).unsqueeze(1)).squeeze(1))
            model_output = {"log_probs": torch.nested.as_nested_tensor(probabilities, layout=torch.jagged)}
            if loss_function is not None:
                loss, _ = loss_function(model_output, micro_batch)
                if not forward_only:
                    loss.backward()
            outputs.extend(value.detach() for value in probabilities)
        return {"model_output": {"log_probs": torch.nested.as_nested_tensor(outputs, layout=torch.jagged)}, "metrics": {}}


def test_native_execution_layouts_preserve_objective_and_parameter_update():
    runtimes = [population(), population()]
    runtimes[1] = replace(runtimes[1], execution=PolicyExecutionBudget(2, 100, 10000),
        capabilities=replace(runtimes[1].capabilities, context_layout="dense-pack"))
    engines = [NativeOperatorEngine(), NativeOperatorEngine()]
    engines[1].model.load_state_dict(engines[0].model.state_dict())
    for runtime, engine in zip(runtimes, engines, strict=True):
        runtime.run_update(engine, 0)
        assert engine.steps == engine.lr_steps == 1
    assert runtimes[0].last_output['loss'] == runtimes[1].last_output['loss']
    for left, right in zip(engines[0].model.parameters(), engines[1].model.parameters(), strict=True):
        torch.testing.assert_close(left, right)
    rows = runtimes[1]._rows(runtimes[1].updates[0].dependencies)
    assert runtimes[1]._pack_sizes(rows) == (2,)
    limited = replace(runtimes[1], execution=PolicyExecutionBudget(2, 5, 10000))
    assert limited._pack_sizes(rows) == (1, 1)


def test_native_dense_packs_require_explicit_physical_layout():
    with pytest.raises(InvalidPolicyUpdate, match="declared physical context layout"):
        replace(population(), execution=PolicyExecutionBudget(2, 100, 10000))


def test_native_population_padding_preserves_context_and_enforces_physical_budget():
    from posttrain.train.backends.verl.policy_updates import resolved_verl_engine_type
    from tensordict import TensorDict
    from verl.utils import tensordict_utils as tu

    original = torch.arange(12).reshape(2, 6)

    class DenseInputs:
        preparations = 0

        def prepare_model_inputs(self, data):
            self.preparations += 1
            return {"input_ids": original.clone(), "attention_mask": torch.ones_like(original),
                    "position_ids": torch.arange(6).expand(2, 6)}, {"original": True}

    engine = resolved_verl_engine_type(DenseInputs)()
    data = TensorDict({"dummy": torch.ones(2, 1)}, batch_size=[2])
    tu.assign_non_tensor(data, resolved_dense_width=9, resolved_context_budget=18)
    values, output_args = engine.prepare_model_inputs(data)
    assert output_args == {"original": True}
    assert values["input_ids"].shape == (2, 9)
    assert torch.equal(values["input_ids"][:, :6], original)
    assert torch.equal(values["position_ids"][:, :6], torch.arange(6).expand(2, 6))
    assert values["attention_mask"][:, :6].eq(1).all()
    assert all(values[key][:, 6:].eq(0).all() for key in values)
    tu.assign_non_tensor(data, resolved_context_budget=17)
    with pytest.raises(InvalidPolicyUpdate, match="physical execution budget"):
        engine.prepare_model_inputs(data)
    assert engine.preparations == 1
    tu.assign_non_tensor(data, resolved_dense_width=10**9)
    with pytest.raises(InvalidPolicyUpdate, match="physical execution budget"):
        engine.prepare_model_inputs(data)
    assert engine.preparations == 1
    tu.assign_non_tensor(data, resolved_context_budget=18, resolved_dense_width=5)
    with pytest.raises(InvalidPolicyUpdate, match="bounded causal-text"):
        engine.prepare_model_inputs(data)


@pytest.mark.parametrize("overflow", [False, True])
def test_native_base_engine_retries_same_occurrence_without_advancing_lr(overflow):
    runtime = population(max_overflow_retries=1)
    engine = NativeOperatorEngine(skip_first=overflow)
    before = [parameter.detach().clone() for parameter in engine.model.parameters()]
    runtime.run_update(engine, 0)
    assert runtime.applied_updates == runtime.next_update == engine.lr_steps == 1
    assert runtime.attempts == engine.steps == (2 if overflow else 1)
    assert any(not torch.equal(parameter, initial) for parameter, initial in zip(engine.model.parameters(), before, strict=True))
    assert all(not value.requires_grad for value in runtime.old.values.values())
    assert runtime.last_output["loss"] == runtime.last_adjoints.evaluation.loss.item()


def test_continuous_verl_populations_keep_native_optimizer_and_attempt_offsets():
    engine = NativeOperatorEngine(skip_first=True)
    collected = []

    def collect(applied, attempts):
        if collected:
            assert collected[-1].next_update == len(collected[-1].updates)
        runtime = population(population_id=f"population-{applied}", applied_update_offset=applied,
                             attempt_offset=attempts, max_overflow_retries=1)
        collected.append(runtime)
        return runtime

    run = ResolvedVeRLRun(collect, lambda checkpoint: pytest.fail("no checkpoint restoration expected"))
    optimizer = engine.optimizer
    run.run_update(engine)
    assert run.active().global_applied_updates == 1 and run.active().global_attempts == 2
    run.run_update(engine)
    assert engine.optimizer is optimizer
    assert engine.lr_scheduler.last_epoch == run.active().global_applied_updates == 2
    assert engine.steps == run.active().global_attempts == 3
    assert [(p.applied_update_offset, p.attempt_offset) for p in collected] == [(0, 0), (1, 2)]
    assert any(not torch.equal(collected[0].old.values[action], collected[1].old.values[action])
               for action in collected[0].old.values)


@pytest.mark.parametrize("actor_rpc", [False, True])
def test_native_collection_host_admits_receipts_reuses_population_and_retains_engine(tmp_path, monkeypatch, actor_rpc):
    import hashlib
    import json
    import math

    import numpy as np
    from posttrain.common import LocalArtifactRef, ProducedArtifact
    from posttrain.train.backends.verl.policy_job import ResolvedVeRLCollectionHost
    from posttrain.train.backends.verl.policy_rollouts import retain_episode_receipt
    from posttrain.train.online_rl import BehaviorPolicySpan
    from posttrain.train.profiles import GRPOSettings
    from verl import DataProto

    from .test_update_admission import source
    from .test_update_resolution import capabilities

    settings = GRPOSettings("host", TrainingLoop(max_steps=4), num_prompts_per_step=1, num_generations=2,
        policy_updates=PolicyUpdateSettings(PolicyUpdateSchedule("episode", 1), PolicyExecutionBudget(1, 100, 1000)))
    artifacts, collected = [], []
    frozen_corrections = {}
    engine = NativeOperatorEngine(skip_first=True)
    before = [parameter.detach().clone() for parameter in engine.model.parameters()]

    def collect(applied, attempts):
        rollouts, evidence, decode = source()
        collected.append((applied, attempts))
        monkeypatch.setattr("posttrain.train.integrations.verifiers_population_artifact.decode_native_population",
            lambda raw, **kwargs: decode(json.dumps([json.loads(line) for line in raw.splitlines()]).encode()))
        receipts = []
        for index, (rollout, record) in enumerate(zip(rollouts, json.loads(evidence), strict=True)):
            trace_id = f"trace-{applied}-{index}"
            record["id"] = trace_id
            rollout = replace(rollout, trace=replace(rollout.trace, external_id=trace_id),
                behavior_policy=BehaviorPolicySpan(applied, applied),
                conditioning_records=tuple(replace(value, trace_id=trace_id) for value in rollout.conditioning_records))
            path = tmp_path / f"{trace_id}.jsonl"
            path.write_text(json.dumps(record) + "\n")
            artifact = ProducedArtifact(trace_id, "evaluation-traces",
                LocalArtifactRef(path, hashlib.sha256(path.read_bytes()).hexdigest()),
                metadata={"format": "verifiers-native-traces", "replay_authority": True, "trace_ids": [trace_id]})
            bridge = SimpleNamespace(policy_update_context_contract="causal-text@1",
                retain_population=lambda selected, artifact=artifact: artifact)
            receipts.append(retain_episode_receipt(bridge, rollout, sampler_step=applied))
        return DataProto(non_tensor_batch={"posttrain_native_episode_receipt": np.array(receipts, dtype=object)})

    context = cast(RunContext, SimpleNamespace(run_id="run", workspace=tmp_path, artifact=artifacts.append))
    session = None
    if actor_rpc:
        from posttrain.train.backends.verl.policy_session import ResolvedVeRLActorSession

        session = ResolvedVeRLActorSession(context, settings, capabilities(), engine,
            lambda path: pytest.fail("unexpected restoration"), "runtime@1", "template@1", 1., "native-fixture@1",
            max_overflow_retries=1)
        host = session.host
    else:
        host = ResolvedVeRLCollectionHost(context, settings, capabilities(), engine, collect,
            lambda path: pytest.fail("unexpected restoration"), "runtime@1", "template@1", 1., "native-fixture@1",
            max_overflow_retries=1)
    optimizer = engine.optimizer
    for slot in range(4):
        if actor_rpc:
            assert session is not None
            state = session.state()
            batch = collect(slot, state["attempts"]) if state["needs_population"] else None
            receipts = tuple(batch.non_tensor_batch["posttrain_native_episode_receipt"].tolist()) if batch else None
            metrics = session.update(receipts, expected_applied=slot)
            assert metrics["train/rl/applied_optimizer_updates"] == slot + 1
            assert metrics["train/rl/advantage_nonzero_fraction"] == 1
            from posttrain.train.grpo_observations import GRPOObservationFeatures, normalize_grpo_metrics

            normalized = normalize_grpo_metrics(backend="verl", step=slot+1, native=metrics,
                                                features=GRPOObservationFeatures())
            assert normalized.metrics == metrics
            if slot == 0:
                with pytest.raises(InvalidPolicyUpdate, match="cannot consume a fresh"):
                    session.update(receipts, expected_applied=1)
                with pytest.raises(InvalidPolicyUpdate, match="native actor boundary"):
                    session.update(None, expected_applied=0)
        else:
            host.run_update()
        active = host.run.active()
        assert active.old is not None and active.sampler_correction is not None
        assert active.prepare_sampler_correction is None
        population_digest = active.updates[0].population.digest
        # Independent episode-product reference; native sampled logps are -1.
        sums = {}
        for action, score in active.old.values.items():
            sums[action.episode_id] = sums.get(action.episode_id, 0.) + float(score) + 1.
        expected = {action: min(max(math.exp(sums[action.episode_id]), .1), 3.) for action in active.old.values}
        assert dict(active.sampler_correction) == pytest.approx(expected)
        previous = frozen_corrections.setdefault(population_digest, dict(active.sampler_correction))
        assert dict(active.sampler_correction) == previous
    assert collected == [(0, 0), (2, 3)]
    assert len(artifacts) == 2
    assert engine.optimizer is optimizer
    assert engine.lr_scheduler.last_epoch == host.run.active().global_applied_updates == 4
    assert engine.steps == host.run.active().global_attempts == 5
    assert any(not torch.equal(parameter, initial)
               for parameter, initial in zip(engine.model.parameters(), before, strict=True))
    retained = host.run.active()
    host.restore_population = lambda path: retained
    assert host.restore(tmp_path) is retained
    host.settings = replace(settings, clip_epsilon_low=.05)
    with pytest.raises(InvalidPolicyUpdate, match="selected job contracts"):
        host.restore(tmp_path)


def test_continuous_verl_rejects_wrong_prior_attempts_before_native_scoring():
    engine = NativeOperatorEngine(skip_first=True)
    run = ResolvedVeRLRun(lambda applied, attempts: population(population_id=f"p-{applied}",
        applied_update_offset=applied, attempt_offset=applied, max_overflow_retries=1),
        lambda checkpoint: pytest.fail("no checkpoint restoration expected"))
    run.run_update(engine)
    previous = run.current
    before = [p.detach().clone() for p in engine.model.parameters()]
    with pytest.raises(InvalidPolicyUpdate, match="exact run boundary"):
        run.run_update(engine)
    assert run.current is previous
    assert engine.steps == 2 and engine.lr_scheduler.last_epoch == 1
    assert all(torch.equal(p, value) for p, value in zip(engine.model.parameters(), before, strict=True))
