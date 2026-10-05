"""Resolved TRL active rounds: TRL post11 arithmetic and durable candidate accounting."""

import json
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, cast

import pytest
from posttrain.common import RunContext
from posttrain.train.backends.trl.policy_job import validate_resolved_job
from posttrain.train.backends.trl.policy_rollouts import collect_active_resolved_population
from posttrain.train.profiles import ActiveGroupSampling, AdaptiveCurriculum, SAMPOSettings, TrainingLoop
from posttrain.train.requests import SAMPORequest
from posttrain.train.update_active_rounds import ActiveRoundPlan
from posttrain.train.update_plan import PolicyExecutionBudget, PolicyUpdateSchedule, PolicyUpdateSettings
from posttrain.train.update_records import InvalidPolicyUpdate, PolicyVersions

from .test_update_resolution import capabilities


def _rounds(plan: ActiveRoundPlan, kept: list[int]) -> list[tuple[int, int, int]]:
    for retained in kept:
        step = plan.next_round()
        if step is None:
            break
        plan.record(*step, retained)
    return plan.round_log


@pytest.mark.parametrize(
    ("target", "rounds", "oversample", "refill", "kept"),
    [
        (1, 3, 0, 0, [0, 1]),
        (2, 3, 1, 0, [1, 0, 1]),
        (3, 4, 2, 1, [1, 1, 2]),
        (2, 2, 0, 2, [0, 2]),
    ],
)
def test_round_plan_matches_native_verl_active_rounds(target, rounds, oversample, refill, kept):
    native_module = pytest.importorskip("verl.trainer.ppo.v1.replay_buffer")
    plan = ActiveRoundPlan(target, rounds, oversample, refill)
    native = native_module.ActiveSamplingRounds(
        target=target, max_rounds=rounds, oversample=oversample, oversample_refill=refill
    )
    assert _rounds(plan, kept) == _rounds(native, kept)
    prefix = native_module.ACTIVE_SAMPLING_METRIC_PREFIX
    expected = {
        f"train/rl/active_sampling_{name.removeprefix(prefix)}": float(value)
        for name, value in native.metrics(4).items()
    }
    assert plan.metrics(4) == expected


def test_round_plan_sizes_rounds_and_rejects_exhaustion():
    plan = ActiveRoundPlan(2, 3, oversample=1, oversample_refill=0)
    assert _rounds(plan, [1, 0, 1]) == [(3, 3, 1), (1, 1, 0), (1, 1, 1)]
    assert plan.metrics(4)["train/rl/active_sampling_candidate_groups_unused"] == 4.0
    with pytest.raises(InvalidPolicyUpdate, match="exhausted"):
        exhausted = ActiveRoundPlan(1, 2)
        _rounds(exhausted, [0, 0])
        exhausted.require_full()


def _settings(**changes: Any) -> SAMPOSettings:
    return SAMPOSettings(
        "settings/active@1",
        TrainingLoop(max_steps=2, per_device_batch_size=1),
        num_prompts_per_step=1,
        num_generations=2,
        active_sampling=ActiveGroupSampling(3),
        policy_updates=PolicyUpdateSettings(PolicyUpdateSchedule("episode", 1), PolicyExecutionBudget(1, 100, 1000)),
        **changes,
    )


def _collect(tmp_path, monkeypatch, rewards: dict[str, list[float]]):
    published: list[Any] = []
    admitted: dict[str, Any] = {}
    collected: list[tuple[str, ...]] = []

    def collector_factory(context, request, tokenizer, totals, *, retain_native):
        assert retain_native

        def collect(prompts, trainer, *, inputs):
            collected.append(tuple(row["example_id"] for row in inputs))
            counts: dict[str, int] = {}
            values = []
            for row in inputs:
                task = row["example_id"]
                index = counts[task] = counts.get(task, -1) + 1
                if index < len(rewards[task]):
                    values.append(
                        SimpleNamespace(
                            example_id=task,
                            reward=rewards[task][index],
                            completion_ids=(1, 2),
                            env_mask=(True, True),
                            is_truncated=False,
                            trace=SimpleNamespace(external_id=f"{task}/{index}"),
                        )
                    )
            return tuple(values)

        return collect

    def admit(artifact, population, settings, capabilities, **kwargs):
        admitted.update(population=population, kwargs=kwargs)
        # A non-SAMPO estimator id: only population reward/shape evidence is reported.
        return SimpleNamespace(
            population=population, resolved=SimpleNamespace(credit=SimpleNamespace(estimator_id="fixture@1"))
        )

    monkeypatch.setattr("posttrain.train.backends.trl.policy_rollouts.rollout_function", collector_factory)
    monkeypatch.setattr(
        "posttrain.train.backends.trl.policy_rollouts.AdmittedNativePopulation.from_retained_artifact", admit
    )
    bridge = SimpleNamespace(
        policy_update_context_contract="causal-text@1",
        retain_population=lambda values: SimpleNamespace(name="retained", values=values),
    )
    request = SimpleNamespace(bridge=bridge, settings=_settings())
    context = SimpleNamespace(
        run_id="run",
        phase=lambda *args: nullcontext(),
        artifact=published.append,
        metrics=lambda values, **kwargs: published.append(("metrics", values)),
    )
    trainer = SimpleNamespace(state=SimpleNamespace(global_step=1), accelerator=SimpleNamespace(num_processes=1))
    reserved = [{"example_id": task, "prompt": []} for task in ("a", "b", "c")]
    run = lambda: collect_active_resolved_population(  # noqa: E731
        cast(RunContext, context),
        cast(SAMPORequest, request),
        object(),
        trainer,
        reserved,
        capabilities(),
        evidence_directory=tmp_path / "evidence",
        population_id="population@1",
        template_revision="template@1",
        versions=PolicyVersions("sampler@1", "old@1", "current@1", None),
        selector_digest="selection@1",
        attempt_offset=1,
    )
    return run, published, admitted, collected


def _snapshots(published):
    return [
        json.loads(item.reference.path.read_text())
        for item in published
        if getattr(item, "kind", None) == "training-collection"
    ]


def test_active_collection_discards_uniform_groups_and_accounts_for_every_candidate(tmp_path, monkeypatch):
    run, published, admitted, collected = _collect(
        tmp_path, monkeypatch, {"a": [0.0, 0.0], "b": [0.0, 1.0], "c": [1.0, 1.0]}
    )
    run()
    assert collected == [("a", "a"), ("b", "b")], "refill dispatches only the missing group; c stays unused"
    assert [rollout.example_id for rollout in admitted["population"]] == ["b", "b"]
    final = _snapshots(published)[-1]
    assert final["status"] == "selected" and final["selected"] == ["candidate-1"]
    assert [(group["uid"], group["spread_eligible"]) for group in final["groups"]] == [
        ("candidate-0", False),
        ("candidate-1", True),
    ]
    assert final["groups"][0]["traces"] == ["a/0", "a/1"]
    assert [item["uids"] for item in final["rounds"]] == [["candidate-0"], ["candidate-1"]]
    statuses = [snapshot["status"] for snapshot in _snapshots(published)]
    assert statuses == ["reserved", "dispatching", "round-observed", "dispatching", "round-observed", "selected"]
    metrics, collection = (values for kind, values in (item for item in published if isinstance(item, tuple)))
    assert metrics["train/rl/active_sampling_candidate_groups_unused"] == 2.0
    # Only the selected group b ([0, 1]) is the admitted population; the uniform
    # group a was generated but never trained on.
    assert collection["train/rl/reward_mean"] == 0.5
    assert collection["train/rl/group_zero_variance_fraction"] == 0.0
    assert "train/rl/turn_credit_share" not in collection
    assert published.index(next(item for item in published if getattr(item, "name", None) == "retained")) > max(
        index for index, item in enumerate(published) if getattr(item, "kind", None) == "training-collection"
    )


def test_active_collection_records_failed_groups_and_exhaustion(tmp_path, monkeypatch):
    run, published, admitted, collected = _collect(
        tmp_path, monkeypatch, {"a": [0.0], "b": [1.0, 1.0], "c": [0.0, 0.0]}
    )
    with pytest.raises(InvalidPolicyUpdate, match="exhausted"):
        run()
    final = _snapshots(published)[-1]
    assert final["status"] == "failed" and final["selected"] is None
    assert [group["terminal"] for group in final["groups"]] == ["failed", "finished", "finished"]
    assert not admitted, "no population is admitted without a full informative selection"


def test_resolved_job_admits_sampo_active_rounds_but_not_curriculum():
    def request(**changes: Any) -> SAMPORequest:
        # Bypass request construction, which keeps the public resolved guard.
        value = object.__new__(SAMPORequest)
        for name, item in {
            "settings": _settings(**changes),
            "training": SimpleNamespace(runtime=SimpleNamespace(nodes=1, devices_per_node=1), backend_options={}),
            "inference": SimpleNamespace(backend="transformers@1", sampling={}),
            "bridge": SimpleNamespace(
                policy_update_context_contract="causal-text@1", retain_population=lambda population: population
            ),
        }.items():
            object.__setattr__(value, name, item)
        return value

    validate_resolved_job(request())
    with pytest.raises(InvalidPolicyUpdate, match="curriculum"):
        validate_resolved_job(request(adaptive_curriculum=AdaptiveCurriculum("domain")))


def test_resolved_arguments_drop_native_refill_and_precomputed_advantage_transport():
    from posttrain.train.backends.trl.policy_optimization import resolved_native_arguments

    arguments = {
        "active_sampling": True,
        "active_sampling_max_batches": 3,
        "active_sampling_reward_std_epsilon": 0.0,
        "active_sampling_oversample": 1,
        "use_precomputed_advantages": True,
        "beta": 0.01,
    }
    assert resolved_native_arguments(arguments) == {"beta": 0.01}
