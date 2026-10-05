"""Native collection hands retained rollouts to the resolved admission boundary."""

from contextlib import nullcontext
from types import SimpleNamespace
from typing import cast

import pytest
from posttrain.common import RunContext
from posttrain.train.backends.trl.policy_rollouts import collect_resolved_population
from posttrain.train.backends.trl.update_totals import RolloutUpdateTotals
from posttrain.train.requests import GRPORequest
from posttrain.train.update_records import InvalidPolicyUpdate, PolicyVersions

from .test_update_admission import source
from .test_update_resolution import capabilities, settings


@pytest.mark.parametrize("corrupt", [False, True])
def test_collector_preserves_native_rollouts_and_publishes_after_admission(tmp_path, monkeypatch, corrupt):
    import hashlib

    from posttrain.common import LocalArtifactRef, ProducedArtifact

    rollouts, evidence, decode = source()
    path = tmp_path / "native.jsonl"
    path.write_bytes(evidence)
    artifact = ProducedArtifact(
        "training/rollouts/populations/test",
        "evaluation-traces",
        LocalArtifactRef(path.resolve(), hashlib.sha256(evidence).hexdigest()),
        metadata={"format": "verifiers-native-traces", "replay_authority": True},
    )
    seen = []

    async def run(batch, generator):
        seen.append(("collect", batch.example_ids))
        return rollouts

    def retain(values):
        assert values == rollouts
        seen.append(("retain", None))
        if corrupt:
            path.write_bytes(evidence + b" ")
        return artifact

    bridge = SimpleNamespace(policy_update_context_contract="causal-text@1", retain_population=retain, run=run)
    request = SimpleNamespace(bridge=bridge, settings=settings(), policy=SimpleNamespace(id="model"), training=object())
    context = SimpleNamespace(
        phase=lambda *args: nullcontext(),
        trace=lambda value: None,
        trace_fact_update=lambda value: None,
        artifact=lambda value: seen.append(("publish", value.name)),
        metrics=lambda values, **kwargs: seen.append(("metrics", (values, kwargs))),
    )
    trainer = SimpleNamespace(state=SimpleNamespace(global_step=3), accelerator=SimpleNamespace(num_processes=1))
    totals = SimpleNamespace(add_batch=lambda *args, **kwargs: None)
    monkeypatch.setattr("posttrain.train.backends.trl.policy_config._rollout_execution_config", lambda request: None)
    monkeypatch.setattr("posttrain.train.backends.trl.online_rl.TrlPolicyGenerator", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        "posttrain.train.integrations.verifiers_population_artifact.decode_native_population",
        lambda raw, **kwargs: decode(raw),
    )

    def collect():
        return collect_resolved_population(
            cast(RunContext, context),
            cast(GRPORequest, request),
            object(),
            trainer,
            [{"example_id": "task", "prompt": []}] * 2,
            capabilities(),
            population_id="population@3",
            template_revision="template@1",
            versions=PolicyVersions("sampler@3", "old@3", "current@3", None),
            selector_digest="complete-groups@1",
            attempt_offset=5,
            totals=cast(RolloutUpdateTotals, totals),
        )

    if corrupt:
        with pytest.raises(InvalidPolicyUpdate, match="declared digest"):
            collect()
        assert [entry[0] for entry in seen] == ["collect", "retain"]
        return
    admitted = collect()
    assert [entry[0] for entry in seen] == ["collect", "retain", "publish", "metrics"]
    assert admitted.applied_update_offset == 3 and admitted.attempt_offset == 5
    assert admitted.resolved.snapshot.native_evidence_ref == artifact.name
    assert len(admitted.resolved.updates) == 2
    # The admitted population is described once, at the first update trained on
    # it. Two one-turn episodes with sparse rewards 0 and 1 share one anchor, so
    # SAMPO "mean" credit is -/+0.5 at both the episode and the turn level.
    values, metadata = seen[-1][1]
    assert metadata == {"step": 4, "attributes": {"measurement_scope": "resolved-collection"}}
    assert values == pytest.approx(
        {
            "train/rl/reward_mean": 0.5,
            "train/rl/reward_std": 2**-0.5,
            "train/rl/group_reward_std_mean": 2**-0.5,
            "train/rl/group_zero_variance_fraction": 0.0,
            "train/rl/completion_tokens_mean": 2.0,
            "train/rl/completion_tokens_max": 2.0,
            "train/rl/completion_truncation_rate": 0.0,
            "train/rl/episode_advantage_mean": 0.0,
            "train/rl/turn_advantage_mean": 0.0,
            "train/rl/anchor_group_size_mean": 2.0,
            "train/rl/sparse_reward_projection_fraction": 1.0,
            "train/rl/episode_advantage_abs_mean": 0.5,
            "train/rl/turn_advantage_abs_mean": 0.5,
            "train/rl/turn_advantage_informative_fraction": 1.0,
            "train/rl/singleton_anchor_fraction": 0.0,
            "train/rl/turn_credit_share": 0.5,
        }
    )


def test_collector_rejects_missing_native_contract_before_collection():
    request = SimpleNamespace(bridge=SimpleNamespace())
    with pytest.raises(InvalidPolicyUpdate, match="native conditioning"):
        collect_resolved_population(
            cast(RunContext, None),
            cast(GRPORequest, request),
            None,
            None,
            [],
            capabilities(),
            population_id="population",
            template_revision="template@1",
            versions=PolicyVersions("a", "a", "a", None),
            selector_digest="selection",
            attempt_offset=0,
        )
