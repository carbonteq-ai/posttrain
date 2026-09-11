"""Real two-rank failure agreement; no generation-runtime claims are implied."""

import os
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest


def _rank(rank: int, rendezvous: str) -> None:
    import torch.distributed as dist
    from posttrain.common import TraceObservation
    from posttrain.train.online_rl import EnvironmentRollout, RolloutBatch
    from posttrain.train.profiles import GDPOSettings, TrainingLoop
    from posttrain.train.reward_admission import admit_reward_groups
    from posttrain.train.reward_advantages import compute_gdpo_advantages
    from posttrain.train.reward_evidence import RewardEvidence, RewardValue

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        settings = GDPOSettings(
            id="test",
            loop=TrainingLoop(max_steps=1, gradient_accumulation_steps=2),
            component_names=("score",),
            component_weights=(1.0,),
        )
        batch = RolloutBatch(("task",), 1, "policy", ("shared-group",), (f"r{rank}",))
        attempts = 0

        def collect(selected):
            nonlocal attempts
            attempts += 1
            if rank == 1 and attempts == 1:
                raise ValueError("one-rank annotation failure")
            identity = selected.rollout_ids[0]
            trace = TraceObservation(trace_type="fixture", external_id=identity, payload={})
            evidence = RewardEvidence(
                "shared-group", identity, identity, "0", "projection@1", (RewardValue("score", "valid", float(rank)),)
            )
            return [
                EnvironmentRollout(
                    "task", (1,), (2,), (-1.0,), (True,), float(rank), False, trace, reward_evidence=evidence
                )
            ]

        def gather(local):
            received: list[Any] = [None, None]
            dist.all_gather_object(received, local)
            assert all(isinstance(shard, list) for shard in received)
            return [item for shard in received if isinstance(shard, list) for item in shard]

        admitted = admit_reward_groups(batch, settings, collect, gather)
        assert admitted.rounds == attempts == 2
        evidence = gather([row.reward_evidence for row in admitted.rollouts])
        assert {item.rollout_id for item in evidence} == {"r0/attempt/1", "r1/attempt/1"}
        values = compute_gdpo_advantages(
            evidence,
            [(True,), (True,)],
            group_size=2,
            component_names=("score",),
            component_weights=(1.0,),
            epsilon=1e-6,
        )
        assert values.token_advantages[0][0] < 0 < values.token_advantages[1][0]
        assert sum(row[0] for row in values.token_advantages) == 0
    finally:
        dist.destroy_process_group()


def test_two_rank_failure_replaces_entire_shared_group(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    torch = pytest.importorskip("torch")
    if not torch.distributed.is_gloo_available():
        pytest.skip("Gloo unavailable")
    repository_root = str(Path(__file__).resolve().parents[3])
    inherited = os.environ.get("PYTHONPATH")
    monkeypatch.setenv("PYTHONPATH", f"{repository_root}:{inherited}" if inherited else repository_root)
    monkeypatch.syspath_prepend(repository_root)
    torch.multiprocessing.spawn(_rank, args=((tmp_path / "rendezvous").as_uri(),), nprocs=2, join=True)
