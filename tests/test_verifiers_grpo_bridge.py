from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from posttrain.train import CompletedRollout
from posttrain.train.integrations import VerifiersOnlineRLEnvironment
from posttrain.train.integrations import verifiers as bridge_module


class FakeTrace:
    def __init__(self, **values: object) -> None:
        self.id = "trace-1"
        self.values: dict[str, Any] = values
        self.rewards: dict[str, float] = {}
        self.metrics: dict[str, float] = {}

    def record_reward(self, name: str, value: float, weight: float = 1.0) -> None:
        self.rewards[name] = value * weight

    def record_metric(self, name: str, value: float) -> None:
        self.metrics[name] = value

    @property
    def reward(self) -> float:
        return sum(self.rewards.values())

    def to_record(self) -> dict[str, object]:
        return {
            "id": self.id,
            "rewards": self.rewards,
            "metrics": self.metrics,
            "is_completed": self.values["is_completed"],
            "stop_condition": self.values["stop_condition"],
            "model": self.values["agent"].values["model"],
            "completion_token_ids": self.values["nodes"][1].values["token_ids"],
            "completion_mask": self.values["nodes"][1].values["mask"],
        }


class FakeValue:
    def __init__(self, **values: object) -> None:
        self.values = values


class FakeRuntime:
    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


class FakeTask:
    data = SimpleNamespace(prompt="Arbitrary environment prompt")

    async def score(self, trace: FakeTrace, runtime: FakeRuntime) -> None:
        del runtime
        trace.record_reward("native", 1.0)


def add_shaping(trace: FakeTrace, rollout: CompletedRollout) -> None:
    del rollout
    trace.record_reward("shape", 0.5, 0.1)


def test_generic_bridge_projects_tasks_scores_and_preserves_native_trace(monkeypatch, tmp_path) -> None:
    task = FakeTask()
    runtime = FakeRuntime()
    monkeypatch.setattr(
        bridge_module,
        "_imports",
        lambda: (FakeTrace, FakeValue, FakeValue, (FakeValue, FakeValue, lambda _: runtime)),
    )
    environment = VerifiersOnlineRLEnvironment(
        dataset_id="custom/train-v1",
        revision="revision",
        tasks={7: task},
        environment_factory=lambda: SimpleNamespace(runtime_for=lambda _: object()),
        trace_path=tmp_path / "traces.jsonl",
        environment_id="custom-v1",
        run_id="run-1",
        enrichers=(add_shaping,),
    )
    assert environment.dataset.examples[0].prompt == "Arbitrary environment prompt"
    assert environment.dataset.examples[0].metadata["task_index"] == 7

    score = asyncio.run(
        environment.score(
            CompletedRollout(
                example_id="train/000007",
                completion="completion",
                token_ids=tuple(range(12)),
                step=0,
                terminated=False,
                model_id="model-profile-v1",
            )
        )
    )
    artifacts = environment.finalize()

    assert score.reward == 1.05
    assert score.trace.external_id == "trace-1"
    assert score.trace.payload["rewards"] == {"native": 1.0, "shape": 0.05}
    assert score.trace.payload["is_completed"] is False
    assert score.trace.payload["stop_condition"] == "max_tokens"
    assert score.trace.payload["model"] == "model-profile-v1"
    assert score.trace.payload["completion_token_ids"] == list(range(12))
    assert score.trace.payload["completion_mask"] == [True] * 12
    assert score.trace.attributes["is_truncated"] is True
    assert score.trace.attributes["model"] == "model-profile-v1"
    assert len(artifacts) == 1
    assert artifacts[0].metadata["trace_count"] == 1
