from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from types import SimpleNamespace

from posttrain.common import (
    EventObservation,
    ExecutionContext,
    Invocation,
    Job,
    JobAction,
    MetricBatchObservation,
    MetricObservation,
    ProducedArtifact,
    RunAttempt,
    TraceObservation,
)
from posttrain.train.integrations import VerifiersGRPOBridge, verifiers_rollout_dataset
from posttrain.train.integrations import verifiers as bridge_module


@dataclass
class Observer:
    traces: list[TraceObservation] = field(default_factory=list)
    artifacts: list[ProducedArtifact] = field(default_factory=list)

    def event(self, observation: EventObservation) -> None:
        del observation

    def metric(self, observation: MetricObservation) -> None:
        del observation

    def metrics(self, observation: MetricBatchObservation) -> None:
        del observation

    def trace(self, observation: TraceObservation) -> None:
        self.traces.append(observation)

    def artifact(self, artifact: ProducedArtifact) -> None:
        self.artifacts.append(artifact)


class FakeTrace:
    def __init__(self, **values: object) -> None:
        self.id = "trace-1"
        self.values = values
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
        return {"id": self.id, "rewards": self.rewards, "metrics": self.metrics}


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


def add_shaping(trace: FakeTrace, completion: str, completion_tokens: int) -> None:
    del completion, completion_tokens
    trace.record_reward("shape", 0.5, 0.1)


def test_generic_bridge_projects_tasks_scores_and_preserves_native_trace(monkeypatch, tmp_path) -> None:
    task = FakeTask()
    dataset = verifiers_rollout_dataset("custom/train-v1", "revision", "custom-v1", {7: task})
    assert dataset.examples[0].prompt == "Arbitrary environment prompt"
    assert dataset.examples[0].metadata["task_index"] == 7

    runtime = FakeRuntime()
    monkeypatch.setattr(
        bridge_module,
        "_imports",
        lambda: (FakeTrace, FakeValue, FakeValue, (FakeValue, lambda _: runtime)),
    )
    observer = Observer()
    context = ExecutionContext(
        Job("custom/job", "a" * 40, "Custom job"),
        JobAction("custom/job", "train/grpo", "reinforcement-learning"),
        Invocation.new(),
        RunAttempt.new(),
        tmp_path.resolve(),
        observer,
    )
    bridge = VerifiersGRPOBridge(
        context=context,
        tasks={7: task},
        environment_factory=lambda: SimpleNamespace(runtime_for=lambda _: object()),
        trace_path=tmp_path / "traces.jsonl",
        environment_id="custom-v1",
        model_profile_id="model",
        training_profile_id="grpo",
        enrichers=(add_shaping,),
    )

    reward = asyncio.run(bridge._score(task, 7, "completion", 12, 0))
    artifact = bridge.publish_native_artifact()

    assert reward == 1.05
    assert observer.traces[0].external_id == "trace-1"
    assert observer.traces[0].payload["rewards"] == {"native": 1.0, "shape": 0.05}
    assert artifact is not None
    assert observer.artifacts == [artifact]
