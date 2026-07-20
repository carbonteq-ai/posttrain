"""Generic Verifiers task scoring bridge for online training backends."""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from posttrain.common import LocalArtifactRef, ProducedArtifact, TraceObservation

from ..data import CompletedRollout, RolloutDataset, RolloutExample, RolloutScore


class TraceEnricher(Protocol):
    def __call__(self, trace: Any, rollout: CompletedRollout) -> None: ...


type EnvironmentFactory = Callable[[], Any]


def _imports() -> tuple[type[Any], type[Any], type[Any], Any]:
    try:
        from verifiers.v1 import (  # pyright: ignore[reportMissingImports]
            AgentInfo,
            MessageNode,
            Trace,
            TraceTask,
            TrainRunInfo,
        )
        from verifiers.v1.runtimes import make_runtime  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise RuntimeError("install the Verifiers integration dependencies") from error
    return Trace, TraceTask, TrainRunInfo, (AgentInfo, MessageNode, make_runtime)


def _rollout_dataset(
    identifier: str,
    revision: str,
    environment_id: str,
    tasks: Mapping[int, Any],
) -> RolloutDataset:
    """Project native Verifiers tasks into task-neutral train prompts and stable keys."""

    examples = tuple(
        RolloutExample(
            id=f"train/{index:06d}",
            prompt=str(task.data.prompt),
            metadata={"task_index": index, "environment_id": environment_id},
        )
        for index, task in sorted(tasks.items())
    )
    return RolloutDataset(identifier, revision, examples)


@dataclass(slots=True)
class VerifiersOnlineRLEnvironment:
    """Expose native Verifiers tasks through the backend-neutral online-RL contract."""

    dataset_id: str
    revision: str
    tasks: Mapping[int, Any]
    environment_factory: EnvironmentFactory
    trace_path: Path
    environment_id: str
    run_id: str
    enrichers: tuple[TraceEnricher, ...] = ()
    _write_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _trace_count: int = field(default=0, init=False)
    _dataset: RolloutDataset = field(init=False, repr=False)
    _tasks_by_example_id: dict[str, tuple[int, Any]] = field(init=False, repr=False)
    _environment: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._dataset = _rollout_dataset(
            self.dataset_id,
            self.revision,
            self.environment_id,
            self.tasks,
        )
        self._tasks_by_example_id = {
            example.id: (index, self.tasks[index])
            for index, example in zip(sorted(self.tasks), self._dataset.examples, strict=True)
        }
        self._environment = self.environment_factory()

    @property
    def dataset(self) -> RolloutDataset:
        return self._dataset

    async def score(self, rollout: CompletedRollout) -> RolloutScore:
        try:
            task_index, task = self._tasks_by_example_id[rollout.example_id]
        except KeyError as error:
            raise ValueError(f"unknown rollout example {rollout.example_id!r}") from error
        Trace, TraceTask, TrainRunInfo, extras = _imports()
        AgentInfo, MessageNode, make_runtime = extras
        trace = Trace(
            task=TraceTask(type=type(task).__name__, data=task.data),
            run=TrainRunInfo(id=self.run_id, step=rollout.step),
            agent=AgentInfo(model=rollout.model_id),
            nodes=[
                MessageNode(message={"role": "user", "content": str(task.data.prompt)}, sampled=False),
                MessageNode(
                    parent=0,
                    message={"role": "assistant", "content": rollout.completion},
                    sampled=True,
                    token_ids=list(rollout.token_ids),
                    mask=[True] * rollout.token_count,
                    is_content=[True] * rollout.token_count,
                ),
            ],
            info={
                "environment_id": self.environment_id,
                "task_index": task_index,
                "example_id": rollout.example_id,
            },
            is_completed=rollout.terminated,
            stop_condition="agent_completed" if rollout.terminated else "max_tokens",
        )
        runtime = make_runtime(self._environment.runtime_for(task))
        await runtime.start()
        try:
            await task.score(trace, runtime)
        finally:
            await runtime.stop()
        for enrich in self.enrichers:
            enrich(trace, rollout)
        trace.record_metric("completion_tokens", float(rollout.token_count))
        record = trace.to_record()
        self._preserve(record)
        observation = TraceObservation(
            trace_type="verifiers",
            external_id=str(trace.id),
            payload=record,
            attributes={
                "environment_id": self.environment_id,
                "task_index": task_index,
                "example_id": rollout.example_id,
                "is_truncated": rollout.is_truncated,
                "model": rollout.model_id,
            },
        )
        return RolloutScore(float(trace.reward), observation)

    def _preserve(self, record: dict[str, Any]) -> None:
        encoded = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        with self._write_lock:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            with self.trace_path.open("a", encoding="utf-8") as stream:
                stream.write(encoded)
            self._trace_count += 1

    def finalize(self) -> tuple[ProducedArtifact, ...]:
        if not self.trace_path.is_file():
            return ()
        digest = hashlib.sha256(self.trace_path.read_bytes()).hexdigest()
        artifact = ProducedArtifact(
            name=f"training/rollouts/{self.dataset.id}/verifiers-traces",
            kind="evaluation-traces",
            reference=LocalArtifactRef(self.trace_path.resolve(), digest),
            metadata={
                "technique": "grpo",
                "environment_id": self.environment_id,
                "dataset_id": self.dataset.id,
                "dataset_revision": self.dataset.revision,
                "trace_count": self._trace_count,
                "schema_version": 2,
            },
        )
        return (artifact,)


__all__ = ["TraceEnricher", "VerifiersOnlineRLEnvironment"]
