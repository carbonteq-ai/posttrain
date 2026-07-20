"""Generic Verifiers task scoring bridge for online training backends."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

from posttrain.common import ExecutionContext, LocalArtifactRef, ProducedArtifact, TraceObservation

from ..data import RolloutDataset, RolloutExample
from ..requests import RewardFunction


class TraceEnricher(Protocol):
    def __call__(self, trace: Any, completion: str, completion_tokens: int) -> None: ...


type EnvironmentFactory = Callable[[], Any]


def _imports() -> tuple[type[Any], type[Any], type[Any], Any]:
    try:
        from verifiers.v1 import MessageNode, Trace, TraceTask, TrainRunInfo  # pyright: ignore[reportMissingImports]
        from verifiers.v1.runtimes import make_runtime  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise RuntimeError("install the Verifiers integration dependencies") from error
    return Trace, TraceTask, TrainRunInfo, (MessageNode, make_runtime)


def verifiers_rollout_dataset(
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


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if not isinstance(completion, list) or not completion:
        raise TypeError("conversational GRPO completions must be non-empty message lists")
    message = completion[-1]
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise TypeError("GRPO completion messages require string content")
    return cast(str, message["content"])


@dataclass(slots=True)
class VerifiersGRPOBridge:
    """Expose any native Verifiers task collection as one asynchronous GRPO reward."""

    context: ExecutionContext
    tasks: Mapping[int, Any]
    environment_factory: EnvironmentFactory
    trace_path: Path
    environment_id: str
    model_profile_id: str
    training_profile_id: str
    enrichers: tuple[TraceEnricher, ...] = ()
    _write_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _trace_count: int = field(default=0, init=False)

    def reward_function(self) -> RewardFunction:
        async def verifiers_reward(
            *,
            completions: list[Any],
            completion_ids: list[list[int]],
            task_index: list[int],
            trainer_state: Any,
            **_: Any,
        ) -> list[float]:
            if len(completions) != len(task_index) or len(completions) != len(completion_ids):
                raise ValueError("completion and task metadata counts differ")
            return list(
                await asyncio.gather(
                    *(
                        self._score(
                            self.tasks[int(index)],
                            int(index),
                            _completion_text(completion),
                            len(token_ids),
                            int(trainer_state.global_step),
                        )
                        for completion, token_ids, index in zip(
                            completions,
                            completion_ids,
                            task_index,
                            strict=True,
                        )
                    )
                )
            )

        return cast(RewardFunction, verifiers_reward)

    async def _score(
        self,
        task: Any,
        task_index: int,
        completion: str,
        completion_tokens: int,
        step: int,
    ) -> float:
        Trace, TraceTask, TrainRunInfo, extras = _imports()
        MessageNode, make_runtime = extras
        trace = Trace(
            task=TraceTask(type=type(task).__name__, data=task.data),
            run=TrainRunInfo(id=self.context.attempt.id, step=step),
            nodes=[
                MessageNode(message={"role": "user", "content": str(task.data.prompt)}, sampled=False),
                MessageNode(
                    parent=0,
                    message={"role": "assistant", "content": completion},
                    sampled=True,
                ),
            ],
            info={
                "environment_id": self.environment_id,
                "task_index": task_index,
                "model_profile_id": self.model_profile_id,
                "training_profile_id": self.training_profile_id,
            },
            is_completed=True,
            stop_condition="agent_completed",
        )
        environment = self.environment_factory()
        runtime = make_runtime(environment.runtime_for(task))
        await runtime.start()
        try:
            await task.score(trace, runtime)
        finally:
            await runtime.stop()
        for enrich in self.enrichers:
            enrich(trace, completion, completion_tokens)
        trace.record_metric("completion_tokens", float(completion_tokens))
        record = trace.to_record()
        self._preserve(record)
        self.context.trace(
            TraceObservation(
                trace_type="verifiers",
                external_id=str(trace.id),
                payload=record,
                attributes={
                    "technique": "grpo",
                    "environment_id": self.environment_id,
                    "task_index": task_index,
                    "model_profile_id": self.model_profile_id,
                    "training_profile_id": self.training_profile_id,
                },
            )
        )
        return float(trace.reward)

    def _preserve(self, record: dict[str, Any]) -> None:
        encoded = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        with self._write_lock:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            with self.trace_path.open("a", encoding="utf-8") as stream:
                stream.write(encoded)
            self._trace_count += 1

    def publish_native_artifact(self) -> ProducedArtifact | None:
        if not self.trace_path.is_file():
            return None
        digest = hashlib.sha256(self.trace_path.read_bytes()).hexdigest()
        artifact = ProducedArtifact(
            name=f"training/{self.model_profile_id}/grpo/verifiers-traces",
            kind="evaluation-traces",
            reference=LocalArtifactRef(self.trace_path.resolve(), digest),
            metadata={
                "technique": "grpo",
                "environment_id": self.environment_id,
                "training_profile_id": self.training_profile_id,
                "trace_count": self._trace_count,
                "schema_version": 2,
            },
        )
        self.context.artifact(artifact)
        return artifact


__all__ = ["TraceEnricher", "VerifiersGRPOBridge", "verifiers_rollout_dataset"]
