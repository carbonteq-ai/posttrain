"""Use the pinned GSM8K Verifiers task as a native GRPO reward source."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from posttrain.common import ExecutionContext, LocalArtifactRef, ProducedArtifact, TraceObservation
from posttrain.train import RewardFunction, RolloutDataset, RolloutExample

VERIFIERS_REVISION = "284a868d6a9022109b749710672a0460e8a996d4"
_FINAL_ANSWER = re.compile(r"(?m)^####\s*[+-]?(?:\d[\d,]*)(?:\.\d+)?\s*$")
_SHAPING_WEIGHT = 0.1


def _imports() -> tuple[type[Any], type[Any], type[Any], type[Any], Any]:
    try:
        from gsm8k_v1.taskset import GSM8KConfig, GSM8KTaskset  # pyright: ignore[reportMissingImports]
        from verifiers.v1 import MessageNode, Trace, TraceTask, TrainRunInfo  # pyright: ignore[reportMissingImports]
        from verifiers.v1.runtimes import make_runtime  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise RuntimeError("install posttrain-lab with the gpu-posttrain extra") from error
    return GSM8KConfig, GSM8KTaskset, Trace, TraceTask, (MessageNode, TrainRunInfo, make_runtime)


def load_gsm8k_rollout_dataset(task_indices: tuple[int, ...]) -> tuple[RolloutDataset, dict[int, Any]]:
    """Load selected native tasks and expose only task-neutral training inputs."""

    if not task_indices or len(task_indices) != len(set(task_indices)) or min(task_indices) < 0:
        raise ValueError("GRPO task indices must be non-empty, unique, and non-negative")
    GSM8KConfig, GSM8KTaskset, _, _, _ = _imports()
    available = GSM8KTaskset(GSM8KConfig(split="train")).load()
    try:
        tasks = {index: available[index] for index in task_indices}
    except IndexError as error:
        raise ValueError("GRPO task index is outside the GSM8K training split") from error
    examples = tuple(
        RolloutExample(
            id=f"train/{index:06d}",
            prompt=str(tasks[index].data.prompt),
            metadata={"task_index": index, "environment_id": "gsm8k-v1"},
        )
        for index in task_indices
    )
    suffix = "-".join(str(index) for index in task_indices)
    return RolloutDataset(f"gsm8k/grpo/train-{suffix}-v1", VERIFIERS_REVISION, examples), tasks


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
class GSM8KRewardBridge:
    """Score completions with Verifiers and preserve queryable plus native traces."""

    context: ExecutionContext
    tasks: dict[int, Any]
    trace_path: Path
    model_profile_id: str
    training_profile_id: str
    _write_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _trace_count: int = field(default=0, init=False)

    def reward_function(self) -> RewardFunction:
        async def gsm8k_verifiers_reward(
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

        return cast(RewardFunction, gsm8k_verifiers_reward)

    async def _score(
        self,
        task: Any,
        task_index: int,
        completion: str,
        completion_tokens: int,
        step: int,
    ) -> float:
        _, _, Trace, TraceTask, extras = _imports()
        MessageNode, TrainRunInfo, make_runtime = extras
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
                "environment_id": "gsm8k-v1",
                "task_index": task_index,
                "model_profile_id": self.model_profile_id,
                "training_profile_id": self.training_profile_id,
            },
            is_completed=True,
            stop_condition="agent_completed",
        )
        environment = _training_environment()
        runtime = make_runtime(environment.runtime_for(task))
        await runtime.start()
        try:
            await task.score(trace, runtime)
        finally:
            await runtime.stop()
        trace.record_reward(
            "final_answer_conciseness",
            _final_answer_conciseness(completion, completion_tokens),
            weight=_SHAPING_WEIGHT,
        )
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
                    "environment_id": "gsm8k-v1",
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
                "environment_id": "gsm8k-v1",
                "training_profile_id": self.training_profile_id,
                "trace_count": self._trace_count,
                "schema_version": 2,
            },
        )
        self.context.artifact(artifact)
        return artifact


def _training_environment() -> Any:
    try:
        from verifiers.v1.env import EnvConfig, Environment  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise RuntimeError("install posttrain-lab with the gpu-posttrain extra") from error
    config = EnvConfig.model_validate(
        {
            "taskset": {"id": "gsm8k-v1", "split": "train"},
            "harness": {"id": "null", "runtime": {"type": "subprocess"}},
            "timeout": {"setup": 120, "rollout": 180, "finalize": 60, "scoring": 120},
        }
    )
    return Environment(config)


def _final_answer_conciseness(completion: str, completion_tokens: int) -> float:
    """Small shaping signal: require the native answer format, then prefer concision."""

    if completion_tokens < 1 or _FINAL_ANSWER.search(completion) is None:
        return 0.0
    return 1.0 / (1.0 + completion_tokens / 256.0)


__all__ = ["GSM8KRewardBridge", "VERIFIERS_REVISION", "load_gsm8k_rollout_dataset"]
