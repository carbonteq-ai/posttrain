"""Pinned GSM8K tasks and job-specific native Verifiers reward shaping."""

from __future__ import annotations

import re
from pathlib import Path
from random import Random
from typing import Any

from posttrain.eval import EnvironmentBinding
from posttrain.train import PolicySampling
from posttrain.train.integrations import VerifiersEnvironmentRolloutBridge

VERIFIERS_REVISION = "284a868d6a9022109b749710672a0460e8a996d4"
_FINAL_ANSWER = re.compile(r"(?m)^####\s*[+-]?(?:\d[\d,]*)(?:\.\d+)?\s*$")
_SHAPING_WEIGHT = 0.1


def _imports() -> tuple[type[Any], type[Any]]:
    try:
        from gsm8k_v1.taskset import GSM8KConfig, GSM8KTaskset  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise RuntimeError("install posttrain-lab with the gpu-posttrain extra") from error
    return GSM8KConfig, GSM8KTaskset


def create_gsm8k_training_bridge(
    task_indices: tuple[int, ...],
    trace_path: Path,
    run_id: str,
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    purpose: str = "grpo",
) -> VerifiersEnvironmentRolloutBridge:
    if not task_indices or len(task_indices) != len(set(task_indices)) or min(task_indices) < 0:
        raise ValueError("training task indices must be non-empty, unique, and non-negative")
    GSM8KConfig, GSM8KTaskset = _imports()
    available = GSM8KTaskset(GSM8KConfig(split="train")).load()
    try:
        tasks = {index: available[index] for index in task_indices}
    except IndexError as error:
        raise ValueError("training task index is outside the GSM8K training split") from error
    suffix = "-".join(str(index) for index in task_indices)
    return VerifiersEnvironmentRolloutBridge(
        dataset_id=f"gsm8k/{purpose}/train-{suffix}-v1",
        revision=VERIFIERS_REVISION,
        tasks=tasks,
        environment_factory=_training_environment,
        trace_path=trace_path,
        environment_id="gsm8k-v1",
        run_id=run_id,
        sampling=PolicySampling(max_tokens=max_tokens, temperature=temperature, top_p=top_p),
        enrichers=(_add_gsm8k_shaping,),
    )


def create_gsm8k_training_bridge_from_environment(
    environment: EnvironmentBinding,
    trace_path: Path,
    run_id: str,
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    purpose: str = "grpo",
) -> VerifiersEnvironmentRolloutBridge:
    """Resolve a bounded GSM8K population from an environment-only GRPO seat."""

    if environment.source.package != "gsm8k-v1":
        raise ValueError("GSM8K bridge requires a gsm8k-v1 environment")
    seed = environment.parameters.get("sampling_seed", 0)
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("GSM8K environment sampling_seed must be a non-negative integer")
    GSM8KConfig, GSM8KTaskset = _imports()
    available = GSM8KTaskset(GSM8KConfig(split="train")).load()
    if environment.num_tasks > len(available):
        raise ValueError(
            f"GSM8K environment requests {environment.num_tasks} tasks, but train exposes {len(available)}"
        )
    indices = tuple(sorted(Random(seed).sample(range(len(available)), environment.num_tasks)))
    return create_gsm8k_training_bridge(
        indices,
        trace_path,
        run_id,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        purpose=purpose,
    )


def _add_gsm8k_shaping(trace: Any) -> None:
    trace.record_reward(
        "final_answer_conciseness",
        _final_answer_conciseness(trace.last_reply, trace.num_output_tokens),
        weight=_SHAPING_WEIGHT,
    )


def _final_answer_conciseness(completion: str, completion_tokens: int) -> float:
    if completion_tokens < 1 or _FINAL_ANSWER.search(completion) is None:
        return 0.0
    return 1.0 / (1.0 + completion_tokens / 256.0)


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


__all__ = [
    "VERIFIERS_REVISION",
    "create_gsm8k_training_bridge",
    "create_gsm8k_training_bridge_from_environment",
]
