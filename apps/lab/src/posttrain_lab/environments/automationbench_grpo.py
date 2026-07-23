"""AutomationBench environment selection and native Verifiers bridge for GRPO."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from random import Random
from typing import Any, Literal

from posttrain.eval import EnvironmentBinding
from posttrain.train import PolicySampling
from posttrain.train.integrations import VerifiersEnvironmentRolloutBridge
from pydantic import BaseModel, ConfigDict, Field

VERIFIERS_REVISION = "284a868d6a9022109b749710672a0460e8a996d4"
AUTOMATIONBENCH_REVISION = "d54dbebabdba6c6eda201694aee8ddcf36ccfc51"
type AutomationBenchDomain = Literal["simple", "sales", "marketing", "operations", "support", "finance", "hr"]


class AutomationBenchTrainingParameters(BaseModel):
    """Strict environment-owned controls; concrete task rows are resolved later."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    domains: tuple[AutomationBenchDomain, ...] = Field(default=("simple",), min_length=1)
    sampling_seed: int = Field(default=0, ge=0)
    search_top_k: int = Field(default=20, gt=0)
    max_turns: int = Field(default=50, gt=0)
    max_total_tokens: int = Field(default=8192, gt=0)
    rollout_timeout_seconds: float = Field(default=1800, gt=0)
    toolset: Literal["zapier", "limited_zapier", "api"] = "zapier"


def create_automationbench_training_bridge(
    environment: EnvironmentBinding,
    trace_path: Path,
    run_id: str,
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> VerifiersEnvironmentRolloutBridge:
    if environment.source.package != "automationbench-v1":
        raise ValueError("AutomationBench bridge requires an automationbench-v1 environment")
    parameters = AutomationBenchTrainingParameters.model_validate(dict(environment.parameters))
    try:
        from automationbench_v1.taskset import (  # pyright: ignore[reportMissingImports]
            AutomationBenchConfig,
            AutomationBenchTaskConfig,
            AutomationBenchTaskset,
        )
    except ImportError as error:
        raise RuntimeError(
            "AutomationBench GRPO requires the pinned automationbench-v1 package; "
            "install posttrain-lab with the gpu-posttrain extra"
        ) from error
    available = AutomationBenchTaskset(
        AutomationBenchConfig(
            domains=list(parameters.domains),
            task=AutomationBenchTaskConfig(
                toolset=parameters.toolset,
                search_top_k=parameters.search_top_k,
            ),
        )
    ).load()
    if environment.num_tasks > len(available):
        raise ValueError(
            f"AutomationBench environment requests {environment.num_tasks} tasks, "
            f"but the selected domains expose {len(available)}"
        )
    selected_indices = sorted(Random(parameters.sampling_seed).sample(range(len(available)), environment.num_tasks))
    tasks = {index: available[index] for index in selected_indices}
    domains = "-".join(parameters.domains)
    return VerifiersEnvironmentRolloutBridge(
        dataset_id=(f"automationbench/{domains}/seed-{parameters.sampling_seed}-limit-{environment.num_tasks}-v1"),
        revision=AUTOMATIONBENCH_REVISION,
        tasks=tasks,
        environment_factory=partial(
            _training_environment,
            max_turns=parameters.max_turns,
            max_total_tokens=parameters.max_total_tokens,
            rollout_timeout_seconds=parameters.rollout_timeout_seconds,
        ),
        trace_path=trace_path,
        environment_id=environment.id,
        run_id=run_id,
        sampling=PolicySampling(max_tokens=max_tokens, temperature=temperature, top_p=top_p),
    )


def _training_environment(
    *, max_turns: int = 50, max_total_tokens: int = 8192, rollout_timeout_seconds: float = 1800
) -> Any:
    try:
        from verifiers.v1.env import EnvConfig, Environment
    except ImportError as error:
        raise RuntimeError("install the AutomationBench v1 environment package") from error
    config = EnvConfig.model_validate(
        {
            "taskset": {"id": "automationbench-v1"},
            "harness": {"id": "null", "runtime": {"type": "subprocess"}},
            "timeout": {
                "setup": 120,
                "rollout": rollout_timeout_seconds,
                "finalize": 60,
                "scoring": 120,
            },
            "max_turns": max_turns,
            "max_total_tokens": max_total_tokens,
        }
    )
    return Environment(config)


def automationbench_training_environment() -> Any:
    """Catalog factory for the default Zapier AutomationBench training environment."""

    return _training_environment()


__all__ = [
    "AUTOMATIONBENCH_REVISION",
    "AutomationBenchTrainingParameters",
    "VERIFIERS_REVISION",
    "automationbench_training_environment",
    "create_automationbench_training_bridge",
]
