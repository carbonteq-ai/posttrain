"""AutomationBench environment selection policy for qualification scenarios."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

VERIFIERS_REVISION = "c6c0097ad21da845c62e4b19aba80ef6633e4d9f"
type AutomationBenchDomain = Literal["simple", "sales", "marketing", "operations", "support", "finance", "hr"]


class AutomationBenchTrainingParameters(BaseModel):
    """Strict environment-owned controls; concrete task rows are resolved later."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    domains: tuple[AutomationBenchDomain, ...] = Field(default=("simple",), min_length=1)
    sampling_seed: int = Field(default=0, ge=0)
    search_top_k: int = Field(default=20, gt=0)
    max_turns: int = Field(default=50, gt=0)
    # ``max_total_tokens`` remains accepted for older catalog selections. New
    # agentic jobs should prefer the output-only limit so replayed history does
    # not consume the generation allowance on every turn.
    max_output_tokens: int | None = Field(default=None, gt=0)
    max_total_tokens: int | None = Field(default=None, gt=0)
    rollout_timeout_seconds: float = Field(default=1800, gt=0)
    toolset: Literal["zapier", "limited_zapier", "api"] = "zapier"


def automationbench_training_environment() -> Any:
    """Catalog factory for the default Zapier AutomationBench training environment."""

    from posttrain.environment.verifiers_runtime import (
        materialize_verifiers_environment,
        verifiers_environment_types,
    )

    EnvConfig, _ = verifiers_environment_types()
    config = EnvConfig.model_validate(
        {
            "taskset": {"id": "automationbench-v1"},
            "agent": {
                "harness": {"id": "null"},
                "runtime": {"type": "subprocess"},
                "timeout": {
                    "setup": 120,
                    "rollout": 1800,
                    "finalize": 60,
                    "scoring": 120,
                },
                "max_turns": 50,
                "max_output_tokens": 8192,
            },
        }
    )
    return materialize_verifiers_environment(config)


__all__ = [
    "AutomationBenchTrainingParameters",
    "VERIFIERS_REVISION",
    "automationbench_training_environment",
]
