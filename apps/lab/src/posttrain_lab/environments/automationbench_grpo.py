"""AutomationBench environment selection policy for qualification scenarios."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

VERIFIERS_REVISION = "284a868d6a9022109b749710672a0460e8a996d4"
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


def automationbench_training_environment() -> Any:
    """Catalog factory for the default Zapier AutomationBench training environment."""

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
                "rollout": 1800,
                "finalize": 60,
                "scoring": 120,
            },
            "max_turns": 50,
            "max_total_tokens": 8192,
        }
    )
    return Environment(config)


__all__ = [
    "AutomationBenchTrainingParameters",
    "VERIFIERS_REVISION",
    "automationbench_training_environment",
]
