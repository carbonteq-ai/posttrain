"""Job-owned Verifiers environment composition."""

from pathlib import Path
from typing import Literal

from posttrain.eval import EnvironmentBinding
from posttrain.train.integrations import VerifiersEnvironmentRolloutBridge

from .automationbench_grpo import (
    AUTOMATIONBENCH_REVISION,
    AutomationBenchTrainingParameters,
    automationbench_training_environment,
    create_automationbench_training_bridge,
)
from .gsm8k_grpo import (
    VERIFIERS_REVISION,
    create_gsm8k_training_bridge,
    create_gsm8k_training_bridge_from_environment,
)


def create_training_bridge(
    environment: EnvironmentBinding,
    trace_path: Path,
    run_id: str,
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    purpose: Literal["grpo", "distill"] = "grpo",
) -> VerifiersEnvironmentRolloutBridge:
    """Dispatch one resolved Verifiers environment to its private trainer bridge."""

    package = environment.source.package
    if package == "automationbench-v1":
        if purpose != "grpo":
            raise ValueError("AutomationBench bridge is currently registered only for GRPO")
        return create_automationbench_training_bridge(
            environment,
            trace_path,
            run_id,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
    if package == "gsm8k-v1":
        return create_gsm8k_training_bridge_from_environment(
            environment,
            trace_path,
            run_id,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            purpose=purpose,
        )
    raise ValueError(f"no GRPO bridge is registered for environment package {package!r}")


__all__ = [
    "AUTOMATIONBENCH_REVISION",
    "AutomationBenchTrainingParameters",
    "VERIFIERS_REVISION",
    "automationbench_training_environment",
    "create_automationbench_training_bridge",
    "create_gsm8k_training_bridge",
    "create_gsm8k_training_bridge_from_environment",
    "create_training_bridge",
]
