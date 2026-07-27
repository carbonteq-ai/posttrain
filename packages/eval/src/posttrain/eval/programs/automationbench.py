"""Reusable agentic and full-domain programs over the native v1 port."""

from __future__ import annotations

from posttrain.common import JsonValue

from ..requests import (
    EnvironmentBinding,
    EnvironmentSource,
    EvaluationPlan,
    SamplingPolicy,
    VerifiersV1ConfigActivation,
)

AUTOMATIONBENCH_REVISION = "d54dbebabdba6c6eda201694aee8ddcf36ccfc51"
AUTOMATIONBENCH_REPOSITORY = "https://github.com/carbonteq-ai/AutomationBench"


def _activation(*domains: str) -> VerifiersV1ConfigActivation:
    config: dict[str, JsonValue] = {
        "taskset": {
            "id": "automationbench-v1",
            "domains": list(domains),
            "task": {"search_top_k": 20},
        },
        "harness": {"id": "null", "runtime": {"type": "subprocess"}},
        "timeout": {"setup": 600, "rollout": 900, "finalize": 120, "scoring": 300},
        "max_turns": 50,
    }
    return VerifiersV1ConfigActivation(config)


SOURCE = EnvironmentSource(
    package="automationbench-v1",
    repository=AUTOMATIONBENCH_REPOSITORY,
    revision=AUTOMATIONBENCH_REVISION,
)

AGENTIC_SMOKE = EvaluationPlan(
    id="agentic-smoke-v1",
    kind="general",
    environments=(
        EnvironmentBinding(
            id="automationbench-simple",
            category="agentic-tool-use",
            source=SOURCE,
            activation=_activation("simple"),
            sampling=SamplingPolicy(max_tokens=2_048),
            num_tasks=2,
            max_concurrent=1,
        ),
    ),
)


def _domain_cell(domain: str) -> EnvironmentBinding:
    return EnvironmentBinding(
        id=f"automationbench-{domain}",
        category=f"business-{domain}",
        source=SOURCE,
        activation=_activation(domain),
        sampling=SamplingPolicy(max_tokens=4_096),
        num_tasks=100,
        max_concurrent=4,
    )


AUTOMATIONBENCH_PUBLIC = EvaluationPlan(
    id="automationbench-public-v1",
    kind="domain",
    environments=tuple(
        _domain_cell(domain) for domain in ("sales", "marketing", "operations", "support", "finance", "hr")
    ),
)

__all__ = [
    "AGENTIC_SMOKE",
    "AUTOMATIONBENCH_PUBLIC",
    "AUTOMATIONBENCH_REPOSITORY",
    "AUTOMATIONBENCH_REVISION",
]
