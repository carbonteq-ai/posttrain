"""Reusable agentic and full-domain programs over the native v1 port."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..requests import EnvironmentProgram, EnvironmentSource, EvaluationProgram, SamplingPolicy

AUTOMATIONBENCH_REVISION = "a321764ace3cfbe42289e6a13abef2f0f4f56fad"
AUTOMATIONBENCH_REPOSITORY = "https://github.com/zapier/AutomationBench"


def _environment(domains: list[str]) -> object:
    try:
        from verifiers.v1.env import EnvConfig  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise RuntimeError("install posttrain-eval with the verifiers extra") from error
    config: dict[str, Any] = {
        "taskset": {
            "id": "automationbench-v1",
            "domains": domains,
            "task": {"search_top_k": 20},
        },
        "harness": {"id": "null", "runtime": {"type": "subprocess"}},
        "timeout": {"setup": 600, "rollout": 900, "finalize": 120, "scoring": 300},
        "max_turns": 50,
    }
    return EnvConfig.model_validate(config)


def _factory(*domains: str) -> Callable[[], object]:
    def create() -> object:
        return _environment(list(domains))

    return create


SOURCE = EnvironmentSource(
    package="automationbench-v1",
    repository=AUTOMATIONBENCH_REPOSITORY,
    revision=AUTOMATIONBENCH_REVISION,
)

AGENTIC_SMOKE = EvaluationProgram(
    id="agentic-smoke-v1",
    kind="general",
    environments=(
        EnvironmentProgram(
            id="automationbench-simple",
            category="agentic-tool-use",
            source=SOURCE,
            factory=_factory("simple"),
            sampling=SamplingPolicy(max_tokens=2_048),
            num_tasks=2,
            max_concurrent=1,
        ),
    ),
)


def _domain_cell(domain: str) -> EnvironmentProgram:
    return EnvironmentProgram(
        id=f"automationbench-{domain}",
        category=f"business-{domain}",
        source=SOURCE,
        factory=_factory(domain),
        sampling=SamplingPolicy(max_tokens=4_096),
        num_tasks=100,
        max_concurrent=4,
    )


AUTOMATIONBENCH_PUBLIC = EvaluationProgram(
    id="automationbench-public-v1",
    kind="domain",
    environments=tuple(
        _domain_cell(domain)
        for domain in ("sales", "marketing", "operations", "support", "finance", "hr")
    ),
)

__all__ = [
    "AGENTIC_SMOKE",
    "AUTOMATIONBENCH_PUBLIC",
    "AUTOMATIONBENCH_REPOSITORY",
    "AUTOMATIONBENCH_REVISION",
]
