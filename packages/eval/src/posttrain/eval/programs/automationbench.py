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

# The selected distribution is the native Verifiers v1 port, not the benchmark
# fork it reads.  The fork publishes `carbonteq-automation-bench` on the legacy
# 0.1 environment API and contains no v1 taskset at any revision, so naming its
# repository here could never resolve `automationbench-v1`.
AUTOMATIONBENCH_REVISION = "4c2f756393b0e44d2587f9e9a5ee1f4704d5d73b"
AUTOMATIONBENCH_REPOSITORY = "https://github.com/carbonteq-ai/posttrain"
AUTOMATIONBENCH_SUBDIRECTORY = "environments/automationbench_v1"


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
    subdirectory=AUTOMATIONBENCH_SUBDIRECTORY,
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
    "AUTOMATIONBENCH_SUBDIRECTORY",
]
