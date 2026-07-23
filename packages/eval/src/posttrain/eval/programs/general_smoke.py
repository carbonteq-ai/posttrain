"""Small, reusable coverage across four general behavior categories."""

from __future__ import annotations

from typing import Any

from ..requests import (
    EnvironmentBinding,
    EnvironmentSource,
    EvaluationPlan,
    SamplingPolicy,
)

VERIFIERS_REVISION = "284a868d6a9022109b749710672a0460e8a996d4"
VERIFIERS_REPOSITORY = "https://github.com/PrimeIntellect-ai/verifiers"


def _environment(config: dict[str, Any]) -> object:
    try:
        from verifiers.v1.env import EnvConfig  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise RuntimeError("install posttrain-eval with the verifiers extra") from error
    return EnvConfig.model_validate(config)


def _gsm8k() -> object:
    return _environment(
        {
            "taskset": {"id": "gsm8k-v1", "split": "test"},
            "harness": {"id": "null", "runtime": {"type": "subprocess"}},
            "timeout": {"setup": 120, "rollout": 180, "finalize": 60, "scoring": 120},
        }
    )


def _reverse_text() -> object:
    return _environment(
        {
            "taskset": {
                "id": "reverse-text-v1",
                "dataset_name": "PrimeIntellect/Reverse-Text-RL",
                "dataset_split": "train",
            },
            "harness": {"id": "null", "runtime": {"type": "subprocess"}},
            "timeout": {"setup": 120, "rollout": 180, "finalize": 60, "scoring": 120},
        }
    )


def _code_golf() -> object:
    return _environment(
        {
            "taskset": {"id": "code-golf-v1"},
            "harness": {
                "id": "null",
                "runtime": {"type": "docker", "image": "lab/verifiers-runtime:vf-284a868d"},
            },
            "timeout": {"setup": 120, "rollout": 180, "finalize": 60, "scoring": 120},
        }
    )


def _alphabet_sort() -> object:
    return _environment(
        {
            "taskset": {
                "id": "alphabet-sort-v1",
                "min_turns": 2,
                "max_turns": 3,
                "min_names_per_turn": 2,
                "max_names_per_turn": 4,
            },
            "harness": {"id": "null", "runtime": {"type": "subprocess"}},
            "timeout": {"setup": 120, "rollout": 180, "finalize": 60, "scoring": 120},
        }
    )


def _source(package: str, subdirectory: str) -> EnvironmentSource:
    return EnvironmentSource(
        package=package,
        repository=VERIFIERS_REPOSITORY,
        revision=VERIFIERS_REVISION,
        subdirectory=subdirectory,
    )


GENERAL_SMOKE = EvaluationPlan(
    id="general-smoke-v1",
    kind="general",
    environments=(
        EnvironmentBinding(
            id="math-gsm8k",
            category="math-reasoning",
            source=_source("gsm8k-v1", "environments/gsm8k_v1"),
            factory=_gsm8k,
            sampling=SamplingPolicy(max_tokens=4_096),
            num_tasks=8,
        ),
        EnvironmentBinding(
            id="instruction-reverse-text",
            category="instruction-following",
            source=_source("reverse-text-v1", "environments/reverse_text_v1"),
            factory=_reverse_text,
            sampling=SamplingPolicy(max_tokens=1_024),
            num_tasks=8,
        ),
        EnvironmentBinding(
            id="code-execution",
            category="code-generation",
            source=_source("code-golf-v1", "environments/code_golf_v1"),
            factory=_code_golf,
            sampling=SamplingPolicy(max_tokens=4_096),
            num_tasks=3,
            num_rollouts=2,
            max_concurrent=1,
        ),
        EnvironmentBinding(
            id="multi-turn-alphabet-sort",
            category="multi-turn-state",
            source=_source("alphabet-sort-v1", "environments/alphabet_sort_v1"),
            factory=_alphabet_sort,
            sampling=SamplingPolicy(max_tokens=2_048),
            num_tasks=4,
        ),
    ),
)

GENERAL_ENVIRONMENT_FACTORIES = {
    "math-gsm8k": _gsm8k,
    "instruction-reverse-text": _reverse_text,
    "code-execution": _code_golf,
    "multi-turn-alphabet-sort": _alphabet_sort,
}

__all__ = [
    "GENERAL_ENVIRONMENT_FACTORIES",
    "GENERAL_SMOKE",
    "VERIFIERS_REPOSITORY",
    "VERIFIERS_REVISION",
]
