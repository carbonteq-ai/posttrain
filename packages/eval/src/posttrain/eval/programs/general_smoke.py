"""Small, reusable coverage across four general behavior categories."""

from __future__ import annotations

from posttrain.common import JsonValue

from ..requests import (
    EnvironmentBinding,
    EnvironmentSource,
    EvaluationPlan,
    SamplingPolicy,
    VerifiersV1ConfigActivation,
)

VERIFIERS_REVISION = "1f6793f7d46e8a650a54b2a585193b4010578fa6"
VERIFIERS_REPOSITORY = "https://github.com/carbonteq-ai/verifiers"
ENVIRONMENTS_REVISION = "b14dfe0ba9d60184f36d78786a543242fabfb765"
ENVIRONMENTS_REPOSITORY = "https://github.com/carbonteq-ai/verifiers-environments"


def _activation(config: dict[str, JsonValue]) -> VerifiersV1ConfigActivation:
    return VerifiersV1ConfigActivation(config)


GSM8K_ACTIVATION = _activation(
    {
        "taskset": {
            "id": "gsm8k-v1",
            "dataset_repo": "openai/gsm8k",
            "dataset_revision": "740312add88f781978c0658806c59bc2815b9866",
            "dataset_config": "main",
            "split": "test",
        },
        "agent": {
            "harness": {"id": "null"},
            "runtime": {"type": "subprocess"},
            "timeout": {"setup": 120, "rollout": 180, "finalize": 60, "scoring": 120},
        },
    }
)

GSM8K_TRAIN_ACTIVATION = _activation(
    {
        "taskset": {
            "id": "gsm8k-v1",
            "dataset_repo": "openai/gsm8k",
            "dataset_revision": "740312add88f781978c0658806c59bc2815b9866",
            "dataset_config": "main",
            "split": "train",
        },
        "agent": {
            "harness": {"id": "null"},
            "runtime": {"type": "subprocess"},
            "timeout": {"setup": 120, "rollout": 180, "finalize": 60, "scoring": 120},
        },
    }
)

REVERSE_TEXT_ACTIVATION = _activation(
    {
        "taskset": {
            "id": "reverse-text",
            "dataset_name": "PrimeIntellect/Reverse-Text-RL",
            "dataset_split": "train",
        },
        "agent": {
            "harness": {"id": "null"},
            "runtime": {"type": "subprocess"},
            "timeout": {"setup": 120, "rollout": 180, "finalize": 60, "scoring": 120},
        },
    }
)

CODE_GOLF_ACTIVATION = _activation(
    {
        "taskset": {"id": "code-golf"},
        "agent": {
            "harness": {"id": "null"},
            "runtime": {"type": "docker", "image": "lab/verifiers-runtime:vf-284a868d"},
            "timeout": {"setup": 120, "rollout": 180, "finalize": 60, "scoring": 120},
        },
    }
)

ALPHABET_SORT_ACTIVATION = _activation(
    {
        "taskset": {
            "id": "alphabet-sort",
            "min_turns": 2,
            "max_turns": 3,
            "min_names_per_turn": 2,
            "max_names_per_turn": 4,
        },
        "agent": {
            "harness": {"id": "null"},
            "runtime": {"type": "subprocess"},
            "timeout": {"setup": 120, "rollout": 180, "finalize": 60, "scoring": 120},
        },
    }
)


def _source(package: str, subdirectory: str) -> EnvironmentSource:
    return EnvironmentSource(
        package=package,
        repository=VERIFIERS_REPOSITORY,
        revision=VERIFIERS_REVISION,
        subdirectory=subdirectory,
    )


def _carbonteq_source(package: str, subdirectory: str) -> EnvironmentSource:
    return EnvironmentSource(
        package=package,
        repository=ENVIRONMENTS_REPOSITORY,
        revision=ENVIRONMENTS_REVISION,
        subdirectory=subdirectory,
    )


GENERAL_SMOKE = EvaluationPlan(
    id="general-smoke-v1",
    kind="general",
    environments=(
        EnvironmentBinding(
            id="math-gsm8k",
            category="math-reasoning",
            source=_carbonteq_source("gsm8k-v1", "environments/gsm8k_v1"),
            activation=GSM8K_ACTIVATION,
            sampling=SamplingPolicy(max_tokens=4_096),
            num_tasks=8,
        ),
        EnvironmentBinding(
            id="instruction-reverse-text",
            category="instruction-following",
            source=_source("reverse-text", "environments/reverse_text"),
            activation=REVERSE_TEXT_ACTIVATION,
            sampling=SamplingPolicy(max_tokens=1_024),
            num_tasks=8,
        ),
        EnvironmentBinding(
            id="code-execution",
            category="code-generation",
            source=_source("code-golf", "environments/code_golf"),
            activation=CODE_GOLF_ACTIVATION,
            sampling=SamplingPolicy(max_tokens=4_096),
            num_tasks=3,
            num_rollouts=2,
            max_concurrent=1,
        ),
        EnvironmentBinding(
            id="multi-turn-alphabet-sort",
            category="multi-turn-state",
            source=_source("alphabet-sort", "environments/alphabet_sort"),
            activation=ALPHABET_SORT_ACTIVATION,
            sampling=SamplingPolicy(max_tokens=2_048),
            num_tasks=4,
        ),
    ),
)

# Compatibility aliases for one catalog migration window. Values are inert,
# serializable activations; loading the catalog does not import Verifiers.
GENERAL_ENVIRONMENT_ACTIVATIONS = {
    "math-gsm8k": GSM8K_ACTIVATION,
    "math-gsm8k-train": GSM8K_TRAIN_ACTIVATION,
    "instruction-reverse-text": REVERSE_TEXT_ACTIVATION,
    "code-execution": CODE_GOLF_ACTIVATION,
    "multi-turn-alphabet-sort": ALPHABET_SORT_ACTIVATION,
}
GENERAL_ENVIRONMENT_FACTORIES = GENERAL_ENVIRONMENT_ACTIVATIONS

__all__ = [
    "ALPHABET_SORT_ACTIVATION",
    "CODE_GOLF_ACTIVATION",
    "GENERAL_ENVIRONMENT_ACTIVATIONS",
    "GENERAL_ENVIRONMENT_FACTORIES",
    "GENERAL_SMOKE",
    "GSM8K_ACTIVATION",
    "GSM8K_TRAIN_ACTIVATION",
    "ENVIRONMENTS_REPOSITORY",
    "ENVIRONMENTS_REVISION",
    "REVERSE_TEXT_ACTIVATION",
    "VERIFIERS_REPOSITORY",
    "VERIFIERS_REVISION",
]
