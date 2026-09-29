"""Every Verifiers environment in the lab catalog must validate against the pinned Verifiers.

Image qualification builds each activation config with
`verifiers.v1.utils.loaders.resolve_env_config`, the native Verifiers entry that
narrows it to the environment's config class (`SingleAgentEnvConfig` for a plain
taskset, the multi-agent config for a paired env) and validates it. A config that
the framework's Verifiers rejects then fails only when someone packs a job, as
the duplicated top-level `harness`/`timeout` keys did. This test runs the same
validation for every environment selection without a GPU or an image.

A taskset package that is not installed in the test environment (the concrete
environments are job dependencies, not workspace dependencies) is validated with
the base `TasksetConfig` fields plus its own taskset, task and judge fields
accepted as given: the environment, seat, harness and timeout structure is still
checked exactly.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
from posttrain.catalog import open_catalog
from posttrain.common import CatalogRef
from posttrain.environment import VerifiersV1ConfigActivation
from posttrain.eval import EnvironmentBinding

WORKSPACE = Path(__file__).resolve().parents[3]

loaders = pytest.importorskip("verifiers.v1.utils.loaders", reason="requires the verifiers extra")


def _installed(taskset_id: str) -> bool:
    return importlib.util.find_spec(taskset_id.replace("-", "_")) is not None


@pytest.fixture
def pinned_verifiers_validation(monkeypatch: pytest.MonkeyPatch) -> Any:
    from pydantic import ConfigDict
    from verifiers.v1.configs.judge import JudgeConfig
    from verifiers.v1.configs.taskset import TasksetConfig
    from verifiers.v1.envs.single_agent import SingleAgentEnvConfig

    class _UninstalledTasksetConfig(TasksetConfig):
        # Only the taskset package knows its own and its tasks' fields; everything
        # around it is checked.
        model_config = ConfigDict(extra="allow")
        task: dict[str, Any] = {}  # pyright: ignore[reportIncompatibleVariableOverride]

    class _UninstalledJudgeConfig(JudgeConfig):
        # A judge plugin shipped by an uninstalled environment package.
        model_config = ConfigDict(extra="allow")

    real_taskset_config_type = loaders.taskset_config_type
    real_env_config_type = loaders.env_config_type
    real_judge_config_type = loaders.judge_config_type

    def taskset_config_type(taskset_id: str) -> type[Any]:
        return real_taskset_config_type(taskset_id) if _installed(taskset_id) else _UninstalledTasksetConfig

    def judge_config_type(judge_id: str) -> type[Any]:
        return real_judge_config_type(judge_id) if _installed(judge_id) else _UninstalledJudgeConfig

    def env_config_type(taskset_id: str, env_id: str = "") -> type[Any]:
        if taskset_id and not _installed(taskset_id) and not env_id:
            return SingleAgentEnvConfig
        return real_env_config_type(taskset_id, env_id)

    monkeypatch.setattr(loaders, "taskset_config_type", taskset_config_type)
    monkeypatch.setattr(loaders, "env_config_type", env_config_type)
    monkeypatch.setattr(loaders, "judge_config_type", judge_config_type)

    def validate(config: dict[str, Any]) -> object:
        # Native validators rewrite nested plugin dictionaries; give them a detached copy.
        return loaders.resolve_env_config(json.loads(json.dumps(config)))

    return validate


def _verifiers_environments() -> list[tuple[str, dict[str, Any]]]:
    # The framework base catalog plus every lab overlay.
    catalog = open_catalog(scope="posttrain-lab", overlays=(WORKSPACE / "apps/lab/.posttrain/catalog",))
    environments: list[tuple[str, dict[str, Any]]] = []
    for ref in catalog.list("environment"):
        value = catalog.resolve(CatalogRef("environment", ref.id)).value
        if isinstance(value, EnvironmentBinding) and isinstance(value.activation, VerifiersV1ConfigActivation):
            environments.append((ref.id, dict(value.activation.config)))
    return environments


def test_every_verifiers_environment_config_validates_against_the_pinned_verifiers(
    pinned_verifiers_validation: Any,
) -> None:
    environments = _verifiers_environments()
    assert len(environments) > 50

    failures: dict[str, str] = {}
    for environment_id, config in environments:
        try:
            pinned_verifiers_validation(config)
        except Exception as error:  # noqa: BLE001 - report every invalid environment at once
            failures[environment_id] = " | ".join(line.strip() for line in str(error).splitlines())
    assert failures == {}, "\n".join(f"{key}: {value}" for key, value in sorted(failures.items()))


def test_pinned_verifiers_rejects_an_environment_level_harness_or_stage_timeout(
    pinned_verifiers_validation: Any,
) -> None:
    config = {
        "agent": {
            "harness": {"id": "null"},
            "runtime": {"type": "subprocess"},
            "timeout": {"setup": 120, "rollout": 180, "finalize": 60, "scoring": 120},
        },
        "taskset": {"id": "gsm8k-v1", "split": "train"},
    }
    pinned_verifiers_validation(config)

    with pytest.raises(ValueError, match="a harness belongs to a seat"):
        pinned_verifiers_validation({**config, "harness": {"id": "null", "runtime": {"type": "subprocess"}}})
    with pytest.raises(ValueError, match="rollout"):
        pinned_verifiers_validation({**config, "timeout": {"setup": 120, "rollout": 180}})
