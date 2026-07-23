"""Environment-only AutomationBench GRPO bridge tests."""

from __future__ import annotations

import sys
from dataclasses import replace
from random import Random
from types import ModuleType
from typing import Any, cast

import pytest
from posttrain_lab.catalog import AUTOMATIONBENCH_ZAPIER_GRPO
from posttrain_lab.environments import automationbench_grpo
from pydantic import ValidationError


def test_bridge_resolves_a_seeded_population_inside_the_environment(monkeypatch, tmp_path) -> None:
    taskset_module = ModuleType("automationbench_v1.taskset")

    class FakeTaskConfig:
        def __init__(self, **values) -> None:
            self.values = values

    class FakeConfig:
        def __init__(self, **values) -> None:
            self.values = values

    class FakeTaskset:
        def __init__(self, config) -> None:
            self.config = config

        def load(self):
            return [f"task-{index}" for index in range(20)]

    taskset_module.__dict__.update(
        {
            "AutomationBenchTaskConfig": FakeTaskConfig,
            "AutomationBenchConfig": FakeConfig,
            "AutomationBenchTaskset": FakeTaskset,
        }
    )
    package = ModuleType("automationbench_v1")
    package.__dict__["taskset"] = taskset_module
    monkeypatch.setitem(sys.modules, "automationbench_v1", package)
    monkeypatch.setitem(sys.modules, "automationbench_v1.taskset", taskset_module)
    monkeypatch.setattr(
        automationbench_grpo,
        "VerifiersEnvironmentRolloutBridge",
        lambda **values: values,
    )

    bridge = automationbench_grpo.create_automationbench_training_bridge(
        AUTOMATIONBENCH_ZAPIER_GRPO,
        tmp_path / "traces.jsonl",
        "run-1",
        max_tokens=256,
        temperature=1.0,
        top_p=0.95,
    )

    bridge_data = cast(dict[str, Any], bridge)
    expected = sorted(Random(17).sample(range(20), 2))
    assert sorted(bridge_data["tasks"]) == expected
    assert bridge_data["dataset_id"] == "automationbench/simple/seed-17-limit-2-v1"
    assert bridge_data["environment_id"] == AUTOMATIONBENCH_ZAPIER_GRPO.id


def test_environment_rejects_exact_task_ids_as_an_unknown_authoring_control(tmp_path) -> None:
    environment = replace(
        AUTOMATIONBENCH_ZAPIER_GRPO,
        parameters={**AUTOMATIONBENCH_ZAPIER_GRPO.parameters, "task_indices": [194, 198]},
    )

    with pytest.raises(ValidationError, match="task_indices"):
        automationbench_grpo.create_automationbench_training_bridge(
            environment,
            tmp_path / "traces.jsonl",
            "run-1",
            max_tokens=256,
            temperature=1.0,
            top_p=0.95,
        )
