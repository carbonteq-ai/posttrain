from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

from posttrain.environment import (
    EnvironmentBinding,
    EnvironmentSource,
    EvaluationFacetField,
    EvaluationObservation,
    PythonFactoryActivation,
    SamplingPolicy,
)
from posttrain.eval.backends.verifiers.inventory import _inventory_from_taskset, _Task


class Data:
    def __init__(self, **values: object) -> None:
        self.values = values

    def model_dump(self, *, mode: str, exclude_none: bool = False) -> dict[str, object]:
        assert mode == "json"
        return {key: value for key, value in self.values.items() if not exclude_none or value is not None}


@dataclass
class Task:
    key: str
    hash: str
    data: Data


def binding() -> EnvironmentBinding:
    return EnvironmentBinding(
        id="math",
        category="reasoning",
        source=EnvironmentSource("math-env", "https://example.com/math", "a" * 40),
        activation=PythonFactoryActivation("builtins:object"),
        sampling=SamplingPolicy(max_tokens=64),
        num_tasks=10,
        parameters={"split": "test"},
        observation=EvaluationObservation(
            facets=(
                EvaluationFacetField("subject", "subject", "Subject"),
                EvaluationFacetField("skill", "skill", "Skill", "prefix_before_colon"),
            )
        ),
    )


def test_native_inventory_retains_identity_facets_and_source_reference() -> None:
    tasks = (
        Task("stable-a", "1" * 64, Data(idx=7, subject="algebra", skill=["solve:linear", "verify"])),
        Task("stable-b", "2" * 64, Data(idx=8, subject="geometry", skill="draw:plane", split="heldout")),
    )

    inventory = _inventory_from_taskset(binding(), cast(Iterable[_Task], tasks))

    assert [item.key for item in inventory] == ["stable-a", "stable-b"]
    assert inventory[0].values("skill") == ("solve", "verify")
    assert inventory[0].split == "test"
    assert inventory[0].source_reference == {"task_key": "stable-a", "index": 7}
    assert inventory[1].split == "heldout"
