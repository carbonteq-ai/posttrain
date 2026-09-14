"""Enumerate stable task descriptors from a pinned Verifiers taskset."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol

from posttrain.common import JsonValue
from posttrain.environment import EnvironmentBinding, TaskDescriptor, TaskFacetValue
from posttrain.environment.verifiers_runtime import materialize_verifiers_environment


class _TaskData(Protocol):
    def model_dump(self, *, mode: str, exclude_none: bool = False) -> dict[str, Any]: ...


class _Task(Protocol):
    @property
    def key(self) -> str: ...

    @property
    def hash(self) -> str: ...

    @property
    def data(self) -> _TaskData: ...


def _values(value: object, transform: str) -> tuple[str, ...]:
    raw = value if isinstance(value, (list, tuple, set, frozenset)) else (value,)
    values: list[str] = []
    for item in raw:
        if item is None:
            continue
        text = str(item).strip()
        if not text:
            continue
        if transform == "prefix_before_colon":
            text = text.partition(":")[0].strip()
        if text:
            values.append(text)
    return tuple(sorted(set(values)))


def _inventory_from_taskset(
    binding: EnvironmentBinding,
    taskset: Iterable[_Task],
) -> tuple[TaskDescriptor, ...]:
    descriptors: list[TaskDescriptor] = []
    declared_split = binding.parameters.get("split")
    for task in taskset:
        data = task.data.model_dump(mode="json", exclude_none=True)
        facets = tuple(
            TaskFacetValue(spec.dimension, value)
            for spec in binding.observation.facets
            for value in _values(data.get(spec.field), spec.transform)
        )
        raw_split = data.get("split", declared_split)
        split = str(raw_split) if isinstance(raw_split, (str, int, float)) else None
        source_reference: dict[str, JsonValue] = {"task_key": task.key}
        index = data.get("idx")
        if isinstance(index, int) and not isinstance(index, bool) and index >= 0:
            source_reference["index"] = index
        descriptors.append(
            TaskDescriptor(
                key=task.key,
                fingerprint="sha256:" + task.hash.removeprefix("sha256:"),
                split=split,
                facets=facets,
                source_reference=source_reference,
            )
        )
    return tuple(descriptors)


def inventory_verifiers_tasks(binding: EnvironmentBinding) -> tuple[TaskDescriptor, ...]:
    """Load a finite native taskset without starting model inference."""

    try:
        from verifiers.v1.utils.loaders import load_taskset  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise RuntimeError("install posttrain-eval with the verifiers extra") from error
    activated = binding.activate()
    taskset_config = getattr(activated, "taskset", None)
    if taskset_config is not None:
        taskset = load_taskset(taskset_config)
    else:
        environment = materialize_verifiers_environment(activated)
        taskset = environment.taskset
    if bool(getattr(taskset, "INFINITE", False)):
        raise ValueError("evaluation inventory requires a finite Verifiers taskset")
    return _inventory_from_taskset(binding, taskset)


__all__ = ["inventory_verifiers_tasks"]
