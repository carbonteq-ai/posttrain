"""Provider-neutral task inventory values used by evaluation selection."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from posttrain.common import JsonValue

_DIMENSION = re.compile(r"^[a-z0-9][a-z0-9._/-]*$")
_SHA256 = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True, order=True)
class TaskFacetValue:
    """One semantic dimension/value attached to a task.

    A task may have several values for one dimension. These values are useful
    for reporting and can form a canonical membership-set allocation stratum.
    """

    dimension: str
    value: str

    def __post_init__(self) -> None:
        if not _DIMENSION.fullmatch(self.dimension):
            raise ValueError(f"task facet dimension must be a lowercase stable identifier, got {self.dimension!r}")
        if not self.value.strip():
            raise ValueError("task facet value cannot be empty")


@dataclass(frozen=True, slots=True)
class TaskDescriptor:
    """One stable task in an environment-owned finite population."""

    key: str
    fingerprint: str
    split: str | None = None
    facets: tuple[TaskFacetValue, ...] = ()
    source_reference: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.key.strip():
            raise ValueError("task key cannot be empty")
        if not _SHA256.fullmatch(self.fingerprint):
            raise ValueError("task fingerprint must be a sha256 digest")
        if self.split is not None and not self.split.strip():
            raise ValueError("task split cannot be empty")
        facets = tuple(sorted(self.facets))
        if len(facets) != len(set(facets)):
            raise ValueError("task facet values must be unique")
        reference = dict(self.source_reference)
        try:
            json.dumps(reference, sort_keys=True)
        except (TypeError, ValueError) as error:
            raise ValueError("task source_reference must contain only JSON values") from error
        object.__setattr__(self, "facets", facets)
        object.__setattr__(self, "source_reference", MappingProxyType(reference))

    @property
    def normalized_fingerprint(self) -> str:
        return self.fingerprint.removeprefix("sha256:")

    def values(self, dimension: str) -> tuple[str, ...]:
        return tuple(item.value for item in self.facets if item.dimension == dimension)

    def to_payload(self) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {
            "key": self.key,
            "fingerprint": f"sha256:{self.normalized_fingerprint}",
            "facets": [
                {"dimension": item.dimension, "value": item.value}
                for item in self.facets
            ],
            "source_reference": dict(self.source_reference),
        }
        if self.split is not None:
            payload["split"] = self.split
        return payload


def task_inventory_digest(tasks: tuple[TaskDescriptor, ...]) -> str:
    """Return an order-independent digest for a finite task inventory."""

    by_key: dict[str, TaskDescriptor] = {}
    fingerprints: defaultdict[str, set[str]] = defaultdict(set)
    for task in tasks:
        fingerprints[task.key].add(task.normalized_fingerprint)
        by_key.setdefault(task.key, task)
    collisions = sorted(key for key, values in fingerprints.items() if len(values) > 1)
    if collisions:
        raise ValueError("task keys resolve to different content: " + ", ".join(collisions))
    if len(by_key) != len(tasks):
        duplicates = sorted(
            key
            for key, values in fingerprints.items()
            if len(values) == 1 and sum(task.key == key for task in tasks) > 1
        )
        raise ValueError("task inventory contains duplicate task keys: " + ", ".join(duplicates))
    encoded = json.dumps(
        [by_key[key].to_payload() for key in sorted(by_key)],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


__all__ = ["TaskDescriptor", "TaskFacetValue", "task_inventory_digest"]
