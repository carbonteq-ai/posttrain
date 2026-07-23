"""Deterministic train, validation, and reserve partitions for SFT data."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from posttrain.common import JsonValue

from .models import SupervisedDataset, SupervisedExample

type PartitionName = Literal["train", "validation", "reserve"]

_ID = re.compile(r"^[a-z0-9][a-z0-9._/-]*$")
_HASH_DENOMINATOR = 1 << 64


@dataclass(frozen=True, slots=True)
class SupervisedPartitionPlan:
    """A reproducible allocation policy over one supervised snapshot."""

    id: str
    revision: str
    validation_fraction: float
    reserve_fraction: float = 0.0
    seed: int = 42
    group_by: str | None = None
    stratify_by: str | None = None

    def __post_init__(self) -> None:
        if not _ID.fullmatch(self.id) or not self.revision.strip():
            raise ValueError("partition plans require a stable lowercase id and revision")
        fractions = (self.validation_fraction, self.reserve_fraction)
        if any(not 0 <= value < 1 for value in fractions):
            raise ValueError("partition fractions must be greater than or equal to zero and less than one")
        if sum(fractions) >= 1:
            raise ValueError("validation and reserve fractions must leave a non-empty training allocation")
        if not any(fractions):
            raise ValueError("partition plans must request validation or reserve records")
        for field_name, value in (("group_by", self.group_by), ("stratify_by", self.stratify_by)):
            if value is not None and not value.strip():
                raise ValueError(f"{field_name} cannot be blank")


@dataclass(frozen=True, slots=True)
class SupervisedPartitionManifest:
    """Explicit example assignments derived from one source snapshot."""

    source_id: str
    source_revision: str
    plan_id: str
    plan_revision: str
    seed: int
    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    reserve_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        populations = (self.train_ids, self.validation_ids, self.reserve_ids)
        if not self.train_ids:
            raise ValueError("partition manifests require a non-empty training population")
        flattened = tuple(identifier for population in populations for identifier in population)
        if len(flattened) != len(set(flattened)):
            raise ValueError("partition manifest populations must be disjoint")

    @property
    def digest(self) -> str:
        encoded = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def as_dict(self) -> dict[str, JsonValue]:
        """Return a JSON-compatible manifest suitable for durable evidence."""

        return {
            "source_id": self.source_id,
            "source_revision": self.source_revision,
            "plan_id": self.plan_id,
            "plan_revision": self.plan_revision,
            "seed": self.seed,
            "train_ids": list(self.train_ids),
            "validation_ids": list(self.validation_ids),
            "reserve_ids": list(self.reserve_ids),
        }

    def ids(self, partition: PartitionName) -> tuple[str, ...]:
        return {
            "train": self.train_ids,
            "validation": self.validation_ids,
            "reserve": self.reserve_ids,
        }[partition]


@dataclass(frozen=True, slots=True)
class PartitionedSupervisedDataset:
    """Materialized partitions plus the manifest proving their assignments."""

    train: SupervisedDataset
    validation: SupervisedDataset | None
    reserve: SupervisedDataset | None
    manifest: SupervisedPartitionManifest

    def __post_init__(self) -> None:
        actual = {
            "train": tuple(example.id for example in self.train.examples),
            "validation": tuple(example.id for example in self.validation.examples) if self.validation else (),
            "reserve": tuple(example.id for example in self.reserve.examples) if self.reserve else (),
        }
        for name in ("train", "validation", "reserve"):
            if actual[name] != self.manifest.ids(name):
                raise ValueError(f"{name} dataset does not match the partition manifest")


@dataclass(frozen=True, slots=True)
class _Group:
    id: str
    stratum: str
    rank: int
    examples: tuple[SupervisedExample, ...]


def partition_supervised_dataset(
    dataset: SupervisedDataset,
    plan: SupervisedPartitionPlan,
) -> PartitionedSupervisedDataset:
    """Partition a canonical snapshot without relying on provider row order."""

    groups = _groups(dataset, plan)
    assigned: dict[PartitionName, list[_Group]] = {"train": [], "validation": [], "reserve": []}
    reserve_limit = plan.reserve_fraction
    validation_limit = reserve_limit + plan.validation_fraction
    for group in groups:
        score = group.rank / _HASH_DENOMINATOR
        partition: PartitionName
        if score < reserve_limit:
            partition = "reserve"
        elif score < validation_limit:
            partition = "validation"
        else:
            partition = "train"
        assigned[partition].append(group)

    examples: dict[PartitionName, tuple[SupervisedExample, ...]] = {
        name: _ordered_examples(values) for name, values in assigned.items()
    }
    _validate_populations(examples, plan)
    manifest = SupervisedPartitionManifest(
        source_id=dataset.id,
        source_revision=dataset.revision,
        plan_id=plan.id,
        plan_revision=plan.revision,
        seed=plan.seed,
        train_ids=tuple(example.id for example in examples["train"]),
        validation_ids=tuple(example.id for example in examples["validation"]),
        reserve_ids=tuple(example.id for example in examples["reserve"]),
    )
    return PartitionedSupervisedDataset(
        train=_partition(dataset, plan, manifest, "train", examples["train"]),
        validation=(
            _partition(dataset, plan, manifest, "validation", examples["validation"])
            if examples["validation"]
            else None
        ),
        reserve=(_partition(dataset, plan, manifest, "reserve", examples["reserve"]) if examples["reserve"] else None),
        manifest=manifest,
    )


def _groups(dataset: SupervisedDataset, plan: SupervisedPartitionPlan) -> tuple[_Group, ...]:
    examples_by_group: dict[str, list[SupervisedExample]] = defaultdict(list)
    strata_by_group: dict[str, str] = {}
    for example in dataset.examples:
        group_id = example.id if plan.group_by is None else _metadata_value(example, plan.group_by)
        stratum = "all" if plan.stratify_by is None else _metadata_value(example, plan.stratify_by)
        previous = strata_by_group.setdefault(group_id, stratum)
        if previous != stratum:
            raise ValueError(
                f"partition group {group_id!r} spans strata {previous!r} and {stratum!r}; "
                "group and stratification metadata must agree"
            )
        examples_by_group[group_id].append(example)

    result = []
    for group_id, examples in examples_by_group.items():
        stratum = strata_by_group[group_id]
        rank = _rank(dataset, plan, group_id, stratum)
        result.append(_Group(group_id, stratum, rank, tuple(sorted(examples, key=lambda item: item.id))))
    return tuple(sorted(result, key=lambda group: (group.rank, group.id)))


def _metadata_value(example: SupervisedExample, key: str) -> str:
    if key not in example.metadata:
        raise ValueError(f"supervised example {example.id!r} is missing partition metadata {key!r}")
    value = example.metadata[key]
    if not isinstance(value, str | int | float | bool) or isinstance(value, float) and not value.is_integer():
        raise ValueError(f"partition metadata {key!r} on {example.id!r} must be a stable scalar")
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _rank(dataset: SupervisedDataset, plan: SupervisedPartitionPlan, group_id: str, stratum: str) -> int:
    payload = {
        "source_id": dataset.id,
        "source_revision": dataset.revision,
        "plan_id": plan.id,
        "plan_revision": plan.revision,
        "seed": plan.seed,
        "stratum": stratum,
        "group_id": group_id,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def _ordered_examples(groups: list[_Group]) -> tuple[SupervisedExample, ...]:
    return tuple(
        example for group in sorted(groups, key=lambda item: (item.rank, item.id)) for example in group.examples
    )


def _validate_populations(
    examples: Mapping[PartitionName, tuple[SupervisedExample, ...]],
    plan: SupervisedPartitionPlan,
) -> None:
    if not examples["train"]:
        raise ValueError("partition plan produced no training records; lower the reserved fractions")
    requested: dict[Literal["validation", "reserve"], float] = {
        "validation": plan.validation_fraction,
        "reserve": plan.reserve_fraction,
    }
    for name, fraction in requested.items():
        if fraction and not examples[name]:
            raise ValueError(
                f"partition plan requested {name} records but produced none; "
                "increase the source population or the allocation fraction"
            )


def _partition(
    source: SupervisedDataset,
    plan: SupervisedPartitionPlan,
    manifest: SupervisedPartitionManifest,
    name: PartitionName,
    examples: tuple[SupervisedExample, ...],
) -> SupervisedDataset:
    metadata: dict[str, JsonValue] = {
        **dict(source.metadata),
        "partition": name,
        "partition_plan_id": plan.id,
        "partition_plan_revision": plan.revision,
        "partition_manifest_digest": manifest.digest,
        "partition_seed": plan.seed,
        "source_dataset_id": source.id,
        "source_dataset_revision": source.revision,
    }
    if plan.group_by is not None:
        metadata["partition_group_by"] = plan.group_by
    if plan.stratify_by is not None:
        metadata["partition_stratify_by"] = plan.stratify_by
    return SupervisedDataset(
        id=f"{source.id}/{name}",
        revision=manifest.digest,
        examples=examples,
        metadata=metadata,
        schema_version=source.schema_version,
    )


__all__ = [
    "PartitionedSupervisedDataset",
    "SupervisedPartitionManifest",
    "SupervisedPartitionPlan",
    "partition_supervised_dataset",
]
