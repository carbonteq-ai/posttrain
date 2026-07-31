"""Typed inventory for Lab-owned framework qualification gates.

The inventory intentionally describes project work packages instead of
duplicating their execution implementation.  It is therefore safe to use in
release checks and CI before a provider is configured.
"""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass
from importlib.resources import as_file
from importlib.resources import files as resource_files
from pathlib import Path
from typing import Literal, cast

from posttrain.catalog import ProjectLayout
from posttrain.common import ContractError
from posttrain.work import Recipe, load_work_package

GateTier = Literal["release", "extended", "experimental"]
GateState = Literal["active", "retired"]

_TIERS = frozenset(("release", "extended", "experimental"))
_STATES = frozenset(("active", "retired"))
_FIELDS = frozenset(("id", "work_package", "job_id", "tier", "state", "job_kind", "acceptance"))


class QualificationGateError(ValueError):
    """The declarative Lab gate registry does not match a project."""


@dataclass(frozen=True, slots=True)
class QualificationGate:
    """One named acceptance gate over a single work-package job."""

    id: str
    work_package: str
    job_id: str
    tier: GateTier
    state: GateState
    job_kind: str
    acceptance: str

    def as_json(self) -> dict[str, str]:
        return cast(dict[str, str], asdict(self))


@dataclass(frozen=True, slots=True)
class QualificationInventory:
    """Validated classification of all work packages in one Lab project."""

    gates: tuple[QualificationGate, ...]
    classified: tuple[str, ...]
    retired: tuple[str, ...]
    excluded: tuple[str, ...]
    unclassified: tuple[str, ...]

    def as_json(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "gates": [gate.as_json() for gate in self.gates],
            "classified_work_packages": list(self.classified),
            "retired_work_packages": list(self.retired),
            "excluded_work_packages": list(self.excluded),
            "unclassified_work_packages": list(self.unclassified),
        }


def load_qualification_gates(path: Path | None = None) -> tuple[QualificationGate, ...]:
    """Load the package-owned gate manifest or an explicit test fixture."""

    if path is not None:
        return _load_gate_manifest(path)
    resource = resource_files("posttrain_lab.qualification").joinpath("gates.toml")
    with as_file(resource) as resource_path:
        return _load_gate_manifest(resource_path)


def validate_qualification_project(
    layout: ProjectLayout,
    gates: tuple[QualificationGate, ...],
) -> QualificationInventory:
    """Require each project work-package YAML to have one valid gate entry."""

    errors: list[str] = []
    paths_by_gate: dict[str, Path] = {}
    seen_gate_ids: set[str] = set()
    seen_paths: dict[Path, str] = {}
    root = layout.root.resolve()
    work_packages = layout.work_packages.resolve()

    for gate in gates:
        _validate_gate(gate, errors)
        if gate.id in seen_gate_ids:
            errors.append(f"duplicate gate id {gate.id!r}")
        seen_gate_ids.add(gate.id)

        configured = Path(gate.work_package)
        if configured.is_absolute() or ".." in configured.parts:
            errors.append(f"gate {gate.id!r} work_package must be a project-relative path")
            continue
        path = (root / configured).resolve()
        if not path.is_relative_to(root) or not path.is_relative_to(work_packages):
            errors.append(f"gate {gate.id!r} work_package is outside the project work-packages directory")
            continue
        if path in seen_paths:
            errors.append(f"duplicate work_package {gate.work_package!r} in gates {seen_paths[path]!r} and {gate.id!r}")
        seen_paths[path] = gate.id
        paths_by_gate[gate.id] = path

    for gate in gates:
        path = paths_by_gate.get(gate.id)
        if path is None:
            continue
        if not path.is_file():
            errors.append(f"gate {gate.id!r} references missing work package {gate.work_package!r}")
            continue
        try:
            package = load_work_package(path)
        except ContractError as error:
            errors.append(f"gate {gate.id!r} cannot load {gate.work_package!r}: {error}")
            continue
        if not isinstance(package.recipe, Recipe):
            errors.append(f"gate {gate.id!r} must reference an inline-recipe work package")
            continue
        jobs = tuple(job for job in package.recipe.jobs if job.id == gate.job_id)
        if not jobs:
            errors.append(f"gate {gate.id!r} references unknown job {gate.job_id!r} in {gate.work_package!r}")
            continue
        if len(jobs) != 1:
            errors.append(f"gate {gate.id!r} job {gate.job_id!r} is ambiguous in {gate.work_package!r}")
            continue
        if jobs[0].kind != gate.job_kind:
            errors.append(f"gate {gate.id!r} expects job kind {gate.job_kind!r}, found {jobs[0].kind!r}")

    all_work_packages = {
        path.resolve(): path.resolve().relative_to(root).as_posix()
        for path in work_packages.rglob("*.yaml")
        if path.is_file()
    }
    classified_paths = set(seen_paths)
    unclassified = tuple(sorted(path for path in all_work_packages.values() if root / path not in classified_paths))
    if unclassified:
        errors.append(f"unclassified work packages: {', '.join(unclassified)}")

    if errors:
        details = "\n".join(f"- {error}" for error in sorted(set(errors)))
        raise QualificationGateError(f"invalid Lab qualification gate registry:\n{details}")

    classified = tuple(sorted(all_work_packages[path] for path in classified_paths))
    retired = tuple(sorted(gate.work_package for gate in gates if gate.state == "retired"))
    return QualificationInventory(
        gates=gates,
        classified=classified,
        retired=retired,
        excluded=(),
        unclassified=(),
    )


def _load_gate_manifest(path: Path) -> tuple[QualificationGate, ...]:
    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise QualificationGateError(f"qualification gate manifest not found: {path}") from error
    except tomllib.TOMLDecodeError as error:
        raise QualificationGateError(f"invalid qualification gate manifest {path}: {error}") from error
    if document.get("schema_version") != 1:
        raise QualificationGateError("qualification gate manifest schema_version must be 1")
    entries = document.get("gates")
    if not isinstance(entries, list):
        raise QualificationGateError("qualification gate manifest must define [[gates]] entries")

    gates: list[QualificationGate] = []
    errors: list[str] = []
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            errors.append(f"gate entry {index} must be a table")
            continue
        unknown = set(entry) - _FIELDS
        missing = _FIELDS - set(entry)
        if unknown:
            errors.append(f"gate entry {index} has unknown fields: {', '.join(sorted(unknown))}")
        if missing:
            errors.append(f"gate entry {index} is missing fields: {', '.join(sorted(missing))}")
        if unknown or missing or not all(isinstance(entry[field], str) for field in _FIELDS if field in entry):
            if not unknown and not missing:
                errors.append(f"gate entry {index} fields must all be strings")
            continue
        values = cast(dict[str, str], entry)
        gates.append(
            QualificationGate(
                id=values["id"],
                work_package=values["work_package"],
                job_id=values["job_id"],
                tier=cast(GateTier, values["tier"]),
                state=cast(GateState, values["state"]),
                job_kind=values["job_kind"],
                acceptance=values["acceptance"],
            )
        )
    if errors:
        details = "\n".join(f"- {error}" for error in errors)
        raise QualificationGateError(f"invalid qualification gate manifest {path}:\n{details}")
    return tuple(gates)


def _validate_gate(gate: QualificationGate, errors: list[str]) -> None:
    for field, value in (
        ("id", gate.id),
        ("work_package", gate.work_package),
        ("job_id", gate.job_id),
        ("job_kind", gate.job_kind),
        ("acceptance", gate.acceptance),
    ):
        if not value.strip():
            errors.append(f"gate {gate.id!r} {field} cannot be empty")
    if gate.tier not in _TIERS:
        errors.append(f"gate {gate.id!r} has unknown tier {gate.tier!r}")
    if gate.state not in _STATES:
        errors.append(f"gate {gate.id!r} has unknown state {gate.state!r}")
    if gate.state == "retired" and gate.tier == "release":
        errors.append(f"gate {gate.id!r} cannot be both retired and release tier")


__all__ = [
    "GateState",
    "GateTier",
    "QualificationGate",
    "QualificationGateError",
    "QualificationInventory",
    "load_qualification_gates",
    "validate_qualification_project",
]
