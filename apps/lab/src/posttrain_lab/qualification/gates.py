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
GateState = Literal["active", "candidate", "retired"]

_TIERS = frozenset(("release", "extended", "experimental"))
_STATES = frozenset(("active", "candidate", "retired"))
_REQUIRED_FIELDS = frozenset(("id", "work_package", "job_id", "tier", "state", "job_kind", "acceptance"))
_CANDIDATE_FIELDS = frozenset(("experiment_family", "hypothesis", "owner", "replacement_condition"))
_FIELDS = _REQUIRED_FIELDS | _CANDIDATE_FIELDS


class QualificationGateError(ValueError):
    """The declarative Lab gate registry does not match a project."""


@dataclass(frozen=True, slots=True)
class QualificationGate:
    """One retained work-package record in the Lab qualification inventory.

    ``active`` entries are maintained release or extended gates.  A
    ``candidate`` entry is a retained experiment: it remains inspectable and
    validates as a work package, but is deliberately excluded from the active
    gate set until its documented replacement condition is met.
    """

    id: str
    work_package: str
    job_id: str
    tier: GateTier
    state: GateState
    job_kind: str
    acceptance: str
    experiment_family: str | None = None
    hypothesis: str | None = None
    owner: str | None = None
    replacement_condition: str | None = None

    def as_json(self) -> dict[str, object]:
        return cast(dict[str, object], asdict(self))


@dataclass(frozen=True, slots=True)
class QualificationInventory:
    """Validated classification of all work packages in one Lab project."""

    entries: tuple[QualificationGate, ...]
    active_gates: tuple[QualificationGate, ...]
    candidate_experiments: tuple[QualificationGate, ...]
    retired_gates: tuple[QualificationGate, ...]
    classified: tuple[str, ...]
    retired: tuple[str, ...]
    excluded: tuple[str, ...]
    unclassified: tuple[str, ...]

    def as_json(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "active_gates": [gate.as_json() for gate in self.active_gates],
            "candidate_experiments": [gate.as_json() for gate in self.candidate_experiments],
            "retired_gates": [gate.as_json() for gate in self.retired_gates],
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
    """Require each project work-package YAML to have one retained record.

    This validates both active qualification gates and retained experiments,
    while intentionally exposing them as separate inventory sets to callers.
    """

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
    active_gates = tuple(gate for gate in gates if gate.state == "active")
    candidate_experiments = tuple(gate for gate in gates if gate.state == "candidate")
    retired_gates = tuple(gate for gate in gates if gate.state == "retired")
    retired = tuple(sorted(gate.work_package for gate in retired_gates))
    return QualificationInventory(
        entries=gates,
        active_gates=active_gates,
        candidate_experiments=candidate_experiments,
        retired_gates=retired_gates,
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
        missing = _REQUIRED_FIELDS - set(entry)
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
                experiment_family=values.get("experiment_family"),
                hypothesis=values.get("hypothesis"),
                owner=values.get("owner"),
                replacement_condition=values.get("replacement_condition"),
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
    candidate_details = {
        "experiment_family": gate.experiment_family,
        "hypothesis": gate.hypothesis,
        "owner": gate.owner,
        "replacement_condition": gate.replacement_condition,
    }
    if gate.state == "candidate":
        if gate.tier != "experimental":
            errors.append(f"candidate {gate.id!r} must use the experimental tier")
        for field, value in candidate_details.items():
            if value is None or not value.strip():
                errors.append(f"candidate {gate.id!r} {field} cannot be empty")
        if gate.replacement_condition is not None and not any(
            outcome in gate.replacement_condition.casefold() for outcome in ("promote", "replace", "retire", "delete")
        ):
            errors.append(
                f"candidate {gate.id!r} replacement_condition must name a promotion, replacement, retirement, or deletion outcome"
            )
    elif any(value is not None for value in candidate_details.values()):
        errors.append(f"non-candidate gate {gate.id!r} cannot declare candidate retention fields")
    if gate.state == "active" and gate.tier == "experimental":
        errors.append(f"active gate {gate.id!r} cannot use the experimental tier; retain it as a candidate")


__all__ = [
    "GateState",
    "GateTier",
    "QualificationGate",
    "QualificationGateError",
    "QualificationInventory",
    "load_qualification_gates",
    "validate_qualification_project",
]
