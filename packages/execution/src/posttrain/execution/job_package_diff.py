"""Explain why two job packages have different identities.

A job package key is a hash over a structured payload, so an unexpected repack
is always attributable to specific fields. Without that attribution the only
observable fact is that two opaque digests differ, which reliably produces the
wrong conclusion: people assume the change was larger than it was, or assume it
was cosmetic when a dependency actually moved.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

type ChangeKind = Literal["added", "removed", "changed"]

_SHA256 = re.compile(r"^[0-9a-f]{64}$")

# What each field means in terms a reader can act on.
_EXPLANATIONS: Mapping[str, str] = {
    "code_requirements_digest": "the project's own dependency requirements changed",
    "datasets": "a materialized dataset changed",
    "environment_activations": "environment activation changed",
    "environment_packages": "a selected environment package changed",
    "expected_artifact_roles": "the job's expected artifacts changed",
    "framework_source_digest": "the framework source packed into the image changed",
    "kind_image": "the job-kind image changed, so the framework release moved",
    "project_config_digest": "project configuration changed",
    "project_source_digest": "project source changed, including comments and formatting",
    "resolved_config_digest": "resolved job configuration changed, such as hyperparameters",
    "resolved_inputs_digest": "resolved catalog selections changed, such as model or settings",
    "runtime_dependencies_digest": "the resolved runtime dependency closure changed",
    "universal_image": "the universal base image changed",
    "worker_contract_version": "the worker contract version changed",
}

# Keys that identify an entry inside a list field, so list changes can be
# reported per entry instead of as one opaque "the list differs".
_ENTRY_IDENTITY: Mapping[str, str] = {
    "datasets": "dataset_id",
    "environment_packages": "name",
    "environment_activations": "name",
    "runtime_dependency_locks": "role",
}


@dataclass(frozen=True, slots=True)
class FieldChange:
    """One field that differs between two job packages."""

    field: str
    kind: ChangeKind
    previous: str
    current: str
    explanation: str

    def describe(self) -> str:
        if self.kind == "added":
            return f"{self.field}: added {self.current} ({self.explanation})"
        if self.kind == "removed":
            return f"{self.field}: removed {self.previous} ({self.explanation})"
        return f"{self.field}: {self.previous} -> {self.current} ({self.explanation})"


def _abbreviate(field: str, value: object) -> str:
    if isinstance(value, str):
        # Any full SHA-256, at any nesting depth. Twelve characters is enough to
        # tell two digests apart by eye and short enough to scan a column of.
        if _SHA256.fullmatch(value):
            return value[:12]
        if "@sha256:" in value:
            repository, digest = value.rsplit("@sha256:", 1)
            return f"{repository.rsplit('/', 1)[-1]}@{digest[:12]}"
        return value
    if value is None:
        return "none"
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _summarize(field: str, value: object) -> str:
    """Describe a composite value without printing the whole thing.

    A reader who is shown two long JSON blobs has to diff them by eye, which is
    the work this module exists to remove.
    """
    if isinstance(value, list):
        if not value:
            return "nothing"
        labels = [_entry_label(field, entry) for entry in value]
        return f"{len(labels)} entr{'y' if len(labels) == 1 else 'ies'}: " + ", ".join(
            sorted(labels)
        )
    if isinstance(value, dict):
        return "{" + ", ".join(sorted(str(key) for key in value)) + "}"
    return _abbreviate(field, value)


def _entry_label(field: str, entry: object) -> str:
    identity = _ENTRY_IDENTITY.get(field)
    if isinstance(entry, dict) and identity is not None and identity in entry:
        return str(entry[identity])
    return _abbreviate(field, entry)


def _compare_sequence(
    field: str,
    previous: Sequence[object],
    current: Sequence[object],
    explanation: str,
) -> list[FieldChange]:
    before = {_entry_label(field, entry): entry for entry in previous}
    after = {_entry_label(field, entry): entry for entry in current}
    changes: list[FieldChange] = []
    for label in sorted(set(before) | set(after)):
        old, new = before.get(label), after.get(label)
        if old == new:
            continue
        if label not in before:
            changes.append(FieldChange(field, "added", "", label, explanation))
        elif label not in after:
            changes.append(FieldChange(field, "removed", label, "", explanation))
        else:
            changes.append(
                FieldChange(
                    f"{field}[{label}]",
                    "changed",
                    _abbreviate(field, old),
                    _abbreviate(field, new),
                    explanation,
                )
            )
    return changes


def _compare_mapping(
    previous: Mapping[str, object],
    current: Mapping[str, object],
    *,
    prefix: str = "",
) -> list[FieldChange]:
    changes: list[FieldChange] = []
    for key in sorted(set(previous) | set(current)):
        if not prefix and key == "schema":
            continue
        field = f"{prefix}.{key}" if prefix else key
        old = previous.get(key)
        new = current.get(key)
        if old == new:
            continue
        explanation = _EXPLANATIONS.get(prefix or key, "this input changed")

        if key in previous and key in current:
            # Recurse into composites so the report names the sub-field that
            # actually moved rather than showing two blobs to compare by eye.
            if isinstance(old, dict) and isinstance(new, dict):
                changes.extend(_compare_mapping(old, new, prefix=field))
                continue
            if isinstance(old, list) and isinstance(new, list):
                changes.extend(_compare_sequence(field, old, new, explanation))
                continue

        if key not in previous:
            changes.append(FieldChange(field, "added", "", _summarize(field, new), explanation))
        elif key not in current:
            changes.append(FieldChange(field, "removed", _summarize(field, old), "", explanation))
        else:
            changes.append(
                FieldChange(
                    field,
                    "changed",
                    _abbreviate(field, old),
                    _abbreviate(field, new),
                    explanation,
                )
            )
    return changes


def compare_job_packages(
    previous: Mapping[str, object],
    current: Mapping[str, object],
) -> tuple[FieldChange, ...]:
    """Return every field that differs between two job package payloads.

    An empty result means the two payloads are identical, which implies the
    same package key; identity here is total, so there is no such thing as an
    unexplained difference.
    """
    return tuple(_compare_mapping(previous, current))


def unchanged_fields(
    previous: Mapping[str, object],
    current: Mapping[str, object],
) -> tuple[str, ...]:
    """Return the fields that are identical, which is the reassuring half."""
    return tuple(
        field
        for field in sorted(set(previous) & set(current))
        if field != "schema" and previous[field] == current[field]
    )


__all__ = [
    "ChangeKind",
    "FieldChange",
    "compare_job_packages",
    "unchanged_fields",
]
