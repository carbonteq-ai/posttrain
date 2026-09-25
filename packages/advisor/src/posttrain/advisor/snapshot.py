"""Read a run's recorded selections (its resolved-input snapshot).

Every run records the same JSON snapshot of its resolved selections: at plan
time it is the job's ``resolved_inputs``, and trackers keep it with the run.
The rules and the calculator read only this snapshot, so the CLI, validation
and Observatory reach the same findings, for new runs and for runs recorded
before the rules existed.

Seats are recognized by the fields their snapshots carry, not by role names,
because roles are recipe-defined (``rollout_inference``, ``evaluation_inference``,
``judge_inference``...).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

type Snapshot = Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class Seat:
    role: str
    selection_id: str
    revision: str | None
    resolved: Mapping[str, Any]

    @property
    def label(self) -> str:
        return f"{self.selection_id}@{self.revision}" if self.revision and "@" not in self.selection_id else self.selection_id


def seats(snapshot: Snapshot) -> Iterator[Seat]:
    for role, entry in sorted(snapshot.items()):
        if not isinstance(entry, Mapping):
            continue
        resolved = entry.get("resolved")
        if not isinstance(resolved, Mapping):
            continue
        selection = entry.get("selection_id")
        revision = entry.get("revision")
        yield Seat(
            role,
            selection if isinstance(selection, str) else role,
            revision if isinstance(revision, str) else None,
            resolved,
        )


def inference_seats(snapshot: Snapshot) -> Iterator[Seat]:
    """Locally served inference bindings (hosted services have no engine)."""

    return (seat for seat in seats(snapshot) if isinstance(seat.resolved.get("engine"), Mapping))


def settings_seat(snapshot: Snapshot) -> Seat | None:
    return next(
        (seat for seat in seats(snapshot) if "learning_rate" in seat.resolved and "max_steps" in seat.resolved), None
    )


def training_seat(snapshot: Snapshot) -> Seat | None:
    return next((seat for seat in seats(snapshot) if isinstance(seat.resolved.get("parameter_update"), Mapping)), None)


def environment_seat(snapshot: Snapshot) -> Seat | None:
    return next(
        (seat for seat in seats(snapshot) if "max_concurrent" in seat.resolved and "activation" in seat.resolved), None
    )


def model_seat(snapshot: Snapshot, variant_id: object) -> Seat | None:
    """The model variant a binding serves, when the run recorded it as a seat."""

    return next(
        (
            seat
            for seat in seats(snapshot)
            if seat.selection_id == variant_id and isinstance(seat.resolved.get("artifact"), Mapping)
        ),
        None,
    )


def served_model(snapshot: Snapshot, seat: Seat) -> Mapping[str, Any]:
    """Facts about the model an inference seat serves: family, precision, base repo, capabilities.

    Runs recorded after model facts joined the inference snapshot carry them
    inline; older runs only have them when the model was also bound as a seat,
    and never record capabilities.
    """

    inline = seat.resolved.get("model")
    if isinstance(inline, Mapping):
        return inline
    model = model_seat(snapshot, seat.resolved.get("model_variant_id"))
    if model is None:
        return {}
    artifact = model.resolved.get("artifact")
    facts: dict[str, Any] = {key: model.resolved[key] for key in ("family", "weight_precision", "parameters", "capabilities") if key in model.resolved}
    if isinstance(artifact, Mapping) and isinstance(artifact.get("repo_id"), str):
        facts["base"] = {"repo_id": artifact["repo_id"], "revision": artifact.get("revision")}
    return facts


def target(snapshot: Snapshot, target_id: object) -> Mapping[str, Any] | None:
    targets = snapshot.get("execution_targets")
    entries = targets.get("targets") if isinstance(targets, Mapping) else None
    if not isinstance(entries, list):
        return None
    return next(
        (entry for entry in entries if isinstance(entry, Mapping) and entry.get("selection_id") == target_id), None
    )


def backend_name(value: object) -> str:
    return value.split("@", 1)[0].strip().lower() if isinstance(value, str) else ""


def mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def items(value: object) -> list[Any]:
    return value if isinstance(value, list) else []


def integer(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def number(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


__all__ = [
    "Seat",
    "Snapshot",
    "backend_name",
    "environment_seat",
    "inference_seats",
    "integer",
    "items",
    "mapping",
    "model_seat",
    "number",
    "seats",
    "served_model",
    "settings_seat",
    "target",
    "training_seat",
]
