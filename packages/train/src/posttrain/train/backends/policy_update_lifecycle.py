"""Shared collection boundaries for explicitly resolved native policy runs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ..update_plan import ResolvedUpdate
from ..update_records import InvalidPolicyUpdate


class ResolvedNativePopulation(Protocol):
    updates: tuple[ResolvedUpdate, ...]
    next_update: int
    applied_updates: int
    attempts: int
    applied_update_offset: int
    attempt_offset: int
    old: Any
    _pending: int | None

    @property
    def global_applied_updates(self) -> int: ...

    @property
    def global_attempts(self) -> int: ...


@dataclass
class ResolvedPolicyRun[Population: ResolvedNativePopulation]:
    """Collect only after all occurrences of the current population have applied.

    Hosts supply collection and retained-input restoration. Native dataloaders
    may prefetch global slot IDs, but collection runs at loss evaluation, after
    previous optimizer work. Restored candidates still require the backend's
    sealed identity/file verification before native model or optimizer loading.
    """

    collect: Callable[[int, int], Population]
    restore: Callable[[Path], Population]
    current: Population | None = None

    def active(self) -> Population:
        if self.current is None:
            raise InvalidPolicyUpdate("native run has no collected or restored population")
        return self.current

    def occurrence(self, slot: int, native_applied_updates: int) -> tuple[Population, int]:
        if type(slot) is not int or slot != native_applied_updates:
            raise InvalidPolicyUpdate("native run slot differs from its applied boundary")
        previous = self.current
        if previous is None or previous.next_update == len(previous.updates):
            if previous is not None and (
                previous._pending is not None or previous.applied_updates != len(previous.updates)
            ):
                raise InvalidPolicyUpdate("native collection cannot advance an incomplete population")
            attempts = previous.global_attempts if previous is not None else 0
            candidate = self.collect(native_applied_updates, attempts)
            if (
                candidate.applied_update_offset != native_applied_updates
                or candidate.attempt_offset != attempts
                or candidate.next_update != 0
                or candidate.applied_updates != 0
                or candidate.attempts != 0
                or candidate.old is not None
                or candidate._pending is not None
            ):
                raise InvalidPolicyUpdate(
                    "new native population must start at the exact run boundary without reused scores"
                )
            if previous is not None and candidate.updates[0].population.digest == previous.updates[0].population.digest:
                raise InvalidPolicyUpdate("native collection cannot regenerate a completed frozen population identity")
            self.current = candidate
        active = self.active()
        if active.global_applied_updates != native_applied_updates or active._pending is not None:
            raise InvalidPolicyUpdate("native run active population differs from its applied boundary")
        return active, active.next_update

    def retain_checkpoint_population(self, checkpoint: Path) -> None:
        if self.current is not None:
            raise InvalidPolicyUpdate("native run resume cannot replace an already collected population")
        self.current = self.restore(checkpoint)
