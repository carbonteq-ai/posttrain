"""World snapshots along an attempt, from the environment's raw-action envelopes.

Every tool-server invocation names the world before and after it by digest; the bytes are
published once per rollout and resolved by `capture.SnapshotStore`. Worlds are rebuilt with
`world_codec.load_world`, which restores the private Sheets update record the graders read.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from .calls import raw_actions
from .episode import Episode


class Timeline:
    def __init__(self, episode: Episode):
        from automationbench_v1.capture import trace_snapshot_store

        self.store = trace_snapshot_store(episode.events)
        actions = raw_actions(episode)
        self.steps = sorted(
            ((a["occurrence_index"], inv, a["before_digest"], a["after_digest"], a.get("tool_name"))
             for inv, a in actions.items()),
            key=lambda s: s[0],
        )
        self._json: dict[str, dict | None] = {}
        self._worlds: dict[str, Any] = {}

    @property
    def available(self) -> bool:
        return bool(self.steps)

    def text(self, digest: str) -> str | None:
        return self.store.text(digest)

    def json(self, digest: str) -> dict | None:
        if digest not in self._json:
            text = self.store.text(digest)
            self._json[digest] = None if text is None else json.loads(text)
        return self._json[digest]

    def world(self, digest: str):
        if digest not in self._worlds:
            data = self.json(digest)
            self._worlds[digest] = None if data is None else load_world_safe(data)
        return self._worlds[digest]

    @property
    def initial_digest(self) -> str | None:
        return self.steps[0][2] if self.steps else None

    @property
    def final_digest(self) -> str | None:
        return self.steps[-1][3] if self.steps else None

    def mutating_steps(self) -> list[tuple[str, str, str, str]]:
        """(invocation id, before, after, tool) for steps that changed the world."""
        return [(inv, b, a, tool) for _, inv, b, a, tool in self.steps if b != a]


def load_world_safe(data: dict):
    from automationbench_v1.world_codec import load_world

    try:
        return load_world(data)
    except Exception:  # noqa: BLE001 - an unloadable snapshot is "unavailable", never guessed
        return None


_QUIET = False


def _quiet_registry() -> None:
    """Probing deliberately calls handlers with partial parameters; keep their warnings out of logs."""
    global _QUIET
    if _QUIET:
        return
    import atexit
    import logging

    import automationbench.rubric as rubric  # noqa: F401  (registers handlers)
    from automationbench.rubric.registry import AssertionRegistry

    logging.getLogger("automationbench.rubric.registry").setLevel(logging.CRITICAL)
    atexit.unregister(rubric.print_assertion_error_summary)
    AssertionRegistry.reset_error_counts()
    _QUIET = True


def check(world, assertion: dict) -> bool | None:
    """The grader's verdict on a world, or None when the handler raised."""
    _quiet_registry()
    from automationbench.rubric.registry import AssertionRegistry

    if world is None:
        return None
    try:
        return bool(AssertionRegistry.check(world, assertion))
    except Exception:  # noqa: BLE001
        return None
