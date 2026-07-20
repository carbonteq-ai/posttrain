"""Preserve AutomationBench 1.0.5 final-state assertion semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from automationbench.rubric import partial_credit, task_completed_correctly
from automationbench.schema.world import WorldState


@dataclass(frozen=True, slots=True)
class ScoreSnapshot:
    partial_credit: float
    task_completed_correctly: float
    assertions_passed: int
    assertions_scored: int
    assertions_excluded: int
    assertion_results: tuple[dict[str, Any], ...]
    end_state: dict[str, Any]


def score_world(
    *,
    world: dict[str, Any],
    initial_state: dict[str, Any],
    assertions: tuple[dict[str, Any], ...],
) -> ScoreSnapshot:
    """Call upstream's authoritative scorer without leaking mutable v0 state."""

    state: dict[str, Any] = {
        "info": {"assertions": [dict(item) for item in assertions]},
        "world": WorldState.model_validate(world),
        "initial_state": initial_state,
    }
    dense = float(partial_credit(state))
    strict = float(task_completed_correctly(state))
    raw_results = state.get("_assertion_results", [])
    results = tuple(dict(item) for item in raw_results if isinstance(item, dict))
    scored = tuple(item for item in results if not item.get("excluded"))
    return ScoreSnapshot(
        partial_credit=dense,
        task_completed_correctly=strict,
        assertions_passed=sum(bool(item.get("passed")) for item in scored),
        assertions_scored=len(scored),
        assertions_excluded=len(results) - len(scored),
        assertion_results=results,
        end_state=state["world"].model_dump(mode="json"),
    )


__all__ = ["ScoreSnapshot", "score_world"]
