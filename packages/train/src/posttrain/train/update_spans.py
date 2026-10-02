"""Derived reasoning/answer spans over original native sampled actions.

Extraction is environment-owned (``native_reasoning_partition``); this module
only addresses its result in the population's action coordinates, so a span is
a derived reference to the native trace rather than a second trajectory store.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from posttrain.environment.verifiers_conditioning import (
    REASONING_PREFIX_REVISION,
    InvalidNativeConditioning,
    native_reasoning_partition,
)

from .online_rl import EnvironmentRollout
from .update_records import ActionInterval, ActionRef, InvalidPolicyUpdate, SemanticSpan


def _intervals(episode: str, branch: str, view: str, indices: tuple[int, ...]) -> tuple[ActionInterval, ...]:
    runs: list[ActionInterval] = []
    start = previous = None
    for index in indices:
        if start is None:
            start = previous = index
        elif index == previous + 1:  # type: ignore[operator]
            previous = index
        else:
            runs.append(ActionInterval(ActionRef(episode, branch, view, start), previous + 1))  # type: ignore[operator]
            start = previous = index
    if start is not None:
        runs.append(ActionInterval(ActionRef(episode, branch, view, start), previous + 1))  # type: ignore[operator]
    return tuple(runs)


def reasoning_answer_spans(
    rollouts: tuple[EnvironmentRollout, ...], traces: Mapping[str, Any],
) -> tuple[SemanticSpan, ...]:
    """One ``reasoning`` and one ``answer`` span per sampled assistant call, when nonempty."""
    spans: list[SemanticSpan] = []
    for rollout in rollouts:
        info = rollout.trace.payload.get("info")
        episode = info.get("posttrain_episode_id") if isinstance(info, Mapping) else None
        if not isinstance(episode, str) or rollout.selected_branch_id is None:
            raise InvalidPolicyUpdate("semantic extraction requires retained episode and branch identities")
        for record in rollout.conditioning_records:
            trace = traces.get(record.trace_id)
            if trace is None:
                raise InvalidPolicyUpdate("semantic extraction requires the retained native trace")
            try:
                reasoning, answer = native_reasoning_partition(trace, record)
            except InvalidNativeConditioning as error:
                raise InvalidPolicyUpdate(f"semantic extraction rejected native evidence: {error}") from error
            view = f"{record.trace_id}/node-{record.node_index}"
            for role, indices in (("reasoning", reasoning), ("answer", answer)):
                if indices:
                    spans.append(SemanticSpan(f"{view}/{role}", role, REASONING_PREFIX_REVISION,
                                              _intervals(episode, rollout.selected_branch_id, view, indices)))
    return tuple(spans)
