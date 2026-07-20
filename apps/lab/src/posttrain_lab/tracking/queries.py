"""Narrow query adapter for evidence consumed by code-defined jobs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class VerifiersRollout:
    trace_id: str
    task_index: int
    response: str
    reward: float


def verifiers_rollout(project: str, run_id: str, trace_id: str) -> VerifiersRollout:
    """Read one completed rollout through Trackio's storage query boundary."""

    try:
        from trackio.sqlite_storage import SQLiteStorage
    except ImportError as error:
        raise RuntimeError("the pinned Trackio package is required") from error
    rows: list[dict[str, Any]] = SQLiteStorage.get_traces(
        project,
        run_id=run_id,
        trace_type="verifiers",
    )
    matches = [row for row in rows if row.get("external_id") == trace_id]
    if len(matches) != 1:
        raise ValueError(f"expected one Verifiers trace {trace_id!r} in run {run_id!r}, found {len(matches)}")
    row = matches[0]
    payload = row.get("payload")
    metadata = row.get("metadata")
    if not isinstance(payload, dict) or not isinstance(metadata, dict):
        raise ValueError("Verifiers trace payload and metadata must be JSON objects")
    if not metadata.get("is_completed") or metadata.get("is_truncated") or metadata.get("error_type") is not None:
        raise ValueError("preference rollouts must be completed, untruncated, and error-free")
    nodes = payload.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("Verifiers trace is missing nodes")
    response = None
    for node in reversed(nodes):
        if not isinstance(node, dict):
            continue
        message = node.get("message")
        if isinstance(message, dict) and message.get("role") == "assistant" and isinstance(message.get("content"), str):
            response = message["content"]
            break
    if response is None or not response.strip():
        raise ValueError("Verifiers trace has no assistant response")
    task_index = metadata.get("task_index")
    reward = metadata.get("reward")
    if not isinstance(task_index, int) or not isinstance(reward, int | float):
        raise ValueError("Verifiers trace lacks task index or scalar reward")
    return VerifiersRollout(trace_id, task_index, response, float(reward))


__all__ = ["VerifiersRollout", "verifiers_rollout"]
