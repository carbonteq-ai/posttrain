"""Project authoritative Verifiers traces into canonical SFT snapshots."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from posttrain.common import JsonValue

from ..models import MessageRecord, SupervisedDataset, SupervisedExample, ToolRecord


@dataclass(frozen=True, slots=True)
class TraceSelection:
    min_reward: float | None = None
    drop_truncated: bool = True
    drop_errors: bool = True


def _dump(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return cast(dict[str, Any], value.model_dump(mode="json", exclude_none=True))
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"cannot convert {type(value).__name__} to a JSON record")


def _tool(value: Any) -> ToolRecord:
    record = _dump(value)
    if record.get("type") == "function" and isinstance(record.get("function"), Mapping):
        return cast(ToolRecord, record)
    return cast(
        ToolRecord,
        {
            "type": "function",
            "function": {
                "name": record.get("name"),
                "description": record.get("description"),
                "parameters": record.get("parameters", {}),
                **({} if record.get("strict") is None else {"strict": record["strict"]}),
            },
        },
    )


def supervised_from_verifiers(
    traces: Iterable[Any],
    *,
    dataset_id: str,
    revision: str,
    selection: TraceSelection | None = None,
    metadata: Mapping[str, JsonValue] | None = None,
) -> SupervisedDataset:
    """Project native traces or episodes, retaining episode and branch lineage.

    Legacy trace objects remain accepted during migration. Episode failures and
    non-policy agents are excluded by default rather than becoming SFT targets.
    The source artifact is never changed or reconstructed from these examples.
    """

    policy = selection or TraceSelection()
    examples: list[SupervisedExample] = []
    for episode_id, trace in _selected_traces(traces, policy):
        if policy.drop_errors and (getattr(trace, "stop_condition", None) == "error" or trace.has_error):
            continue
        if policy.drop_truncated and trace.is_truncated:
            continue
        reward = float(trace.reward)
        if policy.min_reward is not None and reward < policy.min_reward:
            continue
        tools = tuple(_tool(tool) for tool in (trace.tools or []))
        for fallback_index, branch in enumerate(trace.branches):
            nodes = list(branch.nodes)
            if not nodes:
                continue
            messages = tuple(cast(MessageRecord, _dump(node.message)) for node in nodes)
            trainable = tuple(index for index, node in enumerate(nodes) if bool(node.sampled))
            if not trainable:
                continue
            branch_index = int(getattr(branch, "index", fallback_index))
            examples.append(
                SupervisedExample(
                    id=(
                        (f"episodes/{episode_id}/" if episode_id is not None else "")
                        + f"traces/{str(trace.id).lower()}/branches/{branch_index}"
                    ),
                    messages=messages,
                    trainable_message_indices=trainable,
                    tools=tools,
                    metadata={
                        "source_format": "verifiers-episode" if episode_id is not None else "verifiers-trace-v2",
                        **({"episode_id": episode_id} if episode_id is not None else {}),
                        "trace_id": str(trace.id),
                        "branch_index": branch_index,
                        "reward": reward,
                        "stop_condition": str(trace.stop_condition or ""),
                        "is_truncated": bool(trace.is_truncated),
                    },
                )
            )
    return SupervisedDataset(dataset_id, revision, tuple(examples), metadata=metadata or {})


def _selected_traces(records: Iterable[Any], policy: TraceSelection) -> Iterable[tuple[str | None, Any]]:
    for record in records:
        if not hasattr(record, "traces"):
            if getattr(getattr(record, "agent", None), "trainable", True) is not False:
                yield None, record
            continue
        # Episode.ok is execution standing, not semantic task reward. A finished
        # but incorrectly solved attempt may still be selected by min_reward.
        if policy.drop_errors and (not record.ok or record.errors):
            continue
        episode_id = str(record.id)
        if not episode_id:
            raise ValueError("native episode requires a non-empty identity")
        for trace in record.traces:
            if getattr(getattr(trace, "agent", None), "trainable", True) is False:
                continue
            yield episode_id, trace


def supervised_from_verifiers_jsonl(
    path: Path,
    *,
    dataset_id: str,
    revision: str,
    selection: TraceSelection | None = None,
    metadata: Mapping[str, JsonValue] | None = None,
) -> SupervisedDataset:
    """Validate native trace records before projecting a completed JSONL artifact."""

    try:
        from verifiers.v1 import WireTrace  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise RuntimeError("install posttrain-data with the verifiers extra") from error

    def records() -> Iterable[Any]:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"native record at line {line_number} must be an object")
                if "traces" in record:
                    try:
                        from verifiers.v1.episode import (
                            WireEpisode,  # pyright: ignore[reportMissingImports, reportAttributeAccessIssue]
                        )
                    except ImportError as error:
                        raise RuntimeError("native episode artifacts require Verifiers v0.3.1 or compatible") from error
                    if "nodes" in record or not isinstance(record.get("task"), dict):
                        raise ValueError(f"ambiguous or incomplete native episode at line {line_number}")
                    yield WireEpisode.model_validate(record)
                else:
                    yield WireTrace.model_validate(record)

    return supervised_from_verifiers(
        records(),
        dataset_id=dataset_id,
        revision=revision,
        selection=selection,
        metadata=metadata,
    )


__all__ = ["TraceSelection", "supervised_from_verifiers", "supervised_from_verifiers_jsonl"]
