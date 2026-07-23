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
    """Create one SFT example per retained trace branch without owning native traces."""

    policy = selection or TraceSelection()
    examples: list[SupervisedExample] = []
    for trace in traces:
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
                    id=f"traces/{str(trace.id).lower()}/branches/{branch_index}",
                    messages=messages,
                    trainable_message_indices=trainable,
                    tools=tools,
                    metadata={
                        "source_format": "verifiers-trace-v2",
                        "trace_id": str(trace.id),
                        "branch_index": branch_index,
                        "reward": reward,
                        "stop_condition": str(trace.stop_condition or ""),
                        "is_truncated": bool(trace.is_truncated),
                    },
                )
            )
    return SupervisedDataset(dataset_id, revision, tuple(examples), metadata=metadata or {})


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
    traces = (
        WireTrace.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    return supervised_from_verifiers(
        traces,
        dataset_id=dataset_id,
        revision=revision,
        selection=selection,
        metadata=metadata,
    )


__all__ = ["TraceSelection", "supervised_from_verifiers", "supervised_from_verifiers_jsonl"]
