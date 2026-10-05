"""Native Verifiers record encoding and decoding, in process or on worker processes.

Serializing an episode to its replay record (Verifiers' ``to_record``, which
normalizes assessment archives) and encoding it as JSON dominate the trainer's
per-episode CPU time; parsing retained records dominates admission. Neither
needs trainer state, so both run on one shared pool of worker processes. The
trainer process keeps every ordered or stateful step: file appends and their
byte spans, identity checks, schema validation of decoded graphs, observation
and projection.

Workers start through a fork server (a fresh interpreter), so they never
inherit a trainer's CUDA context, threads or locks. Episodes travel to them as
pickles, which costs a few milliseconds where serialization costs tens to
hundreds.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import multiprocessing
import os
import pickle
import threading
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

from posttrain.common import TraceObservation

MAX_RECORD_WORKERS = 8
_pool: ProcessPoolExecutor | None = None
_pool_lock = threading.Lock()


def record_workers() -> ProcessPoolExecutor:
    """The process's shared record pool, started on first use."""
    global _pool
    with _pool_lock:
        if _pool is None:
            workers = max(2, min(MAX_RECORD_WORKERS, (os.cpu_count() or 2) - 1))
            _pool = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("forkserver"))
        return _pool


def native_record(value: Any) -> dict[str, Any]:
    """Serialize replay authority without reducing policy-float precision.

    Newer Verifiers releases round JSON record floats by default for storage
    efficiency. Posttrain replays these records for training/evidence audits, so
    it opts out when the runtime exposes that setting while remaining readable
    against the older v0.3.1 contract during the pin migration.
    """

    to_record = value.to_record
    if "float_decimals" in inspect.signature(to_record).parameters:
        return to_record(float_decimals=None)
    return to_record()


def encode_line(record: Mapping[str, Any]) -> bytes:
    """One canonical JSONL line (sorted keys, compact separators)."""
    return (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def episode_assessment_observation(record: Mapping[str, Any]) -> TraceObservation | None:
    """Derived episode evidence, separate from solver traces and reward metrics.

    The native record remains authoritative. Tracking adapters apply their
    results-only export policy; this internal observation retains producer data.
    """
    batches = record.get("assessment_batches")
    assignments = record.get("credit_assignments")
    diagnostics = record.get("assessment_errors")
    credit_diagnostics = record.get("credit_errors")
    if not batches and not assignments and not diagnostics and not credit_diagnostics:
        return None
    episode_id = record.get("id")
    if not isinstance(episode_id, str) or not episode_id:
        raise ValueError("episode assessments require native episode identity")
    children = record.get("traces", [])
    if not isinstance(children, list):
        raise ValueError("native episode children must be a list")
    return TraceObservation(
        trace_type="verifiers.assessment-results",
        external_id=f"episode:{episode_id}:assessment-results",
        payload={
            "schema_version": 1,
            "episode_id": episode_id,
            "child_trace_ids": [str(child["id"]) for child in children if isinstance(child, Mapping)],
            "execution_ok": bool(record.get("ok", False)),
            "assessment_finalization_state": record.get("assessment_finalization_state"),
            "assessment_error_count": len(diagnostics) if isinstance(diagnostics, list) else 0,
            "assessment_batches": batches or [],
            "assessment_sources": record.get("assessment_sources", []),
            "assessment_views": record.get("assessment_views", []),
            "credit_assignments": assignments or [],
            "credit_error_count": len(credit_diagnostics) if isinstance(credit_diagnostics, list) else 0,
            "archive_status": "pending_publication",
        },
        attributes={"episode_id": episode_id, "evidence_scope": "episode"},
    )


@dataclass(frozen=True, slots=True)
class EncodedEpisode:
    """An episode's replay line and what the trainer derives from its record.

    ``trace_record``/``trace_line`` are the derived terminal view of one
    trainable trace (with its run and task facets), when one was requested.
    """

    line: bytes
    digest: str
    trace_ids: tuple[str, ...]
    assessment: TraceObservation | None
    trace_record: dict[str, Any] | None = None
    trace_line: bytes | None = None


def encode_episode(
    episode: Any,
    *,
    trace_id: str | None = None,
    task_facets: Mapping[str, Any] | None = None,
) -> EncodedEpisode:
    record = native_record(episode)
    line = encode_line(record)
    traces = record.get("traces", ())
    trace_ids = tuple(
        str(trace["id"]) for trace in traces if isinstance(trace, Mapping) and isinstance(trace.get("id"), str)
    )
    trace_record = trace_line = None
    if trace_id is not None:
        trace = next(item for item in episode.traces if item.id == trace_id)
        trace_record = native_record(trace)
        trace_info = getattr(trace, "info", {})
        if isinstance(trace_info.get("posttrain_run"), Mapping):
            # Versioned derived trace view for existing observation consumers;
            # the untouched native episode is retained alongside this view.
            trace_record["run"] = dict(trace_info["posttrain_run"])
        info = trace_record.setdefault("info", {})
        if not isinstance(info, dict):
            raise ValueError("Verifiers trace info must be an object")
        info["task_facets"] = dict(task_facets or {})
        trace_line = encode_line(trace_record)
    return EncodedEpisode(
        line,
        hashlib.sha256(line).hexdigest(),
        trace_ids,
        episode_assessment_observation(record),
        trace_record,
        trace_line,
    )


def _encode_pickled(blob: bytes, trace_id: str | None, task_facets: Mapping[str, Any] | None) -> EncodedEpisode:
    return encode_episode(pickle.loads(blob), trace_id=trace_id, task_facets=task_facets)


async def encode_episode_on_workers(
    episode: Any,
    *,
    trace_id: str | None = None,
    task_facets: Mapping[str, Any] | None = None,
) -> EncodedEpisode:
    """``encode_episode`` on the shared pool; the caller's event loop stays free."""
    blob = pickle.dumps(episode, protocol=pickle.HIGHEST_PROTOCOL)
    future = record_workers().submit(_encode_pickled, blob, trace_id, dict(task_facets or {}))
    return await asyncio.wrap_future(future)


def message_graphs(line: bytes, episodes: bool) -> list[tuple[Any, Any, Any]] | str:
    """Parse one native record; return each trace's (id, nodes, calls) or a rejection reason.

    Only parses JSON and selects fields, so the returned values are plain data
    and the caller keeps every check.
    """
    try:
        record = json.loads(line)
    except ValueError:
        return "native population records must be JSON objects"
    if not isinstance(record, dict):
        return "native population records must be objects"
    raw_traces = record.get("traces") if episodes else [record]
    if not isinstance(raw_traces, list) or not raw_traces:
        return "native population record lacks original traces"
    return [
        (raw.get("id"), raw.get("nodes"), raw.get("calls", [])) if isinstance(raw, dict) else (None, None, None)
        for raw in raw_traces
    ]
