"""Best-effort synchronization of authoritative Verifiers JSONL traces."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TraceBatchEmitter = Callable[[list[dict[str, Any]]], None]
TraceValidator = Callable[[dict[str, Any]], dict[str, Any]]


def validate_verifiers_record(record: dict[str, Any]) -> dict[str, Any]:
    """Validate a record against the schema at the pinned Verifiers revision."""

    try:
        from verifiers.v1.trace import WireTrace  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise RuntimeError("install posttrain-eval with the verifiers extra") from error
    return WireTrace.model_validate(record).to_record()


@dataclass(slots=True)
class TraceSyncStats:
    observed_records: int = 0
    emitted_records: int = 0
    duplicate_records: int = 0
    invalid_records: int = 0
    failed_batches: int = 0
    unsynchronized_records: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return self.invalid_records == 0 and self.unsynchronized_records == 0


class VerifiersTraceSynchronizer:
    """Read complete lines, deduplicate trace ids, and emit small retryable batches."""

    def __init__(
        self,
        path: Path,
        emit: TraceBatchEmitter,
        *,
        batch_size: int = 16,
        validate: TraceValidator = validate_verifiers_record,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.path = path
        self.emit = emit
        self.batch_size = batch_size
        self.validate = validate
        self.stats = TraceSyncStats()
        self._offset = 0
        self._pending: list[dict[str, Any]] = []
        self._failed: list[list[dict[str, Any]]] = []
        self._seen: set[str] = set()

    def drain(self) -> TraceSyncStats:
        if not self.path.is_file():
            return self.stats
        if self.path.stat().st_size < self._offset:
            self._offset = 0
        with self.path.open("rb") as stream:
            stream.seek(self._offset)
            while True:
                line_start = stream.tell()
                raw = stream.readline()
                if not raw:
                    break
                if not raw.endswith(b"\n"):
                    self._offset = line_start
                    break
                self._offset = stream.tell()
                self.stats.observed_records += 1
                try:
                    decoded = json.loads(raw)
                    if not isinstance(decoded, dict):
                        raise TypeError("trace record must be a JSON object")
                    record = self.validate(decoded)
                    trace_id = record.get("id")
                    if not isinstance(trace_id, str) or not trace_id:
                        raise ValueError("trace record requires a non-empty id")
                    if trace_id in self._seen:
                        self.stats.duplicate_records += 1
                        continue
                    self._seen.add(trace_id)
                    self._pending.append(record)
                except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
                    self.stats.invalid_records += 1
                    self._remember_error(error)
                if len(self._pending) >= self.batch_size:
                    batch = self._pending[: self.batch_size]
                    del self._pending[: self.batch_size]
                    self._send(batch)
        return self.stats

    def finalize(self) -> TraceSyncStats:
        self.drain()
        if self._pending:
            batch = self._pending
            self._pending = []
            self._send(batch)
        failed = self._failed
        self._failed = []
        for batch in failed:
            self._send(batch)
        self.stats.unsynchronized_records = sum(len(batch) for batch in self._failed)
        return self.stats

    def _send(self, batch: list[dict[str, Any]]) -> None:
        try:
            self.emit(batch)
        except Exception as error:  # observation failure cannot erase the native bundle
            self.stats.failed_batches += 1
            self._remember_error(error)
            self._failed.append(batch)
        else:
            self.stats.emitted_records += len(batch)

    def _remember_error(self, error: BaseException) -> None:
        self.stats.errors.append(f"{type(error).__name__}: {error}")
        del self.stats.errors[:-10]


__all__ = [
    "TraceSyncStats",
    "VerifiersTraceSynchronizer",
    "validate_verifiers_record",
]
