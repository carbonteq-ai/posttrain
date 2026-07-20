"""Best-effort, idempotent synchronization of native Verifiers JSONL traces."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TraceBatchUploader = Callable[[list[dict[str, Any]]], None]
TraceValidator = Callable[[dict[str, Any]], dict[str, Any]]


def validate_verifiers_record(record: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize a record with the pinned Verifiers wire schema."""
    try:
        from verifiers.v1.trace import WireTrace  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise RuntimeError(
            "Verifiers dependencies are unavailable; sync packages/eval with --extra verifiers"
        ) from error
    return WireTrace.model_validate(record).to_record()


@dataclass(slots=True)
class TraceSyncStats:
    observed_records: int = 0
    synced_records: int = 0
    invalid_records: int = 0
    failed_batches: int = 0
    unsynced_records: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return self.invalid_records == 0 and self.unsynced_records == 0

    def metrics(self) -> dict[str, int]:
        return {
            "eval/trace_records_observed": self.observed_records,
            "eval/trace_records_synced": self.synced_records,
            "eval/trace_records_invalid": self.invalid_records,
            "eval/trace_sync_failed_batches": self.failed_batches,
            "eval/trace_records_unsynced": self.unsynced_records,
            "eval/trace_sync_complete": int(self.complete),
        }


class VerifiersTraceSynchronizer:
    """Tail complete JSONL lines and upload validated records in small batches."""

    def __init__(
        self,
        path: Path,
        upload: TraceBatchUploader,
        *,
        batch_size: int = 16,
        validate: TraceValidator = validate_verifiers_record,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.path = path
        self.upload = upload
        self.batch_size = batch_size
        self.validate = validate
        self.stats = TraceSyncStats()
        self._offset = 0
        self._pending: list[dict[str, Any]] = []
        self._failed: list[list[dict[str, Any]]] = []

    def drain(self) -> TraceSyncStats:
        """Read newly completed lines and upload each full batch."""
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
                    self._pending.append(self.validate(decoded))
                except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
                    self.stats.invalid_records += 1
                    self._remember_error(error)
                if len(self._pending) >= self.batch_size:
                    batch = self._pending[: self.batch_size]
                    del self._pending[: self.batch_size]
                    self._send(batch)
        return self.stats

    def finalize(self) -> TraceSyncStats:
        """Drain, flush the short batch, and retry every failed batch once."""
        self.drain()
        if self._pending:
            batch = self._pending
            self._pending = []
            self._send(batch)
        failed = self._failed
        self._failed = []
        for batch in failed:
            self._send(batch)
        self.stats.unsynced_records = sum(len(batch) for batch in self._failed)
        return self.stats

    def _send(self, batch: list[dict[str, Any]]) -> None:
        try:
            self.upload(batch)
        except Exception as error:  # telemetry must not invalidate a completed evaluation
            self.stats.failed_batches += 1
            self._remember_error(error)
            self._failed.append(batch)
        else:
            self.stats.synced_records += len(batch)

    def _remember_error(self, error: BaseException) -> None:
        self.stats.errors.append(f"{type(error).__name__}: {error}")
        del self.stats.errors[:-10]
