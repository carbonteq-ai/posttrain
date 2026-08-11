"""Small, backend-neutral tailer for append-only JSONL evidence journals."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

type JsonRecordEmitter = Callable[[dict[str, Any]], None]


@dataclass(slots=True)
class JsonlJournalStats:
    observed_records: int = 0
    emitted_records: int = 0
    duplicate_records: int = 0
    invalid_records: int = 0
    failed_records: int = 0
    unsynchronized_records: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return self.invalid_records == 0 and self.unsynchronized_records == 0


class AppendOnlyJsonlTailer:
    """Emit complete JSON object lines exactly once per live tailer lifetime.

    A failed emit never advances the cursor for that line, so a later poll can
    retry it.  The source journal remains authoritative; this class owns no
    provider-specific transport, credentials, or schema.
    """

    def __init__(self, path: Path, emit: JsonRecordEmitter) -> None:
        self.path = path
        self.emit = emit
        self.stats = JsonlJournalStats()
        self._offset = 0
        self._seen: set[str] = set()

    def poll(self) -> JsonlJournalStats:
        if not self.path.is_file():
            return self.stats
        if self.path.stat().st_size < self._offset:
            # A new attempt owns a new journal.  Never silently retain stale
            # cursor state across truncation.
            self._offset = 0
            self._seen.clear()
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
                try:
                    decoded = json.loads(raw)
                    if not isinstance(decoded, dict):
                        raise TypeError("journal record must be a JSON object")
                    external_id = decoded.get("id")
                    if not isinstance(external_id, str) or not external_id:
                        raise ValueError("journal record requires a non-empty id")
                except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
                    self.stats.observed_records += 1
                    self.stats.invalid_records += 1
                    self._remember_error(error)
                    self._offset = stream.tell()
                    continue
                if external_id in self._seen:
                    self.stats.observed_records += 1
                    self.stats.duplicate_records += 1
                    self._offset = stream.tell()
                    continue
                try:
                    self.emit(decoded)
                except Exception as error:  # native journal must outlive observer failures
                    self.stats.failed_records += 1
                    self.stats.unsynchronized_records = 1
                    self._remember_error(error)
                    self._offset = line_start
                    break
                self.stats.observed_records += 1
                self.stats.emitted_records += 1
                self.stats.unsynchronized_records = 0
                self._seen.add(external_id)
                self._offset = stream.tell()
        return self.stats

    @property
    def offset(self) -> int:
        """The last byte acknowledged by this tailer."""

        return self._offset

    def _remember_error(self, error: BaseException) -> None:
        self.stats.errors.append(f"{type(error).__name__}: {error}")
        del self.stats.errors[:-10]


__all__ = ["AppendOnlyJsonlTailer", "JsonlJournalStats"]
