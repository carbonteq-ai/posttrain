"""Tests for Verifiers trace synchronization."""

from __future__ import annotations

import json
from pathlib import Path

from posttrain.eval.backends.verifiers.synchronization import VerifiersTraceSynchronizer


def _record(trace_id: str) -> dict:
    return {"id": trace_id, "version": 2, "nodes": []}


def _identity(record: dict) -> dict:
    if not isinstance(record.get("id"), str) or not isinstance(record.get("version"), int):
        raise ValueError("invalid trace")
    return record


def test_tailer_waits_for_complete_lines_and_batches(tmp_path: Path):
    path = tmp_path / "traces.jsonl"
    uploaded: list[list[dict]] = []
    first = json.dumps(_record("one"))
    second = json.dumps(_record("two"))
    path.write_bytes(f"{first}\n{second}".encode())
    sync = VerifiersTraceSynchronizer(path, uploaded.append, batch_size=2, validate=_identity)

    sync.drain()
    assert sync.stats.observed_records == 1
    assert uploaded == []

    with path.open("ab") as stream:
        stream.write(b"\n")
    stats = sync.drain()

    assert [record["id"] for record in uploaded[0]] == ["one", "two"]
    assert stats.emitted_records == 2


def test_finalization_retries_failed_batches(tmp_path: Path):
    path = tmp_path / "traces.jsonl"
    path.write_text(json.dumps(_record("one")) + "\n", encoding="utf-8")
    calls = 0

    def upload(records: list[dict]) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ConnectionError("temporary")

    sync = VerifiersTraceSynchronizer(path, upload, validate=_identity)
    stats = sync.finalize()

    assert calls == 2
    assert stats.emitted_records == 1
    assert stats.failed_batches == 1
    assert stats.unsynchronized_records == 0
    assert stats.complete


def test_invalid_records_are_reported_without_stopping_valid_sync(tmp_path: Path):
    path = tmp_path / "traces.jsonl"
    path.write_text(
        json.dumps({"bad": True}) + "\n" + json.dumps(_record("valid")) + "\n",
        encoding="utf-8",
    )
    uploaded: list[list[dict]] = []
    stats = VerifiersTraceSynchronizer(path, uploaded.append, validate=_identity).finalize()

    assert stats.observed_records == 2
    assert stats.invalid_records == 1
    assert stats.emitted_records == 1
    assert not stats.complete
    assert uploaded[0][0]["id"] == "valid"


def test_duplicate_external_ids_are_emitted_once(tmp_path: Path):
    path = tmp_path / "traces.jsonl"
    path.write_text(
        json.dumps(_record("same")) + "\n" + json.dumps(_record("same")) + "\n",
        encoding="utf-8",
    )
    uploaded: list[list[dict]] = []
    stats = VerifiersTraceSynchronizer(path, uploaded.append, validate=_identity).finalize()

    assert stats.observed_records == 2
    assert stats.duplicate_records == 1
    assert stats.emitted_records == 1
    assert uploaded == [[_record("same")]]
