from __future__ import annotations

import json

from posttrain.common import AppendOnlyJsonlTailer


def _append(path, record: object, *, newline: bool = True) -> None:
    encoded = json.dumps(record).encode()
    with path.open("ab") as stream:
        stream.write(encoded + (b"\n" if newline else b""))


def test_tailer_emits_only_complete_lines_and_deduplicates(tmp_path) -> None:
    path = tmp_path / "traces.jsonl"
    emitted: list[str] = []
    tailer = AppendOnlyJsonlTailer(path, lambda record: emitted.append(str(record["id"])))

    _append(path, {"id": "one"}, newline=False)
    assert tailer.poll().emitted_records == 0
    with path.open("ab") as stream:
        stream.write(b"\n")
    _append(path, {"id": "one"})
    _append(path, {"id": "two"})

    stats = tailer.poll()
    assert emitted == ["one", "two"]
    assert stats.emitted_records == 2
    assert stats.duplicate_records == 1
    assert stats.complete


def test_tailer_retries_the_same_record_after_an_observer_failure(tmp_path) -> None:
    path = tmp_path / "traces.jsonl"
    _append(path, {"id": "one"})
    attempts = 0
    emitted: list[str] = []

    def emit(record: dict[str, object]) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary provider failure")
        emitted.append(str(record["id"]))

    tailer = AppendOnlyJsonlTailer(path, emit)
    assert tailer.poll().failed_records == 1
    assert emitted == []
    assert tailer.poll().emitted_records == 1
    assert emitted == ["one"]
    assert tailer.stats.complete
