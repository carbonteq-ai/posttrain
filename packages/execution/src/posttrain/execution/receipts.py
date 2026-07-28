"""Append-only compact provider receipts for pre-Trackio reconciliation."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

from .contracts import ExecutionRecord, RuntimeImageRef


class ExecutionJournal:
    def __init__(self, path: Path) -> None:
        if not path.is_absolute():
            raise ValueError("execution journal path must be absolute")
        self._path = path

    def append(self, record: ExecutionRecord) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(record)
        payload["observed_at"] = record.observed_at.isoformat()
        encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
        descriptor = os.open(
            self._path,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o600,
        )
        try:
            os.write(descriptor, encoded)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def latest_runtime_image(
    receipt_root: Path,
    *,
    profile: str | None = None,
) -> RuntimeImageRef:
    """Resolve the newest immutable runtime image for an optional profile."""

    receipts = list(receipt_root.glob("*.json"))
    if not receipts:
        raise FileNotFoundError(f"runtime image receipt is missing under {receipt_root}")
    candidates: list[tuple[Path, dict[str, object]]] = []
    for receipt in receipts:
        try:
            payload = json.loads(receipt.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"runtime image receipt is invalid: {receipt}") from error
        if not isinstance(payload, dict):
            raise ValueError(f"runtime image receipt is invalid: {receipt}")
        if profile is None or payload.get("profile") == profile:
            candidates.append((receipt, payload))
    if not candidates:
        raise FileNotFoundError(f"runtime image receipt for profile {profile!r} is missing under {receipt_root}")
    receipt, payload = max(
        candidates,
        key=lambda item: (item[0].stat().st_mtime_ns, item[0].name),
    )
    value = payload.get("image")
    if not isinstance(value, str):
        raise ValueError(f"runtime image receipt has no image: {receipt}")
    return RuntimeImageRef(value)
