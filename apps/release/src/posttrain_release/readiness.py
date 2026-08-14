"""Deterministic source-readiness receipts for release promotion."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from .fork_ledger import render_fork_ledger
from .versioning import check_release, load_release_manifest

_SCHEMA = "posttrain.release-readiness.v2"
_REQUIRED_CHECKS = (
    ("tests", ("pytest", "--cov", "--cov-report=term-missing")),
    ("lint", ("ruff", "check", ".")),
    ("format", ("ruff", "format", "--check", ".")),
    ("types", ("pyright",)),
    ("imports", ("lint-imports",)),
)


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create_readiness_receipt(
    repository_root: Path,
    *,
    checks: list[dict[str, object]],
    runtime_lock_pending: bool,
) -> dict[str, object]:
    """Create evidence that a source tree passed the deterministic release gate."""

    root = repository_root.resolve()
    return {
        "schema": _SCHEMA,
        "source_sha": _git(root, "rev-parse", "HEAD"),
        "source_tree": _git(root, "rev-parse", "HEAD^{tree}"),
        "framework_version": load_release_manifest(root).version,
        "uv_lock_sha256": _sha256(root / "uv.lock"),
        "fork_ledger": render_fork_ledger(root),
        "runtime_lock_pending": runtime_lock_pending,
        "checks": checks,
        "created_at": datetime.now(UTC).isoformat(),
    }


def write_readiness_receipt(receipt: dict[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_readiness(
    repository_root: Path,
    *,
    allow_pending_runtime_lock: bool,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, object]:
    """Run exactly the deterministic local/PR checks and return their receipt."""

    root = repository_root.resolve()
    release = check_release(root, allow_pending_runtime_lock=allow_pending_runtime_lock)
    checks: list[dict[str, object]] = [{"name": "release", "status": "success", "duration_seconds": 0.0}]
    for name, command in _REQUIRED_CHECKS:
        started = time.monotonic()
        runner(list(command), cwd=root, check=True, text=True)
        checks.append(
            {
                "name": name,
                "status": "success",
                "duration_seconds": round(time.monotonic() - started, 3),
            }
        )
    return create_readiness_receipt(root, checks=checks, runtime_lock_pending=release.runtime_lock_pending)


def verify_readiness_receipt(receipt_path: Path, repository_root: Path) -> dict[str, object]:
    """Reject readiness evidence that does not exactly describe this checkout."""

    root = repository_root.resolve()
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != _SCHEMA:
        raise ValueError("unsupported release readiness receipt")
    expected: dict[str, object] = {
        "source_sha": _git(root, "rev-parse", "HEAD"),
        "source_tree": _git(root, "rev-parse", "HEAD^{tree}"),
        "framework_version": load_release_manifest(root).version,
        "uv_lock_sha256": _sha256(root / "uv.lock"),
        "fork_ledger": render_fork_ledger(root),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"readiness receipt {key} does not match this checkout")
    checks = payload.get("checks")
    if not isinstance(checks, list):
        raise ValueError("readiness receipt has no checks")
    observed = {
        item.get("name")
        for item in checks
        if isinstance(item, dict) and item.get("status") == "success" and isinstance(item.get("name"), str)
    }
    required = {"release", *(name for name, _ in _REQUIRED_CHECKS)}
    missing = sorted(required - observed)
    if missing:
        raise ValueError(f"readiness receipt is missing successful checks: {', '.join(missing)}")
    return payload


def required_check_names() -> Sequence[str]:
    """Expose the public check set for tests and workflow documentation."""

    return ("release", *(name for name, _ in _REQUIRED_CHECKS))
