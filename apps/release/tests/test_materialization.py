"""Candidate-to-final runtime materialization contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from posttrain_release.materialization import (
    apply_materialization,
    create_materialization,
    verify_materialization,
)

_RUNTIME_ROOT = Path("packages/runtime-images/src/posttrain/runtime_images")
_LOCK_ROOT = _RUNTIME_ROOT / "containers/posttrain-job-kinds/locks"
_SHA = "a" * 40
_TREE = "b" * 40


def _source(root: Path) -> Path:
    runtime = root / _RUNTIME_ROOT
    locks = root / _LOCK_ROOT
    locks.mkdir(parents=True)
    (runtime / "published.toml").write_text('framework_version = "0.4.2"\n', encoding="utf-8")
    (locks / "base.lock.txt").write_text("example==1 --hash=sha256:abc\n", encoding="utf-8")
    readiness = root / "readiness.json"
    readiness.write_text(
        json.dumps(
            {
                "schema": "posttrain.release-readiness.v2",
                "framework_version": "0.4.2",
                "source_sha": _SHA,
                "source_tree": _TREE,
                "checks": [
                    {"name": name, "status": "success"}
                    for name in ("release", "tests", "lint", "format", "types", "imports")
                ],
            }
        ),
        encoding="utf-8",
    )
    return readiness


def _staged(root: Path) -> Path:
    runtime = root / _RUNTIME_ROOT
    locks = root / _LOCK_ROOT
    locks.mkdir(parents=True)
    (runtime / "published.toml").write_text("old manifest\n", encoding="utf-8")
    (locks / "old.lock.txt").write_text("old lock\n", encoding="utf-8")
    return root


def test_materialization_replaces_stale_generated_source_only_after_verification(tmp_path: Path) -> None:
    source = tmp_path / "source"
    readiness = _source(source)
    materialization = tmp_path / "materialization"
    receipt = create_materialization(source, readiness, materialization, candidate_version="0.4.2rc1")

    staged = _staged(tmp_path / "staged")
    applied = apply_materialization(
        materialization,
        staged,
        target_version="0.4.2",
        source_sha=_SHA,
        source_tree=_TREE,
    )

    assert applied == receipt
    assert (staged / _RUNTIME_ROOT / "published.toml").read_text() == 'framework_version = "0.4.2"\n'
    assert not (staged / _LOCK_ROOT / "old.lock.txt").exists()
    assert (staged / _LOCK_ROOT / "base.lock.txt").read_text() == "example==1 --hash=sha256:abc\n"


def test_materialization_rejects_tampered_bytes_without_changing_destination(tmp_path: Path) -> None:
    source = tmp_path / "source"
    readiness = _source(source)
    materialization = tmp_path / "materialization"
    create_materialization(source, readiness, materialization, candidate_version="0.4.2rc3")
    (materialization / "published.toml").write_text("tampered\n", encoding="utf-8")
    staged = _staged(tmp_path / "staged")

    with pytest.raises(ValueError, match="does not match receipt"):
        apply_materialization(materialization, staged, target_version="0.4.2")

    assert (staged / _RUNTIME_ROOT / "published.toml").read_text() == "old manifest\n"
    assert (staged / _LOCK_ROOT / "old.lock.txt").read_text() == "old lock\n"


def test_materialization_rejects_undeclared_and_unsafe_files(tmp_path: Path) -> None:
    source = tmp_path / "source"
    readiness = _source(source)
    materialization = tmp_path / "materialization"
    create_materialization(source, readiness, materialization, candidate_version="0.4.2rc2")
    (materialization / "runtime-locks" / "extra.txt").write_text("not declared\n", encoding="utf-8")

    with pytest.raises(ValueError, match="file set differs"):
        verify_materialization(materialization)

    receipt_path = materialization / "materialization.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["files"][0]["path"] = "../published.toml"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe materialization file path"):
        verify_materialization(materialization)


def test_materialization_rejects_a_swapped_readiness_receipt(tmp_path: Path) -> None:
    source = tmp_path / "source"
    readiness = _source(source)
    materialization = tmp_path / "materialization"
    create_materialization(source, readiness, materialization, candidate_version="0.4.2rc4")
    swapped = tmp_path / "swapped-readiness.json"
    swapped.write_text(readiness.read_text().replace(_SHA, "c" * 40), encoding="utf-8")

    with pytest.raises(ValueError, match="readiness receipt digest does not match"):
        verify_materialization(materialization, readiness_receipt=swapped)
