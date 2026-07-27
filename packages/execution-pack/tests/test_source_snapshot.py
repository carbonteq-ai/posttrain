from __future__ import annotations

from pathlib import Path

import pytest
from posttrain.execution_pack import (
    ImmutableSourceSnapshotter,
    SourceSnapshotRequest,
)


def _source(root: Path) -> None:
    package = root / "packages" / "example"
    (package / "src" / "example").mkdir(parents=True)
    (package / "pyproject.toml").write_text(
        "[project]\nname='example'\nversion='1.0.0'\n",
        encoding="utf-8",
    )
    (package / "src" / "example" / "__init__.py").write_text(
        "VALUE = 1\n",
        encoding="utf-8",
    )
    config = root / ".posttrain"
    config.mkdir()
    (config / "project.toml").write_text(
        "schema_version=1\nproject_id='example'\n",
        encoding="utf-8",
    )


def test_snapshot_is_content_addressed_and_selective(tmp_path: Path) -> None:
    source = (tmp_path / "source").resolve()
    source.mkdir()
    _source(source)
    request = SourceSnapshotRequest(
        root=source,
        includes=("packages/example",),
        install_roots=("packages/example",),
    )
    snapshotter = ImmutableSourceSnapshotter(
        cache_root=(tmp_path / "cache").resolve()
    )

    first = snapshotter.materialize(request)
    second = snapshotter.materialize(request)

    assert snapshotter.inspect(request) == first.digest
    assert first.created is True
    assert second.created is False
    assert first.digest == second.digest
    assert first.package.root == second.package.root
    assert (
        first.package.root / "packages/example/src/example/__init__.py"
    ).is_file()
    assert not (first.package.root / ".posttrain").exists()


def test_snapshot_digest_changes_with_selected_working_tree_bytes(
    tmp_path: Path,
) -> None:
    source = (tmp_path / "source").resolve()
    source.mkdir()
    _source(source)
    request = SourceSnapshotRequest(
        root=source,
        includes=("packages/example",),
        install_roots=("packages/example",),
    )
    snapshotter = ImmutableSourceSnapshotter(
        cache_root=(tmp_path / "cache").resolve()
    )
    first = snapshotter.materialize(request)

    (source / "packages/example/src/example/__init__.py").write_text(
        "VALUE = 2\n",
        encoding="utf-8",
    )
    second = snapshotter.materialize(request)

    assert first.digest != second.digest


def test_snapshot_ignores_generated_python_cache_files(tmp_path: Path) -> None:
    source = (tmp_path / "source").resolve()
    source.mkdir()
    _source(source)
    request = SourceSnapshotRequest(
        root=source,
        includes=("packages/example",),
        install_roots=("packages/example",),
    )
    snapshotter = ImmutableSourceSnapshotter(
        cache_root=(tmp_path / "cache").resolve()
    )
    before = snapshotter.inspect(request)
    cache = source / "packages/example/src/example/__pycache__"
    cache.mkdir()
    (cache / "__init__.cpython-312.pyc").write_bytes(b"generated")

    assert snapshotter.inspect(request) == before
    assert snapshotter.materialize(request).digest == before


def test_snapshot_rejects_overlap_and_secret_files(tmp_path: Path) -> None:
    source = (tmp_path / "source").resolve()
    source.mkdir()
    _source(source)

    with pytest.raises(ValueError, match="cannot overlap"):
        SourceSnapshotRequest(
            root=source,
            includes=("packages", "packages/example"),
            install_roots=("packages/example",),
        )

    (source / "packages/example/.env").write_text("TOKEN=secret\n")
    request = SourceSnapshotRequest(
        root=source,
        includes=("packages/example",),
        install_roots=("packages/example",),
    )
    with pytest.raises(ValueError, match="forbidden filename"):
        ImmutableSourceSnapshotter(
            cache_root=(tmp_path / "cache").resolve()
        ).materialize(request)
