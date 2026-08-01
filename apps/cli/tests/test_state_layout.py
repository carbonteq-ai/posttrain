from __future__ import annotations

import json
from pathlib import Path

import pytest
from posttrain.catalog import load_project_layout
from posttrain.common import ContractError
from posttrain_cli.state_layout import cache_path, migrate_state, prune_cache


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    control = root / ".posttrain"
    control.mkdir(parents=True)
    (control / "project.toml").write_text('schema_version = 1\nproject_id = "state-test"\n', encoding="utf-8")
    return root


def test_in_place_migration_moves_only_known_rebuildable_entries(tmp_path: Path) -> None:
    root = _project(tmp_path)
    state = root / ".posttrain" / "state"
    (state / "pack").mkdir(parents=True)
    (state / "pack" / "context.json").write_text("{}\n", encoding="utf-8")
    (state / "execution.toml").write_text("secret-free fixture\n", encoding="utf-8")

    report = migrate_state(load_project_layout(root))

    assert report.moved_cache_entries == ("pack",)
    assert report.protected_entries == ("execution.toml",)
    assert (state / "cache" / "pack" / "context.json").is_file()
    assert not (state / "pack").exists()
    assert migrate_state(load_project_layout(root)).changed is False


def test_relocation_refuses_submission_without_terminal_reconciliation(tmp_path: Path) -> None:
    destination = _project(tmp_path)
    source = tmp_path / "old-project"
    run = source / ".posttrain" / "state" / "executions" / "still-running"
    run.mkdir(parents=True)
    (run / "submission.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ContractError, match="still-running"):
        migrate_state(load_project_layout(destination), source_project_root=source)


def test_relocation_copies_only_verified_execution_records_and_never_deletes_source(tmp_path: Path) -> None:
    destination = _project(tmp_path)
    source = tmp_path / "old-project"
    run = source / ".posttrain" / "state" / "executions" / "terminal"
    run.mkdir(parents=True)
    (run / "submission.json").write_text("{}\n", encoding="utf-8")
    (run / "reconciliation.json").write_text(
        json.dumps(
            {
                "schema": "posttrain.execution-reconciliation.v1",
                "provider_record": {"state": "succeeded"},
            }
        ),
        encoding="utf-8",
    )
    (source / ".posttrain" / "state" / "pack").mkdir(parents=True)

    report = migrate_state(load_project_layout(destination), source_project_root=source)

    assert report.copied_execution_files == 2
    assert (destination / ".posttrain" / "state" / "executions" / "terminal" / "submission.json").is_file()
    assert (source / ".posttrain" / "state" / "pack").is_dir()


def test_cache_path_cannot_escape_project_cache(tmp_path: Path) -> None:
    layout = load_project_layout(_project(tmp_path))

    with pytest.raises(ContractError, match="escapes"):
        cache_path(layout, "..", "outside")


def test_cache_prune_is_dry_run_first_and_preserves_control_and_unknown_state(tmp_path: Path) -> None:
    root = _project(tmp_path)
    state = root / ".posttrain" / "state"
    cache = state / "cache" / "pack"
    cache.mkdir(parents=True)
    (cache / "layout.json").write_text("cache", encoding="utf-8")
    (state / "executions" / "retained").mkdir(parents=True)
    (state / "executions" / "retained" / "receipt.json").write_text("receipt", encoding="utf-8")
    (state / "unknown").mkdir()
    (state / "unknown" / "note").write_text("preserve", encoding="utf-8")

    dry_run = prune_cache(load_project_layout(root))

    assert dry_run.reclaimable_bytes == len("cache")
    assert dry_run.removed_bytes == 0
    assert cache.is_dir()
    assert (state / "executions" / "retained" / "receipt.json").is_file()
    assert (state / "unknown" / "note").is_file()

    applied = prune_cache(load_project_layout(root), apply=True)

    assert applied.removed_bytes == len("cache")
    assert not cache.exists()
    assert (state / "executions" / "retained" / "receipt.json").is_file()
    assert (state / "unknown" / "note").is_file()


def test_cache_prune_handles_legacy_cache_and_rejects_unscoped_root(tmp_path: Path) -> None:
    root = _project(tmp_path)
    state = root / ".posttrain" / "state"
    legacy = state / "pack"
    legacy.mkdir(parents=True)
    (legacy / "layout.json").write_text("legacy", encoding="utf-8")

    report = prune_cache(load_project_layout(root), apply=True)

    assert report.removed_bytes == len("legacy")
    assert not legacy.exists()
    with pytest.raises(ContractError, match=".posttrain/state"):
        prune_cache(load_project_layout(root), state_root=tmp_path)
