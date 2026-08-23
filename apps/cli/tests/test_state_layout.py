from __future__ import annotations

import json
from pathlib import Path

import pytest
from posttrain.catalog import load_project_layout
from posttrain.common import ContractError
from posttrain.execution import JobPackageManifest, RuntimeImageRef
from posttrain.execution_pack import CacheLease
from posttrain_cli.state_layout import cache_path, explain_cache, migrate_legacy_pack, migrate_state, prune_cache


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    control = root / ".posttrain"
    control.mkdir(parents=True)
    (control / "project.toml").write_text('schema_version = 1\nproject_id = "state-test"\n', encoding="utf-8")
    return root


def _manifest(*, project_id: str = "state-test") -> JobPackageManifest:
    return JobPackageManifest(
        project_id=project_id,
        work_package_id="work",
        job_id="job",
        job_definition_id="definition",
        job_kind="train.sft",
        resolved_inputs_digest="a" * 64,
        framework_source_digest="a" * 64,
        project_source_digest="a" * 64,
        runtime_dependencies_digest="a" * 64,
        code_requirements_digest="a" * 64,
        resolved_config_digest="a" * 64,
        project_config_digest="a" * 64,
        universal_image=RuntimeImageRef(f"registry.example/base@sha256:{'b' * 64}"),
        kind_image=RuntimeImageRef(f"registry.example/kind@sha256:{'c' * 64}"),
        runtime_variant="supervised",
    )


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
    cache = state / "cache" / "pack" / "contexts"
    cache.mkdir(parents=True)
    (cache / ".job-context-stage-old").mkdir()
    (cache / ".job-context-stage-old" / "payload").write_text("cache", encoding="utf-8")
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
    assert not (cache / ".job-context-stage-old").exists()
    assert cache.is_dir()
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


def test_cache_prune_removes_internal_local_layouts_regardless_of_remote_receipt(tmp_path: Path) -> None:
    root = _project(tmp_path)
    publications = root / ".posttrain" / "state" / "cache" / "pack" / "publications"
    layouts = publications / "local-layouts"
    verified = layouts / "verified"
    unpublished = layouts / "unpublished"
    verified.mkdir(parents=True)
    unpublished.mkdir()
    (verified / "blob").write_text("published", encoding="utf-8")
    (unpublished / "blob").write_text("keep", encoding="utf-8")
    (publications / "verified.json").write_text(
        json.dumps(
            {
                "schema": "posttrain.job-image-publication-receipt.v1",
                "image": "registry.example/job@sha256:" + "a" * 64,
            }
        ),
        encoding="utf-8",
    )

    report = prune_cache(load_project_layout(root), apply=True)

    assert report.removed_bytes == len("published") + len("keep")
    assert not verified.exists()
    assert not unpublished.exists()
    assert (publications / "verified.json").is_file()


def test_cache_prune_handles_local_layouts_with_receipts_outside_cache(tmp_path: Path) -> None:
    root = _project(tmp_path)
    state = root / ".posttrain" / "state"
    layouts = state / "cache" / "pack" / "local-layouts"
    verified = layouts / "verified"
    verified.mkdir(parents=True)
    (verified / "blob").write_text("published", encoding="utf-8")
    publications = state / "publications"
    publications.mkdir()
    (publications / "verified.json").write_text(
        json.dumps(
            {
                "schema": "posttrain.job-image-publication-receipt.v1",
                "image": "registry.example/job@sha256:" + "a" * 64,
            }
        ),
        encoding="utf-8",
    )

    report = prune_cache(load_project_layout(root), apply=True)

    assert report.removed_bytes == len("published")
    assert not verified.exists()
    assert (publications / "verified.json").is_file()


def test_cache_prune_protects_a_local_layout_with_an_active_lease(tmp_path: Path) -> None:
    root = _project(tmp_path)
    state = root / ".posttrain" / "state"
    layouts = state / "cache" / "pack" / "local-layouts"
    verified = layouts / ("a" * 64)
    verified.mkdir(parents=True)
    (verified / "blob").write_text("published", encoding="utf-8")
    publications = state / "publications"
    publications.mkdir()
    (publications / f"{'a' * 64}.json").write_text(
        json.dumps(
            {
                "schema": "posttrain.job-image-publication-receipt.v1",
                "image": "registry.example/job@sha256:" + "a" * 64,
            }
        ),
        encoding="utf-8",
    )
    lease = CacheLease.acquire(state / "cache" / "pack" / "leases", "a" * 64)

    report = prune_cache(load_project_layout(root), apply=True)

    assert report.removed_bytes == 0
    assert verified.is_dir()
    lease.release()


def test_cache_explain_matches_a_content_key_and_reports_protection(tmp_path: Path) -> None:
    root = _project(tmp_path)
    state = root / ".posttrain" / "state"
    context = state / "cache" / "pack" / "contexts" / ("b" * 64)
    context.mkdir(parents=True)
    (context / "package.json").write_text("{}", encoding="utf-8")

    entries = explain_cache(load_project_layout(root), "b" * 64)

    assert len(entries) == 1
    assert entries[0].classification == "protected"
    assert "compact package records" in entries[0].reason


def test_legacy_pack_migration_commits_verified_record_then_prune_removes_context(tmp_path: Path) -> None:
    root = _project(tmp_path)
    state = root / ".posttrain" / "state"
    manifest = _manifest()
    context = state / "cache" / "pack" / "contexts" / manifest.package_key
    context.mkdir(parents=True)
    (context / "package.json").write_bytes(manifest.to_bytes())
    (context / "payload").write_text("legacy", encoding="utf-8")
    publication_key = "d" * 64
    publications = state / "cache" / "pack" / "publications"
    publications.mkdir()
    receipt = publications / f"{publication_key}.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "posttrain.job-image-publication-receipt.v1",
                "package_key": manifest.package_key,
                "publication_key": publication_key,
                "image": f"registry.example/job@sha256:{'e' * 64}",
            }
        ),
        encoding="utf-8",
    )
    layout = load_project_layout(root)

    dry_run = migrate_legacy_pack(layout, verify_remote=lambda _image: True)

    assert dry_run.migratable_bytes == len(manifest.to_bytes()) + len("legacy")
    assert dry_run.records_committed == 0
    assert not (state / "packages" / "materializations").exists()

    applied = migrate_legacy_pack(layout, verify_remote=lambda _image: True, apply=True)

    assert applied.records_committed == 1
    assert applied.receipts_imported == 1
    assert (state / "packages" / "materializations" / f"{manifest.package_key}.json").is_file()
    assert (state / "publications" / receipt.name).is_file()
    pruned = prune_cache(layout, apply=True)
    assert pruned.removed_bytes >= len("legacy")
    assert not context.exists()


def test_legacy_pack_migration_journals_context_without_a_receipt(tmp_path: Path) -> None:
    root = _project(tmp_path)
    state = root / ".posttrain" / "state"
    manifest = _manifest()
    context = state / "cache" / "pack" / "contexts" / manifest.package_key
    context.mkdir(parents=True)
    (context / "package.json").write_bytes(manifest.to_bytes())
    layout = load_project_layout(root)

    report = migrate_legacy_pack(layout, verify_remote=lambda _image: False, apply=True)

    assert report.discard_records_committed == 1
    assert report.records_committed == 0
    marker = state / "migrations" / "legacy-pack" / "contexts" / f"{manifest.package_key}.json"
    assert marker.is_file()
    prune_cache(layout, apply=True)
    assert not context.exists()


def test_legacy_pack_migration_protects_an_active_context_lease(tmp_path: Path) -> None:
    root = _project(tmp_path)
    state = root / ".posttrain" / "state"
    manifest = _manifest()
    context = state / "cache" / "pack" / "contexts" / manifest.package_key
    context.mkdir(parents=True)
    (context / "package.json").write_bytes(manifest.to_bytes())
    lease = CacheLease.acquire(state / "cache" / "pack" / "leases", manifest.package_key)
    try:
        report = migrate_legacy_pack(load_project_layout(root), verify_remote=lambda _image: True, apply=True)
    finally:
        lease.release()

    assert report.protected_bytes == len(manifest.to_bytes())
    assert report.discard_records_committed == 0
    assert context.is_dir()


def test_legacy_pack_migration_uses_discard_record_when_context_digest_is_unavailable(tmp_path: Path) -> None:
    root = _project(tmp_path)
    state = root / ".posttrain" / "state"
    manifest = _manifest()
    context = state / "cache" / "pack" / "contexts" / manifest.package_key
    context.mkdir(parents=True)
    (context / "package.json").write_bytes(manifest.to_bytes())
    (context / ".env").write_text("not-retained", encoding="utf-8")
    publication_key = "d" * 64
    publications = state / "cache" / "pack" / "publications"
    publications.mkdir()
    (publications / f"{publication_key}.json").write_text(
        json.dumps(
            {
                "schema": "posttrain.job-image-publication-receipt.v1",
                "package_key": manifest.package_key,
                "publication_key": publication_key,
                "image": f"registry.example/job@sha256:{'e' * 64}",
            }
        ),
        encoding="utf-8",
    )

    report = migrate_legacy_pack(load_project_layout(root), verify_remote=lambda _image: True, apply=True)

    assert report.discard_records_committed == 1
    assert report.records_committed == 0
    assert "filesystem drift" in report.entries[0].reason
