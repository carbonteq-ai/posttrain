"""Classify and safely migrate project-local Posttrain state."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from posttrain.catalog import ProjectLayout
from posttrain.common import ContractError
from posttrain.execution import JobPackageManifest
from posttrain.execution_pack import (
    PackageMaterializationRecord,
    PackageMaterializationStore,
    digest_job_context,
    has_active_lease,
)

_CACHE_CHILDREN = frozenset({"datasets", "pack", "runs", "runtime-builds", "scratch"})
_TERMINAL_STATES = frozenset({"succeeded", "failed", "cancelled", "lost"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REMOTE_RECEIPT_SCHEMA = "posttrain.job-image-publication-receipt.v1"


def cache_root(layout: ProjectLayout) -> Path:
    """Return the rebuildable state root for one project."""

    return (layout.state / "cache").resolve()


def cache_path(layout: ProjectLayout, *parts: str) -> Path:
    """Return an owned rebuildable state path without permitting escapes."""

    root = cache_root(layout)
    result = root.joinpath(*parts).resolve()
    if not result.is_relative_to(root):
        raise ContractError("project cache path escapes the state cache root")
    return result


@dataclass(frozen=True, slots=True)
class StateMigrationReport:
    source: Path
    destination: Path
    dry_run: bool
    copied_execution_files: int
    moved_cache_entries: tuple[str, ...]
    protected_entries: tuple[str, ...]
    unresolved_runs: tuple[str, ...]

    @property
    def changed(self) -> bool:
        return self.copied_execution_files > 0 or bool(self.moved_cache_entries)

    def as_json(self) -> dict[str, object]:
        return {
            "source": str(self.source),
            "destination": str(self.destination),
            "dry_run": self.dry_run,
            "copied_execution_files": self.copied_execution_files,
            "moved_cache_entries": list(self.moved_cache_entries),
            "protected_entries": list(self.protected_entries),
            "unresolved_runs": list(self.unresolved_runs),
            "changed": self.changed,
        }


@dataclass(frozen=True, slots=True)
class CachePruneEntry:
    """One classified local-state entry considered by cache pruning."""

    path: Path
    classification: str
    reason: str
    bytes: int
    removed: bool


@dataclass(frozen=True, slots=True)
class CachePruneReport:
    """Explain a dry-run or applied cache prune without hiding protected state."""

    state_root: Path
    apply: bool
    entries: tuple[CachePruneEntry, ...]

    @property
    def reclaimable_bytes(self) -> int:
        return sum(entry.bytes for entry in self.entries if entry.classification == "rebuildable")

    @property
    def removed_bytes(self) -> int:
        return sum(entry.bytes for entry in self.entries if entry.removed)

    @property
    def protected_bytes(self) -> int:
        return sum(entry.bytes for entry in self.entries if entry.classification == "protected")

    @property
    def total_bytes(self) -> int:
        return sum(entry.bytes for entry in self.entries)

    def as_json(self) -> dict[str, object]:
        return {
            "state_root": str(self.state_root),
            "apply": self.apply,
            "reclaimable_bytes": self.reclaimable_bytes,
            "removed_bytes": self.removed_bytes,
            "protected_bytes": self.protected_bytes,
            "total_bytes": self.total_bytes,
            "entries": [
                {
                    "path": str(entry.path),
                    "classification": entry.classification,
                    "reason": entry.reason,
                    "bytes": entry.bytes,
                    "removed": entry.removed,
                }
                for entry in self.entries
            ],
        }


@dataclass(frozen=True, slots=True)
class LegacyPackMigrationEntry:
    """One historical context considered by the project-local migration."""

    path: Path
    package_key: str
    classification: str
    reason: str
    bytes: int
    record_committed: bool
    receipt_imported: bool
    discard_record_committed: bool


@dataclass(frozen=True, slots=True)
class LegacyPackMigrationReport:
    """Dry-run or applied result for compacting historical pack state."""

    state_root: Path
    apply: bool
    entries: tuple[LegacyPackMigrationEntry, ...]

    @property
    def migratable_bytes(self) -> int:
        return sum(entry.bytes for entry in self.entries if entry.classification == "migratable")

    @property
    def protected_bytes(self) -> int:
        return sum(entry.bytes for entry in self.entries if entry.classification == "protected")

    @property
    def records_committed(self) -> int:
        return sum(entry.record_committed for entry in self.entries)

    @property
    def receipts_imported(self) -> int:
        return sum(entry.receipt_imported for entry in self.entries)

    @property
    def discard_records_committed(self) -> int:
        return sum(entry.discard_record_committed for entry in self.entries)

    def as_json(self) -> dict[str, object]:
        return {
            "state_root": str(self.state_root),
            "apply": self.apply,
            "migratable_bytes": self.migratable_bytes,
            "protected_bytes": self.protected_bytes,
            "records_committed": self.records_committed,
            "receipts_imported": self.receipts_imported,
            "discard_records_committed": self.discard_records_committed,
            "entries": [
                {
                    "path": str(entry.path),
                    "package_key": entry.package_key,
                    "classification": entry.classification,
                    "reason": entry.reason,
                    "bytes": entry.bytes,
                    "record_committed": entry.record_committed,
                    "receipt_imported": entry.receipt_imported,
                    "discard_record_committed": entry.discard_record_committed,
                }
                for entry in self.entries
            ],
        }


def _protected_legacy_entry(path: Path, size: int, reason: str) -> LegacyPackMigrationEntry:
    return LegacyPackMigrationEntry(path, path.name, "protected", reason, size, False, False, False)


def migrate_legacy_pack(
    layout: ProjectLayout,
    *,
    verify_remote: Callable[[str], bool],
    apply: bool = False,
) -> LegacyPackMigrationReport:
    """Bridge verified legacy contexts into compact project-local records.

    The selected project's fixed state paths are the only read/write scope.
    Registry inspection is read-only. Context deletion remains the job of the
    ordinary cache pruner after this function has durably committed the record
    and copied its verified receipt out of the cache namespace.
    """

    state = layout.state.resolve()
    _validate_state_root(state)
    contexts = cache_path(layout, "pack", "contexts")
    if not contexts.exists():
        return LegacyPackMigrationReport(state, apply, ())
    if not contexts.is_dir() or contexts.is_symlink():
        raise ContractError("legacy context root must be a non-symlink directory")

    legacy_receipts = cache_path(layout, "pack", "publications")
    durable_receipts = state / "publications"
    receipts = _legacy_remote_receipts(legacy_receipts)
    verified: dict[str, bool] = {}
    store = PackageMaterializationStore((state / "packages" / "materializations").resolve())
    entries: list[LegacyPackMigrationEntry] = []
    for context in sorted(contexts.iterdir(), key=lambda value: value.name):
        size = _tree_bytes(context)
        if context.is_symlink() or not context.is_dir() or _SHA256.fullmatch(context.name) is None:
            entries.append(_protected_legacy_entry(context, size, "legacy context path is ambiguous or symlinked"))
            continue
        if has_active_lease(cache_path(layout, "pack", "leases"), context.name):
            entries.append(_protected_legacy_entry(context, size, "legacy context has an active lease"))
            continue
        try:
            manifest = JobPackageManifest.from_bytes((context / "package.json").read_bytes())
        except (OSError, ContractError):
            entries.append(_protected_legacy_entry(context, size, "legacy context package manifest is invalid"))
            continue
        if manifest.package_key != context.name or manifest.project_id != layout.project_id:
            entries.append(
                _protected_legacy_entry(context, size, "legacy context identity differs from the selected project")
            )
            continue
        candidates = receipts.get(context.name, ())
        receipt_path: Path | None = None
        receipt: dict[str, object] | None = None
        remote_verified = False
        if len(candidates) == 1:
            receipt_path, receipt = candidates[0]
            image = str(receipt["image"])
            if image not in verified:
                try:
                    verified[image] = verify_remote(image)
                except Exception:  # A diagnostic must not trust a failed registry probe.
                    verified[image] = False
            remote_verified = verified[image]
        record_committed = False
        receipt_imported = False
        discard_record_committed = False
        if remote_verified and receipt_path is not None and receipt is not None:
            try:
                record = PackageMaterializationRecord(
                    package_key=context.name,
                    context_digest=digest_job_context(context),
                    publication_key=str(receipt["publication_key"]),
                    manifest=manifest,
                )
            except ContractError:
                if apply:
                    _commit_legacy_discard_record(
                        state,
                        project_id=layout.project_id,
                        package_key=context.name,
                        manifest=manifest,
                        reason="context-digest-unavailable",
                    )
                    discard_record_committed = True
                reason = "validated legacy context is disposable despite filesystem drift"
            else:
                if apply:
                    store.commit(record)
                    record_committed = True
                    receipt_imported = _commit_verified_receipt(receipt_path, durable_receipts / receipt_path.name)
                reason = "verified remote publication can replace the assembled context"
        else:
            if apply:
                _commit_legacy_discard_record(
                    state,
                    project_id=layout.project_id,
                    package_key=context.name,
                    manifest=manifest,
                    reason=(
                        "ambiguous-publications"
                        if len(candidates) > 1
                        else "remote-publication-unverified"
                        if candidates
                        else "no-remote-publication-receipt"
                    ),
                )
                discard_record_committed = True
            reason = "durable run evidence makes the legacy assembled context disposable"
        entries.append(
            LegacyPackMigrationEntry(
                context,
                context.name,
                "migratable",
                reason,
                size,
                record_committed,
                receipt_imported,
                discard_record_committed,
            )
        )
    return LegacyPackMigrationReport(state, apply, tuple(entries))


def explain_cache(
    layout: ProjectLayout,
    selector: str,
    *,
    state_root: Path | None = None,
) -> tuple[CachePruneEntry, ...]:
    """Explain one cache object by path, basename, or content key."""

    if not selector.strip():
        raise ContractError("cache object selector cannot be empty")
    report = prune_cache(layout, state_root=state_root)
    root = report.state_root
    requested = Path(selector).expanduser()
    candidates = tuple(
        entry
        for entry in report.entries
        if entry.path.name == selector
        or entry.path.stem == selector
        or entry.path == requested
        or entry.path == (root / requested).resolve()
    )
    if not candidates:
        raise ContractError(f"no cache object matches {selector!r}")
    return candidates


def migrate_state(
    layout: ProjectLayout,
    *,
    source_project_root: Path | None = None,
    dry_run: bool = False,
) -> StateMigrationReport:
    """Split local state and copy verified execution control records.

    A source project is never deleted.  Cache entries are only reorganized
    when migrating the selected project in place; a relocation copies durable
    `executions/` records and deliberately leaves the old cache rebuildable.
    """

    destination = layout.state.resolve()
    source = (
        (source_project_root.resolve() / ".posttrain" / "state") if source_project_root is not None else destination
    )
    if source_project_root is not None and not source.is_relative_to(source_project_root.resolve()):
        raise ContractError("source project state path is invalid")
    if not source.exists():
        return StateMigrationReport(source, destination, dry_run, 0, (), (), ())
    if not source.is_dir() or source.is_symlink():
        raise ContractError(f"project state root is not a safe directory: {source}")

    unresolved = _unresolved_runs(source / "executions")
    if unresolved:
        raise ContractError("state migration refuses unresolved executions: " + ", ".join(unresolved))

    copied = _copy_executions(source / "executions", destination / "executions", dry_run=dry_run)
    moved: list[str] = []
    protected: list[str] = []
    if source == destination:
        for child in sorted(source.iterdir(), key=lambda value: value.name):
            if child.name in {"cache", "executions"}:
                continue
            if child.name in _CACHE_CHILDREN:
                target = cache_root(layout) / child.name
                if target.exists() and not _same_tree(child, target):
                    raise ContractError(f"state cache migration conflicts at {target}")
                moved.append(child.name)
                if not dry_run and not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(child), str(target))
            else:
                protected.append(child.name)
    return StateMigrationReport(
        source,
        destination,
        dry_run,
        copied,
        tuple(moved),
        tuple(protected),
        (),
    )


def prune_cache(
    layout: ProjectLayout,
    *,
    state_root: Path | None = None,
    apply: bool = False,
) -> CachePruneReport:
    """Classify and optionally remove only recognized rebuildable cache trees.

    The default project state supports the current ``state/cache`` layout and
    the old direct-child layout so a maintainer can clean up after migration.
    Execution receipts and any unknown or symlinked entry are reported as
    protected.  ``apply`` must be explicit; dry-runs never mutate the tree.
    """

    root = (state_root or layout.state).resolve()
    _validate_state_root(root)
    if not root.exists():
        return CachePruneReport(root, apply, ())

    entries: list[CachePruneEntry] = []
    for child in sorted(root.iterdir(), key=lambda value: value.name):
        if child.is_symlink():
            entries.append(CachePruneEntry(child, "protected", "symlinked state entries are never traversed", 0, False))
            continue
        if child.name == "cache" and child.is_dir():
            entries.extend(_classify_cache_root(child, apply=apply))
            continue
        if child.name in _CACHE_CHILDREN and child.is_dir():
            entries.append(_prune_entry(child, "legacy rebuildable cache", apply=apply))
            continue
        entries.append(CachePruneEntry(child, "protected", "durable or unknown state entry", _tree_bytes(child), False))
    return CachePruneReport(root, apply, tuple(entries))


def _validate_state_root(root: Path) -> None:
    if root.name != "state" or root.parent.name != ".posttrain":
        raise ContractError("cache prune state root must be a .posttrain/state directory")
    if root.exists() and (not root.is_dir() or root.is_symlink()):
        raise ContractError("cache prune state root must be a non-symlink directory")


def _classify_cache_root(cache: Path, *, apply: bool) -> list[CachePruneEntry]:
    entries: list[CachePruneEntry] = []
    for child in sorted(cache.iterdir(), key=lambda value: value.name):
        if child.is_symlink():
            entries.append(CachePruneEntry(child, "protected", "symlinked cache entry is never traversed", 0, False))
        elif child.name == "pack" and child.is_dir():
            entries.extend(_classify_pack_cache(child, apply=apply))
        elif child.name in _CACHE_CHILDREN and child.is_dir():
            entries.append(_prune_entry(child, "rebuildable cache", apply=apply))
        else:
            entries.append(CachePruneEntry(child, "protected", "unknown cache entry", _tree_bytes(child), False))
    return entries


def _classify_pack_cache(pack: Path, *, apply: bool) -> list[CachePruneEntry]:
    """Classify pack material without treating durable evidence as cache.

    Package contexts and source snapshots still lack compact replacement
    records, so they remain protected during this transitional migration.  A
    local OCI layout is reclaimable only when its sibling receipt proves that
    the same image was published to the configured registry.  Hidden staging
    directories are always disposable once no pack process is using them.
    """

    entries: list[CachePruneEntry] = []
    for child in sorted(pack.iterdir(), key=lambda value: value.name):
        if child.is_symlink():
            entries.append(CachePruneEntry(child, "protected", "symlinked pack entries are never traversed", 0, False))
            continue
        if child.name == "contexts" and child.is_dir():
            for staged in sorted(child.iterdir(), key=lambda value: value.name):
                if staged.is_symlink():
                    entries.append(
                        CachePruneEntry(staged, "protected", "symlinked context entries are never traversed", 0, False)
                    )
                elif staged.is_dir() and staged.name.startswith((".job-context-stage-", ".job-context-work-")):
                    entries.append(_prune_entry(staged, "abandoned pack staging directory", apply=apply))
                else:
                    if has_active_lease(pack / "leases", staged.name):
                        entries.append(
                            CachePruneEntry(
                                staged,
                                "protected",
                                "assembled context has an active lease",
                                _tree_bytes(staged),
                                False,
                            )
                        )
                    elif _has_migrated_context_record(pack, staged.name) or _has_legacy_discard_record(
                        pack, staged.name
                    ):
                        entries.append(_prune_entry(staged, "migrated assembled context", apply=apply))
                    else:
                        entries.append(
                            CachePruneEntry(
                                staged,
                                "protected",
                                "assembled contexts need compact package records before pruning",
                                _tree_bytes(staged),
                                False,
                            )
                        )
            continue
        if child.name == "local-layouts" and child.is_dir():
            entries.extend(
                _classify_local_layouts(
                    child,
                    lease_root=pack / "leases",
                    apply=apply,
                )
            )
            continue
        if child.name == "publications" and child.is_dir():
            entries.extend(_classify_publications(child, apply=apply))
            continue
        entries.append(
            CachePruneEntry(
                child,
                "protected",
                "pack material needs a durable record or migration policy",
                _tree_bytes(child),
                False,
            )
        )
    return entries


def _classify_publications(publications: Path, *, apply: bool) -> list[CachePruneEntry]:
    entries: list[CachePruneEntry] = []
    layouts = publications / "local-layouts"
    for child in sorted(publications.iterdir(), key=lambda value: value.name):
        if child.is_symlink():
            entries.append(
                CachePruneEntry(child, "protected", "symlinked publication entries are never traversed", 0, False)
            )
            continue
        if child != layouts or not child.is_dir():
            entries.append(
                CachePruneEntry(
                    child,
                    "protected",
                    "publication receipts are durable evidence",
                    _tree_bytes(child),
                    False,
                )
            )
            continue
        entries.extend(
            _classify_local_layouts(
                child,
                lease_root=publications.parent / "leases",
                apply=apply,
            )
        )
    return entries


def _classify_local_layouts(
    layouts: Path,
    *,
    lease_root: Path,
    apply: bool,
) -> list[CachePruneEntry]:
    entries: list[CachePruneEntry] = []
    for layout in sorted(layouts.iterdir(), key=lambda value: value.name):
        if layout.is_symlink():
            entries.append(
                CachePruneEntry(layout, "protected", "symlinked local layouts are never traversed", 0, False)
            )
        elif layout.is_dir() and has_active_lease(lease_root, layout.name):
            entries.append(
                CachePruneEntry(
                    layout,
                    "protected",
                    "local layout has an active lease",
                    _tree_bytes(layout),
                    False,
                )
            )
        elif layout.is_dir():
            entries.append(_prune_entry(layout, "internal local OCI transport layout", apply=apply))
        else:
            entries.append(
                CachePruneEntry(
                    layout,
                    "protected",
                    "local layout has no verified registry publication receipt",
                    _tree_bytes(layout),
                    False,
                )
            )
    return entries


def _has_registry_publication_receipt(
    publications: Path,
    publication_key: str,
    *,
    package_key: str | None = None,
) -> bool:
    receipt = publications / f"{publication_key}.json"
    if not receipt.is_file() or receipt.is_symlink():
        return False
    try:
        payload = json.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    image = payload.get("image")
    return (
        payload.get("schema") == _REMOTE_RECEIPT_SCHEMA
        and isinstance(image, str)
        and "@sha256:" in image
        and payload.get("publication_key", publication_key) == publication_key
        and (package_key is None or payload.get("package_key") == package_key)
    )


def _has_migrated_context_record(pack: Path, package_key: str) -> bool:
    state = pack.parent.parent
    record_path = state / "packages" / "materializations" / f"{package_key}.json"
    if not record_path.is_file() or record_path.is_symlink():
        return False
    try:
        record = PackageMaterializationRecord.from_bytes(record_path.read_bytes())
    except (OSError, ContractError):
        return False
    return record.package_key == package_key and _has_registry_publication_receipt(
        state / "publications",
        record.publication_key,
        package_key=package_key,
    )


def _has_legacy_discard_record(pack: Path, package_key: str) -> bool:
    state = pack.parent.parent
    path = state / "migrations" / "legacy-pack" / "contexts" / f"{package_key}.json"
    if not path.is_file() or path.is_symlink():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return (
        isinstance(payload, dict)
        and payload.get("schema") == "posttrain.legacy-pack-discard.v1"
        and payload.get("package_key") == package_key
        and isinstance(payload.get("project_id"), str)
        and _SHA256.fullmatch(str(payload.get("manifest_digest"))) is not None
    )


def _legacy_remote_receipts(root: Path) -> dict[str, tuple[tuple[Path, dict[str, object]], ...]]:
    found: dict[str, list[tuple[Path, dict[str, object]]]] = {}
    if not root.is_dir() or root.is_symlink():
        return {}
    for path in sorted(root.glob("*.json")):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or payload.get("schema") != _REMOTE_RECEIPT_SCHEMA:
            continue
        package_key = payload.get("package_key")
        publication_key = payload.get("publication_key")
        image = payload.get("image")
        if (
            not isinstance(package_key, str)
            or _SHA256.fullmatch(package_key) is None
            or not isinstance(publication_key, str)
            or _SHA256.fullmatch(publication_key) is None
            or path.name != f"{publication_key}.json"
            or not isinstance(image, str)
            or "@sha256:" not in image
        ):
            continue
        found.setdefault(package_key, []).append((path, payload))
    return {key: tuple(value) for key, value in found.items()}


def _commit_verified_receipt(source: Path, destination: Path) -> bool:
    encoded = source.read_bytes()
    if destination.exists():
        if destination.is_symlink() or not destination.is_file() or destination.read_bytes() != encoded:
            raise ContractError(f"durable publication receipt conflicts at {destination}")
        return False
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    destination.parent.chmod(0o700)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o600)
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.is_symlink() or not destination.is_file() or destination.read_bytes() != encoded:
                raise ContractError(f"durable publication receipt conflicts at {destination}") from None
    finally:
        temporary.unlink(missing_ok=True)
    return True


def _commit_legacy_discard_record(
    state: Path,
    *,
    project_id: str,
    package_key: str,
    manifest: JobPackageManifest,
    reason: str,
) -> Path:
    root = state / "migrations" / "legacy-pack" / "contexts"
    payload = {
        "schema": "posttrain.legacy-pack-discard.v1",
        "project_id": project_id,
        "package_key": package_key,
        "manifest_digest": hashlib.sha256(manifest.to_bytes()).hexdigest(),
        "reason": reason,
    }
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    destination = root / f"{package_key}.json"
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root.chmod(0o700)
    if destination.exists():
        if destination.is_symlink() or not destination.is_file() or destination.read_bytes() != encoded:
            raise ContractError(f"legacy discard record conflicts at {destination}")
        return destination
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{package_key}.", dir=root)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o600)
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.is_symlink() or not destination.is_file() or destination.read_bytes() != encoded:
                raise ContractError(f"legacy discard record conflicts at {destination}") from None
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _prune_entry(path: Path, reason: str, *, apply: bool) -> CachePruneEntry:
    bytes_before = _tree_bytes(path)
    if apply:
        shutil.rmtree(path)
    return CachePruneEntry(path, "rebuildable", reason, bytes_before, apply)


def _tree_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if not path.is_dir():
        return 0
    total = 0
    for value in path.rglob("*"):
        if value.is_symlink():
            continue
        if value.is_file():
            total += value.stat().st_size
    return total


def _unresolved_runs(root: Path) -> tuple[str, ...]:
    if not root.is_dir():
        return ()
    unresolved: list[str] = []
    for run in sorted(path for path in root.iterdir() if path.is_dir() and not path.is_symlink()):
        submission = run / "submission.json"
        if not submission.is_file():
            continue
        reconciliation = run / "reconciliation.json"
        try:
            payload = json.loads(reconciliation.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            unresolved.append(run.name)
            continue
        record = payload.get("provider_record")
        state = record.get("state") if isinstance(record, dict) else None
        if payload.get("schema") != "posttrain.execution-reconciliation.v1" or state not in _TERMINAL_STATES:
            unresolved.append(run.name)
    return tuple(unresolved)


def _copy_executions(source: Path, destination: Path, *, dry_run: bool) -> int:
    if not source.is_dir():
        return 0
    copied = 0
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ContractError(f"execution state contains a symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        target = destination / relative
        if target.exists():
            if not target.is_file() or target.read_bytes() != path.read_bytes():
                raise ContractError(f"execution state migration conflicts at {target}")
            continue
        copied += 1
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    return copied


def _same_tree(left: Path, right: Path) -> bool:
    if left.is_file() or right.is_file():
        return left.is_file() and right.is_file() and left.read_bytes() == right.read_bytes()
    left_files = {path.relative_to(left): path for path in left.rglob("*") if path.is_file()}
    right_files = {path.relative_to(right): path for path in right.rglob("*") if path.is_file()}
    return set(left_files) == set(right_files) and all(
        left_files[relative].read_bytes() == right_files[relative].read_bytes() for relative in left_files
    )
