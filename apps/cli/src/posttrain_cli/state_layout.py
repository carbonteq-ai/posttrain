"""Classify and safely migrate project-local Posttrain state."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from posttrain.catalog import ProjectLayout
from posttrain.common import ContractError
from posttrain.execution import JobPackageManifest
from posttrain.execution_pack import (
    PackageMaterializationRecord,
    PackageMaterializationStore,
    digest_context_tree,
    has_active_lease,
)

_CACHE_CHILDREN = frozenset({"datasets", "pack", "runs", "runtime-builds", "scratch"})
_TERMINAL_STATES = frozenset({"succeeded", "failed", "cancelled", "lost"})


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
    """One retained legacy context considered for compact-record migration."""

    package_key: str
    path: Path
    bytes: int
    classification: str
    reason: str
    publication_key: str | None = None
    image: str | None = None
    record_path: Path | None = None
    applied: bool = False


@dataclass(frozen=True, slots=True)
class LegacyPackMigrationReport:
    """Dry-run-first result for legacy package-record migration."""

    state_root: Path
    apply: bool
    entries: tuple[LegacyPackMigrationEntry, ...]

    @property
    def migratable_bytes(self) -> int:
        return sum(entry.bytes for entry in self.entries if entry.classification in {"migratable", "migrated"})

    @property
    def migrated_bytes(self) -> int:
        return sum(entry.bytes for entry in self.entries if entry.applied)

    def as_json(self) -> dict[str, object]:
        return {
            "state_root": str(self.state_root),
            "apply": self.apply,
            "migratable_bytes": self.migratable_bytes,
            "migrated_bytes": self.migrated_bytes,
            "entries": [
                {
                    "package_key": entry.package_key,
                    "path": str(entry.path),
                    "bytes": entry.bytes,
                    "classification": entry.classification,
                    "reason": entry.reason,
                    "publication_key": entry.publication_key,
                    "image": entry.image,
                    "record_path": str(entry.record_path) if entry.record_path is not None else None,
                    "applied": entry.applied,
                }
                for entry in self.entries
            ],
        }


def migrate_legacy_pack_cache(
    layout: ProjectLayout,
    *,
    verify_registry_image: Callable[[str], bool],
    apply: bool = False,
) -> LegacyPackMigrationReport:
    """Create compact records only for contexts backed by a live registry image.

    The registry verifier is supplied by the application so this state module
    remains independent of Docker and BuildKit. The context itself is never
    removed here; a subsequent cache prune independently rechecks the compact
    record and durable receipt before classifying it as rebuildable.
    """

    state = layout.state.resolve()
    _validate_state_root(state)
    contexts = state / "cache" / "pack" / "contexts"
    if not contexts.is_dir():
        return LegacyPackMigrationReport(state, apply, ())
    if contexts.is_symlink():
        raise ContractError("legacy context root must be a non-symlink directory")

    records_root = state / "packages" / "materializations"
    store = PackageMaterializationStore(records_root.resolve())
    receipt_roots = (state / "publications", state / "cache" / "pack" / "publications")
    entries: list[LegacyPackMigrationEntry] = []
    for context in sorted(contexts.iterdir(), key=lambda value: value.name):
        size = _tree_bytes(context)
        if context.is_symlink() or not context.is_dir():
            entries.append(
                LegacyPackMigrationEntry(context.name, context, size, "protected", "context is not a regular directory")
            )
            continue
        if has_active_lease(state / "cache" / "pack" / "leases", context.name):
            entries.append(
                LegacyPackMigrationEntry(context.name, context, size, "protected", "context has an active lease")
            )
            continue
        try:
            manifest = JobPackageManifest.from_bytes((context / "package.json").read_bytes())
        except (OSError, ContractError) as error:
            entries.append(
                LegacyPackMigrationEntry(
                    context.name,
                    context,
                    size,
                    "protected",
                    f"package manifest is invalid: {error}",
                )
            )
            continue
        if manifest.package_key != context.name:
            entries.append(
                LegacyPackMigrationEntry(
                    context.name,
                    context,
                    size,
                    "protected",
                    "context name differs from the manifest package key",
                )
            )
            continue

        receipts = _registry_receipts_for_package(receipt_roots, context.name)
        existing_records = tuple(
            record
            for path in (
                records_root / f"{context.name}.json",
                *(records_root / f"{publication_key}.json" for publication_key, _ in receipts),
            )
            if (record := _load_materialization_record(path)) is not None and record.package_key == context.name
        )
        if existing_records:
            matching = next(
                (
                    item
                    for existing in existing_records
                    for item in receipts
                    if item[0] == existing.publication_key and verify_registry_image(item[1])
                ),
                None,
            )
            if matching is not None:
                entries.append(
                    LegacyPackMigrationEntry(
                        context.name,
                        context,
                        size,
                        "migrated",
                        "compact record and live registry image already exist",
                        matching[0],
                        matching[1],
                        records_root / f"{context.name}.json",
                    )
                )
            else:
                entries.append(
                    LegacyPackMigrationEntry(
                        context.name,
                        context,
                        size,
                        "protected",
                        "existing compact record has no matching live registry image",
                    )
                )
            continue

        verified = next((item for item in receipts if verify_registry_image(item[1])), None)
        if verified is None:
            entries.append(
                LegacyPackMigrationEntry(
                    context.name,
                    context,
                    size,
                    "protected",
                    "no matching registry receipt resolves to a live immutable image",
                )
            )
            continue

        record_path = records_root / f"{context.name}.json"
        applied = False
        if apply:
            record_path = store.commit(
                PackageMaterializationRecord(
                    package_key=context.name,
                    context_digest=digest_context_tree(context),
                    publication_key=verified[0],
                    manifest=manifest,
                )
            )
            applied = True
        entries.append(
            LegacyPackMigrationEntry(
                context.name,
                context,
                size,
                "migrated" if applied else "migratable",
                "live registry image permits compact-record migration",
                verified[0],
                verified[1],
                record_path,
                applied,
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
                elif _context_has_registry_backed_record(pack, staged.name):
                    entries.append(_prune_entry(staged, "registry-backed migrated package context", apply=apply))
                else:
                    reason = (
                        "assembled context has an active lease"
                        if has_active_lease(pack / "leases", staged.name)
                        else "assembled contexts need compact package records before pruning"
                    )
                    entries.append(
                        CachePruneEntry(
                            staged,
                            "protected",
                            reason,
                            _tree_bytes(staged),
                            False,
                        )
                    )
            continue
        if child.name == "local-layouts" and child.is_dir():
            receipt_root = pack.parent.parent / "publications"
            entries.extend(
                _classify_local_layouts(
                    child,
                    receipt_root=receipt_root,
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
                receipt_root=publications,
                lease_root=publications.parent / "leases",
                apply=apply,
            )
        )
    return entries


def _classify_local_layouts(
    layouts: Path,
    *,
    receipt_root: Path,
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
        elif layout.is_dir() and _has_registry_publication_receipt(receipt_root, layout.name):
            entries.append(_prune_entry(layout, "registry-backed local OCI layout", apply=apply))
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


def _has_registry_publication_receipt(publications: Path, publication_key: str) -> bool:
    receipt = publications / f"{publication_key}.json"
    if not receipt.is_file() or receipt.is_symlink():
        return False
    try:
        payload = json.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    image = payload.get("image")
    return (
        payload.get("schema") == "posttrain.job-image-publication-receipt.v1"
        and isinstance(image, str)
        and "@sha256:" in image
    )


def _context_has_registry_backed_record(pack: Path, package_key: str) -> bool:
    state = pack.parent.parent
    receipts = _registry_receipts_for_package(
        (state / "publications", pack / "publications"),
        package_key,
    )
    records_root = state / "packages" / "materializations"
    for publication_key, _image in receipts:
        record = _load_materialization_record(records_root / f"{publication_key}.json")
        if record is not None and record.package_key == package_key and record.publication_key == publication_key:
            return True
    legacy_record = _load_materialization_record(records_root / f"{package_key}.json")
    return (
        legacy_record is not None
        and legacy_record.package_key == package_key
        and any(publication_key == legacy_record.publication_key for publication_key, _image in receipts)
    )


def _load_materialization_record(path: Path) -> PackageMaterializationRecord | None:
    if not path.is_file() or path.is_symlink():
        return None
    try:
        return PackageMaterializationRecord.from_bytes(path.read_bytes())
    except (OSError, ContractError):
        return None


def _registry_receipts_for_package(
    roots: tuple[Path, ...],
    package_key: str,
) -> tuple[tuple[str, str], ...]:
    found: set[tuple[str, str]] = set()
    for root in roots:
        if not root.is_dir() or root.is_symlink():
            continue
        for path in root.glob("*.json"):
            if path.is_symlink() or not path.is_file() or path.name.endswith(".local.json"):
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            publication_key = payload.get("publication_key") if isinstance(payload, dict) else None
            image = payload.get("image") if isinstance(payload, dict) else None
            if (
                payload.get("schema") == "posttrain.job-image-publication-receipt.v1"
                and payload.get("package_key") == package_key
                and isinstance(publication_key, str)
                and path.name == f"{publication_key}.json"
                and isinstance(image, str)
                and "@sha256:" in image
            ):
                found.add((publication_key, image))
    return tuple(sorted(found))


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
