"""Resolve run identities by exact id, unambiguous prefix, or most recent."""

from __future__ import annotations

from dataclasses import dataclass

from posttrain.catalog import ProjectLayout
from posttrain.common import ContractError
from posttrain.execution import AdmissionEntry, ExecutionSubmissionStore, PurgeStore

from .execution_provider import execution_admission_service


@dataclass(frozen=True, slots=True)
class _RunRow:
    run_id: str
    state: str
    stamp: str


def purged_run_ids(layout: ProjectLayout) -> set[str]:
    """Return runs with a completed, unblocked cross-plane purge receipt."""

    store = PurgeStore(layout.state)
    if not store.root.is_dir():
        return set()
    purged: set[str] = set()
    for directory in store.root.iterdir():
        if not directory.is_dir() or not (directory / "receipt.json").is_file():
            continue
        try:
            plan = store.load_plan(directory.name)
            receipt = store.load_receipt(directory.name)
        except Exception:
            continue
        if plan.blockers or receipt.failed_action is not None:
            continue
        purged.update(plan.run_ids)
    return purged


def project_admission_entries(
    layout: ProjectLayout,
    *,
    include_purged: bool = False,
    entries: tuple[AdmissionEntry, ...] | None = None,
) -> dict[str, AdmissionEntry]:
    """Return current-project admission entries, excluding purged receipts.

    Admission state is machine-scoped and therefore contains runs submitted
    by other projects. A project run list must not present those as local
    work. Completed entries without a project submission are retained in the
    machine ledger for audit, but are omitted from the operational list after
    purge has removed their project receipt.
    """

    submissions = ExecutionSubmissionStore(layout.state).list_submissions()
    submission_ids = {submission.run_id for submission in submissions}
    purged = set() if include_purged else purged_run_ids(layout)
    selected: dict[str, AdmissionEntry] = {}
    admission_entries = entries if entries is not None else tuple(execution_admission_service(layout).list())
    for entry in admission_entries:
        try:
            project_id = entry.plan.request.run_spec.project_id
        except AttributeError:
            project_id = None
        if project_id != layout.project_id:
            continue
        if entry.run_id in purged:
            continue
        if entry.state == "completed" and entry.run_id not in submission_ids and not include_purged:
            continue
        selected[entry.run_id] = entry
    return selected


def known_run_ids(layout: ProjectLayout) -> tuple[str, ...]:
    """Return every known run id in strictly newest-first chronological order."""
    purged = purged_run_ids(layout)
    submissions = tuple(
        submission
        for submission in ExecutionSubmissionStore(layout.state).list_submissions()
        if submission.run_id not in purged
    )
    admission_entries = project_admission_entries(layout)
    submission_by_run = {submission.run_id: submission for submission in submissions}
    rows: list[_RunRow] = []
    for run_id in set(submission_by_run) | set(admission_entries):
        submission = submission_by_run.get(run_id)
        entry = admission_entries.get(run_id)
        queued = entry.queued_at.isoformat() if entry is not None else ""
        submitted = submission.submitted_at.isoformat() if submission is not None else ""
        rows.append(
            _RunRow(
                run_id=run_id,
                state=(entry.state if entry is not None else "legacy-submitted"),
                stamp=queued or submitted or "",
            )
        )
    rows.sort(key=lambda item: (item.stamp, item.run_id), reverse=True)
    return tuple(item.run_id for item in rows)


def resolve_run_id(
    layout: ProjectLayout,
    run_id: str | None,
    *,
    last: bool = False,
    known: tuple[str, ...] | None = None,
    exact_only: bool = False,
) -> str:
    """Resolve a run id from ``--last``, an exact id, or an unambiguous prefix.

    An exact id that is not yet in the shared ledger still passes through so
    submission-only and mocked admission paths keep working; downstream loaders
    report a precise missing-run error when it truly does not exist.
    """
    if exact_only and last:
        raise ContractError("mutating run commands require the complete canonical run id; --last is read-only")
    if last and run_id is not None:
        raise ContractError("pass either a run id or --last, not both")
    ids = known if known is not None else known_run_ids(layout)
    if last:
        if not ids:
            raise ContractError("no submitted runs to select with --last")
        return ids[0]
    if run_id is None:
        raise ContractError("pass a run id or --last")
    if run_id in ids:
        return run_id
    matches = [item for item in ids if item.startswith(run_id)]
    if exact_only and matches:
        raise ContractError("mutating run commands require the complete canonical run id")
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        preview = ", ".join(match[:16] for match in matches[:5])
        raise ContractError(f"{run_id!r} matches several runs: {preview}")
    return run_id


__all__ = ["known_run_ids", "project_admission_entries", "resolve_run_id"]
