"""Resolve run identities by exact id, unambiguous prefix, or most recent."""

from __future__ import annotations

from dataclasses import dataclass

from posttrain.catalog import ProjectLayout
from posttrain.common import ContractError
from posttrain.execution import ExecutionSubmissionStore

from .execution_provider import execution_admission_service


@dataclass(frozen=True, slots=True)
class _RunRow:
    run_id: str
    state: str
    stamp: str


def known_run_ids(layout: ProjectLayout) -> tuple[str, ...]:
    """Return every known run id, newest first within each admission priority."""
    submissions = ExecutionSubmissionStore(layout.state).list_submissions()
    admission_entries = {entry.run_id: entry for entry in execution_admission_service(layout).list()}
    submission_by_run = {submission.run_id: submission for submission in submissions}
    admission_priority = {
        "waiting": 0,
        "submitting": 1,
        "submission_failed": 2,
        "submitted": 3,
        "terminal_pending_evidence": 4,
        "completed": 5,
        "cancelled": 6,
    }
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
    rows.sort(key=lambda item: admission_priority.get(item.state, 7))
    return tuple(item.run_id for item in rows)


def resolve_run_id(
    layout: ProjectLayout,
    run_id: str | None,
    *,
    last: bool = False,
) -> str:
    """Resolve a run id from ``--last``, an exact id, or an unambiguous prefix."""
    if last and run_id is not None:
        raise ContractError("pass either a run id or --last, not both")
    ids = known_run_ids(layout)
    if last:
        if not ids:
            raise ContractError("no submitted runs to select with --last")
        return ids[0]
    if run_id is None:
        raise ContractError("pass a run id or --last")
    if run_id in ids:
        return run_id
    matches = [item for item in ids if item.startswith(run_id)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ContractError(f"no run matches {run_id!r}")
    preview = ", ".join(match[:16] for match in matches[:5])
    raise ContractError(f"{run_id!r} matches several runs: {preview}")


__all__ = ["known_run_ids", "resolve_run_id"]
