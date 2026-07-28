"""Show machine admission placements and who waits behind them."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from ..context import CliState
from ..execution_config import resolve_admission_state_root
from ..execution_provider import execution_admission_service
from ..output import emit

_ACTIVE_ORPHAN_STATES = frozenset(
    {
        "waiting",
        "submitted",
        "submitting",
        "submission_failed",
        "terminal_pending_evidence",
    }
)


def register(app: typer.Typer) -> None:
    @app.command(
        "workers",
        help="show which runs hold machine placements and who waits behind them",
    )
    def workers_cmd(ctx: typer.Context) -> None:
        state: CliState = ctx.obj
        layout = state.layout()
        admission_root = resolve_admission_state_root()
        admission = execution_admission_service(layout)
        placements = admission.placements()
        orphan = _orphaned_project_ledger(layout.state)

        payload = {
            "admission_root": str(admission_root),
            "placements": [
                {
                    "key": placement.key,
                    "provider": placement.provider,
                    "holder": placement.holder,
                    "holder_state": placement.holder_state,
                    "holder_since": (
                        placement.holder_since.isoformat() if placement.holder_since is not None else None
                    ),
                    "holder_message": placement.holder_message,
                    "waiting": list(placement.waiting),
                }
                for placement in placements
            ],
            "orphaned_project_ledger": orphan,
        }

        if state.json_output:
            emit(state, payload, "")
            return

        lines = [f"Admission root: {admission_root}"]
        if not placements:
            lines.append("No placements held.")
        for placement in placements:
            holder = placement.holder or "(none)"
            state_label = placement.holder_state or "empty"
            since = placement.holder_since.isoformat() if placement.holder_since is not None else "-"
            lines.append(
                f"{placement.key}  provider={placement.provider or '-'}  "
                f"holder={holder}  state={state_label}  since={since}"
            )
            if placement.holder_message:
                lines.append(f"  message: {placement.holder_message}")
            if placement.waiting:
                lines.append(f"  waiting ({len(placement.waiting)}): {', '.join(placement.waiting)}")
            else:
                lines.append("  waiting: none")
        if orphan is not None:
            lines.append(
                f"WARN: project still has an active host placement in {orphan}; "
                "reconcile or cancel those runs, then remove the stale project ledger"
            )
        emit(state, payload, "\n".join(lines))


def _orphaned_project_ledger(project_state: Path) -> str | None:
    """Return a stale project admission path if it still holds a host key."""
    snapshot = project_state / "admission" / "queue.json"
    if not snapshot.is_file():
        return None
    try:
        payload = json.loads(snapshot.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    active = payload.get("active_by_key") or {}
    if isinstance(active, dict) and any(str(key).startswith("host:") for key in active):
        return str(snapshot.resolve())
    entries = payload.get("entries") or []
    if not isinstance(entries, list):
        return None
    if any(
        isinstance(entry, dict)
        and str(entry.get("admission_key", "")).startswith("host:")
        and entry.get("state") in _ACTIVE_ORPHAN_STATES
        for entry in entries
    ):
        return str(snapshot.resolve())
    return None


__all__ = ["register"]
