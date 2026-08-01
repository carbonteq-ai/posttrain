"""Foreground lifecycle controller over machine-scoped admission state."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Annotated, Any

import typer
from posttrain.catalog import ProjectLayout, load_project_layout
from posttrain.execution import ExecutionSubmissionStore, reconcile_execution, save_reconciliation

from ..context import CliState
from ..execution_provider import execution_admission_service, reconciliation_source_for_run
from ..output import emit


def _owner(entry: object) -> ProjectLayout:
    locator = getattr(entry, "control_locator", None)
    if locator is None:
        raise RuntimeError("admission entry has no project owner locator")
    owner = load_project_layout(locator.project_root)
    if owner.project_id != locator.project_id or owner.state.resolve() != locator.control_store:
        raise RuntimeError("admission entry project ownership no longer matches its durable locator")
    return owner


async def controller_sweep(layout: ProjectLayout) -> list[dict[str, Any]]:
    """Advance every safely-owned run once without widening its launch intent."""

    admission = execution_admission_service(layout)
    events: list[dict[str, Any]] = []
    while True:
        try:
            promoted = admission.pump_available()
        except Exception as error:
            events.append({"action": "submit", "state": "attention", "message": f"{type(error).__name__}: {error}"})
            break
        if promoted is None:
            break
        events.append(
            {
                "run_id": promoted.entry.run_id,
                "action": "submit",
                "state": promoted.entry.state,
            }
        )

    for initial in admission.list():
        if initial.state not in {"submitted", "terminal_pending_evidence"}:
            continue
        try:
            owner = _owner(initial)
            store = ExecutionSubmissionStore(owner.state)
            service = admission.service(initial.run_id)
            if store.cancel_intent_path(initial.run_id).is_file() and initial.state == "submitted":
                service.cancel(initial.run_id)
                events.append({"run_id": initial.run_id, "action": "cancel", "state": "delivered"})
            entry, record = admission.status(initial.run_id)
            if record is None or entry.state != "terminal_pending_evidence":
                events.append(
                    {
                        "run_id": initial.run_id,
                        "action": "observe",
                        "state": entry.state,
                        "provider_state": record.state if record is not None else None,
                    }
                )
                continue
            result = await reconcile_execution(
                service,
                reconciliation_source_for_run(owner, initial.run_id),
                initial.run_id,
            )
            receipt = save_reconciliation(store, result)
            event: dict[str, Any] = {
                "run_id": initial.run_id,
                "action": "reconcile",
                "state": result.state,
                "outcome": result.outcome,
                "receipt": str(receipt),
            }
            if result.settled:
                next_admission = admission.acknowledge_reconciled(initial.run_id)
                event["released"] = True
                event["next_run_id"] = next_admission.entry.run_id if next_admission is not None else None
            events.append(event)
        except Exception as error:
            events.append(
                {
                    "run_id": initial.run_id,
                    "action": "reconcile",
                    "state": "attention",
                    "message": f"{type(error).__name__}: {error}",
                }
            )
    return events


def register(app: typer.Typer) -> None:
    controller_app = typer.Typer(
        rich_markup_mode=None,
        no_args_is_help=True,
        help="reconcile durable runs after submitting clients exit",
    )
    app.add_typer(controller_app, name="controller")

    @controller_app.command("run", help="run one reconciliation sweep or a foreground loop")
    def controller_run_cmd(
        ctx: typer.Context,
        once: Annotated[bool, typer.Option("--once", help="run one sweep and exit")] = False,
        interval_seconds: Annotated[
            float,
            typer.Option("--interval-seconds", min=0.1, help="delay between foreground sweeps"),
        ] = 5.0,
        health_file: Annotated[
            Path | None,
            typer.Option("--health-file", help="write the last successful sweep time to this file"),
        ] = None,
    ) -> None:
        state: CliState = ctx.obj
        layout = state.layout()
        all_events: list[dict[str, Any]] = []
        while True:
            events = asyncio.run(controller_sweep(layout))
            all_events.extend(events)
            if health_file is not None:
                health_file.parent.mkdir(parents=True, exist_ok=True)
                health_file.write_text(f"{time.time()}\n", encoding="utf-8")
            if once:
                break
            time.sleep(interval_seconds)
        lines = [
            f"{event.get('run_id', '-')}  action={event['action']}  state={event['state']}" for event in all_events
        ]
        emit(state, all_events, "\n".join(lines) if lines else "No lifecycle actions required.")


__all__ = ["controller_sweep", "register"]
