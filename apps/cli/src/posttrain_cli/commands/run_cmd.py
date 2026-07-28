"""Run inspection and lifecycle commands."""

from __future__ import annotations

import asyncio
import importlib
import json
import time
from typing import Annotated, Any

import click
import typer
from posttrain.execution import (
    ExecutionSubmissionStore,
    LogCursor,
    cleanup_execution,
    reconcile_execution,
    recover_cancelled_tracking,
    save_reconciliation,
    save_tracking_recovery,
)

from ..constants import RUN_MODE_CHOICES
from ..context import CliState
from ..execution_provider import (
    cancelled_tracking_writer_for_run,
    evidence_source_for_run,
    execution_admission_service,
    execution_service_for_run,
    reconciliation_source_for_run,
    tracking_source_for_run,
)
from ..output import emit, json_value
from ..run_resolve import resolve_run_id
from ..tracking_config import project_observatory_settings

RUN_MODE_CHOICE = click.Choice(RUN_MODE_CHOICES)

_RUN_ID_ARGUMENT = Annotated[
    str | None,
    typer.Argument(help="canonical run id, or an unambiguous prefix"),
]
_LAST_OPTION = Annotated[
    bool,
    typer.Option("--last", help="select the most recent known run"),
]


def _resolved_run_id(layout, run_id: str | None, *, last: bool) -> str:
    return resolve_run_id(layout, run_id, last=last)


def register(app: typer.Typer) -> None:
    run_app = typer.Typer(
        rich_markup_mode=None,
        no_args_is_help=True,
        help="inspect and control submitted runs",
    )
    app.add_typer(run_app, name="run")

    @run_app.command("list", help="list durable submitted run identities")
    def run_list_cmd(
        ctx: typer.Context,
        limit: Annotated[
            int,
            typer.Option("--limit", min=1, max=1000),
        ] = 50,
    ) -> None:
        state: CliState = ctx.obj
        layout = state.layout()
        submissions = ExecutionSubmissionStore(layout.state).list_submissions()
        admission = execution_admission_service(layout)
        admission_entries = {entry.run_id: entry for entry in admission.list()}
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
        payload: list[dict[str, Any]] = []
        for run_id in set(submission_by_run) | set(admission_entries):
            submission = submission_by_run.get(run_id)
            entry = admission_entries.get(run_id)
            if submission is None:
                assert entry is not None
                provider = entry.plan.provider
                job_image = entry.plan.request.image.value
                request = entry.plan.request
            else:
                provider = submission.provider
                job_image = submission.job_image
                request = entry.plan.request if entry is not None else None
            queued_at = entry.queued_at if entry is not None else None
            submitted_at = submission.submitted_at if submission is not None else None
            payload.append(
                {
                    "run_id": run_id,
                    "submitted_at": (submitted_at.isoformat() if submitted_at is not None else None),
                    "queued_at": (queued_at.isoformat() if queued_at is not None else None),
                    "provider": provider,
                    "provider_id": (submission.provider_id if submission is not None else None),
                    "job_image": job_image,
                    "job_kind": (request.run_spec.job_kind if request is not None else None),
                    "work_package_id": (request.run_spec.work_package_id if request is not None else None),
                    "target_id": (request.target.id if request is not None else None),
                    "target": (f"{request.target.id}@{request.target.revision}" if request is not None else None),
                    "message": (entry.message if entry is not None else None),
                    "tracking": (
                        submission.evidence_source.provider
                        if submission is not None and submission.evidence_source is not None
                        else (
                            entry.evidence_source.provider
                            if entry is not None and entry.evidence_source is not None
                            else None
                        )
                    ),
                    "admission_state": entry.state if entry is not None else None,
                    "queue_position": entry.position if entry is not None else None,
                }
            )
        payload.sort(
            key=lambda item: (
                item["queued_at"] or item["submitted_at"] or "",
                item["run_id"],
            ),
            reverse=True,
        )
        payload.sort(key=lambda item: (admission_priority.get(item["admission_state"], 7),))
        payload = payload[:limit]
        lines = []
        for item in payload:
            kind = item["job_kind"] or "-"
            target = item["target_id"] or "-"
            package = item["work_package_id"] or "-"
            position = (
                f"  queue={item['queue_position']}"
                if item["admission_state"] == "waiting" and item["queue_position"] is not None
                else ""
            )
            lines.append(
                f"{item['run_id']}  {item['provider']}  "
                f"state={item['admission_state'] or 'legacy-submitted'}  "
                f"kind={kind}  package={package}  target={target}  "
                f"submitted={item['submitted_at'] or '-'}{position}"
            )
        emit(state, payload, "\n".join(lines) if lines else "No submitted runs.")

    @run_app.command("status", help="show provider lifecycle state for a submitted run")
    def run_status_cmd(
        ctx: typer.Context,
        run_id: _RUN_ID_ARGUMENT = None,
        last: _LAST_OPTION = False,
    ) -> None:
        state: CliState = ctx.obj
        layout = state.layout()
        run_id = _resolved_run_id(layout, run_id, last=last)
        try:
            admission_entry, record = execution_admission_service(layout).status(run_id)
        except KeyError:
            admission_entry = None
            record = execution_service_for_run(layout, run_id).status(run_id)
        if record is None:
            assert admission_entry is not None
            payload = {
                "run_id": run_id,
                "state": admission_entry.state,
                "queue_position": admission_entry.position,
                "provider": admission_entry.plan.provider,
                "message": admission_entry.message,
            }
            emit(
                state,
                payload,
                "\n".join(
                    (
                        f"Run: {run_id}",
                        f"State: {admission_entry.state}",
                        f"Queue position: {admission_entry.position}",
                        f"Provider: {admission_entry.plan.provider}",
                        f"Detail: {admission_entry.message or 'none'}",
                    )
                ),
            )
            return
        payload = {
            **json_value(record),
            "admission_state": (admission_entry.state if admission_entry is not None else None),
        }
        emit(
            state,
            payload,
            "\n".join(
                (
                    f"Run: {run_id}",
                    f"State: {record.state}",
                    f"Provider state: {record.native_state}",
                    f"Target: {record.target_id}",
                    f"Attempt: {record.attempt}",
                )
            ),
        )

    @run_app.command("wait", help="wait for a submitted run to reach terminal state")
    def run_wait_cmd(
        ctx: typer.Context,
        run_id: _RUN_ID_ARGUMENT = None,
        last: _LAST_OPTION = False,
        timeout_seconds: Annotated[
            float,
            typer.Option("--timeout-seconds", min=0.001),
        ] = 3600,
        poll_interval_seconds: Annotated[
            float,
            typer.Option("--poll-interval-seconds", min=0),
        ] = 5,
        cancel_on_timeout: Annotated[
            bool,
            typer.Option(
                "--cancel-on-timeout",
                help="request provider cancellation if the wait deadline expires",
            ),
        ] = False,
    ) -> None:
        state: CliState = ctx.obj
        layout = state.layout()
        run_id = _resolved_run_id(layout, run_id, last=last)
        record = execution_service_for_run(layout, run_id).wait(
            run_id,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
            cancel_on_timeout=cancel_on_timeout,
        )
        payload = json_value(record)
        emit(
            state,
            payload,
            "\n".join(
                (
                    f"Run: {run_id}",
                    f"State: {record.state}",
                    f"Provider state: {record.native_state}",
                    f"Target: {record.target_id}",
                )
            ),
        )
        if record.state != "succeeded":
            raise typer.Exit(code=1)

    @run_app.command("logs", help="read a bounded provider log page")
    def run_logs_cmd(
        ctx: typer.Context,
        run_id: _RUN_ID_ARGUMENT = None,
        last: _LAST_OPTION = False,
        offset: Annotated[
            int,
            typer.Option("--offset", min=0, help="zero-based log cursor"),
        ] = 0,
        limit: Annotated[
            int,
            typer.Option("--limit", min=1, max=2000),
        ] = 200,
        follow: Annotated[
            bool,
            typer.Option("--follow", "-f", help="keep polling until the run is terminal"),
        ] = False,
        poll_interval_seconds: Annotated[
            float,
            typer.Option("--poll-interval-seconds", min=0.1),
        ] = 2.0,
    ) -> None:
        state: CliState = ctx.obj
        layout = state.layout()
        run_id = _resolved_run_id(layout, run_id, last=last)
        service = execution_service_for_run(layout, run_id)
        cursor = offset
        collected: list[str] = []
        truncated = False
        while True:
            page = service.logs(run_id, LogCursor(cursor), limit=limit)
            if page.lines:
                if state.json_output:
                    collected.extend(page.lines)
                else:
                    print("\n".join(page.lines), flush=True)
            truncated = truncated or page.truncated
            cursor = page.next_cursor.offset
            if not follow:
                break
            record = service.status(run_id)
            if record.state in {"succeeded", "failed", "cancelled"} and not page.truncated:
                break
            time.sleep(poll_interval_seconds)
        if state.json_output:
            emit(
                state,
                {
                    "run_id": run_id,
                    "lines": collected,
                    "next_offset": cursor,
                    "truncated": truncated,
                    "followed": follow,
                },
                "",
            )

    @run_app.command("cancel", help="persist cancellation intent and stop the provider run")
    def run_cancel_cmd(
        ctx: typer.Context,
        run_id: _RUN_ID_ARGUMENT = None,
        last: _LAST_OPTION = False,
    ) -> None:
        state: CliState = ctx.obj
        layout = state.layout()
        run_id = _resolved_run_id(layout, run_id, last=last)
        try:
            admission_service = execution_admission_service(layout)
            before = admission_service.get(run_id)
            admission = admission_service.cancel(run_id)
        except KeyError:
            execution_service_for_run(layout, run_id).cancel(run_id)
            status = "cancellation-requested"
        else:
            if before.state == "completed":
                status = "already-complete"
            elif before.state == "terminal_pending_evidence":
                status = "already-terminal"
            elif admission.state == "cancelled":
                status = "cancelled-before-submission"
            else:
                status = "cancellation-requested"
        emit(
            state,
            {"run_id": run_id, "status": status},
            f"Cancellation status: {status} ({run_id})",
        )

    @run_app.command(
        "retry-submit",
        help="retry one ambiguous provider submission with the same run identity",
    )
    def run_retry_submit_cmd(
        ctx: typer.Context,
        run_id: _RUN_ID_ARGUMENT = None,
        last: _LAST_OPTION = False,
    ) -> None:
        state: CliState = ctx.obj
        layout = state.layout()
        run_id = _resolved_run_id(layout, run_id, last=last)
        result = execution_admission_service(layout).retry_submission(run_id)
        emit(
            state,
            {
                "run_id": run_id,
                "state": result.entry.state,
                "provider_id": (result.submission.provider_id if result.submission is not None else None),
            },
            f"Submission retry: {result.entry.state} ({run_id})",
        )

    @run_app.command(
        "recover-cancelled-tracking",
        help="audit and finalize tracking stranded by provider cancellation",
    )
    def run_recover_cancelled_tracking_cmd(
        ctx: typer.Context,
        run_id: _RUN_ID_ARGUMENT = None,
        last: _LAST_OPTION = False,
    ) -> None:
        state: CliState = ctx.obj
        layout = state.layout()
        run_id = _resolved_run_id(layout, run_id, last=last)
        recovery = asyncio.run(
            recover_cancelled_tracking(
                execution_service_for_run(layout, run_id),
                tracking_source_for_run(layout, run_id),
                cancelled_tracking_writer_for_run(layout, run_id),
                run_id,
                project_id=layout.project_id,
            )
        )
        receipt = save_tracking_recovery(
            ExecutionSubmissionStore(layout.state),
            recovery,
        )
        emit(
            state,
            recovery,
            "\n".join(
                (
                    f"Run: {run_id}",
                    f"Recovery: {recovery.disposition}",
                    f"Execution provider: {recovery.execution_provider}",
                    f"Tracking provider: {recovery.tracking_provider}",
                    f"Tracking run: {recovery.tracking_provider_run_id}",
                    f"Audit receipt: {receipt}",
                    "Next: reconcile the run to verify retained evidence.",
                )
            ),
        )

    @run_app.command(
        "reconcile",
        help="join terminal provider state with retained tracking evidence",
    )
    def run_reconcile_cmd(
        ctx: typer.Context,
        run_id: _RUN_ID_ARGUMENT = None,
        last: _LAST_OPTION = False,
    ) -> None:
        state: CliState = ctx.obj
        layout = state.layout()
        run_id = _resolved_run_id(layout, run_id, last=last)
        service = execution_service_for_run(layout, run_id)
        result = asyncio.run(
            reconcile_execution(
                service,
                reconciliation_source_for_run(layout, run_id),
                run_id,
            )
        )
        save_reconciliation(ExecutionSubmissionStore(layout.state), result)
        next_admission = None
        if result.settled:
            try:
                admission = execution_admission_service(layout)
                admission.status(run_id)
                next_admission = admission.acknowledge_reconciled(run_id)
            except KeyError:
                pass
        retained = len(result.retained_artifacts)
        missing = ", ".join(result.missing_artifact_roles) or "none"
        payload = json_value(result)
        assert isinstance(payload, dict)
        payload["next_admission"] = (
            {
                "run_id": next_admission.entry.run_id,
                "state": next_admission.entry.state,
                "queue_position": next_admission.entry.position,
                "message": next_admission.entry.message,
            }
            if next_admission is not None
            else None
        )
        lines = [
            f"Run: {run_id}",
            f"Reconciliation: {result.state}",
            f"Outcome: {result.outcome}",
            f"Provider: {result.provider_record.state}",
            f"Tracking: {result.tracking_status or 'unavailable'}",
            f"Retained artifacts: {retained}",
            f"Missing required roles: {missing}",
            f"Detail: {result.message}",
        ]
        if next_admission is not None:
            detail = next_admission.entry.message or "none"
            lines.append(f"Next admission: {next_admission.entry.run_id} ({next_admission.entry.state}; {detail})")
        emit(
            state,
            payload,
            "\n".join(lines),
        )
        if result.state != "consistent":
            raise typer.Exit(code=2)

    @run_app.command(
        "cleanup",
        help="release one terminal execution after retaining its evidence",
    )
    def run_cleanup_cmd(
        ctx: typer.Context,
        run_id: _RUN_ID_ARGUMENT = None,
        last: _LAST_OPTION = False,
        diagnostic_limit: Annotated[
            int,
            typer.Option(
                "--diagnostic-limit",
                min=1,
                max=2000,
                help="maximum provider lines retained for a pre-tracking failure",
            ),
        ] = 500,
    ) -> None:
        state: CliState = ctx.obj
        layout = state.layout()
        run_id = _resolved_run_id(layout, run_id, last=last)
        result = asyncio.run(
            cleanup_execution(
                execution_service_for_run(layout, run_id),
                ExecutionSubmissionStore(layout.state),
                reconciliation_source_for_run(layout, run_id),
                run_id,
                diagnostic_limit=diagnostic_limit,
            )
        )
        emit(
            state,
            result,
            "\n".join(
                (
                    f"Run: {run_id}",
                    f"Evidence: {result.evidence_state}",
                    f"Outcome: {result.outcome}",
                    f"Provider cleanup: {result.provider_disposition}",
                    f"Workspace cleanup: {result.workspace_disposition}",
                    f"Workspace bytes reclaimed: {result.workspace_reclaimed_bytes}",
                    f"Retained artifacts: {result.retained_artifact_count}",
                )
            ),
        )

    @run_app.command("show", help="show one recorded run view")
    def run_show_cmd(
        ctx: typer.Context,
        run_id: _RUN_ID_ARGUMENT = None,
        last: _LAST_OPTION = False,
        source: Annotated[
            str | None,
            typer.Option(
                "--source",
                help="Observatory source id; defaults to the project's tracking source",
            ),
        ] = None,
        mode: Annotated[
            str,
            typer.Option("--mode", click_type=RUN_MODE_CHOICE),
        ] = "auto",
    ) -> None:
        state: CliState = ctx.obj
        layout = state.layout()
        run_id = _resolved_run_id(layout, run_id, last=last)
        evidence_source = evidence_source_for_run(layout, run_id)
        try:
            observatory = importlib.import_module("posttrain_observatory")
        except ImportError as error:
            raise RuntimeError(
                "Observatory is not installed; run `uv add 'posttrain[observatory]'` "
                "or install the posttrain-observatory package"
            ) from error
        settings_type = getattr(observatory, "ObservatorySettings", None)
        create_service = getattr(observatory, "create_service", None)
        run_locator_type = getattr(observatory, "RunLocator", None)
        if settings_type is None or not callable(create_service) or run_locator_type is None:
            raise RuntimeError("installed Observatory does not expose its query API; upgrade posttrain-observatory")
        settings = project_observatory_settings(
            layout,
            settings_type,
            evidence_source=evidence_source,
        )
        source_id = source or settings.source_id
        service: Any = create_service(settings)

        async def _load() -> Any:
            return await service.get_run_view_response(
                run_locator_type(source_id=source_id, run_id=run_id),
                mode,
            )

        view = asyncio.run(_load())
        payload = view.model_dump(mode="json") if hasattr(view, "model_dump") else view
        emit(
            state,
            payload,
            json.dumps(payload, indent=2, sort_keys=True, default=str),
        )
