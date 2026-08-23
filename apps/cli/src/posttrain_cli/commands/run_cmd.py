"""Run inspection and lifecycle commands."""

from __future__ import annotations

import asyncio
import importlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Annotated, Any

import click
import typer
from posttrain.common import ContractError, StoredArtifactRef
from posttrain.execution import (
    ExecutionSubmissionStore,
    LogCursor,
    PurgeReason,
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
    tracking_source_for_project,
    tracking_source_for_run,
)
from ..output import emit, json_value
from ..purge_surface import render_plan, save_run_preview
from ..run_resolve import project_admission_entries, purged_run_ids, purged_run_tombstones, resolve_run_id
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


def _resolved_run_id(layout, run_id: str | None, *, last: bool, exact_only: bool = False) -> str:
    return resolve_run_id(layout, run_id, last=last, exact_only=exact_only)


def _requested_hostnames(request: object) -> list[str]:
    target = getattr(request, "target", None)
    placement = getattr(target, "placement", {})
    instances = placement.get("instances") if isinstance(placement, Mapping) else None
    if not isinstance(instances, Sequence) or isinstance(instances, str):
        return []
    hostnames: list[str] = []
    for item in instances:
        if isinstance(item, str):
            hostnames.append(item)
        elif isinstance(item, Mapping) and isinstance(item.get("hostname"), str):
            hostnames.append(str(item["hostname"]))
    return hostnames


def _assigned_hostname(target_id: str | None) -> str | None:
    return target_id if target_id and target_id != "unassigned" else None


def register(app: typer.Typer) -> None:
    run_app = typer.Typer(
        rich_markup_mode=None,
        no_args_is_help=True,
        help="inspect and control submitted runs",
    )
    app.add_typer(run_app, name="run")
    checkpoint_app = typer.Typer(
        rich_markup_mode=None,
        no_args_is_help=True,
        help="inspect committed checkpoint views for one run",
    )
    run_app.add_typer(checkpoint_app, name="checkpoint")

    def _checkpoint_links(layout, run_id: str):
        source = tracking_source_for_project(layout)
        return source, asyncio.run(source.artifacts(run_id)).outputs

    def _checkpoint_records(layout, run_id: str) -> tuple[dict[str, object], ...]:
        _source, links = _checkpoint_links(layout, run_id)
        return tuple(link.model_dump(mode="json") for link in links)

    @checkpoint_app.command("list", help="list bounded checkpoint snapshot summaries")
    def checkpoint_list_cmd(
        ctx: typer.Context,
        run_id: _RUN_ID_ARGUMENT = None,
        last: _LAST_OPTION = False,
        limit: Annotated[int, typer.Option("--limit", min=1, max=200)] = 50,
    ) -> None:
        state: CliState = ctx.obj
        layout = state.layout()
        resolved = _resolved_run_id(layout, run_id, last=last)
        from posttrain.train import inspect_checkpoint_artifacts

        inspections = inspect_checkpoint_artifacts(_checkpoint_records(layout, resolved))[:limit]
        payload = [
            {
                "run_id": resolved,
                "checkpoint_snapshot_id": item.snapshot_id,
                "step": item.step,
                "ready": item.ready,
                "recovery": (item.recovery is not None),
                "model": (item.model is not None),
                "recovery_version": item.recovery.version if item.recovery is not None else None,
                "model_version": item.model.version if item.model is not None else None,
            }
            for item in inspections
        ]
        lines = [
            f"step={item['step']}  state={'ready' if item['ready'] else 'partial'}  "
            f"recovery={'yes' if item['recovery'] else 'no'}  model={'yes' if item['model'] else 'no'}"
            for item in payload
        ]
        emit(state, payload, "\n".join(lines) if lines else "No committed checkpoint views.")

    @checkpoint_app.command("show", help="show one checkpoint snapshot and its two views")
    def checkpoint_show_cmd(
        ctx: typer.Context,
        run_id: _RUN_ID_ARGUMENT = None,
        last: _LAST_OPTION = False,
        step: Annotated[int, typer.Option("--step", min=0)] = 0,
        files: Annotated[bool, typer.Option("--files", help="include bounded manifest metadata")] = False,
    ) -> None:
        state: CliState = ctx.obj
        layout = state.layout()
        resolved = _resolved_run_id(layout, run_id, last=last)
        from posttrain.train import inspect_checkpoint_artifacts

        inspection = next(
            (item for item in inspect_checkpoint_artifacts(_checkpoint_records(layout, resolved)) if item.step == step),
            None,
        )
        if inspection is None:
            raise ContractError(f"run {resolved!r} has no checkpoint at step {step}")
        payload: dict[str, object] = {
            "run_id": resolved,
            "checkpoint_snapshot_id": inspection.snapshot_id,
            "step": inspection.step,
            "ready": inspection.ready,
            "recovery": (asdict(inspection.recovery) if inspection.recovery is not None else None),
            "model": (asdict(inspection.model) if inspection.model is not None else None),
        }
        if not files:
            for key in ("recovery", "model"):
                value = payload[key]
                if isinstance(value, dict):
                    value.pop("metadata", None)
        emit(state, payload, json.dumps(payload, indent=2, sort_keys=True, default=str))

    @checkpoint_app.command("verify", help="verify checkpoint metadata and committed digests")
    def checkpoint_verify_cmd(
        ctx: typer.Context,
        run_id: _RUN_ID_ARGUMENT = None,
        last: _LAST_OPTION = False,
        step: Annotated[int, typer.Option("--step", min=0)] = 0,
        deep: Annotated[bool, typer.Option("--deep", help="request provider blob verification")] = False,
    ) -> None:
        state: CliState = ctx.obj
        layout = state.layout()
        resolved = _resolved_run_id(layout, run_id, last=last)
        from posttrain.train import inspect_checkpoint_artifacts

        inspection = next(
            (item for item in inspect_checkpoint_artifacts(_checkpoint_records(layout, resolved)) if item.step == step),
            None,
        )
        if inspection is None:
            raise ContractError(f"run {resolved!r} has no checkpoint at step {step}")
        checks: dict[str, object] = {
            "paired_views": inspection.ready,
            "recovery_digest": inspection.recovery is not None and inspection.recovery.digest is not None,
            "model_digest": inspection.model is not None and inspection.model.digest is not None,
            "deep": "unsupported" if deep else "not_requested",
        }
        if deep:
            source, _links = _checkpoint_links(layout, resolved)
            deep_results: dict[str, object] = {}
            for label, record in (("recovery", inspection.recovery), ("model", inspection.model)):
                if record is None or record.digest is None:
                    continue
                integrity = asyncio.run(
                    source.verify_artifact(
                        StoredArtifactRef(
                            provider=record.provider,
                            namespace=record.namespace,
                            name=record.name,
                            version=record.version,
                            digest=record.digest,
                        ),
                        deep=True,
                    )
                )
                deep_results[label] = asdict(integrity)
            checks["deep"] = deep_results
        deep_states = (
            [value.get("state") for value in checks["deep"].values() if isinstance(value, Mapping)]
            if isinstance(checks["deep"], dict)
            else []
        )
        metadata_ok = all(value is True for value in checks.values() if isinstance(value, bool))
        deep_ok = not deep or (bool(deep_states) and all(state == "verified" for state in deep_states))
        payload = {
            "run_id": resolved,
            "step": step,
            "checkpoint_snapshot_id": inspection.snapshot_id,
            "state": (
                "verified"
                if metadata_ok and deep_ok
                else "unsupported"
                if deep and deep_states and any(state == "unsupported" for state in deep_states)
                else "failed"
            ),
            "checks": checks,
        }
        emit(state, payload, json.dumps(payload, indent=2, sort_keys=True))
        if payload["state"] != "verified":
            raise typer.Exit(code=2)

    @checkpoint_app.command("diff", help="compare checkpoint view metadata between two steps")
    def checkpoint_diff_cmd(
        ctx: typer.Context,
        run_id: _RUN_ID_ARGUMENT = None,
        last: _LAST_OPTION = False,
        from_step: Annotated[int, typer.Option("--from-step", min=0)] = 0,
        to_step: Annotated[int, typer.Option("--to-step", min=0)] = 0,
    ) -> None:
        state: CliState = ctx.obj
        layout = state.layout()
        resolved = _resolved_run_id(layout, run_id, last=last)
        from posttrain.train import inspect_checkpoint_artifacts

        by_step = {item.step: item for item in inspect_checkpoint_artifacts(_checkpoint_records(layout, resolved))}
        before = by_step.get(from_step)
        after = by_step.get(to_step)
        if before is None or after is None:
            raise ContractError(f"run {resolved!r} does not contain both requested checkpoint steps")
        payload = {
            "run_id": resolved,
            "from_step": from_step,
            "to_step": to_step,
            "model_digest_changed": (
                before.model is not None and after.model is not None and before.model.digest != after.model.digest
            ),
            "recovery_digest_changed": (
                before.recovery is not None
                and after.recovery is not None
                and before.recovery.digest != after.recovery.digest
            ),
        }
        emit(state, payload, json.dumps(payload, indent=2, sort_keys=True))

    @run_app.command("list", help="list durable submitted run identities")
    def run_list_cmd(
        ctx: typer.Context,
        limit: Annotated[
            int,
            typer.Option("--limit", min=1, max=1000),
        ] = 50,
        include_purged: Annotated[
            bool,
            typer.Option("--include-purged", help="include labeled runs with completed purge receipts"),
        ] = False,
    ) -> None:
        state: CliState = ctx.obj
        layout = state.layout()
        admission = execution_admission_service(layout)
        purged_ids = purged_run_ids(layout)
        tombstones = purged_run_tombstones(layout) if include_purged else {}
        purged = set() if include_purged else purged_ids
        submissions = tuple(
            submission
            for submission in ExecutionSubmissionStore(layout.state).list_submissions()
            if submission.run_id not in purged
        )
        admission_entries = project_admission_entries(
            layout,
            include_purged=include_purged,
            entries=tuple(admission.list()),
        )
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
                    "requested_hostnames": (_requested_hostnames(request) if request is not None else []),
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
                    "purged": run_id in purged_ids,
                    "purge": (
                        {
                            "status": tombstone.status,
                            "reason": tombstone.reason.payload(),
                            "plane_outcomes": dict(tombstone.plane_outcomes),
                        }
                        if (tombstone := tombstones.get(run_id)) is not None
                        else None
                    ),
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
                f"{'purged  ' if item['purged'] else ''}"
                f"kind={kind}  package={package}  target={target}  "
                f"submitted={item['submitted_at'] or '-'}{position}"
            )
        emit(state, payload, "\n".join(lines) if lines else "No submitted runs.")

    @run_app.command(
        "queue",
        help="show framework admission and provider-native capacity queues",
    )
    def run_queue_cmd(ctx: typer.Context) -> None:
        state: CliState = ctx.obj
        layout = state.layout()
        admission = execution_admission_service(layout)
        submissions = {
            submission.run_id: submission for submission in ExecutionSubmissionStore(layout.state).list_submissions()
        }
        rows: list[dict[str, Any]] = []
        for initial in admission.list():
            entry = initial
            record = None
            detail = entry.message
            if entry.state in {"waiting", "submitting"}:
                queue_scope = "framework"
            elif entry.state == "submitted":
                queue_scope = "provider"
                try:
                    entry, record = admission.status(entry.run_id)
                except Exception as error:
                    detail = f"{type(error).__name__}: {error}"
                if record is not None and record.state != "queued":
                    continue
                if entry.state != "submitted":
                    continue
            else:
                continue
            submission = submissions.get(entry.run_id)
            rows.append(
                {
                    "run_id": entry.run_id,
                    "queue_scope": queue_scope,
                    "queue_position": entry.position if queue_scope == "framework" else None,
                    "admission_state": entry.state,
                    "provider": entry.plan.provider,
                    "provider_id": submission.provider_id if submission is not None else None,
                    "provider_state": record.native_state if record is not None else None,
                    "requested_target_id": entry.plan.request.target.id,
                    "requested_hostnames": _requested_hostnames(entry.plan.request),
                    "assigned_hostname": (_assigned_hostname(record.target_id) if record is not None else None),
                    "queued_at": entry.queued_at.isoformat(),
                    "message": record.message if record is not None else detail,
                }
            )
        rows.sort(key=lambda item: (item["queued_at"], item["run_id"]))
        lines = [
            (
                f"{item['run_id']}  scope={item['queue_scope']}  "
                f"provider={item['provider']}  position={item['queue_position'] or '-'}  "
                f"requested={','.join(item['requested_hostnames']) or item['requested_target_id']}  "
                f"assigned={item['assigned_hostname'] or '-'}  "
                f"provider_state={item['provider_state'] or '-'}"
            )
            for item in rows
        ]
        emit(state, rows, "\n".join(lines) if lines else "No queued runs.")

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
                "requested_target_id": admission_entry.plan.request.target.id,
                "requested_hostnames": _requested_hostnames(admission_entry.plan.request),
                "assigned_hostname": None,
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
            "requested_target_id": (admission_entry.plan.request.target.id if admission_entry is not None else None),
            "requested_hostnames": (
                _requested_hostnames(admission_entry.plan.request) if admission_entry is not None else []
            ),
            "assigned_hostname": _assigned_hostname(record.target_id),
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
        stream: Annotated[
            str,
            typer.Option("--stream", click_type=click.Choice(("workload", "diagnostic"))),
        ] = "workload",
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
            page = service.logs(run_id, LogCursor(cursor), limit=limit, stream=stream)  # type: ignore[arg-type]
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
                    "stream": stream,
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
        run_id = _resolved_run_id(layout, run_id, last=last, exact_only=True)
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
        run_id = _resolved_run_id(layout, run_id, last=last, exact_only=True)
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
        run_id = _resolved_run_id(layout, run_id, last=last, exact_only=True)
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
        run_id = _resolved_run_id(layout, run_id, last=last, exact_only=True)
        service = execution_service_for_run(layout, run_id)
        result = asyncio.run(
            reconcile_execution(
                service,
                reconciliation_source_for_run(layout, run_id),
                run_id,
            )
        )
        store = ExecutionSubmissionStore(layout.state)
        save_reconciliation(store, result)
        next_admission = None
        cleanup = None
        if result.settled:
            cleanup = asyncio.run(
                cleanup_execution(
                    service,
                    store,
                    reconciliation_source_for_run(layout, run_id),
                    run_id,
                    diagnostic_limit=500,
                )
            )
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
        payload["cleanup"] = json_value(cleanup) if cleanup is not None else None
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
        if cleanup is not None:
            lines.append(f"Cleanup: {cleanup.provider_disposition}; workspace={cleanup.workspace_disposition}")
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
        run_id = _resolved_run_id(layout, run_id, last=last, exact_only=True)
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

    @run_app.command("purge", help="preview destructive evidence and resource deletion for one exact run id")
    def run_purge_cmd(
        ctx: typer.Context,
        run_id: Annotated[str, typer.Argument(help="full canonical run id; prefixes and --last are unsupported")],
        reason: Annotated[str, typer.Option("--reason", help="safe reason category, for example disposable-smoke")],
        note: Annotated[str | None, typer.Option("--note", help="optional safe one-line audit note")] = None,
        cascade: Annotated[
            bool, typer.Option("--cascade", help="include the complete same-project consumer closure")
        ] = False,
    ) -> None:
        state: CliState = ctx.obj
        layout = state.layout()
        plan = save_run_preview(layout, run_id, cascade=cascade, reason=PurgeReason(category=reason, note=note))
        emit(state, plan, render_plan(plan))

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
