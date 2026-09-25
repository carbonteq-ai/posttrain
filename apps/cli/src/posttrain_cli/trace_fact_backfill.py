"""Bounded historical projection of retained Verifiers traces."""

from __future__ import annotations

import asyncio
import importlib
import json
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Annotated, Any

import typer
from posttrain.common import ContractError, TraceFactSet
from posttrain.environment import project_verifiers_trace_facts
from posttrain.tracking import TraceQuery

from .context import CliState
from .output import emit, json_value
from .tracking_config import project_tracking_environment

_TRACE_FACT_READ_CHUNK_SIZE = 1000


def fill_renderer_reasoning_tokens(payload: Mapping[str, Any], renderer: Any) -> tuple[dict[str, Any], int]:
    """Return a copy of a native trace whose calls carry renderer reasoning counts.

    Traces recorded before the renderer reported ``reasoning_tokens`` keep each
    call's generated tokens on its committed node. Re-parsing them with the
    model's renderer gives the same count a new rollout would record. The prompt
    is the node's ancestor chain plus its unsampled prefix, which lets the
    renderer see reasoning the generation prompt opened. Calls that already
    report the count, failed, or have no token evidence are left unchanged.
    Returns the updated payload and how many calls were filled.
    """

    nodes = payload.get("nodes")
    calls = payload.get("calls")
    if not isinstance(nodes, list) or not isinstance(calls, list):
        return dict(payload), 0
    filled = 0
    updated_calls: list[Any] = []
    for call in calls:
        count = _renderer_reasoning_count(call, nodes, renderer)
        if count is None:
            updated_calls.append(call)
            continue
        usage = dict(call["usage"])
        usage["reasoning_tokens"] = count
        updated_calls.append({**call, "usage": usage})
        filled += 1
    return {**payload, "calls": updated_calls}, filled


def _has_unscored_thinking(payload: Mapping[str, Any], facts: TraceFactSet) -> bool:
    """True when a trace shows reasoning text but its facts carry no thinking count."""

    if facts.provenance.get("thinking_tokens") != "unsupported":
        return False
    nodes = payload.get("nodes")
    return isinstance(nodes, list) and any(
        isinstance(node, Mapping)
        and isinstance(message := node.get("message"), Mapping)
        and isinstance(message.get("reasoning_content"), str)
        and bool(message.get("reasoning_content"))
        for node in nodes
    )


def _renderer_reasoning_count(call: Any, nodes: list[Any], renderer: Any) -> int | None:
    if not isinstance(call, Mapping) or call.get("error") not in (None, False, ""):
        return None
    usage = call.get("usage")
    index = call.get("node")
    if not isinstance(usage, Mapping) or usage.get("reasoning_tokens") is not None:
        return None
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(nodes):
        return None
    node = nodes[index]
    token_ids = node.get("token_ids") if isinstance(node, Mapping) else None
    mask = node.get("mask") if isinstance(node, Mapping) else None
    if not isinstance(token_ids, list) or not isinstance(mask, list) or len(token_ids) != len(mask):
        return None
    first = next((offset for offset, sampled in enumerate(mask) if sampled is True), None)
    if first is None or not all(sampled is True for sampled in mask[first:]):
        return None
    prompt: list[int] = []
    seen = {index}
    parent = node.get("parent")
    while isinstance(parent, int) and not isinstance(parent, bool) and 0 <= parent < len(nodes) and parent not in seen:
        seen.add(parent)
        ancestor = nodes[parent]
        ids = ancestor.get("token_ids") if isinstance(ancestor, Mapping) else None
        if not isinstance(ids, list):
            return None
        prompt[:0] = ids
        parent = ancestor.get("parent")
    prompt.extend(token_ids[:first])
    parsed = renderer.parse_response(list(token_ids[first:]), prompt_ids=prompt)
    count = getattr(parsed, "reasoning_tokens", None)
    return count if isinstance(count, int) and not isinstance(count, bool) and count >= 0 else None


@dataclass(frozen=True, slots=True)
class TraceFactBackfillPage:
    """Sanitized result of one bounded, replay-safe historical fact page."""

    project: str
    provider_run_id: str
    cursor: str | None
    next_cursor: str | None
    inspected: int
    projected: int
    complete: int
    partial: int
    applied: int
    preview: bool
    reasoning_filled: int = 0
    thinking_unscored: int = 0


@dataclass(frozen=True, slots=True)
class TraceFactBackfillWindow:
    """Sanitized result of one bounded sequence of checkpointed pages."""

    project: str
    provider_run_id: str
    cursor: str | None
    next_cursor: str | None
    inspected: int
    projected: int
    complete: int
    partial: int
    applied: int
    preview: bool
    pages: tuple[TraceFactBackfillPage, ...]
    reasoning_filled: int = 0
    thinking_unscored: int = 0


def backfill_verifiers_trace_page(
    *,
    project: str,
    server_url: str,
    write_token: str | None,
    provider_run_id: str,
    cursor: str | None,
    page_size: int,
    apply: bool,
) -> TraceFactBackfillPage:
    """Compatibility wrapper for one physical checkpoint page."""

    if page_size < 1 or page_size > _TRACE_FACT_READ_CHUNK_SIZE:
        raise ContractError("trace-fact backfill physical page size must be between 1 and 1000")
    window = backfill_verifiers_trace_window(
        project=project,
        server_url=server_url,
        write_token=write_token,
        provider_run_id=provider_run_id,
        cursor=cursor,
        window_size=page_size,
        apply=apply,
    )
    return window.pages[0]


def backfill_verifiers_trace_window(
    *,
    project: str,
    server_url: str,
    write_token: str | None,
    provider_run_id: str,
    cursor: str | None,
    window_size: int,
    apply: bool,
    checkpoint: Callable[[TraceFactBackfillPage], None] | None = None,
    renderer: Any = None,
) -> TraceFactBackfillWindow:
    """Process a bounded window while checkpointing every physical page.

    Source and writer construction are deliberately hoisted out of the loop.
    Only one at-most-1,000-trace payload page is resident at a time.  The
    callback runs after that page's write succeeds, so its next cursor is a
    safe resume point even when a later page is interrupted.
    """

    if window_size < 1 or window_size > 5000:
        raise ContractError("trace-fact backfill window size must be between 1 and 5000")
    try:
        adapter = importlib.import_module("posttrain_tracking_trackio")
    except ImportError as error:
        raise RuntimeError("trace-fact backfill requires the posttrain[trackio] extra") from error

    source = adapter.TrackioDataSource(project, server_url=server_url)
    provider_run = source._provider_run_by_id(provider_run_id)
    writer = adapter.TrackioTraceFactWriter(server_url, write_token=write_token) if apply else None
    pages: list[TraceFactBackfillPage] = []
    next_cursor = cursor
    remaining = window_size

    while remaining:
        read_limit = min(remaining, _TRACE_FACT_READ_CHUNK_SIZE)
        raw_page = asyncio.run(
            source.traces_by_provider_run_id(
                provider_run_id,
                TraceQuery(
                    trace_type="verifiers",
                    cursor=next_cursor,
                    limit=read_limit,
                    include_payload=True,
                ),
            )
        )
        complete = 0
        partial = 0
        reasoning_filled = 0
        thinking_unscored = 0
        updates: list[tuple[str, TraceFactSet]] = []
        for trace in raw_page.items:
            payload = trace.payload
            if renderer is not None:
                payload, filled = fill_renderer_reasoning_tokens(payload, renderer)
                reasoning_filled += filled
            facts = project_verifiers_trace_facts(payload, attributes=trace.attributes)
            if _has_unscored_thinking(payload, facts):
                thinking_unscored += 1
            if facts.state == "complete":
                complete += 1
            else:
                partial += 1
            if writer is not None:
                updates.append((trace.external_id, facts))
        if writer is not None and renderer is None and thinking_unscored:
            # Facts are replaced per trace. Without a renderer, reasoning that
            # earlier calculators or providers counted would be overwritten as
            # unsupported, so refuse before writing anything from this page.
            raise ContractError(
                f"{thinking_unscored} traces in this page contain reasoning but no reasoning-token usage; "
                "apply with --renderer-model so their thinking counts are re-scored instead of erased"
            )
        if writer is not None:
            writer.upsert_many(
                project=project,
                run_name=str(provider_run.name),
                provider_run_id=str(provider_run.id),
                trace_type="verifiers",
                updates=updates,
            )

        page = TraceFactBackfillPage(
            project=project,
            provider_run_id=provider_run_id,
            cursor=next_cursor,
            next_cursor=raw_page.next_cursor,
            inspected=len(raw_page.items),
            projected=len(raw_page.items),
            complete=complete,
            partial=partial,
            applied=len(raw_page.items) if apply else 0,
            preview=not apply,
            reasoning_filled=reasoning_filled,
            thinking_unscored=thinking_unscored,
        )
        pages.append(page)
        if checkpoint is not None:
            checkpoint(page)

        remaining -= len(raw_page.items)
        next_cursor = raw_page.next_cursor
        if next_cursor is None or len(raw_page.items) < read_limit:
            break

    return TraceFactBackfillWindow(
        project=project,
        provider_run_id=provider_run_id,
        cursor=cursor,
        next_cursor=next_cursor,
        inspected=sum(page.inspected for page in pages),
        projected=sum(page.projected for page in pages),
        complete=sum(page.complete for page in pages),
        partial=sum(page.partial for page in pages),
        applied=sum(page.applied for page in pages),
        preview=not apply,
        pages=tuple(pages),
        reasoning_filled=sum(page.reasoning_filled for page in pages),
        thinking_unscored=sum(page.thinking_unscored for page in pages),
    )


def register(app: typer.Typer) -> None:
    trace_facts_app = typer.Typer(
        rich_markup_mode=None,
        no_args_is_help=True,
        help="project retained native traces into generic evidence facts",
    )
    app.add_typer(trace_facts_app, name="trace-facts")

    @trace_facts_app.command("backfill", help="preview or apply one checkpointed Verifiers trace-fact window")
    def trace_facts_backfill_cmd(
        ctx: typer.Context,
        provider_run_id: Annotated[str, typer.Argument(help="exact Trackio provider run id")],
        cursor: Annotated[str | None, typer.Option(help="resume cursor returned by the prior page")] = None,
        window_size: Annotated[
            int,
            typer.Option(
                "--window-size",
                "--page-size",
                min=1,
                max=5000,
                help="bounded orchestration window; checkpoints remain at most 1,000 traces",
            ),
        ] = 200,
        apply: Annotated[
            bool, typer.Option("--apply", help="persist this page; default is a non-mutating preview")
        ] = False,
        trackio_project: Annotated[
            str | None,
            typer.Option(
                "--trackio-project", help="override the configured Trackio project for cross-project maintenance"
            ),
        ] = None,
        renderer_model: Annotated[
            str | None,
            typer.Option(
                "--renderer-model",
                help=(
                    "Hugging Face model id whose renderer re-parses each call's generated tokens to fill "
                    "reasoning_tokens missing from traces recorded before renderers reported them"
                ),
            ),
        ] = None,
    ) -> None:
        state: CliState = ctx.obj
        layout = state.layout()
        if layout.tracking != "trackio":
            raise ContractError("trace-fact backfill requires a project configured with Trackio")
        environment = project_tracking_environment(layout)
        server_url = environment.get("POSTTRAIN_TRACKIO_SERVER_URL")
        if not server_url:
            raise ContractError("trace-fact backfill requires POSTTRAIN_TRACKIO_SERVER_URL")
        project = trackio_project or environment.get("POSTTRAIN_TRACKIO_PROJECT") or layout.project_id
        renderer = None
        if renderer_model is not None:
            try:
                from renderers import create_renderer  # pyright: ignore[reportMissingImports]
                from renderers.base import load_tokenizer  # pyright: ignore[reportMissingImports]
            except ImportError as error:
                raise ContractError("--renderer-model requires the carbonteq-renderers package") from error
            renderer = create_renderer(load_tokenizer(renderer_model))

        def report_checkpoint(page: TraceFactBackfillPage) -> None:
            if state.json_output:
                print(
                    json.dumps({"checkpoint": json_value(page)}, sort_keys=True),
                    file=sys.stderr,
                    flush=True,
                )
                return
            mode = "applied" if apply else "previewed"
            print(
                f"Trace-fact page {mode}: {page.project}/{page.provider_run_id} "
                f"({page.inspected} traces, {page.complete} complete, {page.partial} partial, "
                f"{page.reasoning_filled} calls given renderer reasoning counts, "
                f"{page.thinking_unscored} traces with reasoning but no count; "
                f"next cursor: {page.next_cursor or 'done'})",
                flush=True,
            )

        receipt = backfill_verifiers_trace_window(
            project=project,
            server_url=server_url,
            write_token=environment.get("TRACKIO_WRITE_TOKEN"),
            provider_run_id=provider_run_id,
            cursor=cursor,
            window_size=window_size,
            apply=apply,
            checkpoint=report_checkpoint,
            renderer=renderer,
        )
        mode = "applied" if apply else "previewed"
        emit(
            state,
            receipt,
            f"Trace-fact window {mode}: {receipt.project}/{receipt.provider_run_id} "
            f"({receipt.inspected} traces, {receipt.complete} complete, {receipt.partial} partial; "
            f"next cursor: {receipt.next_cursor or 'done'})",
        )


__all__ = [
    "TraceFactBackfillPage",
    "TraceFactBackfillWindow",
    "backfill_verifiers_trace_page",
    "backfill_verifiers_trace_window",
    "fill_renderer_reasoning_tokens",
    "register",
]
