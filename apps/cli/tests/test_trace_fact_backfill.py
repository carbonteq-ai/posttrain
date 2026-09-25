from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from posttrain.common import TraceFactSet
from posttrain.tracking import TracePage, TraceRecord
from posttrain_cli import trace_fact_backfill


def _trace(external_id: str = "trace-1") -> TraceRecord:
    return TraceRecord(
        trace_type="verifiers",
        external_id=external_id,
        payload={
            "id": external_id,
            "version": 2,
            "agent": {"model": "models/example"},
            "calls": [{"usage": {"prompt_tokens": 4, "completion_tokens": 9}}],
            "nodes": [
                {
                    "sampled": True,
                    "message": {"role": "assistant", "content": "answer"},
                    "token_ids": [1, 2, 3],
                    "mask": [True, True, True],
                }
            ],
            "rewards": {"task": 1.0},
        },
        attributes={"optimizer_step": 7},
    )


def test_preview_projects_a_bounded_page_without_constructing_a_writer(monkeypatch) -> None:
    class Source:
        def __init__(self, project: str, *, server_url: str) -> None:
            assert (project, server_url) == ("ambient-agent", "https://trackio.invalid")

        def _provider_run_by_id(self, run_id: str):
            assert run_id == "provider-run-1"
            return SimpleNamespace(name="run-name", id=run_id)

        async def traces_by_provider_run_id(self, run_id: str, query):
            assert run_id == "provider-run-1"
            assert query.cursor == "200"
            assert query.limit == 25
            assert query.include_payload is True
            return TracePage(items=(_trace(),), next_cursor="225")

    fake_adapter = SimpleNamespace(
        TrackioDataSource=Source,
        TrackioTraceFactWriter=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("preview must not write")),
    )
    monkeypatch.setattr(trace_fact_backfill.importlib, "import_module", lambda _: fake_adapter)

    receipt = trace_fact_backfill.backfill_verifiers_trace_page(
        project="ambient-agent",
        server_url="https://trackio.invalid",
        write_token=None,
        provider_run_id="provider-run-1",
        cursor="200",
        page_size=25,
        apply=False,
    )

    assert receipt.preview is True
    assert receipt.inspected == receipt.projected == 1
    assert receipt.applied == 0
    assert receipt.partial == 1
    assert receipt.next_cursor == "225"


def test_backfill_pipelines_a_larger_window_with_page_checkpoints(monkeypatch) -> None:
    writes: list[tuple[str, ...]] = []
    checkpoints: list[trace_fact_backfill.TraceFactBackfillPage] = []

    class Source:
        calls: list[tuple[str | None, int]] = []
        constructions = 0

        def __init__(self, project: str, *, server_url: str) -> None:
            del project, server_url
            self.__class__.constructions += 1

        def _provider_run_by_id(self, run_id: str):
            return SimpleNamespace(name="run-name", id=run_id)

        async def traces_by_provider_run_id(self, run_id: str, query):
            assert run_id == "provider-run-1"
            assert query.limit == 1000
            self.calls.append((query.cursor, query.limit))
            next_cursor = str(len(self.calls)) if len(self.calls) < 5 else None
            start = (len(self.calls) - 1) * 1000
            return TracePage(
                items=tuple(_trace(f"trace-{start + offset}") for offset in range(1000)),
                next_cursor=next_cursor,
            )

    class Writer:
        constructions = 0

        def __init__(self, server_url: str, *, write_token: str | None) -> None:
            assert (server_url, write_token) == ("https://trackio.invalid", "write-token")
            self.__class__.constructions += 1

        def upsert_many(self, **kwargs: object) -> None:
            updates = cast(tuple[tuple[str, object], ...], kwargs["updates"])
            writes.append(tuple(update[0] for update in updates))

    fake_adapter = SimpleNamespace(TrackioDataSource=Source, TrackioTraceFactWriter=Writer)
    monkeypatch.setattr(trace_fact_backfill.importlib, "import_module", lambda _: fake_adapter)

    receipt = trace_fact_backfill.backfill_verifiers_trace_window(
        project="ambient-agent",
        server_url="https://trackio.invalid",
        write_token="write-token",
        provider_run_id="provider-run-1",
        cursor=None,
        window_size=5000,
        apply=True,
        checkpoint=checkpoints.append,
    )

    assert receipt.inspected == 5000
    assert Source.calls == [(None, 1000), ("1", 1000), ("2", 1000), ("3", 1000), ("4", 1000)]
    assert Source.constructions == Writer.constructions == 1
    assert len(writes) == len(checkpoints) == len(receipt.pages) == 5
    assert [page.next_cursor for page in checkpoints] == ["1", "2", "3", "4", None]
    assert writes[0][0] == "trace-0"
    assert writes[-1][-1] == "trace-4999"


def test_backfill_checkpoints_only_successful_pages_before_interruption(monkeypatch) -> None:
    checkpoints: list[trace_fact_backfill.TraceFactBackfillPage] = []

    class Source:
        calls = 0

        def __init__(self, project: str, *, server_url: str) -> None:
            del project, server_url

        def _provider_run_by_id(self, run_id: str):
            return SimpleNamespace(name="run-name", id=run_id)

        async def traces_by_provider_run_id(self, run_id: str, query):
            del run_id
            self.__class__.calls += 1
            assert query.limit == 1000
            return TracePage(
                items=tuple(_trace(f"trace-{self.calls}-{offset}") for offset in range(1000)),
                next_cursor=str(self.calls),
            )

    class Writer:
        calls = 0

        def __init__(self, server_url: str, *, write_token: str | None) -> None:
            del server_url, write_token

        def upsert_many(self, **kwargs: object) -> None:
            del kwargs
            self.__class__.calls += 1
            if self.calls == 2:
                raise RuntimeError("interrupted write")

    fake_adapter = SimpleNamespace(TrackioDataSource=Source, TrackioTraceFactWriter=Writer)
    monkeypatch.setattr(trace_fact_backfill.importlib, "import_module", lambda _: fake_adapter)

    try:
        trace_fact_backfill.backfill_verifiers_trace_window(
            project="ambient-agent",
            server_url="https://trackio.invalid",
            write_token="write-token",
            provider_run_id="provider-run-1",
            cursor="start",
            window_size=2000,
            apply=True,
            checkpoint=checkpoints.append,
        )
    except RuntimeError as error:
        assert str(error) == "interrupted write"
    else:
        raise AssertionError("the second write must interrupt the window")

    assert len(checkpoints) == 1
    assert checkpoints[0].cursor == "start"
    assert checkpoints[0].next_cursor == "1"


def test_apply_uses_exact_provider_identity_and_shared_projection(monkeypatch) -> None:
    writes: list[dict[str, object]] = []

    class Source:
        def __init__(self, project: str, *, server_url: str) -> None:
            del project, server_url

        def _provider_run_by_id(self, run_id: str):
            return SimpleNamespace(name="run-name", id=run_id)

        async def traces_by_provider_run_id(self, run_id: str, query):
            del run_id, query
            return TracePage(items=(_trace(),), next_cursor=None)

    class Writer:
        def __init__(self, server_url: str, *, write_token: str | None) -> None:
            assert (server_url, write_token) == ("https://trackio.invalid", "write-token")

        def upsert_many(self, **kwargs: object) -> None:
            writes.append(kwargs)

    fake_adapter = SimpleNamespace(TrackioDataSource=Source, TrackioTraceFactWriter=Writer)
    monkeypatch.setattr(trace_fact_backfill.importlib, "import_module", lambda _: fake_adapter)

    receipt = trace_fact_backfill.backfill_verifiers_trace_page(
        project="ambient-agent",
        server_url="https://trackio.invalid",
        write_token="write-token",
        provider_run_id="provider-run-1",
        cursor=None,
        page_size=25,
        apply=True,
    )

    assert receipt.preview is False
    assert receipt.applied == 1
    assert writes[0]["project"] == "ambient-agent"
    assert writes[0]["run_name"] == "run-name"
    assert writes[0]["provider_run_id"] == "provider-run-1"
    assert writes[0]["trace_type"] == "verifiers"
    assert writes[0]["trace_type"] == "verifiers"
    updates = cast(tuple[tuple[str, object], ...], writes[0]["updates"])
    assert updates[0][0] == "trace-1"
    facts = cast(TraceFactSet, updates[0][1])
    assert facts.dimensions["rollout_step"] == 7


class _CountingRenderer:
    """Reports every sampled token before token 99 as reasoning."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[int], list[int]]] = []

    def parse_response(self, token_ids, *, prompt_ids):
        self.calls.append((list(token_ids), list(prompt_ids)))
        return SimpleNamespace(reasoning_tokens=token_ids.index(99) + 1 if 99 in token_ids else len(token_ids))


def test_renderer_fills_missing_reasoning_tokens_from_each_calls_node() -> None:
    from posttrain_cli.trace_fact_backfill import fill_renderer_reasoning_tokens

    payload = {
        "nodes": [
            {"token_ids": [1, 2], "mask": [False, False], "parent": None},
            {"token_ids": [3, 10, 11, 99, 12], "mask": [False, True, True, True, True], "parent": 0},
            {"token_ids": [4, 20, 99], "mask": [False, True, True], "parent": 1},
        ],
        "calls": [
            {"node": 1, "usage": {"completion_tokens": 4, "reasoning_tokens": None}},
            {"node": 2, "usage": {"completion_tokens": 2}},
            {"node": 2, "usage": {"completion_tokens": 2, "reasoning_tokens": 1}},
            {"node": 2, "error": {"type": "timeout"}, "usage": {"completion_tokens": 2}},
        ],
    }
    renderer = _CountingRenderer()

    filled, count = fill_renderer_reasoning_tokens(payload, renderer)

    assert count == 2
    assert [call["usage"].get("reasoning_tokens") for call in filled["calls"]] == [3, 2, 1, None]
    # The prompt is the ancestor chain plus the node's unsampled prefix.
    assert renderer.calls == [([10, 11, 99, 12], [1, 2, 3]), ([20, 99], [1, 2, 3, 10, 11, 99, 12, 4])]
    assert payload["calls"][0]["usage"]["reasoning_tokens"] is None


def _reasoning_trace() -> TraceRecord:
    return TraceRecord(
        trace_type="verifiers",
        external_id="qwen-trace",
        payload={
            "id": "qwen-trace",
            "version": 3,
            "agent": {"model": "models/qwen3.5-2b@bf16"},
            "calls": [{"node": 1, "usage": {"prompt_tokens": 2, "completion_tokens": 3}}],
            "nodes": [
                {"token_ids": [1, 2], "mask": [False, False], "parent": None},
                {
                    "sampled": True,
                    "parent": 0,
                    "message": {"role": "assistant", "reasoning_content": "plan", "content": "answer"},
                    "token_ids": [10, 99, 11],
                    "mask": [True, True, True],
                },
            ],
            "rewards": {"task": 1.0},
        },
    )


def _reasoning_adapter(written: list):
    class Source:
        def __init__(self, project: str, *, server_url: str) -> None:
            pass

        def _provider_run_by_id(self, run_id: str):
            return SimpleNamespace(name="run-name", id=run_id)

        async def traces_by_provider_run_id(self, run_id: str, query):
            return TracePage(items=(_reasoning_trace(),), next_cursor=None)

    class Writer:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def upsert_many(self, **kwargs) -> None:
            written.extend(kwargs["updates"])

    return SimpleNamespace(TrackioDataSource=Source, TrackioTraceFactWriter=Writer)


def test_apply_refuses_to_erase_reasoning_counts_without_a_renderer(monkeypatch) -> None:
    import pytest
    from posttrain.common import ContractError

    written: list = []
    monkeypatch.setattr(trace_fact_backfill.importlib, "import_module", lambda _: _reasoning_adapter(written))
    arguments: dict[str, Any] = dict(
        project="p",
        server_url="https://trackio.invalid",
        write_token=None,
        provider_run_id="run",
        cursor=None,
        window_size=10,
    )

    preview = trace_fact_backfill.backfill_verifiers_trace_window(**arguments, apply=False)
    assert preview.thinking_unscored == 1
    with pytest.raises(ContractError, match="--renderer-model"):
        trace_fact_backfill.backfill_verifiers_trace_window(**arguments, apply=True)
    assert written == []

    applied = trace_fact_backfill.backfill_verifiers_trace_window(**arguments, apply=True, renderer=_CountingRenderer())
    assert applied.reasoning_filled == 1
    assert applied.thinking_unscored == 0
    [(external_id, facts)] = written
    assert external_id == "qwen-trace"
    assert facts.measures["thinking_tokens"] == 2
