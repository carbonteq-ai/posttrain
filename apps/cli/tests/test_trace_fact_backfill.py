from __future__ import annotations

from types import SimpleNamespace

from posttrain.tracking import TracePage, TraceRecord
from posttrain_cli import trace_fact_backfill


def _trace() -> TraceRecord:
    return TraceRecord(
        trace_type="verifiers",
        external_id="trace-1",
        payload={
            "id": "trace-1",
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

        def upsert(self, **kwargs: object) -> None:
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
    assert writes[0]["external_id"] == "trace-1"
    assert writes[0]["facts"].dimensions["rollout_step"] == 7
