"""HTTP and schema contract tests for the packaged product."""

from __future__ import annotations

from typing import Any, cast

import pytest
from fastapi.testclient import TestClient
from posttrain.tracking import InMemoryRunNoteStore
from posttrain_observatory import FixtureRunDataSource, FixtureSemanticSummaryProvider, ObservatoryService
from posttrain_observatory.discovery import TrackioSourceDiscovery
from posttrain_observatory.http import create_http_app
from posttrain_observatory.mcp import create_mcp
from posttrain_observatory.models import RunLocator
from posttrain_observatory.settings import ObservatorySettings
from posttrain_observatory.sources import RunSourceRegistry
from posttrain_tracking_trackio import TrackioDataSource


def _service() -> ObservatoryService:
    return ObservatoryService(
        {"fixture": FixtureRunDataSource()},
        semantic_provider=FixtureSemanticSummaryProvider(),
    )


def _client() -> TestClient:
    return TestClient(create_http_app(_service(), ObservatorySettings()))


def test_run_list_view_metrics_semantics_and_traces_share_one_api() -> None:
    with _client() as client:
        assert client.get("/health/live").json() == {"status": "ok"}
        runs = client.get("/api/v1/runs").json()
        sft = next(run for run in runs if run["run"]["job_kind"] == "train.sft")
        direct = client.get(f"/api/v1/runs/{sft['run_key']}").json()
        assert direct["run_key"] == sft["run_key"]
        assert direct["locator"] == sft["locator"]
        located = client.get(
            "/api/v1/runs/locate",
            params={"run_id": sft["run"]["run_id"]},
        ).json()
        assert [item["run_key"] for item in located] == [sft["run_key"]]
        view = client.get(f"/api/v1/runs/{sft['run_key']}/view").json()
        assert view["view"]["view_kind"] == "job.metrics"
        assert view["view"]["trace_evaluation_enabled"] is False
        loss_help = next(item for item in view["view"]["metric_help"] if item["metric"] == "train/loss")
        assert loss_help["interpretation"]
        system = client.get(f"/api/v1/runs/{sft['run_key']}/system-metrics").json()
        assert system["state"] == "available"
        assert all(metric["description"] for metric in system["summary"])
        semantic = client.post(
            f"/api/v1/runs/{sft['run_key']}/semantic-summary",
            json={"scope": "run", "metric_names": [], "trace_id": None},
        ).json()
        assert semantic["status"] == "ready"
        evaluation = next(run for run in runs if run["run"]["job_kind"] == "eval.domain")
        evaluation_view = client.get(f"/api/v1/runs/{evaluation['run_key']}/view").json()
        assert evaluation_view["view"]["trace_evaluation_enabled"] is True
        comparison_key = client.get(f"/api/v1/runs/{evaluation['run_key']}/comparison-key").json()
        assert comparison_key["job_kind"] == "eval.domain"
        assert comparison_key["comparison_key"]
        traces = client.get(f"/api/v1/runs/{evaluation['run_key']}/traces-evaluation").json()
        assert traces["included"] == 12
        aggregate_only = client.get(
            f"/api/v1/runs/{evaluation['run_key']}/traces-evaluation",
            params={"include_traces": False},
        ).json()
        assert aggregate_only["included"] == 12
        assert aggregate_only["traces"] == []
        first_page = client.get(
            f"/api/v1/runs/{evaluation['run_key']}/traces",
            params={"limit": 3},
        ).json()
        assert first_page["total"] == 12
        assert len(first_page["items"]) == 3
        assert first_page["next_cursor"] == "3"
        filters = client.get(f"/api/v1/runs/{evaluation['run_key']}/trace-filters").json()
        assert filters["total"] == 12
        assert filters["slices"]
        selected_slice = next(item["key"] for item in filters["slices"] if not item["key"].startswith("facet:"))
        filtered = client.get(
            f"/api/v1/runs/{evaluation['run_key']}/traces",
            params={"slice_key": selected_slice, "limit": 3},
        ).json()
        assert filtered["total"] > 0
        assert all(item["task"] == selected_slice for item in filtered["items"])


def test_openapi_contains_bounded_product_routes() -> None:
    schema = _client().get("/openapi.json").json()
    assert "/api/v1/runs/locate" in schema["paths"]
    assert "/api/v1/runs/{run_key}" in schema["paths"]
    assert "/api/v1/runs/{run_key}/view" in schema["paths"]
    assert "/api/v1/runs/{run_key}/system-metrics" in schema["paths"]
    assert "/api/v1/runs/{run_key}/semantic-summary" in schema["paths"]
    assert "/api/v1/runs/{run_key}/traces-evaluation" in schema["paths"]
    assert "/api/v1/runs/{run_key}/rollout-behavior" in schema["paths"]
    assert "/api/v1/runs/{run_key}/traces" in schema["paths"]
    assert "/api/v1/runs/{run_key}/trace-filters" in schema["paths"]
    assert "/api/v1/runs/{run_key}/trace-group-rewards" in schema["paths"]
    assert "/api/v1/runs/{run_key}/comparison-key" in schema["paths"]
    assert "/api/v1/serving-capacity/work-packages/{work_package_id}" in schema["paths"]
    assert set(schema["paths"]["/api/v1/sources/refresh"]) == {"post"}
    schemas = schema["components"]["schemas"]
    assert "EvidenceCompleteness" in schemas
    assert schemas["RunView"]["properties"]["completeness"] == {"$ref": "#/components/schemas/EvidenceCompleteness"}
    assert schemas["TraceEvaluationView"]["properties"]["performance"] == {
        "$ref": "#/components/schemas/EvaluationPerformance"
    }
    assert schemas["TraceSummaryPage"]["properties"]["items"] == {
        "default": [],
        "items": {"$ref": "#/components/schemas/TraceSummary"},
        "title": "Items",
        "type": "array",
    }
    assert schemas["JsonValue"] == {
        "description": "Any JSON-compatible value.",
        "title": "JsonValue",
    }


def test_disabled_source_refresh_has_an_explicit_status() -> None:
    with _client() as client:
        assert client.post("/api/v1/sources/refresh").json() == {
            "enabled": False,
            "state": "disabled",
            "last_attempt_at": None,
            "last_success_at": None,
            "error": None,
            "discovered_source_ids": [],
        }


def test_lifespan_discovers_sources_and_post_refreshes_again() -> None:
    class Catalog:
        calls = 0

        def list_projects(self) -> tuple[str, ...]:
            self.calls += 1
            return ("alpha", "beta")

    registry = RunSourceRegistry({})
    catalog = Catalog()
    discovery = TrackioSourceDiscovery(
        registry,
        catalog,  # type: ignore[arg-type]
        lambda _: FixtureRunDataSource(),
        interval_seconds=300,
    )
    service = ObservatoryService(registry, source_discovery=discovery)

    with TestClient(create_http_app(service, ObservatorySettings())) as client:
        assert {source["source_id"] for source in client.get("/api/v1/sources").json()} == {"alpha", "beta"}
        assert client.post("/api/v1/sources/refresh").json()["discovered_source_ids"] == ["alpha", "beta"]
        assert client.get("/health/ready").json()["source_refresh"]["state"] == "succeeded"

    assert catalog.calls == 2


def test_new_job_metric_schemas_are_available_through_existing_routes() -> None:
    with _client() as client:
        job_kinds = {item["job_kind"] for item in client.get("/api/v1/job-kinds").json()}
        assert {"train.sampo", "train.distill", "serve.smoke", "data.prepare"}.issubset(job_kinds)

        sampo = client.get("/api/v1/job-kinds/train.sampo").json()
        smoke = client.get("/api/v1/job-kinds/serve.smoke").json()
        prepare = client.get("/api/v1/job-kinds/data.prepare").json()

    assert any(field["metric"] == "train/rl/turn_advantage_mean" for field in sampo["summary_fields"])
    assert {field["metric"] for field in smoke["summary_fields"]} == {
        "serve/probe_healthy",
        "serve/probe_model_available",
        "serve/probe_latency_seconds",
    }
    assert {field["metric"] for field in prepare["summary_fields"]} == {
        "data/examples",
        "data/bytes",
    }


@pytest.mark.asyncio
async def test_serving_capacity_http_export_and_mcp_use_the_same_projection() -> None:
    with _client() as client:
        params = {"project_id": "projects/automation-agent", "source_id": "fixture"}
        http_view = client.get(
            "/api/v1/serving-capacity/work-packages/screen/serving-capacity-v1",
            params=params,
        ).json()
        exported = client.post(
            "/api/v1/exports",
            json={
                "view": "serving_capacity",
                "work_package_id": "screen/serving-capacity-v1",
                **params,
            },
        ).json()["view"]

    mcp_result = await create_mcp(_service()).call_tool(
        "get_serving_capacity_view",
        {
            "work_package_id": "screen/serving-capacity-v1",
            **params,
        },
    )

    assert exported == http_view
    assert cast(Any, mcp_result).structured_content == http_view


def _sql_client(project: TrackioDataSource, **kwargs: Any) -> TestClient:
    service = ObservatoryService({"local": project}, **kwargs)
    return TestClient(create_http_app(service, ObservatorySettings()))


def test_semantic_routes_run_sql_in_storage_and_explain_mistakes(
    trackio_project: TrackioDataSource, trackio_engine: str
) -> None:
    with _sql_client(trackio_project) as client:
        model = client.get("/api/v1/semantic/model").json()
        assert any(measure["name"] == "rollout_seconds" for measure in model["measures"])
        described = client.post("/api/v1/semantic/describe", json={"job_kinds": ["train.grpo"]}).json()
        assert described["job_kinds"] == ["train.grpo"]
        grpo = {"run.job_kind": "train.grpo"}
        result = client.post("/api/v1/semantic/query", json={"measures": ["entropy"], "by": ["run.id"], "runs": grpo})
        assert result.status_code == 200, result.text
        assert result.json()["rows"] == [["grpo-a", pytest.approx(0.21)]]
        assert result.json()["sql"].startswith("SELECT r.`id` AS `run.id`")
        sql = client.post("/api/v1/semantic/query", json={"sql": "select count(*) as n from runs"})
        assert sql.status_code == 200, sql.text
        assert sql.json()["rows"] == [[2]] and sql.json()["engine"] == trackio_engine
        unknown = client.post("/api/v1/semantic/query", json={"measures": ["no_such_measure"], "runs": grpo})
        assert unknown.status_code == 422
        assert "no_such_measure" in unknown.json()["message"]
    with _client() as fixture:
        refused = fixture.post("/api/v1/semantic/query", json={"sql": "select 1"})
        assert refused.status_code == 501 and refused.json()["code"] == "sql_unavailable"


@pytest.mark.asyncio
async def test_mcp_semantic_tools_answer_like_http(trackio_project: TrackioDataSource) -> None:
    query = {"measures": ["entropy"], "by": ["run.id"], "runs": {"run.job_kind": "train.grpo"}}
    with _sql_client(trackio_project) as client:
        http_result = client.post("/api/v1/semantic/query", json=query).json()
    server = create_mcp(ObservatoryService({"local": trackio_project}))
    described = await server.call_tool("describe_semantics", {"job_kinds": ["train.grpo"]})
    assert "train.grpo" in cast(Any, described).structured_content["job_kinds"]
    queried = await server.call_tool("query_semantics", query)
    assert cast(Any, queried).structured_content == http_result


def _notes_client(*, writes: bool = True) -> tuple[TestClient, InMemoryRunNoteStore, ObservatoryService]:
    store = InMemoryRunNoteStore()
    service = ObservatoryService(
        {"fixture": FixtureRunDataSource()},
        note_store_factory=lambda source_id, source: store,
        note_writes=writes,
    )
    return TestClient(create_http_app(service, ObservatorySettings())), store, service


def test_run_notes_are_added_rendered_revised_and_deleted_over_http(trackio_project: TrackioDataSource) -> None:
    store = InMemoryRunNoteStore()
    client = _sql_client(trackio_project, note_store_factory=lambda source_id, source: store, note_writes=True)
    key = RunLocator(source_id="local", run_id="grpo-a").key
    with client:
        card = client.get(f"/api/v1/runs/{key}/card").json()
        assert card["template"] == "group-policy@2" and card["unresolved"] == []
        body = (
            "```sql r\nselect max_by(entropy, step) as last from updates\n```\n"
            "Final entropy {{r.last | round 2}}. <script>x</script>"
        )
        added = client.post(f"/api/v1/runs/{key}/notes", json={"kind": "finding", "body_md": body})
        assert added.status_code == 200, added.text
        note = added.json()
        assert (note["revision"], note["source"]) == (1, "observatory")
        listed = client.get(f"/api/v1/runs/{key}/notes").json()
        assert listed[0]["note"]["note_id"] == note["note_id"]
        assert listed[0]["rendered"]["unresolved"] == [] and "Final entropy 0.25." in listed[0]["rendered"]["text"]
        revised = client.put(
            f"/api/v1/runs/{key}/notes/{note['note_id']}",
            json={"expected_revision": 1, "body_md": "Corrected."},
        )
        assert revised.json()["revision"] == 2
        stale = client.put(
            f"/api/v1/runs/{key}/notes/{note['note_id']}", json={"expected_revision": 1, "body_md": "Stale."}
        )
        assert stale.status_code == 409 and stale.json()["current_revision"] == 2
        history = client.get(f"/api/v1/runs/{key}/notes/{note['note_id']}/history").json()
        assert [item["revision"] for item in history] == [1, 2]
        other = RunLocator(source_id="local", run_id="sampo-b").key
        assert client.get(f"/api/v1/runs/{other}/notes/{note['note_id']}/history").status_code == 404
        deleted = client.delete(f"/api/v1/runs/{key}/notes/{note['note_id']}", params={"expected_revision": 2})
        assert deleted.json()["deleted"] is True
        assert client.get(f"/api/v1/runs/{key}/notes").json() == []
        preview = client.post("/api/v1/notes/preview", json={"run_key": key, "body_md": "{{run.nothing}}"}).json()
        assert preview["unresolved"] and "⟦unresolved" in preview["text"]


def test_note_writes_are_refused_unless_enabled_and_need_a_store() -> None:
    client, _, _ = _notes_client(writes=False)
    key = RunLocator(source_id="fixture", run_id="runs/grpo-silver-pine").key
    with client:
        assert client.get("/api/v1/notes/settings").json() == {"writes": False}
        refused = client.post(f"/api/v1/runs/{key}/notes", json={"kind": "finding", "body_md": "x"})
        assert refused.status_code == 403 and refused.json()["code"] == "notes_read_only"
    with _client() as plain:
        assert plain.get(f"/api/v1/runs/{key}/notes").status_code == 501


@pytest.mark.asyncio
async def test_mcp_writes_notes_with_the_mcp_source_only_when_enabled() -> None:
    _, store, service = _notes_client()
    server = create_mcp(service)
    added = await server.call_tool(
        "add_run_note",
        {"source_id": "fixture", "run_id": "runs/grpo-silver-pine", "kind": "summary", "body_md": "Summary."},
    )
    assert cast(Any, added).structured_content["source"] == "mcp"
    assert [note.source for note in await store.list_notes("runs/grpo-silver-pine")] == ["mcp"]
    _, _, read_only = _notes_client(writes=False)
    names = {tool.name for tool in await create_mcp(read_only).list_tools()}
    assert "list_run_notes" in names and "add_run_note" not in names
