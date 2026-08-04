"""HTTP and schema contract tests for the packaged product."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from posttrain_observatory import FixtureRunDataSource, FixtureSemanticSummaryProvider, ObservatoryService
from posttrain_observatory.discovery import TrackioSourceDiscovery
from posttrain_observatory.http import create_http_app
from posttrain_observatory.mcp import create_mcp
from posttrain_observatory.settings import ObservatorySettings
from posttrain_observatory.sources import RunSourceRegistry


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


def test_openapi_contains_bounded_product_routes() -> None:
    schema = _client().get("/openapi.json").json()
    assert "/api/v1/runs/locate" in schema["paths"]
    assert "/api/v1/runs/{run_key}/view" in schema["paths"]
    assert "/api/v1/runs/{run_key}/system-metrics" in schema["paths"]
    assert "/api/v1/runs/{run_key}/semantic-summary" in schema["paths"]
    assert "/api/v1/runs/{run_key}/traces-evaluation" in schema["paths"]
    assert "/api/v1/runs/{run_key}/comparison-key" in schema["paths"]
    assert "/api/v1/serving-capacity/work-packages/{work_package_id}" in schema["paths"]
    assert set(schema["paths"]["/api/v1/sources/refresh"]) == {"post"}
    schemas = schema["components"]["schemas"]
    assert "EvidenceCompleteness" in schemas
    assert schemas["RunView"]["properties"]["completeness"] == {"$ref": "#/components/schemas/EvidenceCompleteness"}
    assert schemas["TraceEvaluationView"]["properties"]["performance"] == {
        "$ref": "#/components/schemas/EvaluationPerformance"
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
    assert isinstance(mcp_result, tuple)
    assert mcp_result[1] == http_view
