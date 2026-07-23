"""HTTP and schema contract tests for the packaged product."""

from __future__ import annotations

from fastapi.testclient import TestClient
from posttrain_observatory import FixtureRunDataSource, FixtureSemanticSummaryProvider, ObservatoryService
from posttrain_observatory.http import create_http_app
from posttrain_observatory.settings import ObservatorySettings


def _client() -> TestClient:
    service = ObservatoryService(
        {"fixture": FixtureRunDataSource()},
        semantic_provider=FixtureSemanticSummaryProvider(),
    )
    return TestClient(create_http_app(service, ObservatorySettings()))


def test_run_list_view_metrics_semantics_and_traces_share_one_api() -> None:
    with _client() as client:
        assert client.get("/health/live").json() == {"status": "ok"}
        runs = client.get("/api/v1/runs").json()
        sft = next(run for run in runs if run["run"]["job_kind"] == "train.sft")
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
        traces = client.get(f"/api/v1/runs/{evaluation['run_key']}/traces-evaluation").json()
        assert traces["included"] == 12


def test_openapi_contains_bounded_product_routes() -> None:
    schema = _client().get("/openapi.json").json()
    assert "/api/v1/runs/{run_key}/view" in schema["paths"]
    assert "/api/v1/runs/{run_key}/system-metrics" in schema["paths"]
    assert "/api/v1/runs/{run_key}/semantic-summary" in schema["paths"]
    assert "/api/v1/runs/{run_key}/traces-evaluation" in schema["paths"]
    schemas = schema["components"]["schemas"]
    assert "EvidenceCompleteness" in schemas
    assert schemas["RunView"]["properties"]["completeness"] == {"$ref": "#/components/schemas/EvidenceCompleteness"}
    assert schemas["JsonValue"] == {
        "description": "Any JSON-compatible value.",
        "title": "JsonValue",
    }
