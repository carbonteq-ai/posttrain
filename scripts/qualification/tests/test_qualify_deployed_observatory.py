from __future__ import annotations

import io
from collections.abc import Mapping
from pathlib import Path
from urllib.request import Request

import pytest

from scripts.qualification.qualify_deployed_observatory import (
    ObservatoryHttpClient,
    QualificationError,
    qualify_retained_runs,
)

RUN_IDS = {
    "data.prepare": "data-run",
    "train.sampo": "sampo-run",
    "serve.smoke": "serve-run",
    "train.distill": "distill-run",
}


def _run(run_id: str, job_kind: str, status: str) -> dict[str, object]:
    return {
        "run_id": run_id,
        "job_kind": job_kind,
        "status": status,
    }


def _metric(metric: str, value: float) -> dict[str, object]:
    return {
        "key": metric.replace("/", "_"),
        "metric": metric,
        "state": "available",
        "value": value,
    }


def _view(
    run_id: str,
    job_kind: str,
    status: str,
    *,
    summary: list[dict[str, object]] | None = None,
    artifact_kind: str | None = None,
    research_ready: bool = False,
    trace_count: int = 0,
    alerts: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    artifacts: list[dict[str, object]] = []
    if artifact_kind is not None:
        artifacts.append(
            {
                "direction": "output",
                "logical_name": "result",
                "kind": artifact_kind,
            }
        )
    return {
        "requested_mode": "job",
        "resolved_mode": "job",
        "fallback_reason": None,
        "view": {
            "view_kind": "job.metrics",
            "run": _run(run_id, job_kind, status),
            "summary": summary or [],
            "completeness": {
                "state": "complete" if status == "succeeded" else "insufficient",
                "research_ready": research_ready,
                "required_available": 4 if status == "succeeded" else 0,
                "required_total": 4,
                "conditional_available": 2 if job_kind == "train.sampo" else 0,
                "conditional_active": 2 if job_kind == "train.sampo" else 0,
            },
            "trace_count": trace_count,
            "trace_evaluation_enabled": job_kind in {"train.sampo", "train.distill"},
            "alerts": alerts or [],
            "artifacts": {"items": artifacts},
        },
    }


def _responses() -> dict[str, object]:
    responses: dict[str, object] = {}
    for index, (job_kind, run_id) in enumerate(RUN_IDS.items()):
        status = "failed" if job_kind == "train.distill" else "succeeded"
        run_key = f"opaque-{index}"
        responses[f"/api/v1/runs/locate?run_id={run_id}"] = [
            {
                "run_key": run_key,
                "run": _run(run_id, job_kind, status),
            }
        ]
        if job_kind == "data.prepare":
            view = _view(
                run_id,
                job_kind,
                status,
                summary=[
                    _metric("data/examples", 128),
                    _metric("data/bytes", 4096),
                ],
                artifact_kind="dataset",
            )
        elif job_kind == "train.sampo":
            view = _view(
                run_id,
                job_kind,
                status,
                research_ready=True,
                trace_count=20,
            )
        elif job_kind == "serve.smoke":
            view = _view(
                run_id,
                job_kind,
                status,
                summary=[
                    _metric("serve/probe_healthy", 1.0),
                    _metric("serve/probe_model_available", 1.0),
                    _metric("serve/probe_latency_seconds", 0.2),
                ],
                artifact_kind="serving-log",
            )
        else:
            view = _view(
                run_id,
                job_kind,
                status,
                trace_count=1,
                alerts=[
                    {
                        "id": "run-failed",
                        "severity": "error",
                        "message": "The run failed.",
                    }
                ],
            )
        responses[f"/api/v1/runs/{run_key}/view?mode=job"] = view
    return responses


def _getter(responses: Mapping[str, object]):
    def get_json(path: str) -> object:
        return responses[path]

    return get_json


def test_qualifies_retained_job_views_without_provider_storage_knowledge() -> None:
    runs = qualify_retained_runs(_getter(_responses()), RUN_IDS)

    assert [(run.job_kind, run.status) for run in runs] == [
        ("data.prepare", "succeeded"),
        ("train.sampo", "succeeded"),
        ("serve.smoke", "succeeded"),
        ("train.distill", "failed"),
    ]
    assert runs[1].research_ready is True
    assert runs[1].trace_count == 20
    assert runs[3].research_ready is False


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload["/api/v1/runs/opaque-1/view?mode=job"].update(
                {"resolved_mode": "generic", "fallback_reason": "unregistered"}
            ),
            "job-aware",
        ),
        (
            lambda payload: payload["/api/v1/runs/opaque-1/view?mode=job"]["view"]["completeness"].update(
                {"research_ready": False}
            ),
            "not research-ready",
        ),
        (
            lambda payload: payload["/api/v1/runs/opaque-2/view?mode=job"]["view"]["summary"][0].update({"value": 0.0}),
            "did not pass",
        ),
        (
            lambda payload: payload["/api/v1/runs/opaque-3/view?mode=job"]["view"].update({"alerts": []}),
            "failed run alert",
        ),
    ],
)
def test_rejects_semantically_incomplete_deployed_views(mutate, message: str) -> None:
    responses = _responses()
    mutate(responses)

    with pytest.raises(QualificationError, match=message):
        qualify_retained_runs(_getter(responses), RUN_IDS)


def test_rejects_ambiguous_retained_run_ids() -> None:
    responses = _responses()
    listed = responses["/api/v1/runs/locate?run_id=data-run"]
    assert isinstance(listed, list)
    listed.append(
        {
            "run_key": "other-source-key",
            "run": _run("data-run", "data.prepare", "succeeded"),
        }
    )

    with pytest.raises(QualificationError, match="ambiguous"):
        qualify_retained_runs(_getter(responses), RUN_IDS)


def test_http_client_injects_credentials_in_header_only(monkeypatch) -> None:
    captured: list[Request] = []

    class Response(io.BytesIO):
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    def urlopen(request: Request, **_kwargs):
        captured.append(request)
        return Response(b'{"version": "0.1.0"}')

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    client = ObservatoryHttpClient(
        "https://observatory.example",
        username="operator",
        password="secret",
        ca_file=None,
        timeout_seconds=1,
    )

    assert client.get_json("/version") == {"version": "0.1.0"}
    assert captured[0].full_url == "https://observatory.example/version"
    assert captured[0].get_header("Authorization", "").startswith("Basic ")
    assert "operator" not in captured[0].full_url
    assert "secret" not in captured[0].full_url


@pytest.mark.parametrize(
    "url",
    (
        "http://observatory.example",
        "https://operator:secret@observatory.example",
    ),
)
def test_http_client_rejects_insecure_or_embedded_credentials(url: str) -> None:
    with pytest.raises(QualificationError):
        ObservatoryHttpClient(
            url,
            username="operator",
            password="secret",
            ca_file=None,
            timeout_seconds=1,
        )


def test_http_client_reports_an_unreadable_ca_without_echoing_its_path() -> None:
    ca_file = Path("/private/credentials/observatory-ca.pem")

    with pytest.raises(QualificationError) as captured:
        ObservatoryHttpClient(
            "https://observatory.example",
            username="operator",
            password="secret",
            ca_file=ca_file,
            timeout_seconds=1,
        )

    assert str(ca_file) not in str(captured.value)


def test_a_deployment_with_no_retained_runs_still_qualifies() -> None:
    """The Observatory is a viewer, so deploying it cannot depend on its data.

    Requiring a run of every shape meant a fresh environment could never
    qualify a deployment, and an established one would stop qualifying as its
    fixtures aged out. What is retained is reported instead of demanded.
    """

    def get_json(path: str) -> object:
        assert path.startswith("/api/v1/runs?"), f"unexpected request: {path}"
        return []

    assert qualify_retained_runs(get_json, {}) == ()


def test_a_pinned_run_is_used_instead_of_discovery() -> None:
    """Naming a run reproduces one observation; it is not the ordinary path."""

    class _Stop(Exception):
        pass

    requested: list[str] = []

    def get_json(path: str) -> object:
        requested.append(path)
        raise _Stop

    with pytest.raises(_Stop):
        qualify_retained_runs(get_json, {"data.prepare": "pinned-run"})

    assert requested, "no request was made"
    assert "runs/locate" in requested[0], "a pinned run must skip discovery"
    assert "pinned-run" in requested[0]
