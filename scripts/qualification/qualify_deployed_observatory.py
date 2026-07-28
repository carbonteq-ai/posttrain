#!/usr/bin/env python3
"""Qualify retained runs through a deployed Observatory's public HTTP API."""

from __future__ import annotations

import argparse
import base64
import json
import os
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

JsonObject = Mapping[str, Any]
JsonGetter = Callable[[str], object]

RUN_ID_ENVIRONMENTS = {
    "data.prepare": "POSTTRAIN_QUALIFY_DATA_PREPARE_RUN_ID",
    "train.sampo": "POSTTRAIN_QUALIFY_SAMPO_RUN_ID",
    "serve.smoke": "POSTTRAIN_QUALIFY_SERVE_SMOKE_RUN_ID",
    "train.distill": "POSTTRAIN_QUALIFY_FAILED_DISTILL_RUN_ID",
}
"""Pin a specific retained run per job kind, overriding what would be found.

Naming a run is for reproducing one observation, not for ordinary use. The
Observatory is a viewer: whether it can be deployed cannot depend on which
runs happen to be retained, or a fresh environment could never deploy one and
every environment would drift as its fixtures aged out.
"""

QUALIFIED_JOB_KINDS = (
    ("data.prepare", "succeeded"),
    ("train.sampo", "succeeded"),
    ("serve.smoke", "succeeded"),
    ("train.distill", "failed"),
)


class QualificationError(RuntimeError):
    """A release-gate failure safe to show in an operator terminal."""


@dataclass(frozen=True, slots=True)
class QualifiedRun:
    run_id: str
    job_kind: str
    status: str
    view_kind: str
    completeness: str
    research_ready: bool
    trace_count: int

    def to_json(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "job_kind": self.job_kind,
            "status": self.status,
            "view_kind": self.view_kind,
            "completeness": self.completeness,
            "research_ready": self.research_ready,
            "trace_count": self.trace_count,
        }


class ObservatoryHttpClient:
    """Small HTTPS client whose credentials never enter URLs or receipts."""

    def __init__(
        self,
        base_url: str,
        *,
        username: str,
        password: str,
        ca_file: Path | None,
        timeout_seconds: float,
    ) -> None:
        parsed = urllib.parse.urlsplit(base_url)
        if parsed.scheme != "https":
            raise QualificationError("the deployed Observatory URL must use HTTPS")
        if parsed.username is not None or parsed.password is not None:
            raise QualificationError("credentials must not be embedded in the Observatory URL")
        if not parsed.netloc or parsed.query or parsed.fragment:
            raise QualificationError("the deployed Observatory URL is invalid")
        self.base_url = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", "")).rstrip("/")
        token = base64.b64encode(f"{username}:{password}".encode()).decode()
        self._authorization = f"Basic {token}"
        try:
            self._context = ssl.create_default_context(cafile=str(ca_file) if ca_file else None)
        except OSError:
            raise QualificationError("the Observatory CA bundle could not be loaded") from None
        self._timeout_seconds = timeout_seconds

    def get_json(self, path: str) -> object:
        if not path.startswith("/"):
            raise ValueError("Observatory API paths must be absolute")
        request = urllib.request.Request(
            self.base_url + path,
            headers={
                "Accept": "application/json",
                "Authorization": self._authorization,
            },
            method="GET",
        )
        try:
            with urllib.request.urlopen(
                request,
                context=self._context,
                timeout=self._timeout_seconds,
            ) as response:
                if response.status != 200:
                    raise QualificationError(f"Observatory request {path} returned HTTP {response.status}")
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise QualificationError(f"Observatory request {path} returned HTTP {error.code}") from None
        except (urllib.error.URLError, TimeoutError, ssl.SSLError, json.JSONDecodeError):
            raise QualificationError(f"Observatory request {path} failed") from None


def _object(value: object, label: str) -> JsonObject:
    if not isinstance(value, Mapping):
        raise QualificationError(f"{label} must be a JSON object")
    return value


def _items(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise QualificationError(f"{label} must be a JSON array")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise QualificationError(f"{label} must be a non-empty string")
    return value


def _integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise QualificationError(f"{label} must be a non-negative integer")
    return value


def _summary_by_metric(view: JsonObject) -> dict[str, JsonObject]:
    result: dict[str, JsonObject] = {}
    for index, raw in enumerate(_items(view.get("summary"), "run view summary")):
        item = _object(raw, f"run view summary item {index}")
        metric = item.get("metric")
        if isinstance(metric, str):
            result[metric] = item
    return result


def _require_available_metrics(
    view: JsonObject,
    metrics: Sequence[str],
    issues: list[str],
) -> dict[str, JsonObject]:
    summary = _summary_by_metric(view)
    for metric in metrics:
        item = summary.get(metric)
        if item is None:
            issues.append(f"summary does not expose {metric}")
        elif item.get("state") != "available":
            issues.append(f"summary metric {metric} is not available")
    return summary


def _require_output_artifact(view: JsonObject, kind: str, issues: list[str]) -> None:
    artifacts = _object(view.get("artifacts"), "run view artifacts")
    items = _items(artifacts.get("items"), "run view artifact items")
    if not any(
        isinstance(item, Mapping) and item.get("direction") == "output" and item.get("kind") == kind for item in items
    ):
        issues.append(f"no retained output artifact has kind {kind}")


def _qualify_view(
    *,
    run_id: str,
    job_kind: str,
    expected_status: str,
    located: JsonObject,
    response: JsonObject,
) -> QualifiedRun:
    issues: list[str] = []
    listed_run = _object(located.get("run"), "located run")
    view = _object(response.get("view"), "run view")
    view_run = _object(view.get("run"), "run view identity")
    completeness = _object(view.get("completeness"), "run view completeness")

    if listed_run.get("run_id") != run_id or view_run.get("run_id") != run_id:
        issues.append("run identity changed between list and view responses")
    if listed_run.get("job_kind") != job_kind or view_run.get("job_kind") != job_kind:
        issues.append(f"expected job kind {job_kind}")
    if listed_run.get("status") != expected_status or view_run.get("status") != expected_status:
        issues.append(f"expected terminal status {expected_status}")
    if response.get("resolved_mode") != "job":
        issues.append("Observatory did not resolve the run to a job-aware view")
    if response.get("fallback_reason") is not None:
        issues.append("Observatory reported a fallback reason")
    if view.get("view_kind") != "job.metrics":
        issues.append("Observatory did not return the job.metrics projection")

    completeness_state = completeness.get("state")
    research_ready = completeness.get("research_ready")
    required_available = completeness.get("required_available")
    required_total = completeness.get("required_total")
    conditional_available = completeness.get("conditional_available")
    conditional_active = completeness.get("conditional_active")
    trace_count = view.get("trace_count")
    if not isinstance(research_ready, bool):
        issues.append("research readiness is not a boolean")

    if job_kind == "data.prepare":
        if completeness_state != "complete":
            issues.append("dataset preparation evidence is not complete")
        _require_available_metrics(view, ("data/examples", "data/bytes"), issues)
        _require_output_artifact(view, "dataset", issues)
    elif job_kind == "train.sampo":
        if completeness_state != "complete":
            issues.append("SAMPO evidence is not complete")
        if research_ready is not True:
            issues.append("SAMPO evidence is not research-ready")
        if view.get("trace_evaluation_enabled") is not True:
            issues.append("SAMPO trace evaluation is not enabled")
        if not isinstance(trace_count, int) or isinstance(trace_count, bool) or trace_count <= 0:
            issues.append("SAMPO has no retained rollout traces")
        if required_available != required_total:
            issues.append("SAMPO required evidence is incomplete")
        if conditional_available != conditional_active:
            issues.append("SAMPO active conditional evidence is incomplete")
    elif job_kind == "serve.smoke":
        if completeness_state != "complete":
            issues.append("serving smoke evidence is not complete")
        summary = _require_available_metrics(
            view,
            (
                "serve/probe_healthy",
                "serve/probe_model_available",
                "serve/probe_latency_seconds",
            ),
            issues,
        )
        for metric in ("serve/probe_healthy", "serve/probe_model_available"):
            item = summary.get(metric)
            if item is not None and item.get("state") == "available" and item.get("value") != 1.0:
                issues.append(f"{metric} did not pass")
        _require_output_artifact(view, "serving-log", issues)
    elif job_kind == "train.distill":
        if research_ready is not False:
            issues.append("the failed distillation run is incorrectly research-ready")
        if view.get("trace_evaluation_enabled") is not True:
            issues.append("distillation trace evaluation is not enabled")
        alert_ids = {
            item.get("id")
            for item in (
                _object(raw, f"run alert {index}")
                for index, raw in enumerate(_items(view.get("alerts"), "run view alerts"))
            )
        }
        if "run-failed" not in alert_ids:
            issues.append("the failed run alert is missing")
    else:  # pragma: no cover - expectations are a closed framework-owned matrix
        raise AssertionError(f"unhandled qualification job kind: {job_kind}")

    if issues:
        raise QualificationError(f"{job_kind} run {run_id} failed qualification: " + "; ".join(issues))
    return QualifiedRun(
        run_id=run_id,
        job_kind=job_kind,
        status=_string(view_run.get("status"), "run status"),
        view_kind=_string(view.get("view_kind"), "view kind"),
        completeness=_string(completeness_state, "completeness state"),
        research_ready=bool(research_ready),
        trace_count=_integer(trace_count, "trace count"),
    )


def _discover_run_id(
    get_json: JsonGetter,
    job_kind: str,
    status: str,
) -> str | None:
    """Return the most recent retained run of this shape, if the deployment has one."""
    query = urllib.parse.urlencode({"job_kind": job_kind, "status": status, "limit": 1})
    listed = _items(get_json(f"/api/v1/runs?{query}"), f"{job_kind} runs")
    if not listed:
        return None
    first = _object(listed[0], f"{job_kind} run")
    identifier = first.get("run_id")
    return identifier if isinstance(identifier, str) and identifier else None


def qualify_retained_runs(
    get_json: JsonGetter,
    run_ids: Mapping[str, str],
) -> tuple[QualifiedRun, ...]:
    """Resolve opaque run keys, then assert job semantics for each retained run."""

    qualified: list[QualifiedRun] = []
    for job_kind, expected_status in QUALIFIED_JOB_KINDS:
        run_id = run_ids.get(job_kind) or _discover_run_id(get_json, job_kind, expected_status)
        if run_id is None:
            # Nothing of this shape is retained here. That is a fact about the
            # deployment's data, not a defect in the deployment.
            continue
        encoded_run_id = urllib.parse.quote(run_id, safe="")
        located = _items(
            get_json(f"/api/v1/runs/locate?run_id={encoded_run_id}"),
            "located runs",
        )
        matches = [_object(item, f"located run item {index}") for index, item in enumerate(located)]
        if not matches:
            raise QualificationError(f"{job_kind} retained run {run_id} is not visible")
        if len(matches) != 1:
            raise QualificationError(f"{job_kind} retained run {run_id} is ambiguous across Observatory sources")
        located = matches[0]
        run_key = _string(located.get("run_key"), "run key")
        encoded_key = urllib.parse.quote(run_key, safe="")
        response = _object(
            get_json(f"/api/v1/runs/{encoded_key}/view?mode=job"),
            "run view response",
        )
        qualified.append(
            _qualify_view(
                run_id=run_id,
                job_kind=job_kind,
                expected_status=expected_status,
                located=located,
                response=response,
            )
        )
    return tuple(qualified)


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise QualificationError(f"required environment variable {name} is not set")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default=os.environ.get("POSTTRAIN_OBSERVATORY_URL"),
        help="deployed HTTPS origin; defaults to POSTTRAIN_OBSERVATORY_URL",
    )
    parser.add_argument(
        "--ca-file",
        type=Path,
        default=(Path(value) if (value := os.environ.get("POSTTRAIN_OBSERVATORY_CA_FILE")) else None),
        help="private CA bundle; defaults to POSTTRAIN_OBSERVATORY_CA_FILE",
    )
    parser.add_argument("--timeout-seconds", type=float, default=15.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if not args.url:
            raise QualificationError("set POSTTRAIN_OBSERVATORY_URL or pass --url")
        username = _required_environment("POSTTRAIN_OBSERVATORY_USERNAME")
        password = _required_environment("POSTTRAIN_OBSERVATORY_PASSWORD")
        run_ids = {
            job_kind: value
            for job_kind, environment in RUN_ID_ENVIRONMENTS.items()
            if (value := os.environ.get(environment, "").strip())
        }
        client = ObservatoryHttpClient(
            args.url,
            username=username,
            password=password,
            ca_file=args.ca_file,
            timeout_seconds=args.timeout_seconds,
        )
        runs = qualify_retained_runs(client.get_json, run_ids)
    except QualificationError as error:
        print(f"Observatory qualification failed: {error}", file=sys.stderr)
        return 2

    receipt = {
        "schema": "posttrain.deployed-observatory-qualification.v1",
        "status": "passed",
        "observatory_url": client.base_url,
        "runs": [run.to_json() for run in runs],
        "job_kinds_covered": sorted({run.job_kind for run in runs}),
        "job_kinds_without_retained_runs": sorted(
            {job_kind for job_kind, _ in QUALIFIED_JOB_KINDS} - {run.job_kind for run in runs}
        ),
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
