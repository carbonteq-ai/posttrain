"""Named inference dependency lifecycle and identity tests."""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import replace
from typing import Any, cast

import pytest
from posttrain.catalog import open_catalog
from posttrain.common import (
    CatalogRef,
    ExecutionTarget,
    ExternalInferenceService,
    HostedModel,
    InferenceBinding,
    ModelVariant,
    NullObserver,
    RunContext,
)
from posttrain.jobs import (
    AttachedInferenceService,
    ExternalInferenceServiceRequest,
    HostedInferenceBinding,
    ManagedInferenceService,
    ResolvedInferenceService,
    bind_inference_services,
)
from posttrain.serve import Endpoint, ProbeResult, ServeLaunchRequest


@pytest.fixture
def service_selection(tmp_path):
    model = cast(
        ModelVariant,
        open_catalog(scope="service-test").resolve(CatalogRef("model", "models/qwen3.5-2b@bf16")).value,
    )
    inference = InferenceBinding(
        "inference/judge@1",
        "1",
        model,
        "vllm@0.25.1",
        model.renderer_contract,
        {"max_model_len": 4096},
        {"max_tokens": 128, "temperature": 0.0},
        ExecutionTarget("targets/judge", "1", "nvidia-cuda"),
        ("eval",),
    )
    context = RunContext(
        project_id="service-test",
        work_package_id="train/judged",
        run_id="run",
        job_kind="train.gdpo",
        job_definition_version="train/gdpo-judged@1",
        workspace=tmp_path.resolve(),
        observer=NullObserver(),
    )
    return context, inference


def _ready(_context: RunContext, endpoint: Endpoint) -> ProbeResult:
    return ProbeResult(True, True, 0.01, (endpoint.model,))


def test_managed_services_use_isolated_workspaces_and_close_in_reverse_order(service_selection):
    context, inference = service_selection
    started: list[tuple[int, object]] = []
    closed: list[int] = []

    @contextmanager
    def provisioner(service_context, request):
        started.append((request.port, service_context.workspace))
        try:
            yield request.endpoint
        finally:
            closed.append(request.port)

    requests = {
        "judge/nanbeige": ManagedInferenceService(ServeLaunchRequest(inference, port=8123)),
        "judge/gemma": ManagedInferenceService(ServeLaunchRequest(inference, port=8124)),
    }
    with bind_inference_services(context, requests, provisioner=provisioner, readiness_probe=_ready) as services:
        assert tuple(services) == ("judge/nanbeige", "judge/gemma")
        assert services["judge/nanbeige"].owned is True
        assert services["judge/nanbeige"].endpoint.api_key == "local"
        assert started[0][1] != started[1][1]
        assert all(str(path).startswith(str(context.workspace / "inference-services")) for _, path in started)
    assert closed == [8124, 8123]


def test_attached_service_resolves_scoped_credential_and_is_never_stopped(service_selection, monkeypatch):
    context, inference = service_selection
    monkeypatch.setenv("TEST_JUDGE_API_KEY", "attached-secret")
    calls: list[str] = []

    def forbidden(*_):
        calls.append("provisioned")
        raise AssertionError("attached endpoint must not be provisioned")

    attached = AttachedInferenceService(
        inference,
        "https://judge.example/v1",
        inference.model.base.repo_id,
        "TEST_JUDGE_API_KEY",
    )
    with bind_inference_services(
        context,
        {"judge/attached": attached},
        provisioner=forbidden,
        readiness_probe=_ready,
    ) as services:
        resolved = services["judge/attached"]
        assert resolved.owned is False
        assert resolved.endpoint.api_key == "attached-secret"
        assert "attached-secret" not in str(resolved.trace_identity())
    assert calls == []
    assert os.environ["TEST_JUDGE_API_KEY"] == "attached-secret"


def test_readiness_failure_closes_all_started_managed_services(service_selection):
    context, inference = service_selection
    closed: list[int] = []

    @contextmanager
    def provisioner(_context, request):
        try:
            yield request.endpoint
        finally:
            closed.append(request.port)

    def readiness(_context, endpoint):
        available = endpoint.model if endpoint.base_url.endswith(":8123/v1") else "different-model"
        return ProbeResult(True, endpoint.model == available, 0.01, (available,))

    with pytest.raises(RuntimeError, match="did not expose selected model"):
        with bind_inference_services(
            context,
            {
                "judge/one": ManagedInferenceService(ServeLaunchRequest(inference, port=8123)),
                "judge/two": ManagedInferenceService(ServeLaunchRequest(inference, port=8124)),
            },
            provisioner=provisioner,
            readiness_probe=readiness,
        ):
            pytest.fail("unready service admitted")
    assert closed == [8124, 8123]


def test_managed_bind_addresses_must_be_explicitly_distinct(service_selection):
    context, inference = service_selection
    request = ServeLaunchRequest(inference, port=8123)
    with pytest.raises(ValueError, match="same bind address"):
        with bind_inference_services(
            context,
            {
                "judge/one": ManagedInferenceService(request),
                "judge/two": ManagedInferenceService(replace(request)),
            },
            readiness_probe=_ready,
        ):
            pytest.fail("conflicting services admitted")


def test_external_service_uses_resolver_without_inventing_model_artifact(service_selection, monkeypatch):
    context, _ = service_selection
    monkeypatch.setenv("TEST_OPENROUTER_API_KEY", "external-secret")
    hosted = HostedInferenceBinding(
        "hosted-inference/deepseek-judge@1",
        "1",
        HostedModel(
            "hosted-models/deepseek-v4-flash-0731@1",
            "0731",
            "deepseek/deepseek-v4-flash-0731",
            1_310_720,
            {"structured-output": True, "reasoning": True},
        ),
        ExternalInferenceService(
            "external-services/openrouter@1",
            "1",
            "https://openrouter.ai/api/v1",
            "TEST_OPENROUTER_API_KEY",
            provider_policy={"allow_fallbacks": False},
        ),
        "open-inference",
        {"temperature": 0.0, "max_tokens": 16_384},
    )
    request = ExternalInferenceServiceRequest(hosted)
    closed: list[str] = []

    @contextmanager
    def resolver(_context, name, selected):
        assert selected is request
        endpoint = Endpoint(hosted.service.base_url, hosted.model.model, os.environ[hosted.service.api_key_var])
        try:
            yield ResolvedInferenceService(
                name,
                hosted,
                endpoint,
                ProbeResult(True, True, 0.02, (hosted.model.model,)),
                owned=False,
                lifecycle="external",
                provider={"slug": "deepseek", "allow_fallbacks": False},
            )
        finally:
            closed.append(name)

    with bind_inference_services(context, {"judge/quality": request}, external_resolver=resolver) as services:
        resolved = services["judge/quality"]
        identity = resolved.trace_identity()
        assert resolved.lifecycle == "external"
        assert resolved.owned is False
        assert cast(dict[str, Any], identity["model"])["api_model"] == "deepseek/deepseek-v4-flash-0731"
        assert "artifact_digest" not in identity
        assert "external-secret" not in str(identity)
    assert closed == ["judge/quality"]


def test_external_service_rejects_an_unregistered_provider(service_selection):
    context, _ = service_selection
    hosted = HostedInferenceBinding(
        "hosted-inference/test@1",
        "1",
        HostedModel("hosted-models/test@1", "1", "provider/model", 4096),
        ExternalInferenceService("external-services/test@1", "1", "https://provider.example/v1", "TEST_API_KEY"),
        "provider",
        {"max_tokens": 128},
    )
    with pytest.raises(ValueError, match="no external inference provider adapter"):
        with bind_inference_services(
            context,
            {"judge/external": ExternalInferenceServiceRequest(hosted)},
        ):
            pytest.fail("unresolved external service admitted")
