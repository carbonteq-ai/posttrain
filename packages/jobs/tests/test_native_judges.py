"""Managed judge composition without importing a model or environment plugin."""

import json
import os
from contextlib import contextmanager
from dataclasses import replace
from typing import Any, cast

import pytest
from posttrain.catalog import open_catalog
from posttrain.common import (
    CatalogRef,
    ExecutionTarget,
    HostedInferenceBinding,
    InferenceBinding,
    ModelVariant,
    NullObserver,
    RunContext,
)
from posttrain.environment import EnvironmentBinding, EnvironmentSource, SamplingPolicy, VerifiersV1ConfigActivation
from posttrain.jobs import ResolvedInferenceService, bind_native_judge_services, bind_native_judges
from posttrain.serve import Endpoint, ProbeResult, ServeLaunchRequest


def _ready(_context, endpoint):
    return ProbeResult(True, True, 0.01, (endpoint.model,))


@pytest.fixture
def selections(tmp_path):
    model = cast(
        ModelVariant, open_catalog(scope="judge-test").resolve(CatalogRef("model", "models/qwen3.5-2b@bf16")).value
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
    request = ServeLaunchRequest(inference, port=8123)
    judge = {
        "id": "project-judge",
        "name": "quality",
        "model": request.endpoint.model,
        "model_revision": model.revision,
        "sampling": dict(inference.sampling),
    }
    environment = EnvironmentBinding(
        "environments/judged",
        "tool-use",
        EnvironmentSource("test", "https://example.test/env", "a" * 40),
        VerifiersV1ConfigActivation({"taskset": {"task": {"judges": [judge]}}}),
        SamplingPolicy(max_tokens=128),
        num_tasks=1,
    )
    context = RunContext(
        project_id="judge-test",
        work_package_id="train/judged",
        run_id="run",
        job_kind="train.gdpo",
        job_definition_version="train/gdpo-judged@1",
        workspace=tmp_path.resolve(),
        observer=NullObserver(),
    )
    return context, environment, request


def test_managed_judge_freezes_inference_without_mutating_plugin_or_retaining_secret(selections):
    context, environment, request = selections
    events = []

    @contextmanager
    def launcher(ctx, selected):
        assert ctx.workspace.parent == context.workspace / "inference-services"
        events.append("start")
        try:
            yield replace(selected.endpoint, api_key="test-secret")
        finally:
            events.append("stop")

    original = environment.activation.to_payload()
    key = ""
    with pytest.raises(RuntimeError, match="training failed"):
        with bind_native_judges(
            context, environment, {"quality": request}, launcher=launcher, readiness_probe=_ready
        ) as bound:
            config = cast(Any, bound.activation).config["taskset"]["task"]["judges"][0]
            key = config["api_key_var"]
            assert os.environ[key] == "test-secret"
            assert "test-secret" not in json.dumps(bound.activation.to_payload())
            identity = cast(Any, bound.parameters)["managed_judge_inference"]["quality"]
            assert identity["engine"] == dict(request.inference.engine)
            assert identity["model"]["model_revision"] == request.inference.model.revision
            assert config["sampling"] == dict(request.inference.sampling)
            assert environment.activation.to_payload() == original
            raise RuntimeError("training failed")
    assert key not in os.environ
    assert events == ["start", "stop"]


@pytest.mark.parametrize(
    "change,match",
    [
        ({"sampling": {"temperature": 1.0}}, "sampling differs"),
        ({"model_revision": "b" * 40}, "revision differs"),
        ({"model": "wrong"}, "model differs"),
    ],
)
def test_invalid_selection_fails_before_loading_any_model(selections, change, match):
    context, environment, request = selections
    raw = json.loads(json.dumps(dict(environment.activation.config)))
    raw["taskset"]["task"]["judges"][0].update(change)
    environment = replace(environment, activation=VerifiersV1ConfigActivation(raw))

    def forbidden(*_):
        raise AssertionError("model must not start")

    with pytest.raises(ValueError, match=match):
        with bind_native_judges(context, environment, {"quality": request}, launcher=forbidden):
            pytest.fail("invalid selection admitted")


def test_second_endpoint_startup_failure_closes_first_and_restores_credentials(selections):
    context, environment, request = selections
    raw = json.loads(json.dumps(dict(environment.activation.config)))
    entries = raw["taskset"]["task"]["judges"]
    entries.append({**entries[0], "name": "outcome"})
    environment = replace(environment, activation=VerifiersV1ConfigActivation(raw))
    before = dict(os.environ)
    closed = []

    @contextmanager
    def launcher(ctx, selected):
        if selected.port == 8124:
            raise RuntimeError("startup failed")
        try:
            yield Endpoint(selected.endpoint.base_url, selected.endpoint.model, "secret")
        finally:
            closed.append(selected.port)

    with pytest.raises(RuntimeError, match="startup failed"):
        with bind_native_judges(
            context,
            environment,
            {
                "quality": request,
                "outcome": replace(request, port=8124),
            },
            launcher=launcher,
            readiness_probe=_ready,
        ):
            pytest.fail("failed startup admitted")
    assert closed == [8123]
    assert dict(os.environ) == before


def test_unmanaged_composition_is_identity_preserving(selections):
    context, environment, _ = selections
    with bind_native_judges(context, environment, {}) as bound:
        assert bound is environment


def test_multiple_native_judges_can_share_one_resolved_service(selections):
    _, environment, request = selections
    raw = json.loads(json.dumps(dict(environment.activation.config)))
    raw["taskset"]["task"]["judges"].append({**raw["taskset"]["task"]["judges"][0], "name": "efficiency"})
    environment = replace(environment, activation=VerifiersV1ConfigActivation(raw))
    endpoint = replace(request.endpoint, api_key="shared-secret")
    service = ResolvedInferenceService(
        "judge/shared",
        request.inference,
        endpoint,
        ProbeResult(True, True, 0.01, (endpoint.model,)),
        True,
    )

    with bind_native_judge_services(
        environment,
        {"judge/shared": service},
        {"quality": "judge/shared", "efficiency": "judge/shared"},
    ) as bound:
        judges = cast(Any, bound.activation).config["taskset"]["task"]["judges"]
        assert judges[0]["base_url"] == judges[1]["base_url"]
        assert judges[0]["api_key_var"] == judges[1]["api_key_var"]
        assert os.environ[judges[0]["api_key_var"]] == "shared-secret"
        assert cast(Any, bound.parameters)["judge_service_bindings"] == {
            "quality": "judge/shared",
            "efficiency": "judge/shared",
        }
    assert judges[0]["api_key_var"] not in os.environ


def test_external_judge_injects_the_explicit_provider_route_without_a_model_artifact(selections):
    context, environment, _ = selections
    binding = cast(
        HostedInferenceBinding,
        open_catalog(scope="judge-test")
        .resolve(CatalogRef("hosted-inference", "hosted-inference/deepseek-v4-flash-openrouter-judge@1"))
        .value,
    )
    raw = json.loads(json.dumps(dict(environment.activation.config)))
    judge = raw["taskset"]["task"]["judges"][0]
    judge.update(
        model=binding.model.model,
        model_revision=binding.model.revision,
        sampling=dict(binding.sampling),
    )
    environment = replace(environment, activation=VerifiersV1ConfigActivation(raw))
    route = {
        "order": [binding.provider],
        "allow_fallbacks": False,
        "require_parameters": True,
    }
    service = ResolvedInferenceService(
        "judge/quality",
        binding,
        Endpoint(binding.service.base_url, binding.model.model, "external-secret"),
        ProbeResult(True, True, 0.01, (binding.model.model,)),
        False,
        lifecycle="external",
        provider={"provider_slug": binding.provider, "route": route},
    )

    with bind_native_judge_services(environment, {"judge/quality": service}, {"quality": "judge/quality"}) as bound:
        config = cast(Any, bound.activation).config["taskset"]["task"]["judges"][0]
        assert config["sampling"]["extra_body"]["provider"] == route
        identity = cast(Any, bound.parameters)["inference_services"]["judge/quality"]
        assert identity["requested_provider"] == "open-inference/fp8"
        assert "artifact_digest" not in identity
        assert "external-secret" not in json.dumps(bound.activation.to_payload())
