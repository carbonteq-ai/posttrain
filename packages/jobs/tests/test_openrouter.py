"""OpenRouter route admission and secret-isolation tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
from posttrain.common import (
    ExternalInferenceService,
    HostedInferenceBinding,
    HostedModel,
    NullObserver,
    RunContext,
)
from posttrain.jobs import ExternalInferenceServiceRequest
from posttrain.jobs.providers.openrouter import OpenRouterResolutionError, OpenRouterResolver


def _binding() -> HostedInferenceBinding:
    return HostedInferenceBinding(
        "hosted-inference/deepseek-v4-flash-openrouter-judge@1",
        "1",
        HostedModel(
            "hosted-models/deepseek-v4-flash-0731@1",
            "0731",
            "deepseek/deepseek-v4-flash-0731",
            1_048_576,
            {"reasoning": True, "structured-output": True},
        ),
        ExternalInferenceService(
            "external-services/openrouter@1",
            "1",
            "https://openrouter.ai/api/v1",
            "TEST_OPENROUTER_API_KEY",
            provider_policy={
                "freeze_provider_per_run": True,
                "allow_fallbacks": False,
                "require_parameters": True,
                "zdr": False,
                "data_collection": "allow",
            },
        ),
        "open-inference/fp8",
        {"temperature": 0.0, "max_tokens": 16_384},
    )


def _context(tmp_path: Path) -> RunContext:
    return RunContext(
        project_id="openrouter-test",
        work_package_id="train/judged",
        run_id="run",
        job_kind="train.gdpo",
        job_definition_version="train/gdpo-judged@1",
        workspace=tmp_path,
        observer=NullObserver(),
    )


def _inventory() -> dict[str, Any]:
    return {
        "data": {
            "id": "deepseek/deepseek-v4-flash-0731",
            "endpoints": [
                {
                    "provider_name": "Slow Expensive",
                    "tag": "slow/fp8",
                    "status": 0,
                    "context_length": 1_048_576,
                    "max_completion_tokens": 393_216,
                    "supported_parameters": ["max_tokens", "reasoning", "response_format"],
                    "pricing": {"prompt": "0.00000009", "completion": "0.00000030"},
                    "quantization": "fp8",
                },
                {
                    "provider_name": "OpenInference",
                    "tag": "open-inference/fp8",
                    "status": 0,
                    "context_length": 1_048_576,
                    "max_completion_tokens": 393_216,
                    "supported_parameters": [
                        "max_tokens",
                        "reasoning",
                        "response_format",
                        "structured_outputs",
                    ],
                    "pricing": {"prompt": "0.00000005", "completion": "0.00000016"},
                    "quantization": "fp8",
                },
            ],
        }
    }


def test_resolver_freezes_explicit_provider_route_and_retains_no_secret(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_OPENROUTER_API_KEY", "secret-value")
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["Authorization"] == "Bearer secret-value"
        if request.method == "GET":
            return httpx.Response(200, json=_inventory())
        payload = __import__("json").loads(request.content)
        assert payload["provider"] == {
            "order": ["open-inference/fp8"],
            "allow_fallbacks": False,
            "require_parameters": True,
            "zdr": False,
            "data_collection": "allow",
        }
        assert payload["response_format"]["type"] == "json_schema"
        return httpx.Response(
            200,
            json={
                "id": "gen-1",
                "model": "deepseek/deepseek-v4-flash-0731",
                "provider": "OpenInference",
                "choices": [{"message": {"content": '{"ready":true}'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 4},
            },
        )

    transport = httpx.MockTransport(handler)

    def client_factory(**kwargs):
        return httpx.Client(transport=transport, **kwargs)

    resolver = OpenRouterResolver(client_factory=client_factory)
    binding = _binding()
    with resolver(_context(tmp_path), "judge/quality", ExternalInferenceServiceRequest(binding)) as resolved:
        identity = resolved.trace_identity()
        assert resolved.provider["provider_slug"] == "open-inference"
        assert resolved.provider["requested_provider"] == "open-inference/fp8"
        assert resolved.provider["pricing"] == {
            "prompt": "0.00000005",
            "completion": "0.00000016",
        }
        assert "artifact_digest" not in identity
        assert "secret-value" not in str(identity)
    assert [request.method for request in requests] == ["GET", "POST"]


def test_resolver_fails_before_network_when_credential_is_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("TEST_OPENROUTER_API_KEY", raising=False)
    resolver = OpenRouterResolver(client_factory=lambda **_: pytest.fail("network must not be opened"))
    with pytest.raises(OpenRouterResolutionError, match="credential variable"):
        with resolver(_context(tmp_path), "judge/quality", ExternalInferenceServiceRequest(_binding())):
            pytest.fail("missing credential admitted")


def test_resolver_rejects_provider_drift(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_OPENROUTER_API_KEY", "secret-value")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json=_inventory())
        return httpx.Response(
            200,
            json={
                "id": "gen-drift",
                "model": "deepseek/deepseek-v4-flash-0731",
                "provider": "Different Provider",
                "choices": [{"message": {"content": '{"ready":true}'}}],
            },
        )

    transport = httpx.MockTransport(handler)
    resolver = OpenRouterResolver(client_factory=lambda **kwargs: httpx.Client(transport=transport, **kwargs))
    with pytest.raises(OpenRouterResolutionError, match="outside the frozen route"):
        with resolver(_context(tmp_path), "judge/quality", ExternalInferenceServiceRequest(_binding())):
            pytest.fail("provider drift admitted")


def test_resolver_rejects_a_probe_that_ignores_the_structured_output_contract(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_OPENROUTER_API_KEY", "secret-value")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json=_inventory())
        return httpx.Response(
            200,
            json={
                "id": "gen-invalid",
                "model": "deepseek/deepseek-v4-flash-0731",
                "provider": "OpenInference",
                "choices": [{"message": {"content": "ready"}, "finish_reason": "stop"}],
            },
        )

    transport = httpx.MockTransport(handler)
    resolver = OpenRouterResolver(client_factory=lambda **kwargs: httpx.Client(transport=transport, **kwargs))
    with pytest.raises(OpenRouterResolutionError, match="invalid structured output"):
        with resolver(_context(tmp_path), "judge/quality", ExternalInferenceServiceRequest(_binding())):
            pytest.fail("invalid capability probe admitted")


def test_resolver_rejects_endpoint_without_required_parameters(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_OPENROUTER_API_KEY", "secret-value")
    inventory = _inventory()
    for endpoint in inventory["data"]["endpoints"]:
        endpoint["supported_parameters"] = ["max_tokens"]
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json=inventory))
    resolver = OpenRouterResolver(client_factory=lambda **kwargs: httpx.Client(transport=transport, **kwargs))
    with pytest.raises(OpenRouterResolutionError, match="no healthy endpoint"):
        with resolver(_context(tmp_path), "judge/quality", ExternalInferenceServiceRequest(_binding())):
            pytest.fail("incompatible endpoint admitted")


def test_resolver_rejects_an_unavailable_explicit_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_OPENROUTER_API_KEY", "secret-value")
    binding = _binding()
    binding = HostedInferenceBinding(
        binding.id,
        binding.revision,
        binding.model,
        binding.service,
        "provider-not-in-inventory",
        binding.sampling,
    )
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json=_inventory()))
    resolver = OpenRouterResolver(client_factory=lambda **kwargs: httpx.Client(transport=transport, **kwargs))
    with pytest.raises(OpenRouterResolutionError, match="provider-not-in-inventory"):
        with resolver(_context(tmp_path), "judge/quality", ExternalInferenceServiceRequest(binding)):
            pytest.fail("unavailable provider admitted")


def test_resolver_does_not_replace_the_explicit_provider_with_a_cheaper_one(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_OPENROUTER_API_KEY", "secret-value")
    binding = _binding()
    binding = HostedInferenceBinding(
        binding.id,
        binding.revision,
        binding.model,
        binding.service,
        "slow",
        binding.sampling,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json=_inventory())
        payload = __import__("json").loads(request.content)
        assert payload["provider"]["order"] == ["slow"]
        return httpx.Response(
            200,
            json={
                "id": "gen-explicit",
                "model": binding.model.model,
                "provider": "Slow Expensive",
                "choices": [{"message": {"content": '{"ready":true}'}}],
            },
        )

    transport = httpx.MockTransport(handler)
    resolver = OpenRouterResolver(client_factory=lambda **kwargs: httpx.Client(transport=transport, **kwargs))
    with resolver(_context(tmp_path), "judge/quality", ExternalInferenceServiceRequest(binding)) as resolved:
        assert resolved.provider["provider_slug"] == "slow"
        assert resolved.provider["pricing"] == {
            "prompt": "0.00000009",
            "completion": "0.00000030",
        }


def test_hosted_binding_requires_an_explicit_provider():
    binding = _binding()
    with pytest.raises(ValueError, match="provider cannot be empty"):
        HostedInferenceBinding(
            binding.id,
            binding.revision,
            binding.model,
            binding.service,
            "",
            binding.sampling,
        )
