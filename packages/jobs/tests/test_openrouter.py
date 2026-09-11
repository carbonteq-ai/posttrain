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
    ProviderEndpointProfile,
    RunContext,
)
from posttrain.jobs import ExternalInferenceServiceRequest, ExternalInferenceUsageProjection
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
            {"reasoning": True},
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
        ProviderEndpointProfile("json-schema"),
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


def _request(binding: HostedInferenceBinding | None = None) -> ExternalInferenceServiceRequest:
    return ExternalInferenceServiceRequest(
        binding or _binding(),
        ExternalInferenceUsageProjection(requests=1, input_tokens=8_192, output_tokens=16_384),
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
        assert payload["response_format"]["type"] in {"json_schema", "json_object"}
        if payload["response_format"]["type"] == "json_schema":
            assert payload["max_tokens"] == 512
            assert "reasoning" not in payload
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
    with resolver(_context(tmp_path), "judge/quality", _request(binding)) as resolved:
        identity = resolved.trace_identity()
        assert resolved.provider["provider_slug"] == "open-inference"
        assert resolved.provider["requested_provider"] == "open-inference/fp8"
        assert resolved.provider["pricing"] == {
            "prompt": "0.00000005",
            "completion": "0.00000016",
        }
        assert "artifact_digest" not in identity
        assert "secret-value" not in str(identity)
        response = httpx.post(
            f"{resolved.endpoint.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {resolved.endpoint.api_key}"},
            json={
                "model": binding.model.model,
                "messages": [{"role": "user", "content": "Return ready JSON"}],
                "max_tokens": 64,
                "response_format": {"type": "json_object"},
                "provider": resolved.provider["route"],
            },
        )
        assert response.status_code == 200
    assert [request.method for request in requests] == ["GET", "POST", "POST"]


def test_resolver_probe_honors_disabled_reasoning(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_OPENROUTER_API_KEY", "secret-value")
    binding = _binding()
    binding = HostedInferenceBinding(
        binding.id,
        binding.revision,
        binding.model,
        binding.service,
        binding.provider,
        binding.provider_profile,
        {"temperature": 0.0, "max_tokens": 16_384, "extra_body": {"reasoning": {"enabled": False}}},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json=_inventory())
        payload = __import__("json").loads(request.content)
        assert payload["reasoning"] == {"enabled": False}
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

    resolver = OpenRouterResolver(
        client_factory=lambda **kwargs: httpx.Client(transport=httpx.MockTransport(handler), **kwargs)
    )
    with resolver(_context(tmp_path), "judge/quality", _request(binding)):
        pass


def test_resolver_fails_before_network_when_credential_is_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("TEST_OPENROUTER_API_KEY", raising=False)
    resolver = OpenRouterResolver(client_factory=lambda **_: pytest.fail("network must not be opened"))
    with pytest.raises(OpenRouterResolutionError, match="credential variable"):
        with resolver(_context(tmp_path), "judge/quality", _request()):
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
        with resolver(_context(tmp_path), "judge/quality", _request()):
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
        with resolver(_context(tmp_path), "judge/quality", _request()):
            pytest.fail("invalid capability probe admitted")


def test_resolver_reports_non_normal_probe_finish_reason(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_OPENROUTER_API_KEY", "secret-value")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json=_inventory())
        return httpx.Response(
            200,
            json={
                "id": "gen-length",
                "model": "deepseek/deepseek-v4-flash-0731",
                "provider": "OpenInference",
                "choices": [{"message": {"content": ""}, "finish_reason": "length"}],
                "usage": {"completion_tokens": 64},
            },
        )

    resolver = OpenRouterResolver(
        client_factory=lambda **kwargs: httpx.Client(transport=httpx.MockTransport(handler), **kwargs)
    )
    with pytest.raises(
        OpenRouterResolutionError,
        match=r"finish_reason='length', completion_tokens=64",
    ):
        with resolver(_context(tmp_path), "judge/quality", _request()):
            pytest.fail("non-normal probe finish admitted")


def test_resolver_rejects_endpoint_without_required_parameters(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_OPENROUTER_API_KEY", "secret-value")
    inventory = _inventory()
    for endpoint in inventory["data"]["endpoints"]:
        endpoint["supported_parameters"] = ["max_tokens"]
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json=inventory))
    resolver = OpenRouterResolver(client_factory=lambda **kwargs: httpx.Client(transport=transport, **kwargs))
    with pytest.raises(OpenRouterResolutionError, match="no healthy endpoint"):
        with resolver(_context(tmp_path), "judge/quality", _request()):
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
        binding.provider_profile,
        binding.sampling,
    )
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json=_inventory()))
    resolver = OpenRouterResolver(client_factory=lambda **kwargs: httpx.Client(transport=transport, **kwargs))
    with pytest.raises(OpenRouterResolutionError, match="provider-not-in-inventory"):
        with resolver(_context(tmp_path), "judge/quality", _request(binding)):
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
        binding.provider_profile,
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
                "usage": {"prompt_tokens": 20, "completion_tokens": 4},
            },
        )

    transport = httpx.MockTransport(handler)
    resolver = OpenRouterResolver(client_factory=lambda **kwargs: httpx.Client(transport=transport, **kwargs))
    with resolver(_context(tmp_path), "judge/quality", _request(binding)) as resolved:
        assert resolved.provider["provider_slug"] == "slow"
        assert resolved.provider["pricing"] == {
            "prompt": "0.00000009",
            "completion": "0.00000030",
        }


def test_resolver_rejects_projected_run_cost_before_paid_probe(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_OPENROUTER_API_KEY", "secret-value")
    methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        return httpx.Response(200, json=_inventory())

    transport = httpx.MockTransport(handler)
    resolver = OpenRouterResolver(client_factory=lambda **kwargs: httpx.Client(transport=transport, **kwargs))
    expensive = ExternalInferenceServiceRequest(
        _binding(),
        ExternalInferenceUsageProjection(
            requests=2_000,
            input_tokens=2_000 * 8_192,
            output_tokens=2_000 * 16_384,
        ),
    )
    with pytest.raises(OpenRouterResolutionError, match="projected judge cost exceeds"):
        with resolver(_context(tmp_path), "judge/quality", expensive):
            pytest.fail("over-budget run admitted")
    assert methods == ["GET"]


def test_resolver_prices_time_window_overrides_conservatively(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_OPENROUTER_API_KEY", "secret-value")
    inventory = _inventory()
    selected = inventory["data"]["endpoints"][1]
    selected["pricing"]["overrides"] = [
        {"utc_start": 600, "utc_end": 1000, "prompt": "0.00000030", "completion": "0.00000120"}
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json=inventory)
        return httpx.Response(
            200,
            json={
                "id": "gen-override",
                "model": "deepseek/deepseek-v4-flash-0731",
                "provider": "OpenInference",
                "choices": [{"message": {"content": '{"ready":true}'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 4},
            },
        )

    transport = httpx.MockTransport(handler)
    resolver = OpenRouterResolver(client_factory=lambda **kwargs: httpx.Client(transport=transport, **kwargs))
    with resolver(_context(tmp_path), "judge/quality", _request()) as resolved:
        assert resolved.provider["pricing"] == {
            "prompt": "0.00000030",
            "completion": "0.00000120",
        }


def test_json_object_transport_preserves_local_schema_request_but_downgrades_upstream(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_OPENROUTER_API_KEY", "secret-value")
    binding = _binding()
    binding = HostedInferenceBinding(
        binding.id,
        binding.revision,
        binding.model,
        binding.service,
        binding.provider,
        ProviderEndpointProfile("json-object"),
        binding.sampling,
    )
    upstream_payloads: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json=_inventory())
        payload = __import__("json").loads(request.content)
        upstream_payloads.append(payload)
        return httpx.Response(
            200,
            json={
                "id": "gen-json-object",
                "model": binding.model.model,
                "provider": "OpenInference",
                "choices": [{"message": {"content": '{"ready":true}'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 4},
            },
        )

    transport = httpx.MockTransport(handler)
    resolver = OpenRouterResolver(client_factory=lambda **kwargs: httpx.Client(transport=transport, **kwargs))
    with resolver(_context(tmp_path), "judge/quality", _request(binding)) as resolved:
        assert resolved.judge_client.capabilities.structured_output_transport == "json-object"
        assert resolved.judge_client.capabilities.structured_output_validation == "schema-instruction-and-local"
        response = httpx.post(
            f"{resolved.endpoint.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {resolved.endpoint.api_key}"},
            json={
                "model": binding.model.model,
                "messages": [{"role": "user", "content": "Return ready."}],
                "max_tokens": 64,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "ready",
                        "strict": True,
                        "schema": {"type": "object"},
                    },
                },
            },
        )
        assert response.status_code == 200
    assert [payload["response_format"]["type"] for payload in upstream_payloads] == [
        "json_object",
        "json_object",
    ]
    runtime_payload = upstream_payloads[1]
    assert runtime_payload["provider"] == {
        "order": ["open-inference/fp8"],
        "allow_fallbacks": False,
        "require_parameters": True,
        "zdr": False,
        "data_collection": "allow",
    }
    schema_instruction = runtime_payload["messages"][0]
    assert schema_instruction["role"] == "system"
    assert "conforms exactly to this JSON Schema" in schema_instruction["content"]
    assert '"type":"object"' in schema_instruction["content"]
    assert runtime_payload["messages"][1] == {"role": "user", "content": "Return ready."}


def test_json_object_downgrade_appends_schema_to_existing_system_instruction(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_OPENROUTER_API_KEY", "secret-value")
    binding = _binding()
    binding = HostedInferenceBinding(
        binding.id,
        binding.revision,
        binding.model,
        binding.service,
        binding.provider,
        ProviderEndpointProfile("json-object"),
        binding.sampling,
    )
    upstream_payloads: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json=_inventory())
        payload = __import__("json").loads(request.content)
        upstream_payloads.append(payload)
        return httpx.Response(
            200,
            json={
                "id": "gen-json-object",
                "model": binding.model.model,
                "provider": "OpenInference",
                "choices": [{"message": {"content": '{"ready":true}'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 4},
            },
        )

    transport = httpx.MockTransport(handler)
    resolver = OpenRouterResolver(client_factory=lambda **kwargs: httpx.Client(transport=transport, **kwargs))
    with resolver(_context(tmp_path), "judge/quality", _request(binding)) as resolved:
        response = httpx.post(
            f"{resolved.endpoint.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {resolved.endpoint.api_key}"},
            json={
                "model": binding.model.model,
                "messages": [
                    {"role": "system", "content": "Judge the episode."},
                    {"role": "user", "content": "Input"},
                ],
                "max_tokens": 64,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "verdict",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {"assessments": {"type": "object"}},
                            "required": ["assessments"],
                            "additionalProperties": False,
                        },
                    },
                },
            },
        )
        assert response.status_code == 200

    runtime = upstream_payloads[1]
    assert len(runtime["messages"]) == 2
    assert runtime["messages"][0]["content"].startswith("Judge the episode.\n\n")
    assert '"required":["assessments"]' in runtime["messages"][0]["content"]
    assert runtime["messages"][1] == {"role": "user", "content": "Input"}


def test_resolver_surfaces_a_bounded_provider_error_detail(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_OPENROUTER_API_KEY", "secret-value")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json=_inventory())
        return httpx.Response(404, json={"error": {"message": "selected endpoint is unavailable"}})

    transport = httpx.MockTransport(handler)
    resolver = OpenRouterResolver(client_factory=lambda **kwargs: httpx.Client(transport=transport, **kwargs))
    with pytest.raises(OpenRouterResolutionError, match="selected endpoint is unavailable"):
        with resolver(_context(tmp_path), "judge/quality", _request()):
            pytest.fail("404 route admitted")


def test_hosted_binding_requires_an_explicit_provider():
    binding = _binding()
    with pytest.raises(ValueError, match="provider cannot be empty"):
        HostedInferenceBinding(
            binding.id,
            binding.revision,
            binding.model,
            binding.service,
            "",
            binding.provider_profile,
            binding.sampling,
        )
