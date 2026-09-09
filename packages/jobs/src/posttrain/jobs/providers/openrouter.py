"""OpenRouter resolution for API-only judge services."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, cast

import httpx
from posttrain.common import JsonValue, RunContext
from posttrain.serve import Endpoint, ProbeResult

from ..inference_services import ExternalInferenceServiceRequest, ResolvedInferenceService

type ClientFactory = Callable[..., httpx.Client]


class OpenRouterResolutionError(RuntimeError):
    """A credential, inventory, capability, or route admission failure."""


class OpenRouterResolver:
    """Validate an explicitly selected OpenRouter model/provider route."""

    def __init__(
        self,
        *,
        client_factory: ClientFactory = httpx.Client,
        timeout_seconds: float = 30.0,
        capability_probe: bool = True,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("OpenRouter timeout must be positive")
        self._client_factory = client_factory
        self._timeout_seconds = timeout_seconds
        self._capability_probe = capability_probe

    @contextmanager
    def __call__(
        self,
        context: RunContext,
        name: str,
        request: ExternalInferenceServiceRequest,
    ) -> Iterator[ResolvedInferenceService]:
        binding = request.binding
        service = binding.service
        if service.origin != "https://openrouter.ai":
            raise OpenRouterResolutionError(
                f"OpenRouter resolver cannot resolve external service origin {service.origin!r}"
            )
        try:
            api_key = os.environ[service.api_key_var]
        except KeyError as error:
            raise OpenRouterResolutionError(
                f"OpenRouter credential variable {service.api_key_var!r} is not set"
            ) from error
        if not api_key.strip():
            raise OpenRouterResolutionError(f"OpenRouter credential variable {service.api_key_var!r} is empty")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            **dict(service.headers),
        }
        started = time.monotonic()
        with self._client_factory(headers=headers, timeout=self._timeout_seconds) as client:
            inventory = _inventory(client, service.base_url, binding.model.model)
            selected = _select_endpoint(inventory, binding)
            requested_provider = binding.provider
            provider_slug = _provider_slug(selected)
            route = _route_policy(service.provider_policy, requested_provider)
            probe_evidence = (
                _probe_capabilities(client, service.base_url, binding.model.model, route)
                if self._capability_probe
                else {}
            )
            reported_model = probe_evidence.get("model")
            if reported_model is not None and reported_model != binding.model.model:
                raise OpenRouterResolutionError(
                    "OpenRouter capability probe returned a model outside the selected hosted-model identity"
                )
            reported_provider = probe_evidence.get("provider")
            if reported_provider is not None and not _provider_matches(str(reported_provider), selected):
                raise OpenRouterResolutionError(
                    "OpenRouter capability probe returned a provider outside the frozen route"
                )
            endpoint = Endpoint(service.base_url, binding.model.model, api_key)
            provider: dict[str, JsonValue] = {
                "resolved_at": datetime.now(UTC).isoformat(),
                "resolution_latency_seconds": time.monotonic() - started,
                "provider_name": str(selected.get("provider_name", "")),
                "provider_slug": provider_slug,
                "requested_provider": requested_provider,
                "endpoint_tag": str(selected.get("tag", "")),
                "quantization": str(selected.get("quantization", "unknown")),
                "context_length": _optional_int(selected.get("context_length")),
                "max_completion_tokens": _optional_int(selected.get("max_completion_tokens")),
                "supported_parameters": cast(JsonValue, _string_list(selected.get("supported_parameters"))),
                "pricing": _safe_pricing(selected.get("pricing")),
                "route": route,
                "capability_probe": probe_evidence,
            }
            readiness = ProbeResult(
                True,
                True,
                cast(float, provider["resolution_latency_seconds"]),
                (binding.model.model,),
            )
            yield ResolvedInferenceService(
                name,
                binding,
                endpoint,
                readiness,
                owned=False,
                lifecycle="external",
                provider=provider,
            )
            context.event(
                "external_inference_service_drained",
                {
                    "service_name": name,
                    "provider_slug": provider_slug,
                    "requested_provider": requested_provider,
                    "model": binding.model.model,
                },
            )


def _inventory(client: httpx.Client, base_url: str, model: str) -> Mapping[str, Any]:
    author, separator, slug = model.partition("/")
    if not separator or not author or not slug:
        raise OpenRouterResolutionError("OpenRouter model identifier must be author/slug")
    response = client.get(f"{base_url.rstrip('/')}/models/{author}/{slug}/endpoints")
    _raise_for_status(response, "endpoint inventory")
    payload = _response_json(response, "endpoint inventory")
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict) or data.get("id") != model:
        raise OpenRouterResolutionError("OpenRouter endpoint inventory did not match the requested model")
    return data


def _required_parameters(binding: Any) -> frozenset[str]:
    required = {"max_tokens"}
    if binding.model.capabilities.get("reasoning") is True:
        required.add("reasoning")
    if binding.model.capabilities.get("structured-output") is True:
        required.add("response_format")
    return frozenset(required)


def _select_endpoint(inventory: Mapping[str, Any], binding: Any) -> Mapping[str, Any]:
    entries = inventory.get("endpoints")
    if not isinstance(entries, list):
        raise OpenRouterResolutionError("OpenRouter endpoint inventory has no endpoint list")
    required = _required_parameters(binding)
    max_tokens = binding.sampling.get("max_tokens", 1)
    compatible: list[Mapping[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("status") != 0:
            continue
        if not _endpoint_matches_provider(entry, binding.provider):
            continue
        supported = set(_string_list(entry.get("supported_parameters")))
        completion_limit = _optional_int(entry.get("max_completion_tokens"))
        if not required.issubset(supported):
            continue
        if isinstance(max_tokens, int) and completion_limit is not None and completion_limit < max_tokens:
            continue
        compatible.append(entry)
    if not compatible:
        missing = ", ".join(sorted(required))
        raise OpenRouterResolutionError(
            f"OpenRouter provider {binding.provider!r} has no healthy endpoint satisfying "
            f"judge parameters for model {binding.model.model!r}: {missing}"
        )
    return min(compatible, key=_endpoint_rank)


def _endpoint_rank(endpoint: Mapping[str, Any]) -> tuple[Decimal, Decimal, str]:
    pricing = endpoint.get("pricing")
    prompt = _decimal(pricing.get("prompt")) if isinstance(pricing, dict) else Decimal("Infinity")
    completion = _decimal(pricing.get("completion")) if isinstance(pricing, dict) else Decimal("Infinity")
    return (completion, prompt, str(endpoint.get("tag", "")))


def _decimal(value: object) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return Decimal("Infinity")


def _provider_slug(endpoint: Mapping[str, Any]) -> str:
    tag = endpoint.get("tag")
    if not isinstance(tag, str) or not tag:
        raise OpenRouterResolutionError("OpenRouter endpoint is missing its provider tag")
    return tag.partition("/")[0]


def _endpoint_matches_provider(endpoint: Mapping[str, Any], provider: str) -> bool:
    tag = endpoint.get("tag")
    if not isinstance(tag, str):
        return False
    return tag == provider if "/" in provider else tag == provider or tag.startswith(f"{provider}/")


def _route_policy(policy: Mapping[str, JsonValue], provider_slug: str) -> dict[str, JsonValue]:
    if policy.get("freeze_provider_per_run") is not True:
        raise OpenRouterResolutionError("OpenRouter judged training requires freeze_provider_per_run=true")
    if policy.get("allow_fallbacks") is not False:
        raise OpenRouterResolutionError("OpenRouter judged training requires allow_fallbacks=false")
    route: dict[str, JsonValue] = {
        "order": [provider_slug],
        "allow_fallbacks": False,
        "require_parameters": policy.get("require_parameters") is True,
    }
    if "zdr" in policy:
        route["zdr"] = policy["zdr"]
    if "data_collection" in policy:
        route["data_collection"] = policy["data_collection"]
    return route


def _probe_capabilities(
    client: httpx.Client,
    base_url: str,
    model: str,
    route: Mapping[str, JsonValue],
) -> dict[str, JsonValue]:
    response = client.post(
        f"{base_url.rstrip('/')}/chat/completions",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": "Return the requested JSON object and nothing else."},
                {"role": "user", "content": 'Return {"ready": true}.'},
            ],
            "temperature": 0,
            "max_tokens": 64,
            "reasoning": {"effort": "low"},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "posttrain_readiness",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"ready": {"type": "boolean", "const": True}},
                        "required": ["ready"],
                        "additionalProperties": False,
                    },
                },
            },
            "provider": dict(route),
        },
    )
    _raise_for_status(response, "capability probe")
    payload = _response_json(response, "capability probe")
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise OpenRouterResolutionError("OpenRouter capability probe returned no completion")
    first = choices[0]
    if first.get("finish_reason") not in {None, "stop"}:
        raise OpenRouterResolutionError("OpenRouter capability probe did not finish normally")
    message = first.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    try:
        structured = json.loads(content) if isinstance(content, str) else None
    except json.JSONDecodeError as error:
        raise OpenRouterResolutionError("OpenRouter capability probe returned invalid structured output") from error
    if structured != {"ready": True}:
        raise OpenRouterResolutionError("OpenRouter capability probe did not satisfy its structured-output contract")
    usage = payload.get("usage")
    return {
        "request_id": str(payload.get("id", "")),
        "model": str(payload.get("model", model)),
        "provider": str(payload.get("provider", "")),
        "usage": dict(usage) if isinstance(usage, dict) else {},
    }


def _raise_for_status(response: httpx.Response, operation: str) -> None:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        status = response.status_code
        category = (
            "credential"
            if status in {401, 403}
            else "service"
            if status in {429, 500, 502, 503, 504}
            else "configuration"
        )
        raise OpenRouterResolutionError(f"OpenRouter {operation} failed ({category}, HTTP {status})") from error


def _response_json(response: httpx.Response, operation: str) -> Any:
    try:
        return response.json()
    except ValueError as error:
        raise OpenRouterResolutionError(f"OpenRouter {operation} returned invalid JSON") from error


def _provider_matches(reported: str, selected: Mapping[str, Any]) -> bool:
    normalized = reported.casefold().replace(" ", "-")
    provider_name = str(selected.get("provider_name", "")).casefold().replace(" ", "-")
    return normalized in {provider_name, _provider_slug(selected).casefold()}


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _string_list(value: object) -> list[str]:
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


def _safe_pricing(value: object) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items() if isinstance(item, str | int | float | bool) or item is None}


__all__ = ["OpenRouterResolutionError", "OpenRouterResolver"]
