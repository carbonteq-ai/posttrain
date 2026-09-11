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
from posttrain.serve import ProbeResult

from ..inference_services import ExternalInferenceServiceRequest, ResolvedInferenceService
from .cost_control import (
    CostLedger,
    InferenceCostLimitError,
    TokenPrices,
    metered_openai_gateway,
    projected_cost,
    usage_cost,
)

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
            prices = _token_prices(selected)
            projection = request.usage
            projected = projected_cost(projection.input_tokens, projection.output_tokens, prices)
            limit = Decimal(binding.max_cost_usd_micros) / Decimal(1_000_000)
            if projected > limit:
                raise OpenRouterResolutionError(
                    "projected judge cost exceeds the selected run ceiling: "
                    f"${projected:.6f} > ${limit:.6f} for {projection.requests} bounded requests"
                )
            requested_provider = binding.provider
            provider_slug = _provider_slug(selected)
            route = _route_policy(service.provider_policy, requested_provider)
            probe_evidence = (
                _probe_capabilities(
                    client,
                    service.base_url,
                    binding.model.model,
                    route,
                    _probe_reasoning(binding),
                    _probe_temperature(binding),
                    _structured_output_transport(binding),
                )
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
            try:
                readiness_cost = usage_cost(probe_evidence.get("usage"), prices) if probe_evidence else Decimal(0)
            except InferenceCostLimitError as error:
                raise OpenRouterResolutionError(str(error)) from error
            if projected + readiness_cost > limit:
                raise OpenRouterResolutionError(
                    "projected judge traffic plus readiness cost exceeds the selected run cost ceiling"
                )
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
                # Retain the effective worst-case rates used by admission,
                # including any provider-declared time-window overrides.
                "pricing": {
                    "prompt": f"{prices.prompt:f}",
                    "completion": f"{prices.completion:f}",
                },
                "route": route,
                "capability_probe": probe_evidence,
                "cost_control": {
                    "limit_usd": f"{limit:f}",
                    "projected_usd": f"{projected:f}",
                    "projected_requests": projection.requests,
                    "projected_input_tokens": projection.input_tokens,
                    "projected_output_tokens": projection.output_tokens,
                    "readiness_cost_usd": f"{readiness_cost:f}",
                    "enforcement": "run-local-reservation-gateway@1",
                },
            }
            readiness = ProbeResult(
                True,
                True,
                cast(float, provider["resolution_latency_seconds"]),
                (binding.model.model,),
            )
            ledger = CostLedger(binding.max_cost_usd_micros, prices, initial_cost=readiness_cost)
            maximum_output_tokens = binding.sampling.get("max_tokens")
            assert isinstance(maximum_output_tokens, int)
            try:
                structured_output_transport = _structured_output_transport(binding)
                with metered_openai_gateway(
                    upstream=client,
                    upstream_base_url=service.base_url,
                    model=binding.model.model,
                    maximum_output_tokens=maximum_output_tokens,
                    ledger=ledger,
                    request_transform=_openrouter_transport(route, structured_output_transport),
                ) as endpoint:
                    yield ResolvedInferenceService(
                        name,
                        binding,
                        endpoint,
                        readiness,
                        owned=False,
                        lifecycle="external",
                        provider=provider,
                        protocol=service.protocol,
                    )
            finally:
                context.event(
                    "external_inference_service_drained",
                    {
                        "service_name": name,
                        "provider_slug": provider_slug,
                        "requested_provider": requested_provider,
                        "model": binding.model.model,
                        "cost_control": ledger.snapshot(),
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
    if _probe_reasoning(binding) is not None:
        required.add("reasoning")
    if _structured_output_transport(binding) is not None:
        required.add("response_format")
    return frozenset(required)


def _structured_output_transport(binding: Any) -> str | None:
    return binding.provider_profile.structured_output


def _probe_reasoning(binding: Any) -> dict[str, JsonValue] | None:
    """Probe the selected reasoning policy instead of overriding it.

    A readiness request must exercise the same provider feature set as the
    judged workload. In particular, enabling low-effort reasoning for a
    no-thinking binding can consume the tiny probe completion budget before
    its structured response is emitted.
    """

    extra = binding.sampling.get("extra_body")
    configured = extra.get("reasoning") if isinstance(extra, Mapping) else None
    if isinstance(configured, Mapping):
        enabled = configured.get("enabled")
        if isinstance(enabled, bool):
            return {"enabled": enabled}
    return None


def _probe_temperature(binding: Any) -> float | None:
    """Return only an explicitly selected temperature for the readiness call."""

    value = binding.sampling.get("temperature")
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


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


def _token_prices(endpoint: Mapping[str, Any]) -> TokenPrices:
    pricing = endpoint.get("pricing")
    if not isinstance(pricing, Mapping):
        raise OpenRouterResolutionError("OpenRouter endpoint has no usable pricing")
    # Provider prices may vary by time window. Admission must reserve against
    # the most expensive declared rate because a run can cross a pricing
    # boundary after endpoint resolution.
    prompt_rates = [_decimal(pricing.get("prompt"))]
    completion_rates = [_decimal(pricing.get("completion"))]
    overrides = pricing.get("overrides")
    if isinstance(overrides, list):
        for override in overrides:
            if not isinstance(override, Mapping):
                continue
            prompt_rates.append(_decimal(override.get("prompt")))
            completion_rates.append(_decimal(override.get("completion")))
    prompt = max(prompt_rates)
    completion = max(completion_rates)
    try:
        return TokenPrices(prompt, completion)
    except ValueError as error:
        raise OpenRouterResolutionError("OpenRouter endpoint has invalid paid-token pricing") from error


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
    reasoning: Mapping[str, JsonValue] | None,
    temperature: float | None,
    structured_output_transport: str | None,
) -> dict[str, JsonValue]:
    request: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return the requested JSON object and nothing else."},
            {"role": "user", "content": 'Return {"ready": true}.'},
        ],
        # A readiness check must admit enough output for providers that spend
        # a small internal budget before emitting a strict JSON object. This
        # is not the judge generation ceiling (which remains binding-owned).
        "max_tokens": 512,
        "provider": dict(route),
    }
    if structured_output_transport == "json-object":
        request["response_format"] = {"type": "json_object"}
    elif structured_output_transport == "json-schema":
        request["response_format"] = {
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
        }
    if temperature is not None:
        request["temperature"] = temperature
    if reasoning is not None:
        request["reasoning"] = dict(reasoning)
    response = client.post(
        f"{base_url.rstrip('/')}/chat/completions",
        json=request,
    )
    _raise_for_status(response, "capability probe")
    payload = _response_json(response, "capability probe")
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise OpenRouterResolutionError("OpenRouter capability probe returned no completion")
    first = choices[0]
    finish_reason = first.get("finish_reason")
    if finish_reason not in {None, "stop"}:
        usage = payload.get("usage")
        completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
        detail = f" (finish_reason={finish_reason!r}"
        if isinstance(completion_tokens, int):
            detail += f", completion_tokens={completion_tokens}"
        detail += ")"
        raise OpenRouterResolutionError(f"OpenRouter capability probe did not finish normally{detail}")
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


def _openrouter_transport(
    route: Mapping[str, JsonValue],
    structured_output_transport: str | None,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Freeze routing and adapt structured output at the OpenRouter boundary."""

    def transform(payload: dict[str, Any]) -> dict[str, Any]:
        transformed = dict(payload)
        # Callers talk to a provider-neutral local OpenAI endpoint. The route
        # is therefore an infrastructure concern and must be injected here,
        # not repeated in every judge implementation.
        transformed["provider"] = dict(route)
        response_format = transformed.get("response_format")
        if (
            structured_output_transport == "json-object"
            and isinstance(response_format, Mapping)
            and response_format.get("type") == "json_schema"
        ):
            transformed["response_format"] = {"type": "json_object"}
            messages = transformed.get("messages")
            if isinstance(messages, list):
                transformed["messages"] = _with_json_schema_instruction(messages, response_format)
        return transformed

    return transform


def _with_json_schema_instruction(messages: list[object], response_format: Mapping[str, Any]) -> list[object]:
    """Preserve the caller's schema when a route supports only JSON-object mode.

    OpenAI-compatible clients validate the returned object locally, but a
    JSON-object-only provider does not otherwise see the schema carried in
    ``response_format``. Dropping it would turn a typed judge request into an
    unconstrained JSON request and make valid multi-field verdicts accidental.
    """

    descriptor = response_format.get("json_schema")
    schema = descriptor.get("schema") if isinstance(descriptor, Mapping) else None
    if not isinstance(schema, Mapping):
        instruction = "Return only one valid JSON object, with no Markdown or surrounding text."
    else:
        encoded = json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        instruction = (
            "Return only one valid JSON object that conforms exactly to this JSON Schema. "
            "Include every required field, obey additionalProperties, and emit no Markdown or "
            f"surrounding text. JSON Schema: {encoded}"
        )

    updated: list[object] = []
    appended = False
    for message in messages:
        if not appended and isinstance(message, Mapping) and message.get("role") == "system":
            content = message.get("content")
            if isinstance(content, str):
                updated.append({**message, "content": f"{content}\n\n{instruction}"})
                appended = True
                continue
        updated.append(message)
    if not appended:
        updated.insert(0, {"role": "system", "content": instruction})
    return updated


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
        detail = _safe_error_detail(response)
        suffix = f": {detail}" if detail else ""
        raise OpenRouterResolutionError(f"OpenRouter {operation} failed ({category}, HTTP {status}){suffix}") from error


def _safe_error_detail(response: httpx.Response) -> str | None:
    """Expose a bounded provider diagnosis without leaking request credentials."""

    try:
        payload = response.json()
    except ValueError:
        return None
    if not isinstance(payload, Mapping):
        return None
    error = payload.get("error")
    value = error.get("message") if isinstance(error, Mapping) else error
    if not isinstance(value, str):
        return None
    normalized = " ".join(value.split())
    return normalized[:300] or None


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


__all__ = ["OpenRouterResolutionError", "OpenRouterResolver"]
