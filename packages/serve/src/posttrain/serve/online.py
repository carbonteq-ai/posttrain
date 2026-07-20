"""OpenAI-compatible endpoint operations with no vLLM implementation dependency."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import httpx
from posttrain.common import ModelProfile

from .profiles import VllmServeProfile


@dataclass(frozen=True, slots=True)
class Endpoint:
    base_url: str
    model: str
    api_key: str = "local"

    def __post_init__(self) -> None:
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("endpoint base_url must use HTTP or HTTPS")
        if not self.model:
            raise ValueError("endpoint model name cannot be empty")

    def url(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"

    @property
    def health_url(self) -> str:
        return self.base_url.rstrip("/").removesuffix("/v1") + "/health"


@dataclass(frozen=True, slots=True)
class LaunchRequest:
    model: ModelProfile
    profile: VllmServeProfile
    host: str = "127.0.0.1"
    port: int = 8000
    startup_timeout_seconds: float = 180.0

    def __post_init__(self) -> None:
        self.profile.validate_model(self.model)
        if not self.host:
            raise ValueError("host cannot be empty")
        if not 0 < self.port < 65_536:
            raise ValueError("port must be between 1 and 65535")
        if self.startup_timeout_seconds <= 0:
            raise ValueError("startup timeout must be positive")

    @property
    def endpoint(self) -> Endpoint:
        return Endpoint(f"http://{self.host}:{self.port}/v1", self.model.artifact.repo_id)


@dataclass(frozen=True, slots=True)
class ProbeResult:
    healthy: bool
    model_available: bool
    latency_seconds: float
    models: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    endpoint: Endpoint
    messages: tuple[Mapping[str, Any], ...]
    max_tokens: int
    temperature: float = 0.0
    reasoning_mode: str | None = None
    tools: tuple[Mapping[str, Any], ...] = ()
    timeout_seconds: float = 120.0

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("generation requires at least one message")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if self.temperature < 0:
            raise ValueError("temperature cannot be negative")
        if self.timeout_seconds <= 0:
            raise ValueError("generation timeout must be positive")


@dataclass(frozen=True, slots=True)
class GenerationResult:
    content: str
    reasoning: str
    tool_call_deltas: tuple[dict[str, Any], ...]
    input_tokens: int | None
    output_tokens: int | None
    latency_seconds: float
    ttft_seconds: float | None
    finish_reason: str | None
    events: tuple[dict[str, Any], ...]

    def as_json(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "reasoning": self.reasoning,
            "tool_call_deltas": list(self.tool_call_deltas),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_seconds": self.latency_seconds,
            "ttft_seconds": self.ttft_seconds,
            "finish_reason": self.finish_reason,
            "events": list(self.events),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "reasoning_characters": len(self.reasoning),
            "tool_call_delta_count": len(self.tool_call_deltas),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_seconds": self.latency_seconds,
            "ttft_seconds": self.ttft_seconds,
            "finish_reason": self.finish_reason,
        }


def probe(endpoint: Endpoint, *, client: httpx.Client | None = None) -> ProbeResult:
    """Check both process health and whether the requested model is exposed."""

    owns_client = client is None
    active = client or httpx.Client(timeout=10)
    started = time.perf_counter()
    try:
        health = active.get(endpoint.health_url)
        health.raise_for_status()
        models_response = active.get(endpoint.url("models"))
        models_response.raise_for_status()
        models = tuple(str(item["id"]) for item in models_response.json().get("data", ()) if "id" in item)
        return ProbeResult(True, endpoint.model in models, time.perf_counter() - started, models)
    finally:
        if owns_client:
            active.close()


def _event_payload(line: str) -> dict[str, Any] | None:
    if not line.startswith("data:"):
        return None
    value = line.removeprefix("data:").strip()
    if not value or value == "[DONE]":
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("stream event must be a JSON object")
    return parsed


def generate(
    request: GenerationRequest,
    model: ModelProfile,
    *,
    client: httpx.Client | None = None,
) -> GenerationResult:
    """Stream one chat request and preserve both normalized fields and raw events."""

    if request.endpoint.model != model.artifact.repo_id:
        raise ValueError("generation endpoint does not match the selected model profile")
    mode = request.reasoning_mode or model.default_reasoning_mode
    extra = model.conversation.reasoning_mode(mode).kwargs()
    payload: dict[str, Any] = {
        "model": request.endpoint.model,
        "messages": list(request.messages),
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
        **extra,
    }
    if request.tools:
        payload.update({"tools": list(request.tools), "tool_choice": "auto"})

    owns_client = client is None
    active = client or httpx.Client(timeout=request.timeout_seconds)
    started = time.perf_counter()
    first_token_at: float | None = None
    content: list[str] = []
    reasoning: list[str] = []
    tool_deltas: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    input_tokens: int | None = None
    output_tokens: int | None = None
    finish_reason: str | None = None
    try:
        with active.stream(
            "POST",
            request.endpoint.url("chat/completions"),
            headers={"Authorization": f"Bearer {request.endpoint.api_key}"},
            json=payload,
            timeout=request.timeout_seconds,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                event = _event_payload(line)
                if event is None:
                    continue
                events.append(event)
                usage = event.get("usage")
                if isinstance(usage, dict):
                    input_tokens = usage.get("prompt_tokens", input_tokens)
                    output_tokens = usage.get("completion_tokens", output_tokens)
                choices = event.get("choices") or ()
                for choice in choices:
                    finish_reason = choice.get("finish_reason") or finish_reason
                    delta = choice.get("delta") or {}
                    content_delta = delta.get("content") or ""
                    reasoning_delta = delta.get("reasoning") or delta.get("reasoning_content") or ""
                    calls = delta.get("tool_calls") or ()
                    if first_token_at is None and (content_delta or reasoning_delta or calls):
                        first_token_at = time.perf_counter()
                    content.append(str(content_delta))
                    reasoning.append(str(reasoning_delta))
                    tool_deltas.extend(dict(item) for item in calls)
        finished = time.perf_counter()
    finally:
        if owns_client:
            active.close()
    return GenerationResult(
        content="".join(content),
        reasoning="".join(reasoning),
        tool_call_deltas=tuple(tool_deltas),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_seconds=finished - started,
        ttft_seconds=(first_token_at - started) if first_token_at is not None else None,
        finish_reason=finish_reason,
        events=tuple(events),
    )


__all__ = [
    "Endpoint",
    "GenerationRequest",
    "GenerationResult",
    "LaunchRequest",
    "ProbeResult",
    "generate",
    "probe",
]
