"""Private external inference provider adapters."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from posttrain.common import RunContext

from ..inference_services import ExternalInferenceServiceRequest, ResolvedInferenceService
from .openrouter import OpenRouterResolutionError, OpenRouterResolver


@contextmanager
def resolve_external_service(
    context: RunContext,
    name: str,
    request: ExternalInferenceServiceRequest,
) -> Iterator[ResolvedInferenceService]:
    """Dispatch a declared external service to its private provider adapter."""

    origin = request.binding.service.origin
    if origin != "https://openrouter.ai":
        raise ValueError(f"no external inference provider adapter is registered for {origin!r}")
    with OpenRouterResolver()(context, name, request) as resolved:
        yield resolved


__all__ = [
    "OpenRouterResolutionError",
    "OpenRouterResolver",
    "resolve_external_service",
]
