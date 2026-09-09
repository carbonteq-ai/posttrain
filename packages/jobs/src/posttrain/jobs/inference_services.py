"""Composition-host ownership for named inference service dependencies."""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, ExitStack, contextmanager
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Literal

from posttrain.common import HostedInferenceBinding, InferenceBinding, JsonValue, RunContext
from posttrain.common.selections import validate_selection_id
from posttrain.serve import Endpoint, ProbeResult, ServeLaunchRequest, launch, probe


@dataclass(frozen=True, slots=True)
class ManagedInferenceService:
    """A local model service whose lifecycle is owned by the composition host."""

    request: ServeLaunchRequest

    @property
    def inference(self) -> InferenceBinding:
        return self.request.inference


@dataclass(frozen=True, slots=True)
class AttachedInferenceService:
    """A pre-existing local-model endpoint that the composition host never stops."""

    inference: InferenceBinding
    base_url: str
    model: str
    api_key_var: str | None = None

    def __post_init__(self) -> None:
        Endpoint(self.base_url, self.model)
        if self.api_key_var is not None and not self.api_key_var.strip():
            raise ValueError("attached service api_key_var cannot be empty")

    def endpoint(self) -> Endpoint:
        api_key = "local"
        if self.api_key_var is not None:
            try:
                api_key = os.environ[self.api_key_var]
            except KeyError as error:
                raise ValueError(f"attached service credential variable {self.api_key_var!r} is not set") from error
        return Endpoint(self.base_url, self.model, api_key)


@dataclass(frozen=True, slots=True)
class ExternalInferenceUsageProjection:
    """Conservative run-wide request and token ceilings for one paid service."""

    requests: int
    input_tokens: int
    output_tokens: int

    def __post_init__(self) -> None:
        values = (self.requests, self.input_tokens, self.output_tokens)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in values):
            raise ValueError("external inference usage projection values must be positive integers")


@dataclass(frozen=True, slots=True)
class ExternalInferenceServiceRequest:
    """An external API service resolved but never deployed by Posttrain."""

    binding: HostedInferenceBinding
    usage: ExternalInferenceUsageProjection

    @property
    def inference(self) -> HostedInferenceBinding:
        return self.binding


type InferenceSelection = InferenceBinding | HostedInferenceBinding
type InferenceServiceRequest = ManagedInferenceService | AttachedInferenceService | ExternalInferenceServiceRequest


@dataclass(frozen=True, slots=True)
class ResolvedInferenceService:
    """One ready ephemeral connection and its exact secret-free selection."""

    name: str
    inference: InferenceSelection
    endpoint: Endpoint
    readiness: ProbeResult
    owned: bool
    lifecycle: Literal["managed", "attached", "external"] | None = None
    provider: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        lifecycle = self.lifecycle or ("managed" if self.owned else "attached")
        if self.owned != (lifecycle == "managed"):
            raise ValueError("only managed inference services may be owned")
        object.__setattr__(self, "lifecycle", lifecycle)
        object.__setattr__(self, "provider", MappingProxyType(dict(self.provider)))

    def trace_identity(self) -> dict[str, JsonValue]:
        identity: dict[str, JsonValue] = {
            "service_name": self.name,
            "lifecycle": self.lifecycle,
            "ownership": "managed" if self.owned else "unowned",
            "inference_binding_id": self.inference.id,
            "inference_binding_revision": self.inference.revision,
            "model": self.inference.model.trace_identity(),
            "sampling": dict(self.inference.sampling),
            "endpoint": {"base_url": self.endpoint.base_url, "model": self.endpoint.model},
            "readiness": {
                "healthy": self.readiness.healthy,
                "model_available": self.readiness.model_available,
                "models": list(self.readiness.models),
            },
            "provider": dict(self.provider),
        }
        if isinstance(self.inference, InferenceBinding):
            identity.update(
                {
                    "artifact_digest": self.inference.model.digest,
                    "backend": self.inference.backend,
                    "renderer": self.inference.renderer,
                    "engine": dict(self.inference.engine),
                    "target": {
                        "id": self.inference.target.id,
                        "revision": self.inference.target.revision,
                        "device_class": self.inference.target.device_class,
                        "memory_gb": self.inference.target.memory_gb,
                        "placement": dict(self.inference.target.placement),
                        "host_constraints": dict(self.inference.target.host_constraints),
                    },
                }
            )
        else:
            identity["external_service"] = self.inference.service.trace_identity()
            identity["requested_provider"] = self.inference.provider
        return identity


type ServiceProvisioner = Callable[[RunContext, ServeLaunchRequest], AbstractContextManager[Endpoint]]
type ReadinessProbe = Callable[[RunContext, Endpoint], ProbeResult]
type ExternalServiceResolver = Callable[
    [RunContext, str, ExternalInferenceServiceRequest], AbstractContextManager[ResolvedInferenceService]
]


@contextmanager
def bind_inference_services(
    context: RunContext,
    services: Mapping[str, InferenceServiceRequest],
    *,
    provisioner: ServiceProvisioner = launch,
    readiness_probe: ReadinessProbe = probe,
    external_resolver: ExternalServiceResolver | None = None,
) -> Iterator[Mapping[str, ResolvedInferenceService]]:
    """Resolve named services once and close only resources owned by each variant."""

    _validate_service_requests(services)
    if not services:
        yield MappingProxyType({})
        return

    resolved: dict[str, ResolvedInferenceService] = {}
    with ExitStack() as stack:
        for name, service in services.items():
            if isinstance(service, ExternalInferenceServiceRequest):
                if external_resolver is None:
                    from .providers import resolve_external_service

                    external_resolver = resolve_external_service
                context.event(
                    "inference_service_resolving",
                    _service_attributes(name, service.inference, lifecycle="external"),
                )
                connection = stack.enter_context(external_resolver(context, name, service))
                if connection.name != name or connection.inference != service.inference:
                    raise ValueError(f"external inference service {name!r} resolved a different selection")
                if connection.lifecycle != "external" or connection.owned:
                    raise ValueError(f"external inference service {name!r} returned invalid lifecycle ownership")
                if not connection.readiness.healthy or not connection.readiness.model_available:
                    raise RuntimeError(
                        f"inference service {name!r} did not expose selected model {connection.endpoint.model!r}"
                    )
                resolved[name] = connection
                context.event(
                    "inference_service_ready",
                    {
                        **_service_attributes(name, service.inference, lifecycle="external"),
                        "base_url": connection.endpoint.base_url,
                        "endpoint_model": connection.endpoint.model,
                    },
                )
                continue

            owned = isinstance(service, ManagedInferenceService)
            lifecycle: Literal["managed", "attached"] = "managed" if owned else "attached"
            if isinstance(service, ManagedInferenceService):
                service_context = _service_context(context, name)
                service_context.workspace.mkdir(parents=True, exist_ok=True)
                service_context.event(
                    "inference_service_starting",
                    _service_attributes(name, service.inference, lifecycle=lifecycle),
                )
                endpoint = stack.enter_context(provisioner(service_context, service.request))
                expected = service.request.endpoint
            else:
                endpoint = service.endpoint()
                expected = Endpoint(service.base_url, service.model)
                context.event(
                    "inference_service_attaching",
                    _service_attributes(name, service.inference, lifecycle=lifecycle),
                )
            if endpoint.base_url != expected.base_url or endpoint.model != expected.model:
                raise ValueError(f"inference service {name!r} endpoint differs from its selected address or model")
            readiness = readiness_probe(context, endpoint)
            if not readiness.healthy or not readiness.model_available:
                raise RuntimeError(f"inference service {name!r} did not expose selected model {endpoint.model!r}")
            connection = ResolvedInferenceService(
                name=name,
                inference=service.inference,
                endpoint=endpoint,
                readiness=readiness,
                owned=owned,
                lifecycle=lifecycle,
            )
            resolved[name] = connection
            context.event(
                "inference_service_ready",
                {
                    **_service_attributes(name, service.inference, lifecycle=lifecycle),
                    "base_url": endpoint.base_url,
                    "endpoint_model": endpoint.model,
                },
            )
        try:
            yield MappingProxyType(resolved)
        finally:
            for connection in reversed(tuple(resolved.values())):
                context.event(
                    "inference_service_released",
                    _service_attributes(connection.name, connection.inference, lifecycle=connection.lifecycle),
                )


def _validate_service_requests(services: Mapping[str, InferenceServiceRequest]) -> None:
    managed_addresses: dict[tuple[str, int], str] = {}
    for name, service in services.items():
        validate_selection_id(name, "inference service name")
        if not isinstance(
            service,
            ManagedInferenceService | AttachedInferenceService | ExternalInferenceServiceRequest,
        ):
            raise TypeError(f"inference service {name!r} has an unsupported request type")
        if isinstance(service, ManagedInferenceService):
            address = (service.request.host, service.request.port)
            previous = managed_addresses.get(address)
            if previous is not None:
                raise ValueError(f"managed inference services {previous!r} and {name!r} use the same bind address")
            managed_addresses[address] = name


def _service_context(context: RunContext, name: str) -> RunContext:
    readable = re.sub(r"[^a-z0-9._-]+", "-", name).strip("-.") or "service"
    suffix = hashlib.sha256(name.encode()).hexdigest()[:12]
    return replace(context, workspace=context.workspace / "inference-services" / f"{readable}-{suffix}")


def _service_attributes(
    name: str,
    inference: InferenceSelection,
    *,
    lifecycle: Literal["managed", "attached", "external"] | None,
) -> dict[str, JsonValue]:
    return {
        "service_name": name,
        "service_lifecycle": lifecycle,
        "inference_binding_id": inference.id,
        "inference_binding_revision": inference.revision,
        "model_selection_id": inference.model.id,
    }


__all__ = [
    "AttachedInferenceService",
    "ExternalInferenceServiceRequest",
    "ExternalInferenceUsageProjection",
    "ExternalServiceResolver",
    "HostedInferenceBinding",
    "InferenceSelection",
    "InferenceServiceRequest",
    "ManagedInferenceService",
    "ResolvedInferenceService",
    "bind_inference_services",
]
