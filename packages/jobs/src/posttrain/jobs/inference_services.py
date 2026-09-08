"""Composition-host ownership for named inference service dependencies."""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, ExitStack, contextmanager
from dataclasses import dataclass, replace
from types import MappingProxyType

from posttrain.common import InferenceBinding, JsonValue, RunContext
from posttrain.common.selections import validate_selection_id
from posttrain.serve import Endpoint, ProbeResult, ServeLaunchRequest, launch, probe


@dataclass(frozen=True, slots=True)
class ManagedInferenceService:
    """A service whose lifecycle is owned by the enclosing composition host."""

    request: ServeLaunchRequest

    @property
    def inference(self) -> InferenceBinding:
        return self.request.inference


@dataclass(frozen=True, slots=True)
class AttachedInferenceService:
    """A pre-existing endpoint that the composition host must never stop.

    Secrets are referenced by environment-variable name and resolved only for
    the ephemeral connection object. They are never copied into observations.
    """

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
                raise ValueError(
                    f"attached service credential variable {self.api_key_var!r} is not set"
                ) from error
        return Endpoint(self.base_url, self.model, api_key)


type InferenceServiceRequest = ManagedInferenceService | AttachedInferenceService


@dataclass(frozen=True, slots=True)
class ResolvedInferenceService:
    """One ready connection and the exact selection it is expected to serve."""

    name: str
    inference: InferenceBinding
    endpoint: Endpoint
    readiness: ProbeResult
    owned: bool

    def trace_identity(self) -> dict[str, JsonValue]:
        return {
            "service_name": self.name,
            "ownership": "managed" if self.owned else "attached",
            "inference_binding_id": self.inference.id,
            "inference_binding_revision": self.inference.revision,
            "model": self.inference.model.trace_identity(),
            "artifact_digest": self.inference.model.digest,
            "backend": self.inference.backend,
            "renderer": self.inference.renderer,
            "engine": dict(self.inference.engine),
            "sampling": dict(self.inference.sampling),
            "target": {
                "id": self.inference.target.id,
                "revision": self.inference.target.revision,
                "device_class": self.inference.target.device_class,
                "memory_gb": self.inference.target.memory_gb,
                "placement": dict(self.inference.target.placement),
                "host_constraints": dict(self.inference.target.host_constraints),
            },
            "endpoint": {
                "base_url": self.endpoint.base_url,
                "model": self.endpoint.model,
            },
            "readiness": {
                "healthy": self.readiness.healthy,
                "model_available": self.readiness.model_available,
                "models": list(self.readiness.models),
            },
        }


type ServiceProvisioner = Callable[
    [RunContext, ServeLaunchRequest], AbstractContextManager[Endpoint]
]
type ReadinessProbe = Callable[[RunContext, Endpoint], ProbeResult]


@contextmanager
def bind_inference_services(
    context: RunContext,
    services: Mapping[str, InferenceServiceRequest],
    *,
    provisioner: ServiceProvisioner = launch,
    readiness_probe: ReadinessProbe = probe,
) -> Iterator[Mapping[str, ResolvedInferenceService]]:
    """Resolve named services once and close only the managed instances.

    Service names are explicit identity. Equal endpoints or models are not
    deduplicated implicitly, so sharing occurs only when consumers reference
    the same resolved service name.
    """

    _validate_service_requests(services)
    if not services:
        yield MappingProxyType({})
        return

    resolved: dict[str, ResolvedInferenceService] = {}
    with ExitStack() as stack:
        for name, service in services.items():
            owned = isinstance(service, ManagedInferenceService)
            if owned:
                service_context = _service_context(context, name)
                service_context.workspace.mkdir(parents=True, exist_ok=True)
                service_context.event(
                    "inference_service_starting",
                    _service_attributes(name, service.inference, owned=True),
                )
                endpoint = stack.enter_context(provisioner(service_context, service.request))
                expected = service.request.endpoint
            else:
                endpoint = service.endpoint()
                expected = Endpoint(service.base_url, service.model)
                context.event(
                    "inference_service_attaching",
                    _service_attributes(name, service.inference, owned=False),
                )
            if endpoint.base_url != expected.base_url or endpoint.model != expected.model:
                raise ValueError(
                    f"inference service {name!r} endpoint differs from its selected address or model"
                )
            readiness = readiness_probe(context, endpoint)
            if not readiness.healthy or not readiness.model_available:
                raise RuntimeError(
                    f"inference service {name!r} did not expose selected model {endpoint.model!r}"
                )
            connection = ResolvedInferenceService(
                name=name,
                inference=service.inference,
                endpoint=endpoint,
                readiness=readiness,
                owned=owned,
            )
            resolved[name] = connection
            context.event(
                "inference_service_ready",
                {
                    **_service_attributes(name, service.inference, owned=owned),
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
                    _service_attributes(connection.name, connection.inference, owned=connection.owned),
                )


def _validate_service_requests(services: Mapping[str, InferenceServiceRequest]) -> None:
    managed_addresses: dict[tuple[str, int], str] = {}
    for name, service in services.items():
        validate_selection_id(name, "inference service name")
        if not isinstance(service, ManagedInferenceService | AttachedInferenceService):
            raise TypeError(f"inference service {name!r} has an unsupported request type")
        if isinstance(service, ManagedInferenceService):
            address = (service.request.host, service.request.port)
            previous = managed_addresses.get(address)
            if previous is not None:
                raise ValueError(
                    f"managed inference services {previous!r} and {name!r} use the same bind address"
                )
            managed_addresses[address] = name


def _service_context(context: RunContext, name: str) -> RunContext:
    readable = re.sub(r"[^a-z0-9._-]+", "-", name).strip("-.") or "service"
    suffix = hashlib.sha256(name.encode()).hexdigest()[:12]
    return replace(context, workspace=context.workspace / "inference-services" / f"{readable}-{suffix}")


def _service_attributes(
    name: str,
    inference: InferenceBinding,
    *,
    owned: bool,
) -> dict[str, JsonValue]:
    return {
        "service_name": name,
        "service_ownership": "managed" if owned else "attached",
        "inference_binding_id": inference.id,
        "inference_binding_revision": inference.revision,
        "model_variant_id": inference.model.id,
    }


__all__ = [
    "AttachedInferenceService",
    "InferenceServiceRequest",
    "ManagedInferenceService",
    "ResolvedInferenceService",
    "bind_inference_services",
]
