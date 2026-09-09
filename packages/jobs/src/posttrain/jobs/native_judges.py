"""Composition-host lifecycle for native judge plugins and selected inference."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, ExitStack, contextmanager
from dataclasses import replace
from typing import Any

from posttrain.common import JsonValue, RunContext
from posttrain.environment import EnvironmentBinding, VerifiersV1ConfigActivation
from posttrain.serve import Endpoint, ProbeResult, ServeLaunchRequest, launch, probe

from .inference_services import (
    ManagedInferenceService,
    ResolvedInferenceService,
    bind_inference_services,
)


@contextmanager
def bind_native_judges(
    context: RunContext,
    environment: EnvironmentBinding,
    requests: Mapping[str, ServeLaunchRequest],
    *,
    launcher: Callable[[RunContext, ServeLaunchRequest], AbstractContextManager[Endpoint]] = launch,
    readiness_probe: Callable[[RunContext, Endpoint], ProbeResult] = probe,
) -> Iterator[EnvironmentBinding]:
    """Compatibility wrapper for one managed service per native judge.

    Callers supply explicit serving selections and ports, including capacity
    suitable for concurrent policy training. No device or model is selected here.
    Each endpoint lives for the enclosing operation and closes on every exit.
    """
    if not requests:
        yield environment
        return
    service_requests = {name: ManagedInferenceService(request) for name, request in requests.items()}
    with bind_managed_native_judge_services(
        context,
        environment,
        service_requests,
        {name: name for name in requests},
        launcher=launcher,
        readiness_probe=readiness_probe,
    ) as bound:
        yield bound


@contextmanager
def bind_managed_native_judge_services(
    context: RunContext,
    environment: EnvironmentBinding,
    requests: Mapping[str, ManagedInferenceService],
    judge_services: Mapping[str, str],
    *,
    launcher: Callable[[RunContext, ServeLaunchRequest], AbstractContextManager[Endpoint]] = launch,
    readiness_probe: Callable[[RunContext, Endpoint], ProbeResult] = probe,
) -> Iterator[EnvironmentBinding]:
    """Provision named services and connect native plugins after full preflight."""

    _validate_native_judge_service_selections(environment, requests, judge_services)
    with bind_inference_services(
        context,
        requests,
        provisioner=launcher,
        readiness_probe=readiness_probe,
    ) as services:
        with bind_native_judge_services(
            environment,
            services,
            judge_services,
        ) as bound:
            yield bound


@contextmanager
def bind_native_judge_services(
    environment: EnvironmentBinding,
    services: Mapping[str, ResolvedInferenceService],
    judge_services: Mapping[str, str],
) -> Iterator[EnvironmentBinding]:
    """Connect native judge plugins to already-resolved named services.

    Plugin names and service names are intentionally separate. Multiple judge
    plugins can share one service without starting the selected model twice.
    The plugin continues to own its rubric, request schema and score parsing.
    """

    if not judge_services:
        yield environment
        return
    if not isinstance(environment.activation, VerifiersV1ConfigActivation):
        raise ValueError("managed native judges require declarative Verifiers activation")
    activation = environment.activation
    raw, judges = _native_judges(environment)
    undeclared = set(judge_services) - set(judges)
    if undeclared:
        raise ValueError("managed inference refers to an undeclared native judge")
    missing_services = set(judge_services.values()) - set(services)
    if missing_services:
        raise ValueError("native judge refers to an unresolved inference service")
    for judge_name, service_name in judge_services.items():
        _validate_judge_selection(judge_name, judges[judge_name], services[service_name])

    identities: dict[str, JsonValue] = {name: service.trace_identity() for name, service in services.items()}
    with ExitStack() as stack:
        credential_vars: dict[str, str] = {}
        for service_name in dict.fromkeys(judge_services.values()):
            service = services[service_name]
            identity = f"{service_name}:{service.inference.id}:{service.inference.revision}"
            suffix = hashlib.sha256(identity.encode()).hexdigest()[:16].upper()
            key = f"POSTTRAIN_JUDGE_{suffix}_API_KEY"
            previous = os.environ.get(key)
            os.environ[key] = service.endpoint.api_key
            stack.callback(_restore_environment, key, previous)
            credential_vars[service_name] = key
        for judge_name, service_name in judge_services.items():
            service = services[service_name]
            judges[judge_name].update(
                model=service.endpoint.model,
                base_url=service.endpoint.base_url,
                api_key_var=credential_vars[service_name],
                sampling=dict(service.inference.sampling),
                headers={},
            )
        yield replace(
            environment,
            activation=VerifiersV1ConfigActivation(raw, activation.resources),
            parameters={
                **environment.parameters,
                "inference_services": identities,
                "judge_service_bindings": dict(judge_services),
                # Compatibility evidence for existing readers. Remove after
                # callers migrate to the named service fields above.
                "managed_judge_inference": {
                    judge_name: identities[service_name] for judge_name, service_name in judge_services.items()
                },
            },
        )


def _validate_native_judge_service_selections(
    environment: EnvironmentBinding,
    requests: Mapping[str, ManagedInferenceService],
    judge_services: Mapping[str, str],
) -> None:
    _, judges = _native_judges(environment)
    if not set(judge_services).issubset(judges):
        raise ValueError("managed inference refers to an undeclared native judge")
    if not set(judge_services.values()).issubset(requests):
        raise ValueError("native judge refers to an unresolved inference service")
    for name, service_name in judge_services.items():
        request = requests[service_name].request
        endpoint = request.endpoint
        synthetic = ResolvedInferenceService(
            name=service_name,
            inference=request.inference,
            endpoint=endpoint,
            readiness=ProbeResult(True, True, 0.0, (endpoint.model,)),
            owned=True,
        )
        _validate_judge_selection(name, judges[name], synthetic)


def _native_judges(
    environment: EnvironmentBinding,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if not isinstance(environment.activation, VerifiersV1ConfigActivation):
        raise ValueError("managed native judges require declarative Verifiers activation")
    raw = json.loads(json.dumps(dict(environment.activation.config)))
    entries = raw.get("taskset", {}).get("task", {}).get("judges", [])
    if not isinstance(entries, list) or any(not isinstance(entry, dict) for entry in entries):
        raise ValueError("native judges must be a list of named plugin configurations")
    judges: dict[str, dict[str, Any]] = {}
    for entry in entries:
        name = entry.get("name") or entry.get("id")
        if not isinstance(name, str) or not name or name in judges:
            raise ValueError("native judge configurations need unique names")
        judges[name] = entry
    return raw, judges


def _validate_judge_selection(
    name: str,
    judge: Mapping[str, Any],
    service: ResolvedInferenceService,
) -> None:
    model = service.inference.model
    declared_revision = judge.get("model_revision")
    if declared_revision is not None and declared_revision != (model.revision or model.digest):
        raise ValueError(f"judge {name!r} revision differs from its selected inference model")
    if judge.get("model") not in (None, service.endpoint.model):
        raise ValueError(f"judge {name!r} model differs from its selected inference model")
    if judge.get("sampling", dict(service.inference.sampling)) != dict(service.inference.sampling):
        raise ValueError(f"judge {name!r} sampling differs from its selected inference")


def _restore_environment(key: str, previous: str | None) -> None:
    if previous is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = previous


__all__ = [
    "bind_managed_native_judge_services",
    "bind_native_judge_services",
    "bind_native_judges",
]
