"""Composition-host lifecycle for native judge plugins and selected inference."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, ExitStack, contextmanager
from dataclasses import replace
from typing import Any, cast

from posttrain.common import InferenceBinding, JsonValue, RunContext
from posttrain.environment import EnvironmentBinding, VerifiersV1ConfigActivation
from posttrain.serve import Endpoint, ProbeResult, ServeLaunchRequest, launch, probe

from .inference_services import (
    ExternalInferenceUsageProjection,
    HostedInferenceBinding,
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
            client = service.judge_client
            sampling = dict(client.sampling)
            headers: dict[str, str] = {}
            if isinstance(service.inference, HostedInferenceBinding):
                headers.update(service.inference.service.headers)
            if client.capabilities.protocol != "openai-chat@1":
                raise ValueError(f"judge {judge_name!r} requires openai-chat@1, got {client.capabilities.protocol!r}")
            judges[judge_name].update(
                model=client.endpoint.model,
                base_url=client.endpoint.base_url,
                api_key_var=credential_vars[service_name],
                sampling=sampling,
                headers=headers,
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


def project_native_judge_usage(
    environment: EnvironmentBinding,
    maximum_trajectories: int,
    judge_services: Mapping[str, str],
) -> dict[str, ExternalInferenceUsageProjection]:
    """Project paid calls from a generic trajectory ceiling and Verifiers config."""

    if not judge_services:
        return {}
    if isinstance(maximum_trajectories, bool) or not isinstance(maximum_trajectories, int) or maximum_trajectories < 1:
        raise ValueError("external judge cost projection requires a positive trajectory ceiling")
    if not isinstance(environment.activation, VerifiersV1ConfigActivation):
        raise ValueError("paid native judges require declarative Verifiers activation")
    raw = cast(Mapping[str, Any], environment.activation.config)
    taskset = raw.get("taskset", {})
    task = taskset.get("task", {}) if isinstance(taskset, Mapping) else {}
    entries = task.get("judges", []) if isinstance(task, Mapping) else []
    if not isinstance(entries, list):
        raise ValueError("native judges must be a list")
    by_name = {str(entry.get("name") or entry.get("id")): entry for entry in entries if isinstance(entry, Mapping)}
    agent = raw.get("agent", {})
    max_turns = agent.get("max_turns", 1) if isinstance(agent, Mapping) else 1
    if isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns < 1:
        raise ValueError("external judge cost projection requires a positive agent max_turns")
    totals: dict[str, list[int]] = {service: [0, 0, 0] for service in set(judge_services.values())}
    for judge_name, service_name in judge_services.items():
        judge = by_name.get(judge_name)
        if judge is None:
            raise ValueError(f"external judge cost projection cannot find plugin {judge_name!r}")
        attempts_value = judge.get("attempts")
        input_tokens_value = judge.get("input_budget_tokens")
        sampling = judge.get("sampling")
        output_tokens_value = sampling.get("max_tokens") if isinstance(sampling, Mapping) else None
        values = (attempts_value, input_tokens_value, output_tokens_value)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in values):
            raise ValueError(
                f"external judge {judge_name!r} must explicitly declare positive attempts and token budgets"
            )
        attempts = cast(int, attempts_value)
        input_tokens = cast(int, input_tokens_value)
        output_tokens = cast(int, output_tokens_value)
        calls_per_trajectory = max_turns if judge.get("context_scope") == "prefix" else 1
        protocol = judge.get("assessment_protocol", "direct@1")
        if protocol not in {"direct@1", "model-native-frame@1"}:
            raise ValueError(f"external judge {judge_name!r} declares unknown assessment protocol")
        if protocol == "model-native-frame@1":
            frame_output_value = judge.get("assessment_frame_max_tokens")
            if (
                isinstance(frame_output_value, bool)
                or not isinstance(frame_output_value, int)
                or frame_output_value < 1
            ):
                raise ValueError(f"external judge {judge_name!r} must declare a positive assessment_frame_max_tokens")
            frame_output_tokens = frame_output_value
            # Two calls share one attempt. The final request includes the
            # bounded frame, so its input can be the original episode budget
            # plus the entire frame. Count both calls rather than assuming the
            # first stage is free.
            requests_per_attempt = 2
            input_tokens_per_attempt = 2 * input_tokens + frame_output_tokens
            output_tokens_per_attempt = frame_output_tokens + output_tokens
        else:
            requests_per_attempt = 1
            input_tokens_per_attempt = input_tokens
            output_tokens_per_attempt = output_tokens
        requests = maximum_trajectories * calls_per_trajectory * attempts * requests_per_attempt
        total = totals[service_name]
        total[0] += requests
        total[1] += maximum_trajectories * calls_per_trajectory * attempts * input_tokens_per_attempt
        total[2] += maximum_trajectories * calls_per_trajectory * attempts * output_tokens_per_attempt
    return {
        name: ExternalInferenceUsageProjection(requests, input_tokens, output_tokens)
        for name, (requests, input_tokens, output_tokens) in totals.items()
    }


def _validate_judge_selection(
    name: str,
    judge: Mapping[str, Any],
    service: ResolvedInferenceService,
) -> None:
    inference = service.inference
    declared_revision = judge.get("model_revision")
    if isinstance(inference, HostedInferenceBinding):
        expected_revision = inference.model.revision
    else:
        expected_revision = inference.model.revision or inference.model.digest
    if declared_revision is not None and declared_revision != expected_revision:
        raise ValueError(f"judge {name!r} revision differs from its selected inference model")
    if judge.get("model") not in (None, service.endpoint.model):
        raise ValueError(f"judge {name!r} model differs from its selected inference model")
    if judge.get("sampling", dict(service.inference.sampling)) != dict(service.inference.sampling):
        raise ValueError(f"judge {name!r} sampling differs from its selected inference")
    if isinstance(inference, InferenceBinding):
        input_budget = judge.get("input_budget_tokens")
        output_budget = inference.sampling.get("max_tokens")
        context_window = inference.engine.get("max_model_len")
        budgets = (input_budget, output_budget, context_window)
        if all(isinstance(value, int) and not isinstance(value, bool) for value in budgets):
            required_context = cast(int, input_budget) + cast(int, output_budget)
            if required_context > cast(int, context_window):
                raise ValueError(
                    f"judge {name!r} input and output budgets exceed selected inference context: "
                    f"{input_budget} + {output_budget} > {context_window}"
                )
            if judge.get("assessment_protocol", "direct@1") == "model-native-frame@1":
                frame_tokens = judge.get("assessment_frame_max_tokens")
                if isinstance(frame_tokens, int) and not isinstance(frame_tokens, bool):
                    final_required_context = required_context + frame_tokens
                    if final_required_context > cast(int, context_window):
                        raise ValueError(
                            f"judge {name!r} model-native frame exceeds selected inference context: "
                            f"{input_budget} + {frame_tokens} + {output_budget} > {context_window}"
                        )


def _restore_environment(key: str, previous: str | None) -> None:
    if previous is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = previous


__all__ = [
    "bind_managed_native_judge_services",
    "bind_native_judge_services",
    "bind_native_judges",
    "project_native_judge_usage",
]
