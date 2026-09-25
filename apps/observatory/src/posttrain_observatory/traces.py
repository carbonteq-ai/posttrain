"""Generic and Verifiers trace projections used by evaluation and GRPO views."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from collections.abc import Awaitable, Callable, Mapping
from itertools import product
from statistics import fmean
from typing import Any, Literal, cast

from posttrain.common import JsonValue
from posttrain.tracking import (
    RunDataSource,
    TraceAggregateResult,
    TraceFactAggregate,
    TraceFactsQuery,
    TraceQuery,
    TraceRecord,
)

from .evaluation_measurement import measure_evaluation
from .models import (
    EvaluationAttemptEvidence,
    EvaluationBreakdown,
    EvaluationBreakdownGroup,
    EvaluationBreakdownSpec,
    EvaluationBreakdownValue,
    EvaluationDistribution,
    EvaluationFacet,
    EvaluationFacetSpec,
    EvaluationMeasurementFacet,
    EvaluationMeasurementPolicyView,
    EvaluationMeasurementTask,
    EvaluationMeasurementView,
    EvaluationMetadata,
    EvaluationMetricDefinition,
    EvaluationPerformance,
    EvaluationSlice,
    PromptGroupReward,
    PromptGroupRewardStats,
    PromptGroupRewardView,
    RewardComponent,
    RolloutBehaviorPoint,
    RolloutBehaviorView,
    TaskFacet,
    TaskSliceMetadata,
    TraceDetail,
    TraceEvaluationView,
    TraceFilterOptions,
    TraceFilterSlice,
    TraceOutcome,
    TraceSummary,
    TraceSummaryPage,
    TraceTiming,
)
from .redaction import RedactionPolicy
from .timeline import trace_timeline

_TRUNCATED_STOP_CONDITIONS = frozenset(
    {
        "max_turns",
        "max_input_tokens",
        "max_output_tokens",
        "max_total_tokens",
        "context_length",
        "harness_timeout",
    }
)


def _manifest_measurement(
    records: list[TraceRecord],
    summaries: tuple[TraceSummary, ...],
) -> EvaluationMeasurementView | None:
    """Project a manifest-backed single-trace episode without guessing missing identities."""

    if not records:
        return None
    manifest_digests = {
        value
        for record in records
        if isinstance((value := record.attributes.get("evaluation_manifest_digest")), str) and value
    }
    repetition_counts = {
        value
        for record in records
        if isinstance((value := record.attributes.get("num_rollouts")), int)
        and not isinstance(value, bool)
        and value > 0
    }
    selected_counts = {
        value
        for record in records
        if isinstance((value := record.attributes.get("evaluation_selected_tasks")), int)
        and not isinstance(value, bool)
        and value > 0
    }
    estimators = {
        value
        for record in records
        if (value := record.attributes.get("evaluation_estimator")) in {"task_mean", "target_weighted"}
    }
    missing_policies = {
        value
        for record in records
        if (value := record.attributes.get("evaluation_missing_policy")) in {"strict", "available"}
    }
    if (
        len(manifest_digests) != 1
        or len(repetition_counts) != 1
        or len(selected_counts) != 1
        or len(estimators) > 1
        or len(missing_policies) > 1
    ):
        return None
    manifest_digest = next(iter(manifest_digests))
    repetitions = next(iter(repetition_counts))
    selected_count = next(iter(selected_counts))
    tasks: dict[str, EvaluationMeasurementTask] = {}
    attempts: list[EvaluationAttemptEvidence] = []
    slots: set[tuple[str, int]] = set()
    for record, summary in zip(records, summaries, strict=True):
        task_key = summary.task
        repetition_index = record.attributes.get("evaluation_repetition_index")
        attempt_index = record.attributes.get("evaluation_execution_attempt_index", 0)
        target_weight = record.attributes.get("evaluation_task_target_weight")
        if (
            not task_key
            or not isinstance(repetition_index, int)
            or isinstance(repetition_index, bool)
            or repetition_index < 0
            or not isinstance(attempt_index, int)
            or isinstance(attempt_index, bool)
            or attempt_index < 0
            or not isinstance(target_weight, int | float)
            or isinstance(target_weight, bool)
            or target_weight <= 0
        ):
            return None
        slot = (task_key, repetition_index)
        # Several traces from one episode require an environment-owned episode
        # reducer. Choosing the first would silently change the evaluation unit.
        if slot in slots:
            return None
        slots.add(slot)
        raw_facets = record.attributes.get("evaluation_task_facets", [])
        facets = (
            tuple(
                EvaluationMeasurementFacet(
                    dimension=str(item["dimension"]),
                    value=str(item["value"]),
                    label=str(item["value"]),
                )
                for item in raw_facets
                if isinstance(item, Mapping)
                and isinstance(item.get("dimension"), str)
                and isinstance(item.get("value"), str)
            )
            if isinstance(raw_facets, list)
            else ()
        )
        candidate = EvaluationMeasurementTask(
            key=task_key,
            label=summary.task_label or task_key,
            target_weight=float(target_weight),
            facets=facets,
        )
        existing = tasks.get(task_key)
        if existing is not None and existing != candidate:
            return None
        tasks[task_key] = candidate
        attempts.append(
            EvaluationAttemptEvidence(
                task_key=task_key,
                repetition_index=repetition_index,
                attempt_index=attempt_index,
                reward=summary.reward,
                success=summary.success,
                execution_error=summary.error,
                truncated=summary.truncated,
                trace_id=summary.external_id,
            )
        )
    if len(tasks) != selected_count:
        return None
    return measure_evaluation(
        manifest_digest=manifest_digest,
        tasks=tuple(tasks.values()),
        repetitions_per_task=repetitions,
        attempts=attempts,
        policy=EvaluationMeasurementPolicyView(
            estimator=cast(
                Literal["task_mean", "target_weighted"],
                next(iter(estimators), "task_mean"),
            ),
            missing=cast(
                Literal["strict", "available"],
                next(iter(missing_policies), "strict"),
            ),
        ),
    )


def _number(value: object) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _integer(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _distribution(values: list[float]) -> EvaluationDistribution | None:
    if not values:
        return None
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * fraction + 0.5)))
        return ordered[index]

    return EvaluationDistribution(
        samples=len(ordered),
        mean=fmean(ordered),
        p50=percentile(0.50),
        p95=percentile(0.95),
        maximum=ordered[-1],
    )


def _wire_reward(payload: Mapping[str, JsonValue]) -> float | None:
    reward = _number(payload.get("reward"))
    if reward is None:
        reward = _number(payload.get("score"))
    if reward is not None:
        return reward
    components = payload.get("rewards")
    if not isinstance(components, Mapping):
        return None
    values = [_wire_reward_component(value) for value in components.values()]
    numbers = [value for value in values if value is not None]
    return sum(numbers) if numbers else None


def _wire_reward_component(value: object) -> float | None:
    """Project legacy scalars and native Verifiers ``Reward`` objects alike."""

    number = _number(value)
    if number is not None or not isinstance(value, Mapping):
        return number
    contribution = _number(value.get("contribution"))
    if contribution is not None:
        return contribution
    score = _number(value.get("score"))
    weight = _number(value.get("weight"))
    return score * (weight if weight is not None else 1.0) if score is not None else None


def _wire_success(payload: Mapping[str, JsonValue]) -> bool | None:
    success = payload.get("success")
    if isinstance(success, bool):
        return success
    for container_name in ("metrics", "rewards"):
        container = payload.get(container_name)
        if not isinstance(container, Mapping):
            continue
        for field in ("success", "correct"):
            value = container.get(field)
            if isinstance(value, bool):
                return value
            number = _number(value)
            if number is not None:
                return number > 0
    return None


def _wire_error(payload: Mapping[str, JsonValue]) -> str | None:
    error = payload.get("error")
    if error not in (None, False, ""):
        return str(error)
    errors = payload.get("errors")
    if isinstance(errors, list) and errors:
        latest = errors[-1]
        if isinstance(latest, Mapping):
            error_type = latest.get("type")
            return str(error_type) if error_type else "trace reported an error"
        return "trace reported an error"
    calls = payload.get("calls")
    if isinstance(calls, list):
        for call in reversed(calls):
            if not isinstance(call, Mapping):
                continue
            call_error = call.get("error")
            if call_error in (None, False, ""):
                continue
            if isinstance(call_error, Mapping):
                error_type = call_error.get("type")
                status = call_error.get("status_code")
                label = str(error_type) if isinstance(error_type, str) and error_type else "model call failed"
                return f"{label} (HTTP {status})" if isinstance(status, int) else label
            return "model call failed"
    return None


def _wire_truncated(
    payload: Mapping[str, JsonValue],
    attributes: Mapping[str, JsonValue] | None = None,
) -> bool:
    for container in (payload, attributes or {}):
        explicit = container.get("truncated")
        if isinstance(explicit, bool):
            return explicit
        explicit = container.get("is_truncated")
        if isinstance(explicit, bool):
            return explicit
    if payload.get("stop_condition") in _TRUNCATED_STOP_CONDITIONS:
        return True
    calls = payload.get("calls")
    if not isinstance(calls, list):
        return False
    for call in reversed(calls):
        if not isinstance(call, Mapping) or call.get("error") not in (None, False, ""):
            continue
        return call.get("finish_reason") == "length"
    return False


def _wire_outcome(
    *,
    success: bool | None,
    reward: float | None,
    truncated: bool,
    error: str | None,
) -> TraceOutcome:
    """Keep reward-based Verifiers traces distinct from failed boolean traces.

    Verifiers environments commonly emit a native reward without a boolean
    ``success`` field. Treating ``success is None`` as failure made scored
    examples appear failed in Observatory, so the projection exposes an
    explicit, provider-neutral outcome instead.
    """

    if error is not None:
        return "error"
    if truncated:
        return "truncated"
    if success is True:
        return "pass"
    if success is False:
        return "review"
    if reward is not None:
        return "scored"
    return "unknown"


def _wire_tool_calls(payload: Mapping[str, JsonValue]) -> int | None:
    direct = _integer(payload.get("num_tool_calls"))
    if direct is not None:
        return direct
    calls = payload.get("tool_calls")
    if isinstance(calls, list):
        return len(calls)
    nodes = payload.get("nodes")
    if not isinstance(nodes, list):
        return None
    count = 0
    for node in nodes:
        if not isinstance(node, Mapping):
            continue
        message = node.get("message")
        if not isinstance(message, Mapping):
            continue
        nested_calls = message.get("tool_calls")
        if isinstance(nested_calls, list):
            count += len(nested_calls)
    return count


def _wire_metrics(payload: Mapping[str, JsonValue]) -> dict[str, float]:
    values: dict[str, float] = {}
    for container_name in ("rewards", "metrics"):
        container = payload.get(container_name)
        if not isinstance(container, Mapping):
            continue
        for name, value in container.items():
            number = _wire_reward_component(value) if container_name == "rewards" else _number(value)
            if number is not None:
                values[str(name)] = number
    return values


def _wire_numeric_container(payload: Mapping[str, JsonValue], name: str) -> dict[str, float]:
    container = payload.get(name)
    if not isinstance(container, Mapping):
        return {}
    project = _wire_reward_component if name in {"rewards", "reward_components"} else _number
    return {str(key): number for key, value in container.items() if (number := project(value)) is not None}


def _wire_reward_components(payload: Mapping[str, JsonValue]) -> dict[str, float]:
    """Project every reward signal recorded on a Verifiers episode.

    Native Verifiers rewards live at the top level. Structured episode judges
    retain their training-consumed scalar projections under ``info`` so the
    complete assessment (including reasons and evidence) can remain alongside
    the trace without changing the Verifiers wire schema.
    """

    components = _wire_numeric_container(payload, "rewards")
    components.update(
        {
            name: value
            for name, value in _wire_numeric_container(payload, "reward_components").items()
            if name not in components
        }
    )
    info = payload.get("info")
    if not isinstance(info, Mapping):
        return components
    prefix = "episode_reward/"
    for key, value in info.items():
        if not isinstance(key, str) or not key.startswith(prefix):
            continue
        name = key.removeprefix(prefix)
        number = _number(value)
        if name and number is not None:
            components.setdefault(name, number)
    return components


def _wire_reward_component_details(payload: Mapping[str, JsonValue]) -> tuple[RewardComponent, ...]:
    components = _wire_reward_components(payload)
    info = payload.get("info")
    info = info if isinstance(info, Mapping) else {}
    panel = info.get("posttrain_episode_rewards")
    panel = panel if isinstance(panel, Mapping) else {}
    assessments = panel.get("assessments")
    assessments = assessments if isinstance(assessments, Mapping) else {}

    details: list[RewardComponent] = []
    for name, value in components.items():
        assessment = assessments.get(name)
        assessment = assessment if isinstance(assessment, Mapping) else {}
        reason = assessment.get("reason")
        raw_evidence = assessment.get("evidence")
        evidence = (
            tuple(str(item) for item in raw_evidence if isinstance(item, str | int))
            if isinstance(raw_evidence, list)
            else ()
        )
        is_episode_judge = f"episode_reward/{name}" in info or name in assessments
        details.append(
            RewardComponent(
                name=name,
                value=value,
                source="episode_judge" if is_episode_judge else "verifier",
                scope="episode" if is_episode_judge else None,
                reason=reason if isinstance(reason, str) and reason else None,
                evidence=evidence,
            )
        )
    return tuple(details)


def _messages(payload: Mapping[str, JsonValue]) -> tuple[Mapping[str, JsonValue], ...]:
    nodes = payload.get("nodes")
    if isinstance(nodes, list):
        messages: list[Mapping[str, JsonValue]] = []
        for node in nodes:
            if not isinstance(node, Mapping):
                continue
            message = node.get("message")
            if isinstance(message, Mapping):
                messages.append(cast(Mapping[str, JsonValue], message))
        if messages:
            return tuple(messages)
    message_values = payload.get("messages")
    if isinstance(message_values, list):
        return tuple(
            cast(Mapping[str, JsonValue], message) for message in message_values if isinstance(message, Mapping)
        )
    return ()


def _wire_text_stats(
    payload: Mapping[str, JsonValue],
) -> tuple[int | None, int, int | None, int]:
    response_text: list[str] = []
    thinking_text: list[str] = []
    for message in _messages(payload):
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str):
            response_text.append(content)
        reasoning = message.get("reasoning_content")
        if isinstance(reasoning, str):
            thinking_text.append(reasoning)
    usage_completion_tokens: list[int] = []
    usage_thinking_tokens: list[int] = []
    calls = payload.get("calls")
    if isinstance(calls, list):
        for call in calls:
            if not isinstance(call, Mapping):
                continue
            usage = call.get("usage")
            if not isinstance(usage, Mapping):
                continue
            completion = _integer(usage.get("completion_tokens"))
            if completion is not None:
                usage_completion_tokens.append(completion)
            reasoning = _integer(usage.get("reasoning_tokens"))
            if reasoning is not None:
                usage_thinking_tokens.append(reasoning)
    # Thinking comes only from per-call usage: providers report it on API paths
    # and the renderer reports it on the train path. No model-specific recovery.
    thinking_tokens = sum(usage_thinking_tokens) if usage_thinking_tokens else None
    response_tokens = None
    if usage_completion_tokens and thinking_tokens is not None:
        # ``completion_tokens`` includes the thought block. Expose output as
        # the user-visible completion instead of double-counting thinking.
        response_tokens = max(0, sum(usage_completion_tokens) - thinking_tokens)
    return (
        response_tokens,
        sum(len(value) for value in response_text),
        thinking_tokens,
        sum(len(value) for value in thinking_text),
    )


def _wire_model_calls(payload: Mapping[str, JsonValue]) -> int | None:
    projected = _integer(payload.get("num_model_calls"))
    if projected is not None:
        return projected
    calls = payload.get("calls")
    return len(calls) if isinstance(calls, list) else None


def _wire_usage(payload: Mapping[str, JsonValue]) -> tuple[int | None, int | None]:
    projected_input = _integer(payload.get("input_tokens"))
    projected_completion = _integer(payload.get("completion_tokens"))
    if projected_input is not None or projected_completion is not None:
        return projected_input, projected_completion
    input_tokens: list[int] = []
    completion_tokens: list[int] = []
    calls = payload.get("calls")
    if isinstance(calls, list):
        for call in calls:
            if not isinstance(call, Mapping):
                continue
            usage = call.get("usage")
            if not isinstance(usage, Mapping):
                continue
            prompt = _integer(usage.get("prompt_tokens"))
            completion = _integer(usage.get("completion_tokens"))
            if prompt is not None:
                input_tokens.append(prompt)
            if completion is not None:
                completion_tokens.append(completion)
    return (
        sum(input_tokens) if input_tokens else None,
        sum(completion_tokens) if completion_tokens else None,
    )


def _wire_latency_ms(payload: Mapping[str, JsonValue]) -> float | None:
    explicit = _number(payload.get("latency_ms"))
    if explicit is not None:
        return explicit
    timing = payload.get("timing")
    if isinstance(timing, Mapping):
        generation = timing.get("generation")
        model = generation.get("model") if isinstance(generation, Mapping) else None
        duration = _number(model.get("duration")) if isinstance(model, Mapping) else None
        if duration is not None:
            return duration * 1000
    starts: list[float] = []
    ends: list[float] = []
    calls = payload.get("calls")
    if isinstance(calls, list):
        for call in calls:
            if not isinstance(call, Mapping):
                continue
            clock = call.get("time")
            if not isinstance(clock, Mapping):
                continue
            start = _number(clock.get("start"))
            end = _number(clock.get("end"))
            if start is not None and end is not None and end >= start:
                starts.append(start)
                ends.append(end)
    if starts and ends:
        return (max(ends) - min(starts)) * 1000
    return None


def _task_scalar(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return None


def _task_from_record(value: object) -> str | None:
    scalar = _task_scalar(value)
    if scalar is not None:
        return scalar
    if not isinstance(value, Mapping):
        return None
    task_type = _task_scalar(value.get("type"))
    data = value.get("data")
    data = data if isinstance(data, Mapping) else value
    identity = next(
        (
            scalar
            for field in ("id", "task_id", "example_id", "name", "idx")
            if (scalar := _task_scalar(data.get(field))) is not None
        ),
        None,
    )
    if task_type is not None and identity is not None:
        return f"{task_type}:{identity}"
    return identity or task_type


def _humanize(value: str) -> str:
    value = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", value)
    value = re.sub(r"Task(?=[:\s-]|$)", " Task", value)
    value = value.replace("_", " ").replace("-", " ").replace(":", " ")
    label = " ".join(part.capitalize() for part in value.split())
    for source, target in (("Ifeval", "IFEval"), ("Gsm8k", "GSM8K"), ("Mmlu", "MMLU")):
        label = label.replace(source, target)
    return label


_COMPATIBILITY_FACET_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("generator", "generator", "Generator"),
    ("category", "category", "Category"),
    ("domain", "domain", "Domain"),
    ("problem_type", "problem_type", "Problem type"),
    ("level", "difficulty", "Difficulty"),
)


def _facet(
    *,
    dimension: str,
    dimension_label: str,
    value: str,
    label: str | None = None,
) -> TaskFacet:
    return TaskFacet(
        key=f"{dimension}:{value}",
        dimension=dimension,
        dimension_label=dimension_label,
        value=value,
        label=label or _humanize(value),
    )


def _declared_facets(value: object) -> tuple[TaskFacet, ...]:
    """Read the portable Verifiers task-facet convention when an env emits it."""

    if not isinstance(value, list):
        return ()
    facets: list[TaskFacet] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        dimension = _task_scalar(item.get("dimension"))
        raw_value = _task_scalar(item.get("value"))
        if dimension is None or raw_value is None:
            continue
        facets.append(
            _facet(
                dimension=dimension,
                dimension_label=_task_scalar(item.get("dimension_label")) or _humanize(dimension),
                value=raw_value,
                label=_task_scalar(item.get("label")),
            )
        )
    return tuple(facets)


def _task_facets(
    data: Mapping[str, object],
    metadata: Mapping[str, object],
    instruction_families: tuple[str, ...],
    facet_specs: tuple[EvaluationFacetSpec, ...] = (),
) -> tuple[TaskFacet, ...]:
    """Project native environment semantics without an environment-name switch.

    New environments can emit ``evaluation_facets`` directly. The field-based
    fallback preserves semantic data already emitted by the current environment
    packages, so historical traces become useful without migration.
    """

    if facet_specs:
        configured: list[TaskFacet] = []
        seen: set[str] = set()
        for spec in facet_specs:
            raw = data.get(spec.field, metadata.get(spec.field))
            values = raw if isinstance(raw, list | tuple) else [raw]
            for value in values:
                text = _task_scalar(value)
                if text is None:
                    continue
                if spec.transform == "prefix_before_colon":
                    text = text.split(":", 1)[0]
                item = _facet(dimension=spec.dimension, dimension_label=spec.label, value=text)
                if item.key not in seen:
                    configured.append(item)
                    seen.add(item.key)
        return tuple(configured)

    declared = _declared_facets(data.get("evaluation_facets")) or _declared_facets(metadata.get("evaluation_facets"))
    facets: list[TaskFacet] = list(declared)
    seen = {item.key for item in facets}
    for family in instruction_families:
        item = _facet(
            dimension="instruction_family",
            dimension_label="Instruction family",
            value=family,
        )
        if item.key not in seen:
            facets.append(item)
            seen.add(item.key)
    for source_field, dimension, dimension_label in _COMPATIBILITY_FACET_FIELDS:
        value = _task_scalar(data.get(source_field)) or _task_scalar(metadata.get(source_field))
        if value is None:
            continue
        item = _facet(
            dimension=dimension,
            dimension_label=dimension_label,
            value=value,
        )
        if item.key not in seen:
            facets.append(item)
            seen.add(item.key)
    return tuple(facets)


def _task_metadata(
    payload: Mapping[str, JsonValue],
    key: str | None,
    fallback: object = None,
    facet_specs: tuple[EvaluationFacetSpec, ...] = (),
) -> TaskSliceMetadata | None:
    task = payload.get("task") or fallback
    if not isinstance(task, Mapping):
        return None
    task_type = _task_scalar(task.get("type"))
    data = task.get("data")
    data = data if isinstance(data, Mapping) else task
    if key is None:
        key = _task_from_record(task)
    if key is None:
        return None
    name = _task_scalar(data.get("name")) or _task_scalar(data.get("generator")) or key
    metadata = data.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    raw_instruction_ids = data.get("instruction_id_list")
    instruction_ids = (
        tuple(str(item) for item in raw_instruction_ids if isinstance(item, (str, int, float)))
        if isinstance(raw_instruction_ids, list)
        else ()
    )
    instruction_families = tuple(dict.fromkeys(item.split(":", 1)[0] for item in instruction_ids if ":" in item))
    instruction_category = " + ".join(_humanize(item) for item in instruction_families)
    facets = _task_facets(data, metadata, instruction_families, facet_specs)
    task_label = _humanize(name)
    if instruction_category:
        task_label = f"{instruction_category} · {task_label}"
    dataset = (
        _task_scalar(data.get("source_repo"))
        or _task_scalar(data.get("source_repository"))
        or _task_scalar(metadata.get("source_dataset"))
    )
    seed = data.get("seed")
    index = data.get("idx")
    return TaskSliceMetadata(
        key=key,
        label=task_label,
        description=_task_scalar(data.get("description")),
        category=instruction_category or (facets[0].label if facets else _humanize(task_type or "evaluation")),
        instruction_ids=instruction_ids,
        instruction_families=instruction_families,
        facets=facets,
        dataset=dataset,
        dataset_revision=_task_scalar(data.get("source_revision")),
        split=_task_scalar(data.get("source_split")),
        seed=seed if isinstance(seed, int) and not isinstance(seed, bool) else None,
        index=index if isinstance(index, int) and not isinstance(index, bool) else None,
    )


def _compound_breakdowns(
    summaries: tuple[TraceSummary, ...],
    specs: tuple[EvaluationBreakdownSpec, ...],
) -> tuple[EvaluationBreakdown, ...]:
    reports: list[EvaluationBreakdown] = []
    for spec in specs:
        buckets: dict[str, tuple[tuple[TaskFacet, ...], list[TraceSummary]]] = {}
        excluded = 0
        dimension_labels: dict[str, str] = {}
        for item in summaries:
            facets = item.task_metadata.facets if item.task_metadata is not None else ()
            by_dimension: dict[str, list[TaskFacet]] = defaultdict(list)
            for facet in facets:
                by_dimension[facet.dimension].append(facet)
                dimension_labels.setdefault(facet.dimension, facet.dimension_label)
            values_by_dimension: list[list[TaskFacet]] = []
            invalid = False
            for dimension in spec.dimensions:
                values = by_dimension.get(dimension, [])
                if not values and spec.missing == "bucket":
                    values = [
                        _facet(
                            dimension=dimension,
                            dimension_label=dimension_labels.get(dimension, _humanize(dimension)),
                            value="(missing)",
                            label="Missing",
                        )
                    ]
                if not values or (spec.multi_value == "reject" and len(values) != 1):
                    invalid = True
                    break
                values_by_dimension.append(values)
            if invalid:
                excluded += 1
                continue
            for combination in product(*values_by_dimension):
                key = json.dumps(
                    {facet.dimension: facet.value for facet in combination},
                    separators=(",", ":"),
                )
                bucket = buckets.get(key)
                if bucket is None:
                    buckets[key] = (tuple(combination), [item])
                else:
                    bucket[1].append(item)
        groups: list[EvaluationBreakdownGroup] = []
        for key, (combination, values) in sorted(buckets.items()):
            rewards = [item.reward for item in values if item.reward is not None]
            successes = [item.success for item in values if item.success is not None]
            groups.append(
                EvaluationBreakdownGroup(
                    key=key,
                    label=" · ".join(facet.label for facet in combination),
                    values=tuple(
                        EvaluationBreakdownValue(
                            dimension=facet.dimension,
                            dimension_label=facet.dimension_label,
                            value=facet.value,
                            label=facet.label,
                        )
                        for facet in combination
                    ),
                    count=len(values),
                    scored=len(successes),
                    failures=sum(1 for item in values if item.error is not None),
                    truncated=sum(1 for item in values if item.truncated),
                    mean_reward=fmean(rewards) if rewards else None,
                    success_rate=(sum(1 for value in successes if value) / len(successes) if successes else None),
                )
            )
        reports.append(
            EvaluationBreakdown(
                id=spec.id,
                label=spec.label,
                dimensions=spec.dimensions,
                dimension_labels=(
                    dimension_labels.get(spec.dimensions[0], _humanize(spec.dimensions[0])),
                    dimension_labels.get(spec.dimensions[1], _humanize(spec.dimensions[1])),
                ),
                presentation=spec.presentation,
                groups=tuple(groups),
                excluded=excluded,
            )
        )
    return tuple(reports)


def _wire_task(
    payload: Mapping[str, JsonValue],
    metadata: Mapping[str, JsonValue],
    info: Mapping[str, JsonValue],
) -> str | None:
    for value in (
        payload.get("task_id"),
        info.get("example_id"),
        info.get("task_id"),
        metadata.get("task_id"),
        metadata.get("task"),
        payload.get("task"),
        info.get("task"),
    ):
        if (task := _task_from_record(value)) is not None:
            return task
    return None


def _wire_prompt_preview(payload: Mapping[str, JsonValue]) -> str | None:
    """Return one bounded human-readable request preview for list surfaces."""

    candidates: list[object] = []
    info = payload.get("info")
    for container in (payload.get("task"), info.get("task") if isinstance(info, Mapping) else None):
        if not isinstance(container, Mapping):
            continue
        data = container.get("data")
        data = data if isinstance(data, Mapping) else container
        candidates.extend(data.get(field) for field in ("prompt", "question", "instruction"))

    transcript = payload.get("messages") or payload.get("transcript") or payload.get("nodes")
    if isinstance(transcript, list):
        for item in transcript:
            if not isinstance(item, Mapping):
                continue
            message = item.get("message")
            message = message if isinstance(message, Mapping) else item
            if str(message.get("role", "")).lower() == "user":
                candidates.append(message.get("content"))
                break

    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        preview = " ".join(candidate.split())
        if preview:
            return preview if len(preview) <= 240 else f"{preview[:239].rstrip()}…"
    return None


def _summary(record: TraceRecord, evaluation_metadata: EvaluationMetadata | None = None) -> TraceSummary:
    payload = record.payload
    metadata = payload.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    info = payload.get("info")
    info = info if isinstance(info, Mapping) else {}
    error = _wire_error(payload)
    reward = None if error is not None else _wire_reward(payload)
    success = None if error is not None else _wire_success(payload)
    truncated = _wire_truncated(payload, record.attributes)
    task = _wire_task(payload, metadata, info)
    response_tokens, response_chars, thinking_tokens, thinking_chars = _wire_text_stats(payload)
    input_tokens, completion_tokens = _wire_usage(payload)
    timeline = trace_timeline(payload)
    if response_tokens is None and completion_tokens is not None and thinking_tokens is not None:
        response_tokens = max(0, completion_tokens - thinking_tokens)
    explicit_tokens = _integer(payload.get("tokens"))
    task_metadata = _task_metadata(
        payload,
        task,
        info.get("task"),
        evaluation_metadata.facet_specs if evaluation_metadata is not None else (),
    )
    optimizer_step = _integer(record.attributes.get("optimizer_step"))
    if optimizer_step is None:
        posttrain_run = info.get("posttrain_run")
        if isinstance(posttrain_run, Mapping):
            optimizer_step = _integer(posttrain_run.get("step"))
    prompt_group_id = info.get("posttrain_prompt_group_id")
    return TraceSummary(
        external_id=record.external_id,
        trace_type=record.trace_type,
        optimizer_step=optimizer_step if optimizer_step and optimizer_step > 0 else None,
        prompt_group_id=prompt_group_id if isinstance(prompt_group_id, str) and prompt_group_id else None,
        prompt_preview=_wire_prompt_preview(payload),
        task=task,
        task_label=task_metadata.label if task_metadata is not None else None,
        task_metadata=task_metadata,
        reward=reward,
        success=success,
        outcome=_wire_outcome(success=success, reward=reward, truncated=truncated, error=error),
        truncated=truncated,
        error=error,
        tool_calls=_wire_tool_calls(payload),
        model_calls=_wire_model_calls(payload),
        input_tokens=input_tokens,
        completion_tokens=completion_tokens,
        latency_ms=_wire_latency_ms(payload),
        tokens=explicit_tokens if explicit_tokens is not None else completion_tokens,
        response_tokens=response_tokens,
        response_chars=response_chars,
        thinking_tokens=thinking_tokens,
        thinking_chars=thinking_chars,
        timing=TraceTiming.model_validate(timeline.model_dump(exclude={"segments"})) if timeline else None,
        reward_components=_wire_reward_components(payload),
        native_metrics=_wire_numeric_container(payload, "metrics"),
        metrics=_wire_metrics(payload),
    )


def _predicate_matches(value: float, metadata: EvaluationMetadata) -> bool | None:
    definition = metadata.success_definition
    if definition is None:
        return None
    threshold = definition.value
    tolerance = definition.tolerance
    if definition.operator == "eq":
        return abs(value - threshold) <= tolerance
    if definition.operator == "gt":
        return value > threshold
    if definition.operator == "gte":
        return value >= threshold
    if definition.operator == "lt":
        return value < threshold
    if definition.operator == "lte":
        return value <= threshold
    assert definition.upper is not None
    return threshold <= value <= definition.upper


def _apply_evaluation_semantics(
    summary: TraceSummary,
    metadata: EvaluationMetadata | None,
) -> TraceSummary:
    """Apply the environment-declared score and pass-rate metrics to one trace."""

    if metadata is None or summary.error is not None:
        return summary
    reward = summary.metrics.get(metadata.primary_metric) if metadata.primary_metric else None
    success = None
    definition = metadata.success_definition
    if definition is not None:
        # A versioned success definition is authoritative. Operationally
        # incomplete traces stay outside the semantic pass-rate denominator;
        # they must not fall through to the legacy binary-metric adapter.
        if summary.error is None and not summary.truncated:
            container = summary.reward_components if definition.namespace == "reward" else summary.native_metrics
            signal_value = container.get(definition.signal)
            if signal_value is not None:
                success = _predicate_matches(signal_value, metadata)
    elif metadata.pass_rate_metric is not None:
        pass_value = summary.metrics.get(metadata.pass_rate_metric)
        if pass_value in (0.0, 1.0):
            success = pass_value == 1.0
    if reward is None and success is summary.success:
        return summary
    resolved_reward = summary.reward if reward is None else reward
    return summary.model_copy(
        update={
            "reward": resolved_reward,
            "success": success,
            "outcome": _wire_outcome(
                success=success,
                reward=resolved_reward,
                truncated=summary.truncated,
                error=summary.error,
            ),
        }
    )


def project_trace(record: TraceRecord, redaction: RedactionPolicy) -> TraceDetail:
    payload = redaction.mapping(record.payload)
    warning = None
    if record.trace_type != "verifiers":
        warning = f"No specialized projector is registered for trace type {record.trace_type!r}."
    components = _wire_reward_component_details(payload)
    transcript_value = payload.get("messages") or payload.get("transcript") or payload.get("nodes")
    transcript: list[dict[str, JsonValue]] = []
    if isinstance(transcript_value, list):
        for item in transcript_value:
            if not isinstance(item, dict):
                continue
            entry = dict(item)
            message = item.get("message")
            if isinstance(message, Mapping):
                entry.setdefault("role", message.get("role"))
                entry.setdefault("content", message.get("content"))
                if "reasoning_content" in message:
                    entry["reasoning_content"] = message.get("reasoning_content")
                if "tool_calls" in message:
                    entry["tool_calls"] = message.get("tool_calls")
            transcript.append(entry)
    return TraceDetail(
        summary=_summary(record),
        timeline=trace_timeline(record.payload),
        reward_components=components,
        transcript=tuple(transcript),
        attributes=redaction.mapping(record.attributes),
        raw=payload,
        projection_warning=warning,
    )


async def trace_evaluation_view(
    source: RunDataSource,
    run_id: str,
    *,
    expected: int | None = None,
    trace_type: str = "verifiers",
    safety_limit: int = 5000,
    metadata: EvaluationMetadata | None = None,
    include_traces: bool = True,
) -> TraceEvaluationView:
    cursor: str | None = None
    records: list[TraceRecord] = []
    live = False
    while len(records) < safety_limit:
        page = await source.traces(
            run_id,
            TraceQuery(trace_type=trace_type, cursor=cursor, limit=min(1000, safety_limit - len(records))),
        )
        live = page.live
        records.extend(page.items)
        if page.next_cursor is None:
            cursor = None
            break
        cursor = page.next_cursor
    summaries = tuple(_summary(record, metadata) for record in records)
    summaries = tuple(_apply_evaluation_semantics(item, metadata) for item in summaries)
    rewards = [item.reward for item in summaries if item.reward is not None]
    successes = [item.success for item in summaries if item.success is not None]
    grouped: dict[str, list[TraceSummary]] = defaultdict(list)
    for item in summaries:
        grouped[item.task or "unspecified"].append(item)
    slices = []
    for key, values in sorted(grouped.items()):
        slice_rewards = [item.reward for item in values if item.reward is not None]
        slice_successes = [item.success for item in values if item.success is not None]
        slices.append(
            EvaluationSlice(
                key=key,
                label=(values[0].task_label or key),
                description=(values[0].task_metadata.description if values[0].task_metadata else None),
                metadata=(values[0].task_metadata if values[0].task_metadata else None),
                count=len(values),
                mean_reward=fmean(slice_rewards) if slice_rewards else None,
                success_rate=(
                    sum(1 for value in slice_successes if value) / len(slice_successes) if slice_successes else None
                ),
            )
        )
    facet_groups: dict[str, tuple[TaskFacet, list[TraceSummary]]] = {}
    for item in summaries:
        metadata_for_item = item.task_metadata
        if metadata_for_item is None:
            continue
        for facet in metadata_for_item.facets:
            facet_bucket = facet_groups.get(facet.key)
            if facet_bucket is None:
                facet_groups[facet.key] = (facet, [item])
            else:
                facet_bucket[1].append(item)
    facets = []
    for key, (facet, values) in sorted(facet_groups.items()):
        facet_rewards = [item.reward for item in values if item.reward is not None]
        facet_successes = [item.success for item in values if item.success is not None]
        facets.append(
            EvaluationFacet(
                key=key,
                label=facet.label,
                dimension=facet.dimension,
                dimension_label=facet.dimension_label,
                count=len(values),
                mean_reward=fmean(facet_rewards) if facet_rewards else None,
                success_rate=(
                    sum(1 for value in facet_successes if value) / len(facet_successes) if facet_successes else None
                ),
            )
        )
    breakdowns = _compound_breakdowns(
        summaries,
        metadata.breakdown_specs if metadata is not None else (),
    )
    complete = cursor is None and (expected is None or len(records) >= expected)
    state = "complete" if complete else "partial"
    if not records and expected in (None, 0):
        state = "unavailable"
    definition = metadata.success_definition if metadata is not None else None
    if state == "complete" and definition is not None and definition.missing == "error":
        missing_success_signal = any(
            item.error is None
            and not item.truncated
            and definition.signal
            not in (item.reward_components if definition.namespace == "reward" else item.native_metrics)
            for item in summaries
        )
        if missing_success_signal:
            state = "partial"
    metric_names = sorted({name for item in summaries for name in item.metrics})
    metric_definitions = tuple(
        EvaluationMetricDefinition(
            name=name,
            label=_humanize(name),
            role=("primary_reward" if name == (metadata.primary_metric if metadata else None) else "diagnostic"),
        )
        for name in metric_names
    )
    resolved_metadata = metadata
    if resolved_metadata is not None:
        known = {item.name for item in resolved_metadata.metrics}
        resolved_metadata = resolved_metadata.model_copy(
            update={
                "metrics": resolved_metadata.metrics
                + tuple(item for item in metric_definitions if item.name not in known)
            }
        )
    return TraceEvaluationView(
        state=state,
        metadata=resolved_metadata,
        scanned=len(records),
        expected=expected,
        included=len(records),
        scored=len(rewards),
        mean_reward=fmean(rewards) if rewards else None,
        success_rate=(sum(1 for value in successes if value) / len(successes) if successes else None),
        passed=sum(1 for value in successes if value),
        pass_scored=len(successes),
        failures=sum(1 for item in summaries if item.error is not None),
        truncated=sum(1 for item in summaries if item.truncated),
        slices=tuple(slices),
        facets=tuple(facets),
        breakdowns=breakdowns,
        performance=EvaluationPerformance(
            latency_ms=_distribution([item.latency_ms for item in summaries if item.latency_ms is not None]),
            completion_tokens=_distribution(
                [
                    float(item.completion_tokens)
                    if item.completion_tokens is not None
                    else float(cast(int, item.tokens))
                    for item in summaries
                    if item.completion_tokens is not None or item.tokens is not None
                ]
            ),
            thinking_tokens=_distribution(
                [float(item.thinking_tokens) for item in summaries if item.thinking_tokens is not None]
            ),
            tool_calls=_distribution([float(item.tool_calls) for item in summaries if item.tool_calls is not None]),
        ),
        traces=summaries if include_traces else (),
        next_cursor=cursor,
        live=live,
        measurement=_manifest_measurement(records, summaries),
    )


async def rollout_behavior_view(
    source: RunDataSource,
    run_id: str,
    *,
    expected: int | None = None,
    trace_type: str = "verifiers",
) -> RolloutBehaviorView:
    """Read persisted rollout facts without reopening native trace payloads."""

    aggregate = cast(
        Callable[[str, TraceFactsQuery], Awaitable[Any]] | None,
        getattr(source, "aggregate_trace_facts", None),
    )
    if callable(aggregate):
        result = await aggregate(
            run_id,
            TraceFactsQuery(
                trace_type=trace_type,
                group_by=("rollout_step",),
                aggregates=(
                    TraceFactAggregate(measure="thinking_tokens"),
                    TraceFactAggregate(measure="model_output_tokens"),
                    TraceFactAggregate(measure="tool_calls"),
                ),
            ),
        )
        if getattr(result, "state", None) == "available":
            points: list[RolloutBehaviorPoint] = []
            unattributed = 0
            scanned = 0
            for bucket in result.buckets:
                scanned += bucket.trace_count
                step = _integer(bucket.dimensions.get("rollout_step"))
                if step is None:
                    unattributed += bucket.trace_count
                    continue
                points.append(
                    RolloutBehaviorPoint(
                        step=step,
                        rollouts=bucket.trace_count,
                        thinking_tokens=_number(bucket.values.get("mean_thinking_tokens")),
                        output_tokens=_number(bucket.values.get("mean_model_output_tokens")),
                        tool_calls=_number(bucket.values.get("mean_tool_calls")),
                    )
                )
            complete = expected is None or scanned >= expected
            return RolloutBehaviorView(
                state="complete" if points and complete else ("partial" if points else "unavailable"),
                scanned=scanned,
                expected=expected,
                included=sum(point.rollouts for point in points),
                unattributed=unattributed,
                points=tuple(sorted(points, key=lambda point: point.step)),
                live=False,
            )

    return RolloutBehaviorView(
        state="unavailable",
        scanned=0,
        expected=expected,
        included=0,
        unattributed=0,
        points=(),
        live=False,
    )


async def trace_summary_page(
    source: RunDataSource,
    run_id: str,
    *,
    total: int,
    cursor: str | None = None,
    limit: int = 100,
    trace_type: str = "verifiers",
    newest_first: bool = False,
    metadata: EvaluationMetadata | None = None,
) -> TraceSummaryPage:
    """Project one provider-bounded trace page without population aggregation."""

    page = await source.traces(
        run_id,
        TraceQuery(
            trace_type=trace_type,
            cursor=cursor,
            limit=limit,
            order="newest_first" if newest_first else "oldest_first",
        ),
    )
    summaries = tuple(_apply_evaluation_semantics(_summary(record, metadata), metadata) for record in page.items)
    return TraceSummaryPage(
        items=summaries,
        next_cursor=page.next_cursor,
        total=total,
        live=page.live,
    )


async def trace_summary_population(
    source: RunDataSource,
    run_id: str,
    *,
    trace_type: str,
    newest_first: bool,
    metadata: EvaluationMetadata | None,
) -> tuple[tuple[TraceSummary, ...], bool]:
    """Read the whole summary population for exact run-wide filters."""

    summaries: list[TraceSummary] = []
    seen: set[str] = set()
    cursor: str | None = None
    live = False
    while True:
        page = await source.traces(
            run_id,
            TraceQuery(
                trace_type=trace_type,
                cursor=cursor,
                limit=1000,
                order="newest_first" if newest_first else "oldest_first",
            ),
        )
        live = live or page.live
        for record in page.items:
            if record.external_id in seen:
                continue
            seen.add(record.external_id)
            summaries.append(_apply_evaluation_semantics(_summary(record, metadata), metadata))
        if page.next_cursor is None:
            break
        if page.next_cursor == cursor:
            raise ValueError("trace provider did not advance its cursor")
        cursor = page.next_cursor
    return tuple(summaries), live


def prompt_group_reward_view(
    result: TraceAggregateResult, *, expected_size: int | None, recorded_traces: int, live: bool
) -> PromptGroupRewardView:
    """Interpret indexed fact buckets without loading native trace rows."""

    if result.state != "available":
        return PromptGroupRewardView(
            expected_group_size=expected_size,
            fact_rows=0,
            recorded_traces=recorded_traces,
            live=live,
        )
    fact_rows = sum(bucket.trace_count for bucket in result.buckets)
    by_id: dict[str, list[tuple[int | None, str | None, int, int, PromptGroupRewardStats | None]]] = defaultdict(list)
    for bucket in result.buckets:
        group_id = bucket.dimensions.get("prompt_group_id")
        if not isinstance(group_id, str) or not group_id:
            continue
        step = _integer(bucket.dimensions.get("rollout_step"))
        task = bucket.dimensions.get("task_id")
        task_id = task if isinstance(task, str) and task else None
        coverage = bucket.coverage.get("sum_task_reward", 0)
        reward_sum = _number(bucket.values.get("sum_task_reward"))
        reward_squares = _number(bucket.values.get("sum_squares_task_reward"))
        current = None
        if (
            expected_size is not None
            and bucket.trace_count == expected_size
            and coverage == expected_size
            and bucket.coverage.get("sum_squares_task_reward") == expected_size
            and reward_sum is not None
            and reward_squares is not None
        ):
            mean = reward_sum / expected_size
            variance = reward_squares / expected_size - mean * mean
            if variance >= -1e-10:
                current = PromptGroupRewardStats(mean=mean, std=math.sqrt(max(0.0, variance)))
        by_id[group_id].append((step, task_id, bucket.trace_count, coverage, current))

    prepared = [(group_id, *entries[0]) for group_id, entries in by_id.items() if len(entries) == 1]
    prepared.sort(key=lambda item: (item[1] is None, item[1] or 0, item[0]))
    latest_by_task: dict[str, tuple[int, PromptGroupRewardStats, int]] = {}
    groups: list[PromptGroupReward] = []
    steps = sorted({step for _, step, _, _, _, _ in prepared if step is not None})
    for step in steps:
        at_step = [item for item in prepared if item[1] == step]
        for group_id, _, task_id, count, coverage, current in at_step:
            prior = latest_by_task.get(task_id) if task_id else None
            groups.append(
                PromptGroupReward(
                    group_id=group_id,
                    step=step,
                    task_id=task_id,
                    rollouts=count,
                    reward_coverage=coverage,
                    current=current,
                    prior=prior[1] if prior else None,
                    prior_step=prior[0] if prior else None,
                    prior_rollouts=prior[2] if prior else None,
                )
            )
        combined: dict[str, tuple[int, float, float]] = {}
        for _, _, task_id, count, _, current in at_step:
            if task_id and current:
                prior_count, prior_sum, prior_squares = combined.get(task_id, (0, 0.0, 0.0))
                combined[task_id] = (
                    prior_count + count,
                    prior_sum + count * current.mean,
                    prior_squares + count * (current.std * current.std + current.mean * current.mean),
                )
        for task_id, (count, reward_sum, reward_squares) in combined.items():
            mean = reward_sum / count
            variance = reward_squares / count - mean * mean
            latest_by_task[task_id] = (
                step,
                PromptGroupRewardStats(mean=mean, std=math.sqrt(max(0.0, variance))),
                count,
            )
    for group_id, step, task_id, count, coverage, current in prepared:
        if step is None:
            groups.append(
                PromptGroupReward(
                    group_id=group_id,
                    task_id=task_id,
                    rollouts=count,
                    reward_coverage=coverage,
                    current=current,
                )
            )
    complete = (
        bool(groups)
        and expected_size is not None
        and fact_rows == recorded_traces
        and sum(group.rollouts for group in groups) == fact_rows
        and all(group.step is not None and group.task_id is not None and group.current is not None for group in groups)
        and not any(len(entries) > 1 for entries in by_id.values())
    )
    return PromptGroupRewardView(
        state="complete" if complete else ("partial" if groups else "unavailable"),
        groups=tuple(groups),
        expected_group_size=expected_size,
        fact_rows=fact_rows,
        recorded_traces=recorded_traces,
        live=live,
    )


def trace_filter_options(summaries: tuple[TraceSummary, ...]) -> TraceFilterOptions:
    slices: dict[str, str] = {}
    outcomes: set[TraceOutcome] = set()
    for item in summaries:
        if item.task:
            slices.setdefault(item.task, item.task_label or item.task)
        if item.task_metadata:
            for facet in item.task_metadata.facets:
                slices.setdefault(f"facet:{facet.key}", f"{facet.label} · {facet.dimension_label}")
        outcomes.add(item.outcome)
    return TraceFilterOptions(
        total=len(summaries),
        steps=tuple(sorted({item.optimizer_step for item in summaries if item.optimizer_step is not None})),
        slices=tuple(TraceFilterSlice(key=key, label=label) for key, label in sorted(slices.items())),
        outcomes=tuple(sorted(outcomes)),
    )


def filtered_trace_summary_page(
    summaries: tuple[TraceSummary, ...],
    *,
    cursor: str | None,
    limit: int,
    step: int | None,
    slice_key: str | None,
    outcome: TraceOutcome | None,
    search: str | None,
    live: bool,
) -> TraceSummaryPage:
    needle = (search or "").strip().casefold()
    matched = []
    for item in summaries:
        if step is not None and item.optimizer_step != step:
            continue
        if slice_key:
            if slice_key.startswith("facet:"):
                facets = item.task_metadata.facets if item.task_metadata else ()
                if not any(facet.key == slice_key[6:] for facet in facets):
                    continue
            elif item.task != slice_key:
                continue
        if outcome and item.outcome != outcome:
            continue
        if (
            needle
            and needle
            not in " ".join(
                (
                    item.external_id,
                    item.task or "",
                    item.task_label or "",
                    item.prompt_preview or "",
                    item.prompt_group_id or "",
                )
            ).casefold()
        ):
            continue
        matched.append(item)
    offset = int(cursor or 0)
    if offset < 0:
        raise ValueError("trace cursor must be nonnegative")
    end = offset + limit
    return TraceSummaryPage(
        items=tuple(matched[offset:end]),
        next_cursor=str(end) if end < len(matched) else None,
        total=len(matched),
        live=live,
    )


async def get_trace_detail(
    source: RunDataSource,
    run_id: str,
    external_id: str,
    redaction: RedactionPolicy,
    metadata: EvaluationMetadata | None = None,
) -> TraceDetail:
    direct_reader = getattr(source, "get_trace", None)
    if callable(direct_reader):
        direct_record = await cast(Any, direct_reader)(run_id, external_id)
        if direct_record is not None:
            detail = project_trace(direct_record, redaction)
            return detail.model_copy(update={"summary": _apply_evaluation_semantics(detail.summary, metadata)})

    cursor: str | None = None
    while True:
        page = await source.traces(run_id, TraceQuery(cursor=cursor, limit=1000))
        for record in page.items:
            if record.external_id == external_id:
                detail = project_trace(record, redaction)
                return detail.model_copy(update={"summary": _apply_evaluation_semantics(detail.summary, metadata)})
        if page.next_cursor is None:
            break
        cursor = page.next_cursor
    raise LookupError(f"trace {external_id!r} was not found in run {run_id!r}")


__all__ = [
    "filtered_trace_summary_page",
    "get_trace_detail",
    "project_trace",
    "rollout_behavior_view",
    "trace_evaluation_view",
    "trace_filter_options",
    "trace_summary_page",
    "trace_summary_population",
]
