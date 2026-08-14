"""Generic and Verifiers trace projections used by evaluation and GRPO views."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Awaitable, Callable, Mapping
from itertools import product
from statistics import fmean
from typing import Any, cast

from posttrain.common import JsonValue
from posttrain.tracking import RunDataSource, TraceFactAggregate, TraceFactsQuery, TraceQuery, TraceRecord

from .models import (
    EvaluationBreakdown,
    EvaluationBreakdownGroup,
    EvaluationBreakdownSpec,
    EvaluationBreakdownValue,
    EvaluationDistribution,
    EvaluationFacet,
    EvaluationFacetSpec,
    EvaluationMetadata,
    EvaluationMetricDefinition,
    EvaluationPerformance,
    EvaluationSlice,
    RewardComponent,
    RolloutBehaviorPoint,
    RolloutBehaviorView,
    TaskFacet,
    TaskSliceMetadata,
    TraceDetail,
    TraceEvaluationView,
    TraceOutcome,
    TraceSummary,
    TraceSummaryPage,
)
from .redaction import RedactionPolicy

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

# Qwen3.5's pinned tokenizer represents ``</think>`` with this single special
# token. A Verifiers node keeps the exact generated token ids and sampled mask,
# which lets historical traces recover a precise thinking-token count even when
# the OpenAI-compatible usage block did not report ``reasoning_tokens``.
_QWEN35_THINKING_END_TOKEN_ID = 248069


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
    values = [_number(value) for value in components.values()]
    numbers = [value for value in values if value is not None]
    return sum(numbers) if numbers else None


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
    if not isinstance(errors, list) or not errors:
        return None
    latest = errors[-1]
    if isinstance(latest, Mapping):
        error_type = latest.get("type")
        return str(error_type) if error_type else "trace reported an error"
    return "trace reported an error"


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
            number = _number(value)
            if number is not None:
                values[str(name)] = number
    return values


def _wire_numeric_container(payload: Mapping[str, JsonValue], name: str) -> dict[str, float]:
    container = payload.get(name)
    if not isinstance(container, Mapping):
        return {}
    return {str(key): number for key, value in container.items() if (number := _number(value)) is not None}


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


def _trace_reasoning_tokens(
    payload: Mapping[str, JsonValue],
    attributes: Mapping[str, JsonValue],
) -> int | None:
    """Recover exact Qwen3.5 reasoning tokens from retained generated ids.

    This intentionally requires a parsed ``reasoning_content`` message and an
    explicit Qwen3.5 model identity. It never estimates from characters.
    """

    model = attributes.get("model")
    agent = payload.get("agent")
    if not isinstance(model, str) and isinstance(agent, Mapping):
        model = agent.get("model")
    if not isinstance(model, str) or "qwen3.5" not in model.lower():
        return None
    nodes = payload.get("nodes")
    if not isinstance(nodes, list):
        return None
    total = 0
    observed = False
    for node in nodes:
        if not isinstance(node, Mapping):
            continue
        message = node.get("message")
        if not isinstance(message, Mapping) or not isinstance(message.get("reasoning_content"), str):
            continue
        token_ids = node.get("token_ids")
        sampled_mask = node.get("mask")
        if not isinstance(token_ids, list) or not isinstance(sampled_mask, list) or len(token_ids) != len(sampled_mask):
            return None
        sampled_start = next((index for index, value in enumerate(sampled_mask) if value is True), None)
        if sampled_start is None:
            return None
        end = next(
            (
                index
                for index in range(sampled_start, len(token_ids))
                if token_ids[index] == _QWEN35_THINKING_END_TOKEN_ID and sampled_mask[index] is True
            ),
            None,
        )
        if end is None:
            return None
        total += sum(1 for value in sampled_mask[sampled_start:end] if value is True)
        observed = True
    return total if observed else None


def _wire_text_stats(
    payload: Mapping[str, JsonValue],
    attributes: Mapping[str, JsonValue],
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
    recovered_thinking_tokens = _trace_reasoning_tokens(payload, attributes)
    thinking_tokens = sum(usage_thinking_tokens) if usage_thinking_tokens else recovered_thinking_tokens
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
    reward = _wire_reward(payload)
    success = _wire_success(payload)
    truncated = _wire_truncated(payload, record.attributes)
    error = _wire_error(payload)
    task = _wire_task(payload, metadata, info)
    response_tokens, response_chars, thinking_tokens, thinking_chars = _wire_text_stats(payload, record.attributes)
    input_tokens, completion_tokens = _wire_usage(payload)
    if response_tokens is None and completion_tokens is not None and thinking_tokens is not None:
        response_tokens = max(0, completion_tokens - thinking_tokens)
    explicit_tokens = _integer(payload.get("tokens"))
    task_metadata = _task_metadata(
        payload,
        task,
        info.get("task"),
        evaluation_metadata.facet_specs if evaluation_metadata is not None else (),
    )
    return TraceSummary(
        external_id=record.external_id,
        trace_type=record.trace_type,
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
        reward_components=_wire_numeric_container(payload, "rewards"),
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

    if metadata is None:
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
    components: list[RewardComponent] = []
    raw_components = payload.get("reward_components") or payload.get("rewards")
    if isinstance(raw_components, Mapping):
        for name, value in sorted(raw_components.items()):
            number = _number(value)
            if number is not None:
                components.append(RewardComponent(name=str(name), value=number))
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
        reward_components=tuple(components),
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
    metadata: EvaluationMetadata | None = None,
) -> TraceSummaryPage:
    """Project one provider-bounded trace page without population aggregation."""

    page = await source.traces(
        run_id,
        TraceQuery(trace_type=trace_type, cursor=cursor, limit=limit),
    )
    summaries = tuple(_apply_evaluation_semantics(_summary(record, metadata), metadata) for record in page.items)
    return TraceSummaryPage(
        items=summaries,
        next_cursor=page.next_cursor,
        total=total,
        live=page.live,
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
    "get_trace_detail",
    "project_trace",
    "rollout_behavior_view",
    "trace_evaluation_view",
    "trace_summary_page",
]
