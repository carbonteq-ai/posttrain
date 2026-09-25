"""Versioned scalar evidence projected from native Verifiers trace records."""

from __future__ import annotations

import math
from collections.abc import Mapping

from posttrain.common import JsonValue, SignalSource, TraceFactSet, TraceRewardComponent

# v5: facts record the task id and prompt-group id, so group rewards aggregate
# from indexed facts. v6: thinking tokens come only from per-call usage, which
# the renderer fills on the train path (carbonteq-renderers); model-specific
# recovery rules are gone.
VERIFIERS_FACT_CALCULATOR_VERSION = "verifiers-trace-facts.v6"

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


def verifiers_trace_attributes(record: Mapping[str, object]) -> dict[str, JsonValue]:
    """Return shared source attributes without importing the Verifiers runtime."""

    return {
        "model": _model(record),
        "is_truncated": verifiers_trace_is_truncated(record),
        "has_error": verifiers_trace_has_error(record),
        "trace_schema_version": _nonnegative_int(record.get("version")),
    }


def verifiers_trace_has_error(record: Mapping[str, object]) -> bool:
    # Modern traces expose execution standing even when no exception was saved.
    # Missing `ok` is the legacy schema, not an implicit unsuccessful episode.
    errors = record.get("errors")
    if record.get("ok") is False or (isinstance(errors, list) and bool(errors)):
        return True
    calls = record.get("calls")
    return isinstance(calls, list) and any(
        isinstance(call, Mapping) and call.get("error") not in (None, False, "") for call in calls
    )


def verifiers_trace_is_truncated(record: Mapping[str, object]) -> bool:
    """Recompute Verifiers' derived truncation property from serialized fields."""

    if verifiers_trace_has_error(record):
        return False
    if record.get("stop_condition") in _TRUNCATED_STOP_CONDITIONS:
        return True
    last = _last_successful_call(record)
    return bool(last and last.get("finish_reason") == "length")


def project_verifiers_trace_facts(
    record: Mapping[str, object],
    *,
    attributes: Mapping[str, JsonValue] | None = None,
    reward_component_sources: Mapping[str, SignalSource] | None = None,
) -> TraceFactSet:
    """Project one native record into facts shared by train, OPD, and eval.

    Thinking tokens come only from each model call's ``usage.reasoning_tokens``:
    reported by the provider on API and chat-completion paths, and by the
    renderer that parsed the completion on the token-in/token-out train path.
    A trace without it records ``thinking_tokens`` as unsupported.
    """

    supplied = attributes or {}
    shared = verifiers_trace_attributes(record)
    model = _string(supplied.get("model")) or _string(shared["model"]) or ""
    trace_version = _nonnegative_int(record.get("version"))
    model_family = _identity_value(record, supplied, "model_family") or _infer_legacy_model_family(model)
    tokenizer_revision = _identity_value(record, supplied, "tokenizer_revision")
    renderer_revision = _identity_value(record, supplied, "renderer_revision")
    template_revision = _identity_value(record, supplied, "template_revision")
    is_truncated = bool(shared["is_truncated"])
    has_error = bool(shared["has_error"])
    info = record.get("info")
    info = info if isinstance(info, Mapping) else {}

    dimensions: dict[str, str | int | float | bool | None] = {
        "model": model or None,
        "model_family": model_family,
        "tokenizer_revision": tokenizer_revision,
        "renderer_revision": renderer_revision,
        "template_revision": template_revision,
        "trace_schema_version": trace_version,
        "task_type": _task_type(record),
        "task_id": _string(supplied.get("example_id")) or _string(info.get("example_id")),
        "prompt_group_id": _string(info.get("posttrain_prompt_group_id")),
        "rollout_step": _rollout_step(record, supplied),
        "is_truncated": is_truncated,
        "has_error": has_error,
    }

    calls = _calls(record)
    input_tokens, input_complete = _usage_sum(calls, "prompt_tokens", include_cached=True)
    completion_tokens, completion_complete = _usage_sum(calls, "completion_tokens")
    reasoning_tokens, reasoning_complete = _usage_sum(calls, "reasoning_tokens")
    provenance: dict[str, str] = {
        "model_input_tokens": "provider_call_usage_sum" if input_tokens is not None else "unsupported",
        "model_calls": "structured_calls" if calls is not None else "unsupported",
        "is_truncated": "verifiers_derived_fields",
        "task_reward": "verifiers_rewards",
    }

    if reasoning_tokens is not None:
        provenance["thinking_tokens"] = (
            "provider_reasoning_usage" if reasoning_complete else "provider_reasoning_usage_partial"
        )
    else:
        provenance["thinking_tokens"] = "unsupported"

    output_tokens, output_complete, output_method = _model_output_tokens(
        record,
        completion_tokens=completion_tokens,
        completion_complete=completion_complete,
        reasoning_tokens=reasoning_tokens,
        reasoning_complete=reasoning_complete,
    )
    provenance["model_output_tokens"] = output_method
    if reasoning_tokens is not None and (output_tokens is None or reasoning_tokens > output_tokens):
        reasoning_tokens = None
        reasoning_complete = False
        provenance["thinking_tokens"] = "incompatible_output_accounting"

    tool_calls, tool_method = _tool_calls(record)
    provenance["tool_calls"] = tool_method
    reward, reward_components = _reward_facts(record, reward_component_sources or {})
    if reward_components:
        provenance["reward_components"] = "verifiers_weighted_contributions"
    latency_ms = _latency_ms(calls)
    provenance["trace_latency_ms"] = "model_call_spans" if latency_ms is not None else "unsupported"

    measures: dict[str, int | float | None] = {
        "model_input_tokens": input_tokens,
        "model_output_tokens": output_tokens,
        "thinking_tokens": reasoning_tokens,
        "tool_calls": tool_calls,
        "model_calls": len(calls) if calls is not None else None,
        "trace_latency_ms": latency_ms,
        "task_reward": reward,
    }
    required_known = all(
        value is not None
        for name, value in measures.items()
        if name in {"model_output_tokens", "thinking_tokens", "tool_calls", "model_calls"}
    )
    usage_complete = input_complete and output_complete and reasoning_complete
    state = "complete" if required_known and usage_complete else "partial"
    return TraceFactSet(
        namespace="verifiers.trace",
        calculator_version=VERIFIERS_FACT_CALCULATOR_VERSION,
        dimensions=dimensions,
        measures=measures,
        reward_components=reward_components,
        provenance=provenance,
        state=state,
    )


def _calls(record: Mapping[str, object]) -> list[Mapping[str, object]] | None:
    values = record.get("calls")
    if not isinstance(values, list):
        return None
    return [value for value in values if isinstance(value, Mapping)]


def _usage_sum(
    calls: list[Mapping[str, object]] | None,
    name: str,
    *,
    include_cached: bool = False,
) -> tuple[int | None, bool]:
    if calls is None:
        return None, False
    values: list[int] = []
    missing = 0
    for call in calls:
        usage = call.get("usage")
        if not isinstance(usage, Mapping):
            missing += 1
            continue
        value = _nonnegative_int(usage.get(name))
        if value is None:
            missing += 1
            continue
        if include_cached:
            value += _nonnegative_int(usage.get("cached_input_tokens")) or 0
        values.append(value)
    return (sum(values), missing == 0) if values else (None, False)


def _model_output_tokens(
    record: Mapping[str, object],
    *,
    completion_tokens: int | None,
    completion_complete: bool,
    reasoning_tokens: int | None,
    reasoning_complete: bool,
) -> tuple[int | None, bool, str]:
    sampled = _sampled_output_tokens(record)
    if sampled is not None:
        return sampled, True, "sampled_token_ids"
    if completion_tokens is None:
        return None, False, "unsupported"
    if reasoning_tokens is not None and reasoning_tokens > completion_tokens:
        return (
            completion_tokens + reasoning_tokens,
            completion_complete and reasoning_complete,
            "provider_visible_plus_reasoning",
        )
    return completion_tokens, completion_complete, "provider_completion_usage"


def _sampled_output_tokens(record: Mapping[str, object]) -> int | None:
    nodes = record.get("nodes")
    if not isinstance(nodes, list):
        return None
    total = 0
    observed = False
    for node in nodes:
        if not isinstance(node, Mapping) or node.get("sampled") is False:
            continue
        message = node.get("message")
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        token_ids = node.get("token_ids")
        mask = node.get("mask")
        if not isinstance(token_ids, list) or not isinstance(mask, list) or len(token_ids) != len(mask):
            return None
        # Managed evaluation traces can retain the assistant message and
        # provider usage while omitting token arrays altogether.  An empty
        # array is absence of sampled-token evidence, not proof that the
        # model emitted zero tokens; fall back to provider completion usage.
        if not token_ids:
            return None
        total += sum(value is True for value in mask)
        observed = True
    return total if observed else None


def _tool_calls(record: Mapping[str, object]) -> tuple[int | None, str]:
    direct = _nonnegative_int(record.get("num_tool_calls"))
    if direct is not None:
        return direct, "native_count"
    nodes = record.get("nodes")
    if not isinstance(nodes, list):
        return None, "unsupported"
    count = 0
    for node in nodes:
        if not isinstance(node, Mapping) or node.get("sampled") is False:
            continue
        message = node.get("message")
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls")
        if isinstance(calls, list):
            count += len(calls)
    return count, "structured_assistant_calls"


def _reward_facts(
    record: Mapping[str, object],
    sources: Mapping[str, SignalSource],
) -> tuple[float | None, tuple[TraceRewardComponent, ...]]:
    rewards = record.get("rewards")
    if not isinstance(rewards, Mapping):
        return None, ()
    total = 0.0
    observed = False
    components: list[TraceRewardComponent] = []
    for raw_name, value in rewards.items():
        name = raw_name if isinstance(raw_name, str) and raw_name else None
        score = None
        weight = None
        contribution = _number(value)
        if contribution is None and isinstance(value, Mapping):
            score = _number(value.get("score"))
            weight = _number(value.get("weight"))
            contribution = score * (weight if weight is not None else 1.0) if score is not None else None
        if contribution is not None:
            total += contribution
            observed = True
            if name is not None:
                components.append(
                    TraceRewardComponent(
                        name=name,
                        contribution=contribution,
                        score=score,
                        weight=weight,
                        source=sources.get(name, SignalSource()),
                    )
                )
    scalar = total if observed and math.isfinite(total) and not verifiers_trace_has_error(record) else None
    return scalar, tuple(components)


def _latency_ms(calls: list[Mapping[str, object]] | None) -> float | None:
    if calls is None:
        return None
    duration = 0.0
    observed = False
    for call in calls:
        time_span = call.get("time")
        if not isinstance(time_span, Mapping):
            continue
        start = _number(time_span.get("start"))
        end = _number(time_span.get("end"))
        if start is not None and end is not None and end >= start:
            duration += end - start
            observed = True
    return duration * 1000 if observed else None


def _rollout_step(record: Mapping[str, object], attributes: Mapping[str, JsonValue]) -> int | None:
    explicit = _nonnegative_int(attributes.get("optimizer_step"))
    if explicit is not None:
        return explicit
    run = record.get("run")
    if not isinstance(run, Mapping) or run.get("type") != "train":
        return None
    return _nonnegative_int(run.get("step"))


def _task_type(record: Mapping[str, object]) -> str | None:
    task = record.get("task")
    return _string(task.get("type")) if isinstance(task, Mapping) else None


def _model(record: Mapping[str, object]) -> str:
    agent = record.get("agent")
    if not isinstance(agent, Mapping):
        return ""
    direct = _string(agent.get("model"))
    if direct:
        return direct
    config = agent.get("config")
    if not isinstance(config, Mapping):
        return ""
    model = config.get("model")
    if isinstance(model, Mapping):
        return _string(model.get("model")) or ""
    return _string(model) or ""


def _identity_value(
    record: Mapping[str, object],
    attributes: Mapping[str, JsonValue],
    name: str,
) -> str | None:
    direct = _string(attributes.get(name))
    if direct:
        return direct
    info = record.get("info")
    return _string(info.get(name)) if isinstance(info, Mapping) else None


def _infer_legacy_model_family(model: str) -> str | None:
    return "qwen3.5" if "qwen3.5" in model.lower() else None


def _last_successful_call(record: Mapping[str, object]) -> Mapping[str, object] | None:
    calls = _calls(record)
    if calls is None:
        return None
    return next((call for call in reversed(calls) if not call.get("error")), None)


def _nonnegative_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _number(value: object) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def _string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


__all__ = [
    "VERIFIERS_FACT_CALCULATOR_VERSION",
    "project_verifiers_trace_facts",
    "verifiers_trace_attributes",
    "verifiers_trace_has_error",
    "verifiers_trace_is_truncated",
]
