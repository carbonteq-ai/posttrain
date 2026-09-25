"""Review one run's recorded selections: rule findings plus calculator advice.

``review(snapshot, load_architecture)`` is the single entry point shared by
``posttrain settings suggest``, work-package validation and Observatory. Rule
findings need nothing but the snapshot. Calculator findings and recommended
settings also need the checkpoint's architecture, read by ``load_architecture``
(``HubModelReader`` in production); without it the review has rule findings only.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from typing import Any

from posttrain.common import ConfigurationIssue, JsonValue
from posttrain.common.validation import ConfigurationSeverity

from .calculator import Architecture, Hardware, Purpose, Suggestion, Task, suggest
from .rules import TRL_ROLLOUT_ENGINE_KEYS, rule_findings
from .snapshot import (
    Seat,
    Snapshot,
    backend_name,
    environment_seat,
    inference_seats,
    integer,
    items,
    mapping,
    number,
    served_model,
    settings_seat,
    target,
    training_seat,
)

type ArchitectureLoader = Callable[[str, str | None], Architecture]


@dataclass(frozen=True, slots=True)
class Review:
    findings: tuple[ConfigurationIssue, ...]
    recommendations: Mapping[str, dict[str, JsonValue]]
    """Calculator payload per inference role; ``{"unavailable": reason}`` when it could not run."""


class _Unavailable(Exception):
    pass


def _hardware(snapshot: Snapshot, seat: Seat) -> Hardware:
    entry = target(snapshot, seat.resolved.get("target_id"))
    if entry is None:
        raise _Unavailable(f"execution target {seat.resolved.get('target_id')} is not recorded with the run")
    memory = number(entry.get("memory_gb"))
    if memory is None:
        raise _Unavailable(f"execution target {entry.get('selection_id')} declares no memory")
    hardware = mapping(entry.get("hardware"))
    model = hardware.get("accelerator_model")
    bf16 = hardware.get("supports_bf16")
    return Hardware(
        memory_gb=memory,
        accelerator_model=model if isinstance(model, str) else None,
        accelerator_count=integer(hardware.get("accelerator_count")) or 1,
        supports_bf16=bf16 if isinstance(bf16, bool) else None,
    )


def colocated_trainer_gb(architecture: Architecture, max_length: int) -> float:
    """Estimate a LoRA trainer's resident memory beside colocated rollouts.

    Two bf16 weight copies (the trained policy and its frozen base view) plus
    checkpointed activations for one max-length sequence. For LFM2.5-2.6B at
    24,576 tokens this gives about 16 GB; the VORTEX v2 run measured 16-19 GB
    beside a 19 GB engine reservation.
    """

    weights = 2 * architecture.parameters * 2
    activations = max_length * architecture.hidden_size * architecture.layers * 4
    return round((weights + activations) / 1024**3, 1)


def binding_task(snapshot: Snapshot, seat: Seat, architecture: Architecture, draft: Architecture | None = None) -> Task:
    """Describe the load one binding serves in its job.

    A rollout binding's demand is the step's rows (prompts x generations); the
    calculator caps in-flight sequences at what fits, so a larger step is
    collected in waves, and reports step options. An evaluation binding's demand
    is its environment's concurrent episodes.
    """

    resolved = seat.resolved
    engine: Mapping[str, Any] = resolved["engine"]
    sampling = mapping(resolved.get("sampling"))
    purposes = items(resolved.get("purpose"))
    purpose: Purpose = "rollout" if "rollout" in purposes else "judge" if "judge" in purposes else "eval"
    settings = settings_seat(snapshot)
    environment = environment_seat(snapshot)
    step = group = None
    oversampling = False
    values = mapping(settings.resolved if settings is not None else None)
    prompts, generations = integer(values.get("num_prompts_per_step")), integer(values.get("num_generations"))
    prompt_limit, completion_limit = (
        integer(values.get("max_prompt_length")),
        integer(values.get("max_completion_length")),
    )
    if purpose == "rollout" and prompts and generations and prompt_limit and completion_limit:
        group, step = generations, prompts * generations
        oversampling = values.get("active_sampling") is not None or values.get("dynamic_sampling") is not None
        concurrency, prompt, completion = step, prompt_limit, completion_limit
    else:
        concurrency = (
            (
                integer(environment.resolved.get("max_concurrent"))
                if purpose == "eval" and environment is not None
                else None
            )
            or integer(engine.get("max_num_seqs"))
            or 8
        )
        max_tokens = integer(sampling.get("max_tokens")) or 1024
        context = integer(engine.get("max_model_len")) or 4 * max_tokens
        prompt, completion = context - max_tokens, max_tokens
    trainer_gb = None
    lora_rank = None
    if purpose == "rollout":
        training = training_seat(snapshot)
        update = mapping(training.resolved["parameter_update"] if training is not None else None)
        if update.get("kind") in {"lora", "qlora"}:
            lora_rank = integer(update.get("rank"))
        if engine.get("mode") == "colocate":
            trainer_gb = colocated_trainer_gb(architecture, integer(values.get("max_length")) or prompt + completion)
    temperature = number(sampling.get("temperature"))
    kv_cache_dtype = engine.get("kv_cache_dtype")
    speculative = mapping(engine.get("speculative_config", engine.get("speculative")))
    return Task(
        purpose,
        concurrency,
        prompt,
        completion,
        temperature if temperature is not None else 1.0,
        colocated_trainer_gb=trainer_gb,
        lora_rank=lora_rank,
        kv_cache_dtype=kv_cache_dtype if isinstance(kv_cache_dtype, str) else "auto",
        reproducible_logprobs=engine.get("batch_invariant") is True,
        speculative_tokens=integer(speculative.get("num_speculative_tokens")) or 0,
        draft=draft,
        step_sequences=step,
        group_size=group,
        oversampling=oversampling,
    )


def recommendation_payload(
    model_id: str,
    hardware: Hardware,
    task: Task,
    suggestion: Suggestion,
    current: Mapping[str, Any] | None = None,
) -> dict[str, JsonValue]:
    """Serialize one suggestion, diffed against the binding's engine when given."""

    settings: list[JsonValue] = []
    for item in suggestion.settings:
        row: dict[str, JsonValue] = {"key": item.key, "suggested": item.value, "reason": item.reason}
        if current is not None:
            row["current"] = current.get(item.key)
            row["changed"] = current.get(item.key) != item.value
        settings.append(row)
    return {
        "model": model_id,
        "hardware": {
            "memory_gb": hardware.memory_gb,
            "accelerator": hardware.accelerator_model,
            "gpus": hardware.accelerator_count,
        },
        "task": {
            "purpose": task.purpose,
            "concurrency": task.concurrency,
            "prompt_tokens": task.prompt_tokens,
            "completion_tokens": task.completion_tokens,
            "colocated_trainer_gb": task.colocated_trainer_gb,
        },
        "engine": dict(suggestion.engine),
        "environment": dict(suggestion.environment),
        "settings": settings,
        "memory_gb": dict(suggestion.memory_gb),
        "max_concurrency": suggestion.max_concurrency,
        "decode_tokens_per_s_upper_bound": suggestion.decode_tokens_per_s_upper_bound,
        "notes": list(suggestion.notes),
        "step": json.loads(json.dumps(asdict(suggestion.step))) if suggestion.step is not None else None,
    }


def _co_tenant_gb(snapshot: Snapshot, seat: Seat, memory_gb: float, load_architecture: ArchitectureLoader) -> float:
    """Memory other engines and a trainer of the same job hold on this seat's target.

    Another engine reserves its ``gpu_memory_utilization`` share of the device.
    A trainer on the same target counts for engines other than its own colocated
    rollout engine, which the calculator sizes against the trainer directly.
    """

    target_id = seat.resolved.get("target_id")
    total = 0.0
    for other in inference_seats(snapshot):
        if other.role == seat.role or other.resolved.get("target_id") != target_id:
            continue
        share = number(mapping(other.resolved.get("engine")).get("gpu_memory_utilization"))
        if share is not None:
            total += share * memory_gb
    training = training_seat(snapshot)
    colocated_rollout = next(
        (
            other
            for other in inference_seats(snapshot)
            if _trl_colocated(snapshot, other)
            or (
                "rollout" in items(other.resolved.get("purpose"))
                and mapping(other.resolved.get("engine")).get("mode") == "colocate"
            )
        ),
        None,
    )
    if (
        training is not None
        and training.resolved.get("target_id") == target_id
        and colocated_rollout is not None
        and colocated_rollout.role != seat.role
    ):
        base = mapping(served_model(snapshot, colocated_rollout).get("base"))
        if isinstance(base.get("repo_id"), str):
            policy = load_architecture(
                base["repo_id"], base.get("revision") if isinstance(base.get("revision"), str) else None
            )
            settings = settings_seat(snapshot)
            length = integer(settings.resolved.get("max_length")) if settings is not None else None
            total += colocated_trainer_gb(policy, length or 8_192)
    return round(total, 1)


def _trl_colocated(snapshot: Snapshot, seat: Seat) -> bool:
    training = training_seat(snapshot)
    return (
        training is not None
        and backend_name(training.resolved.get("backend")) == "trl"
        and "rollout" in items(seat.resolved.get("purpose"))
        and mapping(seat.resolved.get("engine")).get("mode") == "colocate"
    )


def _only_consumed_keys(snapshot: Snapshot, seat: Seat, suggestion: Suggestion) -> Suggestion:
    """Drop suggestions the colocated TRL rollout engine would ignore (it follows the checkpoint dtype)."""

    if not _trl_colocated(snapshot, seat):
        return suggestion
    keep = TRL_ROLLOUT_ENGINE_KEYS - {"dtype"}
    settings = tuple(item for item in suggestion.settings if item.key in keep)
    return replace(suggestion, settings=settings, engine={item.key: item.value for item in settings})


def draft_reference(engine: Mapping[str, Any]) -> tuple[str, str | None] | None:
    """The separate drafter an engine speculates with, as (repository, revision).

    MTP assistants are recorded as ``draft_model``; vLLM's own schema (DSpark,
    DFlash, EAGLE) names the drafter ``model`` at ``revision``.
    """
    speculative = mapping(engine.get("speculative_config", engine.get("speculative")))
    draft_model = mapping(speculative.get("draft_model"))
    repository, revision = draft_model.get("repo_id"), draft_model.get("revision")
    if not isinstance(repository, str):
        repository, revision = speculative.get("model"), speculative.get("revision")
    if not isinstance(repository, str):
        return None
    return repository, revision if isinstance(revision, str) else None


def recommend(snapshot: Snapshot, seat: Seat, load_architecture: ArchitectureLoader) -> dict[str, JsonValue]:
    """The calculator's payload for one inference seat, with the seats its findings point at."""

    hardware = _hardware(snapshot, seat)
    variant = seat.resolved.get("model_variant_id")
    base = mapping(served_model(snapshot, seat).get("base"))
    if not isinstance(base.get("repo_id"), str):
        raise _Unavailable(f"model {variant} is not recorded with the run")
    revision = base.get("revision")
    architecture = load_architecture(base["repo_id"], revision if isinstance(revision, str) else None)
    engine: Mapping[str, Any] = seat.resolved["engine"]
    draft = None
    draft_note = None
    if (drafter := draft_reference(engine)) is not None:
        try:
            draft = load_architecture(*drafter)
        except Exception as error:  # noqa: BLE001 - size the target alone and say so
            draft_note = (
                f"draft model {drafter[0]} could not be read ({type(error).__name__}); memory excludes the drafter"
            )
    task = replace(
        binding_task(snapshot, seat, architecture, draft),
        co_tenant_gb=_co_tenant_gb(snapshot, seat, hardware.memory_gb, load_architecture),
    )
    suggestion = suggest(hardware, architecture, task)
    if draft_note is not None:
        suggestion = replace(suggestion, notes=(*suggestion.notes, draft_note))
    suggestion = _only_consumed_keys(snapshot, seat, suggestion)
    payload = recommendation_payload(str(variant), hardware, task, suggestion, seat.resolved["engine"])
    payload["binding_id"] = seat.label
    settings, environment = settings_seat(snapshot), environment_seat(snapshot)
    if task.purpose == "rollout" and settings is not None:
        payload["settings_role"] = settings.role
    if environment is not None:
        payload["environment_role"] = environment.role
        payload["environment_max_concurrent"] = environment.resolved.get("max_concurrent")
    return payload


# The rules already fail these outright; the calculator does not repeat them.
_GUARDED_KEYS = frozenset({"enforce_eager", "enable_prefix_caching", "dtype", "max_model_len", "kv_cache_dtype"})
# Keys with a dedicated calculator finding below.
_SPECIFIC_KEYS = frozenset({"max_num_seqs", "max_num_batched_tokens", "kv_cache_memory_bytes"})
_UTILIZATION_TOLERANCE = 0.05
_KV_SHORTFALL = 0.8


def calculator_findings(role: str, recommendation: Mapping[str, Any]) -> tuple[ConfigurationIssue, ...]:
    """Turn one recommendation into findings where the run falls short of it.

    Warnings are shortfalls with a measured or computed cost (queued sequences,
    extra prefill chunks, a KV cache smaller than the step). Other differences,
    and spare capacity a larger step could use, are recommendations.
    """

    rows, task = recommendation.get("settings"), recommendation.get("task")
    if "unavailable" in recommendation or not isinstance(rows, list) or not isinstance(task, Mapping):
        return ()
    concurrency = number(task.get("concurrency")) or 0
    issues: list[ConfigurationIssue] = []

    def issue(code: str, severity: ConfigurationSeverity, key: str, message: str, hint: str) -> None:
        issues.append(ConfigurationIssue(code, severity, "static", role, f"{role}.engine.{key}", message, hint))

    suggested_seqs = None
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        key, current, suggested = str(row.get("key")), row.get("current"), row.get("suggested")
        if key == "max_num_seqs":
            suggested_seqs = number(suggested)
        if not row.get("changed") or key in _GUARDED_KEYS or current is None:
            continue
        reason = str(row.get("reason") or "")
        current_number, suggested_number = number(current), number(suggested)
        if key == "max_num_seqs" and current_number is not None and suggested_number is not None:
            if current_number < suggested_number:
                issue(
                    "CALCULATOR_MAX_NUM_SEQS_BELOW_CONCURRENCY",
                    "warning",
                    key,
                    f"max_num_seqs {current} is below the {suggested} sequences this job can run at once on this "
                    "target, so the engine queues the rest and each collection takes longer",
                    f"Set max_num_seqs to {suggested}.",
                )
        elif key == "max_num_batched_tokens" and current_number is not None and suggested_number is not None:
            if current_number < suggested_number:
                issue(
                    "CALCULATOR_BATCHED_TOKENS_BELOW_RECOMMENDED",
                    "warning",
                    key,
                    f"max_num_batched_tokens {current} is below the calculator's {suggested}: {reason}",
                    f"Set max_num_batched_tokens to {suggested}.",
                )
        elif key == "kv_cache_memory_bytes" and current_number is not None and suggested_number:
            if current_number < _KV_SHORTFALL * suggested_number:
                fits = concurrency * current_number / suggested_number
                issue(
                    "CALCULATOR_KV_CACHE_BELOW_CONCURRENCY",
                    "warning",
                    key,
                    f"the pinned KV cache ({current_number / 1024**3:.1f} GiB) holds about {fits:.0f} of the "
                    f"{concurrency:g} typical-length sequences in flight; the calculator sizes it at "
                    f"{suggested_number / 1024**3:.1f} GiB ({reason})",
                    f"Set kv_cache_memory_bytes to {suggested}, or run fewer sequences at once.",
                )
        elif key == "gpu_memory_utilization" and current_number is not None and suggested_number is not None:
            if abs(current_number - suggested_number) > _UTILIZATION_TOLERANCE:
                issue(
                    "CALCULATOR_SETTING_DIFFERS",
                    "recommendation",
                    key,
                    f"gpu_memory_utilization {current} differs from the calculator's {suggested}: {reason}",
                    f"Use {suggested} unless another engine shares this device.",
                )
        elif key not in _SPECIFIC_KEYS:
            issue(
                "CALCULATOR_SETTING_DIFFERS",
                "recommendation",
                key,
                f"{key} is {json.dumps(current)}; the calculator suggests {json.dumps(suggested)}: {reason}",
                f"Set {key} to {json.dumps(suggested)}.",
            )
    environment_role = recommendation.get("environment_role")
    environment_concurrency = number(recommendation.get("environment_max_concurrent"))
    if (
        isinstance(environment_role, str)
        and environment_concurrency is not None
        and suggested_seqs is not None
        and task.get("purpose") == "rollout"
        and environment_concurrency < suggested_seqs
    ):
        issues.append(
            ConfigurationIssue(
                "CALCULATOR_ENVIRONMENT_CONCURRENCY_BELOW_ENGINE",
                "warning",
                "static",
                environment_role,
                f"{environment_role}.max_concurrent",
                f"the environment runs at most {environment_concurrency:g} episodes at once, below the "
                f"{suggested_seqs:g} the engine can serve for this step, so rollouts queue on the environment",
                f"Set the environment's max_concurrent to {suggested_seqs:g}.",
            )
        )
    step, settings_role = recommendation.get("step"), recommendation.get("settings_role")
    if isinstance(step, Mapping) and isinstance(settings_role, str):
        if (number(step.get("margin_sequences")) or 0) > 0 and number(step.get("extra_prompts_per_step")):
            issues.append(
                ConfigurationIssue(
                    "CALCULATOR_STEP_BELOW_USEFUL_CONCURRENCY",
                    "recommendation",
                    "static",
                    settings_role,
                    f"{settings_role}.num_prompts_per_step",
                    f"{step.get('reason')}; see the step-sizing options for waves and step time",
                    (
                        f"Raise num_prompts_per_step to {step.get('recommended_prompts_per_step')}"
                        + (
                            f" and oversample {step.get('oversample_groups')} groups (at most 20% of the step)"
                            if step.get("oversample_groups")
                            else ""
                        )
                        + f", then set max_num_seqs and the environment's max_concurrent to "
                        f"{step.get('recommended_in_flight')}. A larger step changes the effective batch; keep the "
                        "learning rate under review."
                    ),
                )
            )
        elif (number(step.get("waves")) or 1) > 1:
            issues.append(
                ConfigurationIssue(
                    "CALCULATOR_STEP_COLLECTS_IN_WAVES",
                    "info",
                    "static",
                    settings_role,
                    f"{settings_role}.num_prompts_per_step",
                    f"{step.get('reason')}; see the step-sizing options for waves and step time",
                    "Keep the step if the batch size matters more than step time; otherwise use the one-wave option.",
                )
            )
    return tuple(issues)


def review(snapshot: Snapshot, load_architecture: ArchitectureLoader | None = None) -> Review:
    """Rule findings, then each inference seat's calculator findings and recommendation."""

    findings = list(rule_findings(snapshot))
    recommendations: dict[str, dict[str, JsonValue]] = {}
    if load_architecture is not None:
        for seat in inference_seats(snapshot):
            try:
                payload = recommend(snapshot, seat, load_architecture)
            except Exception as error:  # noqa: BLE001 - advice never blocks a view or a plan
                message = str(error) if isinstance(error, _Unavailable) else f"{type(error).__name__}: {error}"
                recommendations[seat.role] = {"unavailable": message}
                continue
            recommendations[seat.role] = payload
            findings.extend(calculator_findings(seat.role, payload))
    return Review(tuple(findings), recommendations)


__all__ = [
    "ArchitectureLoader",
    "Review",
    "binding_task",
    "calculator_findings",
    "colocated_trainer_gb",
    "recommend",
    "recommendation_payload",
    "review",
]
