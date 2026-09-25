"""Configuration rules over a run's recorded selections.

Two rule sets:

- Performance rules: settings that undo a measured inference optimization. A
  violation is an error unless the binding acknowledges the code with a reason
  (``performance_acknowledgements``), which turns it into information, so the
  trade-off stays visible. Evidence: ``docs/plan/agentic-workload-inference-optimization.md``.
- Training rules: LoRA reinforcement-learning settings far from the reference
  recipes. They are recipe choices, so they warn. References: Thinking Machines,
  "LoRA Without Regret", and the Tinker cookbook RL recipes (constant learning
  rates of 1e-5 to 4e-5 at LoRA alpha 32, no KL penalty). With Adam and a
  zero-initialized B matrix the first-order LoRA update scales with learning
  rate x alpha, so the recipes correspond to lr x alpha of 3.2e-4 to 1.28e-3.

Paths name fields in the snapshot (``<role>.engine.enforce_eager``,
``<role>.learning_rate``), so a reader can show each finding beside its value.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

from posttrain.common import ConfigurationIssue

from .snapshot import (
    Seat,
    Snapshot,
    backend_name,
    inference_seats,
    integer,
    number,
    served_model,
    settings_seat,
    training_seat,
)

_EVIDENCE = "docs/plan/agentic-workload-inference-optimization.md"
_GENERATION_PURPOSES = frozenset({"rollout", "eval", "screen", "teacher-score", "smoke"})
_DRAFT_FREE_SPECULATIVE = frozenset({"ngram", "ngram_gpu", "suffix"})
_ACKNOWLEDGE = (
    "If this is deliberate, acknowledge {code} in the binding's performance_acknowledgements with the reason."
)

# Every rollout-engine key the colocated TRL path reads (packages/train, TRL
# backend). A binding key outside this set is silently ignored by TRL. The drift
# test in packages/train/tests/test_trl_rollout_engine_keys.py scans the backend
# source so this set cannot fall behind it.
TRL_ROLLOUT_ENGINE_KEYS = frozenset(
    {
        "attention_backend_priority",
        "batch_invariant",
        "disable_torch_compile",
        "dtype",
        "enable_chunked_prefill",
        "enable_prefix_caching",
        "enforce_eager",
        "flash_attn_version",
        "gpu_memory_utilization",
        "kv_cache_dtype",
        "kv_cache_memory_bytes",
        "max_model_len",
        "max_num_batched_tokens",
        "max_num_seqs",
        "mode",
        "request_mode",
        "skip_mm_profiling",
        "sleep_during_optimization",
        "speculative_config",
        "tensor_parallel_size",
        "text_only",
        "weight_name_prefix",
        "weight_sync_mode",
    }
)

# Tinker RL recipes: 1e-5 .. 4e-5 at alpha 32.
REFERENCE_LR_TIMES_ALPHA = (1e-5 * 32, 4e-5 * 32)
_HIGH_MULTIPLE = 4.0
_SHORT_RUN_STEPS = 50


def _performance_issue(
    seat: Seat, code: str, path: str, message: str, hint: str, related_paths: tuple[str, ...] = ()
) -> ConfigurationIssue:
    acknowledgements = seat.resolved.get("performance_acknowledgements")
    reason = acknowledgements.get(code) if isinstance(acknowledgements, Mapping) else None
    if isinstance(reason, str) and reason.strip():
        return ConfigurationIssue(
            code,
            "info",
            "static",
            seat.role,
            path,
            f"{message} Acknowledged by {seat.label}: {reason}",
            None,
            related_paths,
        )
    return ConfigurationIssue(
        code, "error", "static", seat.role, path, message, f"{hint} {_ACKNOWLEDGE.format(code=code)}", related_paths
    )


def _sampled(sampling: Mapping[str, Any]) -> bool:
    temperature = sampling.get("temperature")
    # An omitted temperature samples at the engine default (1.0).
    return temperature is None or (number(temperature) or 0.0) > 0


def performance_findings(snapshot: Snapshot) -> tuple[ConfigurationIssue, ...]:
    """Report inference settings that disable a measured optimization."""

    training = training_seat(snapshot)
    trl_training = training is not None and backend_name(training.resolved.get("backend")) == "trl"
    settings = settings_seat(snapshot)
    episode_budget = None
    if settings is not None:
        prompt, completion = (
            integer(settings.resolved.get(key)) for key in ("max_prompt_length", "max_completion_length")
        )
        if prompt is not None and completion is not None:
            episode_budget = prompt + completion
    issues: list[ConfigurationIssue] = []
    for seat in inference_seats(snapshot):
        purposes = seat.resolved.get("purpose")
        purposes = set(purposes) if isinstance(purposes, list) else set()
        if backend_name(seat.resolved.get("backend")) != "vllm" or not purposes & _GENERATION_PURPOSES:
            continue
        issues.extend(_binding_findings(snapshot, seat, "rollout" in purposes, trl_training, episode_budget))
    return tuple(issues)


def _binding_findings(
    snapshot: Snapshot, seat: Seat, rollout: bool, trl_training: bool, episode_budget: int | None
) -> Iterator[ConfigurationIssue]:
    engine: Mapping[str, Any] = seat.resolved["engine"]
    sampling = seat.resolved.get("sampling")
    sampling = sampling if isinstance(sampling, Mapping) else {}
    role, label = seat.role, seat.label
    max_model_len = integer(engine.get("max_model_len"))
    if rollout and episode_budget is not None and max_model_len is not None and max_model_len < episode_budget:
        yield _performance_issue(
            seat,
            "ROLLOUT_CONTEXT_BELOW_EPISODE_BUDGET",
            f"{role}.engine.max_model_len",
            (
                f"{label} admits {max_model_len} tokens but the training settings allow prompts plus completions "
                f"of {episode_budget}; longer episodes fail at the engine instead of training (a 24,576-token "
                f"LFM2.5 profile failed 4% of AutomationBench episodes, {_EVIDENCE})."
            ),
            "Raise max_model_len to at least max_prompt_length + max_completion_length.",
        )
    if rollout and (engine.get("kv_offloading_size") is not None or engine.get("kv_transfer_config") is not None):
        yield _performance_issue(
            seat,
            "VLLM_KV_OFFLOAD_IN_ROLLOUT",
            f"{role}.engine.kv_offloading_size",
            (
                f"{label} offloads KV cache to host memory during rollouts; the offload tier survives the "
                "prefix-cache reset at each policy update, so later rollouts can reuse KV computed by the previous "
                f"policy, and it was not faster at concurrency 16 ({_EVIDENCE})."
            ),
            "Remove kv_offloading_size and kv_transfer_config from rollout bindings.",
        )
    kv_cache_dtype = engine.get("kv_cache_dtype")
    turboquant = isinstance(kv_cache_dtype, str) and kv_cache_dtype.startswith("turboquant_")
    variant = seat.resolved.get("model_variant_id")
    bf16 = served_model(snapshot, seat).get("weight_precision") == "bf16"
    if engine.get("dtype") in {"float16", "half"} and bf16 and not turboquant:
        yield _performance_issue(
            seat,
            "VLLM_FLOAT16_ON_BF16_CHECKPOINT",
            f"{role}.engine.dtype",
            (
                f"{label} computes a bf16 checkpoint ({variant}) in float16, whose narrower exponent range can "
                "overflow activations and changes logprobs relative to training."
            ),
            "Set dtype to bfloat16 or auto.",
        )
    if engine.get("enforce_eager") is True:
        yield _performance_issue(
            seat,
            "VLLM_EAGER_DISABLES_CUDA_GRAPHS",
            f"{role}.engine.enforce_eager",
            (
                f"{label} forces eager execution, which disables CUDA graphs for every decode step; on the LFM2.5 "
                "AutomationBench rollout replay eager decode was 2.32x slower (150.8 s against 64.9 s per "
                f"collection at concurrency 16, {_EVIDENCE})."
            ),
            "Remove enforce_eager or set it to false.",
        )
    if engine.get("enable_prefix_caching") is False:
        yield _performance_issue(
            seat,
            "VLLM_PREFIX_CACHING_DISABLED",
            f"{role}.engine.enable_prefix_caching",
            (
                f"{label} disables prefix caching, so every turn of a multi-turn episode and every sample of a group "
                "re-prefills its shared context; the LFM2.5 AutomationBench replay served 80-89% of prompt tokens "
                f"from the prefix cache ({_EVIDENCE})."
            ),
            (
                "Remove enable_prefix_caching or set it to true; policy updates already reset the cache, so reuse "
                "across weight versions cannot occur."
            ),
        )
    speculative = engine.get("speculative_config", engine.get("speculative"))
    method = speculative.get("method") if isinstance(speculative, Mapping) else None
    if method in _DRAFT_FREE_SPECULATIVE and _sampled(sampling):
        yield _performance_issue(
            seat,
            "SPECULATIVE_DRAFT_FREE_WITH_SAMPLING",
            f"{role}.engine.speculative_config.method",
            (
                f"{label} uses draft-free {method} speculative decoding while sampling; drafts copied from the "
                "context are rarely accepted at nonzero temperature, and on the LFM2.5 AutomationBench replay "
                f"ngram_gpu made collections 0.84x as fast ({_EVIDENCE})."
            ),
            "Remove the speculative configuration, or use it only with greedy decoding.",
        )
    if trl_training and engine.get("mode") == "colocate" and rollout:
        ignored = sorted(set(engine) - TRL_ROLLOUT_ENGINE_KEYS)
        if ignored:
            yield _performance_issue(
                seat,
                "TRL_ROLLOUT_ENGINE_KEYS_IGNORED",
                f"{role}.engine",
                (
                    f"{label} sets {', '.join(ignored)}, which the colocated TRL rollout engine does not read; these "
                    "values have no effect on the run."
                ),
                "Remove the keys, or move them to a binding for a backend that consumes them.",
                tuple(f"{role}.engine.{key}" for key in ignored),
            )


def training_findings(snapshot: Snapshot) -> tuple[ConfigurationIssue, ...]:
    """Warn about LoRA RL settings far from the reference recipes."""

    settings, training = settings_seat(snapshot), training_seat(snapshot)
    if settings is None or training is None:
        return ()
    # Reinforcement learning settings sample groups; distillation records its divergence instead.
    if "num_generations" not in settings.resolved or "divergence" in settings.resolved:
        return ()
    update: Mapping[str, Any] = training.resolved["parameter_update"]
    alpha = integer(update.get("alpha"))
    learning_rate = number(settings.resolved.get("learning_rate"))
    if update.get("kind") not in {"lora", "qlora"} or alpha is None or learning_rate is None:
        return ()
    role, training_role = settings.role, training.role
    issues: list[ConfigurationIssue] = []
    effective = learning_rate * alpha
    low, high = REFERENCE_LR_TIMES_ALPHA
    tinker_equivalent = effective / 32
    if effective < low:
        issues.append(
            ConfigurationIssue(
                "LORA_RL_LEARNING_RATE_BELOW_REFERENCE",
                "warning",
                "static",
                role,
                f"{role}.learning_rate",
                (
                    f"learning rate {learning_rate:g} at LoRA alpha {alpha} equals {tinker_equivalent:.2g} in the "
                    f"alpha-32 convention, {low / 32 / tinker_equivalent:.1f}x below the lowest Tinker RL recipe "
                    "(1e-5); LoRA needs about 10x the full fine-tuning rate, and VORTEX v1/v2 at this setting left "
                    "held-out results unchanged"
                ),
                (
                    f"Use a learning rate between {low / alpha:.2g} and {high / alpha:.2g} for alpha {alpha} (the "
                    "recipes' 1e-5 to 4e-5 at alpha 32), with a constant schedule."
                ),
                (f"{training_role}.parameter_update.alpha",),
            )
        )
    elif effective > high * _HIGH_MULTIPLE:
        issues.append(
            ConfigurationIssue(
                "LORA_RL_LEARNING_RATE_ABOVE_REFERENCE",
                "warning",
                "static",
                role,
                f"{role}.learning_rate",
                (
                    f"learning rate {learning_rate:g} at LoRA alpha {alpha} equals {tinker_equivalent:.2g} in the "
                    f"alpha-32 convention, more than {_HIGH_MULTIPLE:g}x the most aggressive Tinker RL recipe (4e-5); "
                    "without a KL term only the clipped objective bounds each update"
                ),
                "Watch per-update KL and entropy on the first steps, or reduce the learning rate.",
                (f"{training_role}.parameter_update.alpha",),
            )
        )
    max_steps = integer(settings.resolved.get("max_steps"))
    # Runs recorded before the schedule was snapshotted leave it unknown; say nothing then.
    if (
        settings.resolved.get("lr_scheduler_type") == "linear"
        and max_steps is not None
        and max_steps <= _SHORT_RUN_STEPS
    ):
        issues.append(
            ConfigurationIssue(
                "LR_SCHEDULE_DECAYS_WITHIN_SHORT_RUN",
                "warning",
                "static",
                role,
                f"{role}.lr_scheduler_type",
                (
                    f"a linear schedule decays the learning rate to zero over only {max_steps} updates, so the mean "
                    "rate is about half the configured value and the last updates barely move the policy; the "
                    "reference RL recipes train at a constant rate"
                ),
                "Set lr_scheduler_type to constant (or constant_with_warmup).",
            )
        )
    targets = update.get("target_modules")
    if isinstance(targets, str) and targets != "all-linear":
        issues.append(
            ConfigurationIssue(
                "LORA_PARTIAL_TARGET_MODULES",
                "recommendation",
                "static",
                training_role,
                f"{training_role}.parameter_update.target_modules",
                (
                    f"LoRA targets {targets!r}; LoRA Without Regret found adapters on every linear layer (including "
                    "the MLP) match full fine-tuning, while attention-only adapters underperform"
                ),
                "Use target_modules: all-linear unless a module set was qualified for this model.",
            )
        )
    return tuple(issues)


def rule_findings(snapshot: Snapshot) -> tuple[ConfigurationIssue, ...]:
    """Every static rule finding for one run's recorded selections."""

    from .compatibility import compatibility_findings

    return performance_findings(snapshot) + compatibility_findings(snapshot) + training_findings(snapshot)


__all__ = [
    "REFERENCE_LR_TIMES_ALPHA",
    "TRL_ROLLOUT_ENGINE_KEYS",
    "performance_findings",
    "rule_findings",
    "training_findings",
]
