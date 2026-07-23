"""Backend-neutral GRPO metric normalization and evidence requirements."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from .requests import GRPORequest

type GRPOBackendProduct = Literal["trl", "verl"]


@dataclass(frozen=True, slots=True)
class GRPOObservationFeatures:
    """Resolved feature switches that determine the evidence a GRPO run owes."""

    reference_kl_enabled: bool = False
    clipping_enabled: bool = True
    decoupled_rollout: bool = False
    asynchronous_rollout: bool = False
    mtp_rollout_enabled: bool = False
    quantized_kv_cache: bool = False
    tool_environment: bool = False

    @classmethod
    def from_request(
        cls,
        request: GRPORequest,
        *,
        tool_environment: bool = False,
    ) -> GRPOObservationFeatures:
        engine = request.inference.engine
        speculative = engine.get("speculative_config")
        kv_cache_dtype = engine.get("kv_cache_dtype")
        inference_product = request.inference.backend.split("@", 1)[0]
        mode = engine.get("mode")
        return cls(
            reference_kl_enabled=request.settings.beta > 0,
            clipping_enabled=True,
            decoupled_rollout=inference_product == "vllm",
            asynchronous_rollout=mode == "async",
            mtp_rollout_enabled=isinstance(speculative, Mapping) and speculative.get("method") == "mtp",
            quantized_kv_cache=isinstance(kv_cache_dtype, str) and kv_cache_dtype.startswith("turboquant_"),
            tool_environment=tool_environment,
        )


@dataclass(frozen=True, slots=True)
class NormalizedGRPOStep:
    """One logical GRPO step after private backend names have been removed."""

    step: int
    metrics: Mapping[str, float]
    attributes: Mapping[str, str]

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ValueError("GRPO logical step cannot be negative")
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
        object.__setattr__(self, "attributes", MappingProxyType(dict(self.attributes)))


@dataclass(frozen=True, slots=True)
class GRPOEvidenceStatus:
    """Run-level completeness result for the selected GRPO feature set."""

    required: frozenset[str]
    missing: frozenset[str]

    @property
    def complete(self) -> bool:
        return not self.missing


_TRL_METRICS: Mapping[str, str] = MappingProxyType(
    {
        "reward": "train/rl/reward_mean",
        "reward_std": "train/rl/reward_std",
        "frac_reward_zero_std": "train/rl/group_zero_variance_fraction",
        "loss": "train/rl/policy_loss",
        "policy_loss": "train/rl/policy_loss",
        "kl": "train/rl/kl",
        "entropy": "train/rl/entropy",
        "clip_ratio": "train/rl/clip_fraction",
        "clip_ratio/region_mean": "train/rl/clip_fraction",
        "grad_norm": "train/grad_norm",
        "learning_rate": "train/learning_rate",
        "step_time": "train/step_time_seconds",
        "num_tokens": "train/num_tokens",
        "completions/mean_length": "train/rl/completion_tokens_mean",
        "completions/max_length": "train/rl/completion_tokens_max",
        "completions/clipped_ratio": "train/rl/completion_truncation_rate",
        "tools/call_frequency": "train/rl/tool_call_frequency",
        "tools/failure_frequency": "train/rl/tool_failure_frequency",
        "sampling/sampling_logp_difference/mean": "train/rl/sampling_logp_delta_mean",
        "sampling/sampling_logp_difference/max": "train/rl/sampling_logp_delta_max",
        "sampling/importance_sampling_ratio/mean": "train/rl/importance_sampling_ratio_mean",
        "sampling/importance_sampling_ratio/min": "train/rl/importance_sampling_ratio_min",
        "sampling/importance_sampling_ratio/max": "train/rl/importance_sampling_ratio_max",
        "rollout/spec_num_drafts": "serve/backend/speculative_drafts",
        "rollout/spec_num_draft_tokens": "serve/backend/speculative_draft_tokens",
        "rollout/spec_num_accepted_tokens": "serve/backend/speculative_accepted_tokens",
        "rollout/spec_accept_rate": "serve/backend/speculative_acceptance_rate",
        "rollout/spec_accept_length": "serve/backend/speculative_accepted_length",
        "rollout/kv_cache_capacity_tokens": "serve/backend/kv_cache_capacity_tokens",
        "rollout/kv_cache_peak_usage_ratio": "serve/backend/kv_cache_peak_usage_ratio",
    }
)

_VERL_METRICS: Mapping[str, str] = MappingProxyType(
    {
        "critic/rewards/mean": "train/rl/reward_mean",
        "critic/rewards/std": "train/rl/reward_std",
        "actor/pg_loss": "train/rl/policy_loss",
        "actor/policy_loss": "train/rl/policy_loss",
        "actor/ppo_kl": "train/rl/kl",
        "actor/entropy": "train/rl/entropy",
        "actor/pg_clipfrac": "train/rl/clip_fraction",
        "actor/grad_norm": "train/grad_norm",
        "actor/lr": "train/learning_rate",
        "perf/time_per_step": "train/step_time_seconds",
        "perf/total_num_tokens": "train/num_tokens",
        "response_length/total": "train/rl/completion_tokens_total",
        "response_length/mean": "train/rl/completion_tokens_mean",
        "response_length/max": "train/rl/completion_tokens_max",
        "response_length/clip_ratio": "train/rl/completion_truncation_rate",
        "timing_s/gen": "train/rl/time/rollout_seconds",
        "timing_s/reward": "train/rl/time/reward_seconds",
        "timing_s/old_log_prob": "train/rl/time/actor_forward_seconds",
        "timing_s/update_actor": "train/rl/time/actor_update_seconds",
        "timing_s/sync": "train/rl/time/weight_sync_seconds",
        "timing_s/save_checkpoint": "train/rl/time/checkpoint_seconds",
        "training/off_policy/trajectory_staleness/mean": "train/rl/policy_staleness_mean",
        "training/off_policy/trajectory_staleness/max": "train/rl/policy_staleness_max",
        "training/off_policy/trajectory_spans/mean": "train/rl/trajectory_version_span_mean",
        "rollout/spec_num_verify_steps": "serve/backend/speculative_drafts",
        "rollout/spec_num_draft_tokens": "serve/backend/speculative_draft_tokens",
        "rollout/spec_num_accepted_tokens": "serve/backend/speculative_accepted_tokens",
        "rollout/spec_accept_rate": "serve/backend/speculative_acceptance_rate",
        "rollout/spec_accept_length": "serve/backend/speculative_accepted_length",
    }
)

_CANONICAL_PASSTHROUGH = frozenset(
    {
        "train/rl/reward_mean",
        "train/rl/reward_std",
        "train/rl/group_zero_variance_fraction",
        "train/rl/policy_loss",
        "train/rl/kl",
        "train/rl/entropy",
        "train/rl/clip_fraction",
        "train/grad_norm",
        "train/learning_rate",
        "train/step_time_seconds",
        "train/num_tokens",
        "train/rl/rollouts_attempted",
        "train/rl/rollouts_completed",
        "train/rl/rollouts_failed",
        "train/rl/rollouts_truncated",
        "train/rl/rollouts_unscorable",
        "train/rl/completion_tokens_total",
        "train/rl/completion_tokens_mean",
        "train/rl/completion_tokens_max",
        "train/rl/completion_truncation_rate",
        "train/rl/tool_call_frequency",
        "train/rl/tool_failure_frequency",
        "train/rl/sampling_logp_delta_mean",
        "train/rl/sampling_logp_delta_max",
        "train/rl/importance_sampling_ratio_mean",
        "train/rl/importance_sampling_ratio_min",
        "train/rl/importance_sampling_ratio_max",
        "train/rl/policy_staleness_mean",
        "train/rl/policy_staleness_max",
        "train/rl/trajectory_version_span_mean",
        "train/rl/rollout_tokens_per_second",
        "train/rl/time/rollout_seconds",
        "train/rl/time/reward_seconds",
        "train/rl/time/actor_forward_seconds",
        "train/rl/time/actor_update_seconds",
        "train/rl/time/weight_sync_seconds",
        "train/rl/time/checkpoint_seconds",
        "serve/backend/speculative_drafts",
        "serve/backend/speculative_draft_tokens",
        "serve/backend/speculative_accepted_tokens",
        "serve/backend/speculative_acceptance_rate",
        "serve/backend/speculative_accepted_length",
        "serve/backend/kv_cache_capacity_tokens",
        "serve/backend/kv_cache_peak_usage_ratio",
    }
)

_RATIO_METRICS = frozenset(
    {
        "train/rl/group_zero_variance_fraction",
        "train/rl/clip_fraction",
        "train/rl/completion_truncation_rate",
        "train/rl/tool_call_frequency",
        "train/rl/tool_failure_frequency",
        "serve/backend/speculative_acceptance_rate",
        "serve/backend/kv_cache_peak_usage_ratio",
    }
)

_NON_NEGATIVE_METRICS = frozenset(
    {
        "train/rl/reward_std",
        "train/grad_norm",
        "train/learning_rate",
        "train/step_time_seconds",
        "train/num_tokens",
        "train/rl/rollouts_attempted",
        "train/rl/rollouts_completed",
        "train/rl/rollouts_failed",
        "train/rl/rollouts_truncated",
        "train/rl/rollouts_unscorable",
        "train/rl/completion_tokens_total",
        "train/rl/completion_tokens_mean",
        "train/rl/completion_tokens_max",
        "train/rl/rollout_tokens_per_second",
        "train/rl/time/rollout_seconds",
        "train/rl/time/reward_seconds",
        "train/rl/time/actor_forward_seconds",
        "train/rl/time/actor_update_seconds",
        "train/rl/time/weight_sync_seconds",
        "train/rl/time/checkpoint_seconds",
        "serve/backend/speculative_drafts",
        "serve/backend/speculative_draft_tokens",
        "serve/backend/speculative_accepted_tokens",
        "serve/backend/speculative_accepted_length",
        "serve/backend/kv_cache_capacity_tokens",
    }
)

_MTP_REQUIRED = frozenset(
    {
        "serve/backend/speculative_draft_tokens",
        "serve/backend/speculative_accepted_tokens",
        "serve/backend/speculative_acceptance_rate",
        "serve/backend/speculative_accepted_length",
    }
)

_CORE_REQUIRED = frozenset(
    {
        "train/rl/reward_mean",
        "train/rl/reward_std",
        "train/rl/group_zero_variance_fraction",
        "train/rl/policy_loss",
        "train/rl/entropy",
        "train/grad_norm",
        "train/learning_rate",
        "train/step_time_seconds",
        "train/rl/rollouts_attempted",
        "train/rl/rollouts_completed",
        "train/rl/rollouts_failed",
        "train/rl/rollouts_truncated",
        "train/rl/rollouts_unscorable",
        "train/rl/completion_tokens_mean",
        "train/rl/completion_tokens_max",
        "train/rl/completion_truncation_rate",
        "train/rl/rollout_tokens_per_second",
    }
)


def normalize_grpo_metrics(
    *,
    backend: GRPOBackendProduct,
    step: int,
    native: Mapping[str, object],
    features: GRPOObservationFeatures,
) -> NormalizedGRPOStep:
    """Translate one native trainer record without importing either trainer."""

    if step < 0:
        raise ValueError("GRPO logical step cannot be negative")
    if backend not in {"trl", "verl"}:
        raise ValueError(f"unsupported GRPO observation backend {backend!r}")
    mapping = _TRL_METRICS if backend == "trl" else _VERL_METRICS
    normalized: dict[str, float] = {}
    origins: dict[str, str] = {}
    for name, raw_value in native.items():
        canonical = name if name in _CANONICAL_PASSTHROUGH else mapping.get(name)
        if canonical is None or isinstance(raw_value, bool) or not isinstance(raw_value, int | float):
            continue
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"non-finite GRPO metric {name}={value}")
        previous = origins.get(canonical)
        if previous is not None:
            raise ValueError(f"native GRPO metrics {previous!r} and {name!r} both map to {canonical!r}")
        _validate_metric_value(canonical, value)
        normalized[canonical] = value
        origins[canonical] = name

    observed_mtp = _MTP_REQUIRED.intersection(normalized)
    if observed_mtp and observed_mtp != _MTP_REQUIRED:
        missing = ", ".join(sorted(_MTP_REQUIRED - observed_mtp))
        raise ValueError(f"partial MTP evidence; missing {missing}")
    if not features.mtp_rollout_enabled and observed_mtp:
        raise ValueError("MTP metrics were emitted for a run without MTP selected")

    rollout_seconds = normalized.get("train/rl/time/rollout_seconds")
    completion_tokens = normalized.get("train/rl/completion_tokens_total")
    if (
        "train/rl/rollout_tokens_per_second" not in normalized
        and rollout_seconds is not None
        and rollout_seconds > 0
        and completion_tokens is not None
    ):
        normalized["train/rl/rollout_tokens_per_second"] = completion_tokens / rollout_seconds

    return NormalizedGRPOStep(
        step=step,
        metrics=normalized,
        attributes={"training_backend": backend},
    )


def required_grpo_metrics(features: GRPOObservationFeatures) -> frozenset[str]:
    """Return the metrics that make the selected GRPO run research-readable."""

    required = set(_CORE_REQUIRED)
    if features.reference_kl_enabled:
        required.add("train/rl/kl")
    if features.clipping_enabled:
        required.add("train/rl/clip_fraction")
    if features.decoupled_rollout:
        required.update(
            {
                "train/rl/sampling_logp_delta_mean",
                "train/rl/sampling_logp_delta_max",
                "train/rl/importance_sampling_ratio_mean",
                "train/rl/importance_sampling_ratio_min",
                "train/rl/importance_sampling_ratio_max",
            }
        )
    if features.asynchronous_rollout:
        required.update(
            {
                "train/rl/policy_staleness_mean",
                "train/rl/policy_staleness_max",
                "train/rl/trajectory_version_span_mean",
            }
        )
    if features.mtp_rollout_enabled:
        required.update(_MTP_REQUIRED)
    if features.quantized_kv_cache:
        required.add("serve/backend/kv_cache_peak_usage_ratio")
    if features.tool_environment:
        required.update({"train/rl/tool_call_frequency", "train/rl/tool_failure_frequency"})
    return frozenset(required)


def assess_grpo_evidence(
    observed_metrics: Mapping[str, object] | set[str] | frozenset[str],
    features: GRPOObservationFeatures,
) -> GRPOEvidenceStatus:
    """Compare a run's observed names with its core and selected-feature obligations."""

    observed = set(observed_metrics)
    required = required_grpo_metrics(features)
    return GRPOEvidenceStatus(required=required, missing=frozenset(required - observed))


def _validate_metric_value(name: str, value: float) -> None:
    if name in _RATIO_METRICS and not 0 <= value <= 1:
        raise ValueError(f"GRPO ratio metric {name!r} must be between zero and one")
    if name in _NON_NEGATIVE_METRICS and value < 0:
        raise ValueError(f"GRPO metric {name!r} cannot be negative")


__all__ = [
    "GRPOEvidenceStatus",
    "GRPOObservationFeatures",
    "NormalizedGRPOStep",
    "assess_grpo_evidence",
    "normalize_grpo_metrics",
    "required_grpo_metrics",
]
