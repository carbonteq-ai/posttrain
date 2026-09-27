"""Versioned job telemetry definitions owned by Observatory."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Any, Literal

from pydantic import Field, model_validator

from .metric_catalog import metric_help
from .models import AlertSeverity, MetricHelp, ObservatoryModel

type Reducer = Literal["last", "min", "max", "mean", "sum"]
type HealthRuleKind = Literal["threshold", "non_finite"]
type ThresholdOperator = Literal["gt", "gte", "lt", "lte", "eq"]
GROUP_POLICY_JOB_KINDS = frozenset({"train.grpo", "train.gdpo", "train.capo"})
type EvidenceCondition = Literal[
    "validation_configured",
    "gradient_clipping_enabled",
    "source_scores_available",
    "distributed",
    "quantized_update",
    "packing_enabled",
    "reference_kl_enabled",
    "decoupled_rollout",
    "asynchronous_rollout",
    "mtp_rollout_enabled",
    "quantized_kv_cache",
    "tool_environment",
    "dapo_algorithm_enabled",
    "olmo3_algorithm_enabled",
]


class SummaryFieldDefinition(ObservatoryModel):
    key: str = Field(min_length=1)
    label: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    reducer: Reducer = "last"
    required: bool = False
    unit: str | None = None


class ChartDefinition(ObservatoryModel):
    key: str = Field(min_length=1)
    title: str = Field(min_length=1)
    question: str | None = Field(default=None, min_length=1)
    metrics: tuple[str, ...] = Field(min_length=1)


class HealthRuleDefinition(ObservatoryModel):
    id: str = Field(min_length=1)
    kind: HealthRuleKind
    message: str = Field(min_length=1)
    severity: AlertSeverity = "warning"
    metric: str = Field(min_length=1)
    operator: ThresholdOperator | None = None
    threshold: float | None = None

    @model_validator(mode="after")
    def validate_threshold(self) -> HealthRuleDefinition:
        if self.kind == "threshold" and (self.operator is None or self.threshold is None):
            raise ValueError("threshold health rules require an operator and threshold")
        if self.kind != "threshold" and (self.operator is not None or self.threshold is not None):
            raise ValueError("non-threshold health rules cannot define an operator or threshold")
        return self


class TraceSectionDefinition(ObservatoryModel):
    trace_type: str = Field(min_length=1)
    label: str = Field(min_length=1)


class ArtifactRoleDefinition(ObservatoryModel):
    kind: str = Field(min_length=1)
    label: str = Field(min_length=1)
    direction: Literal["input", "output"]


class EvidenceRequirementDefinition(ObservatoryModel):
    key: str = Field(min_length=1)
    label: str = Field(min_length=1)
    level: Literal["required", "conditional", "diagnostic"]
    metrics: tuple[str, ...] = Field(min_length=1)
    condition: EvidenceCondition | None = None
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_condition(self) -> EvidenceRequirementDefinition:
        if self.level == "conditional" and self.condition is None:
            raise ValueError("conditional evidence requirements need a condition")
        if self.level != "conditional" and self.condition is not None:
            raise ValueError("only conditional evidence requirements can declare a condition")
        return self


class JobTelemetryDefinition(ObservatoryModel):
    schema_version: int = Field(default=1, ge=1)
    job_kind: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    summary_fields: tuple[SummaryFieldDefinition, ...]
    charts: tuple[ChartDefinition, ...]
    metric_help: tuple[MetricHelp, ...]
    health_rules: tuple[HealthRuleDefinition, ...] = ()
    comparison_keys: tuple[str, ...]
    trace_sections: tuple[TraceSectionDefinition, ...] = ()
    artifact_roles: tuple[ArtifactRoleDefinition, ...] = ()
    delta_tip_metrics: tuple[str, ...] = ()
    projection_metrics: tuple[str, ...] = ()
    evidence_requirements: tuple[EvidenceRequirementDefinition, ...] = ()

    @model_validator(mode="after")
    def validate_references(self) -> JobTelemetryDefinition:
        summary_keys = [field.key for field in self.summary_fields]
        chart_keys = [chart.key for chart in self.charts]
        rule_ids = [rule.id for rule in self.health_rules]
        requirement_keys = [requirement.key for requirement in self.evidence_requirements]
        help_metrics = [item.metric for item in self.metric_help]
        if len(summary_keys) != len(set(summary_keys)):
            raise ValueError("job telemetry summary keys must be unique")
        if len(chart_keys) != len(set(chart_keys)):
            raise ValueError("job telemetry chart keys must be unique")
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("job telemetry health rule ids must be unique")
        if len(requirement_keys) != len(set(requirement_keys)):
            raise ValueError("job telemetry evidence requirement keys must be unique")
        if len(help_metrics) != len(set(help_metrics)):
            raise ValueError("job telemetry metric help entries must be unique")
        unknown = set(self.comparison_keys) - set(summary_keys)
        if unknown:
            raise ValueError(f"comparison keys are not summary fields: {sorted(unknown)}")
        unknown_tips = set(self.delta_tip_metrics) - self.metric_names
        if unknown_tips:
            raise ValueError(f"delta tip metrics are not otherwise declared: {sorted(unknown_tips)}")
        missing_help = self.metric_names - set(help_metrics)
        extra_help = set(help_metrics) - self.metric_names
        if missing_help:
            raise ValueError(f"job telemetry metrics are missing help: {sorted(missing_help)}")
        if extra_help:
            raise ValueError(f"metric help is not otherwise declared: {sorted(extra_help)}")
        return self

    @property
    def metric_names(self) -> set[str]:
        return {
            *(field.metric for field in self.summary_fields),
            *(metric for chart in self.charts for metric in chart.metrics),
            *(rule.metric for rule in self.health_rules),
            *(metric for requirement in self.evidence_requirements for metric in requirement.metrics),
            *self.projection_metrics,
        }


def _training_artifacts() -> tuple[ArtifactRoleDefinition, ...]:
    return (
        ArtifactRoleDefinition(kind="model-adapter", label="Trained adapter", direction="output"),
        ArtifactRoleDefinition(kind="model-weights", label="Trained weights", direction="output"),
        ArtifactRoleDefinition(kind="training-summary", label="Native training summary", direction="output"),
    )


def _help_for(*metrics: str) -> tuple[MetricHelp, ...]:
    return metric_help(*metrics)


SFT_TELEMETRY = JobTelemetryDefinition(
    job_kind="train.sft",
    display_name="Supervised fine-tuning",
    summary_fields=(
        SummaryFieldDefinition(key="final_loss", label="Latest loss", metric="train/loss", required=True),
        SummaryFieldDefinition(
            key="validation_loss",
            label="Validation loss",
            metric="train/validation/loss",
        ),
        SummaryFieldDefinition(
            key="token_accuracy", label="Token accuracy", metric="train/mean_token_accuracy", unit="ratio"
        ),
        SummaryFieldDefinition(key="grad_norm", label="Gradient norm", metric="train/grad_norm"),
        SummaryFieldDefinition(
            key="tokens_per_second",
            label="Non-padding tokens / second",
            metric="train/non_padding_tokens_per_second",
            unit="tokens/s",
        ),
        SummaryFieldDefinition(
            key="step_time",
            label="Step time",
            metric="train/step_time_seconds",
            unit="s",
        ),
        SummaryFieldDefinition(
            key="supervision_ratio",
            label="Supervision-token ratio",
            metric="train/data/supervision_token_ratio",
            unit="ratio",
        ),
        SummaryFieldDefinition(
            key="truncation_rate",
            label="Truncated examples",
            metric="train/data/truncation_rate",
            unit="ratio",
        ),
        SummaryFieldDefinition(
            key="max_length_utilization",
            label="Max-length utilization",
            metric="train/data/max_length_utilization",
            unit="ratio",
        ),
    ),
    charts=(
        ChartDefinition(
            key="learning",
            title="Learning",
            metrics=("train/loss", "train/validation/loss", "train/mean_token_accuracy"),
        ),
        ChartDefinition(
            key="stability",
            title="Stability",
            metrics=("train/grad_norm", "train/learning_rate", "train/gradient_clipped"),
        ),
        ChartDefinition(
            key="efficiency",
            title="Efficiency",
            metrics=("train/non_padding_tokens_per_second", "train/step_time_seconds"),
        ),
    ),
    metric_help=_help_for(
        "train/loss",
        "train/validation/loss",
        "train/mean_token_accuracy",
        "train/grad_norm",
        "train/learning_rate",
        "train/gradient_clipped",
        "train/non_padding_tokens_per_second",
        "train/step_time_seconds",
        "train/data/supervision_token_ratio",
        "train/data/truncation_rate",
        "train/data/max_length_utilization",
    ),
    health_rules=(
        HealthRuleDefinition(
            id="sft-loss-non-finite",
            kind="non_finite",
            metric="train/loss",
            message="Training loss contains a non-finite value.",
            severity="error",
        ),
        HealthRuleDefinition(
            id="sft-validation-loss-non-finite",
            kind="non_finite",
            metric="train/validation/loss",
            message="Validation loss contains a non-finite value.",
            severity="error",
        ),
        HealthRuleDefinition(
            id="sft-truncation-high",
            kind="threshold",
            metric="train/data/truncation_rate",
            operator="gt",
            threshold=0.1,
            message="More than 10% of rendered training examples are truncated.",
        ),
    ),
    comparison_keys=(
        "final_loss",
        "validation_loss",
        "token_accuracy",
        "tokens_per_second",
        "truncation_rate",
    ),
    artifact_roles=_training_artifacts(),
    delta_tip_metrics=(
        "train/loss",
        "train/validation/loss",
        "train/grad_norm",
        "train/non_padding_tokens_per_second",
        "train/step_time_seconds",
    ),
)

DPO_TELEMETRY = JobTelemetryDefinition(
    job_kind="train.dpo",
    display_name="Direct preference optimization",
    summary_fields=(
        SummaryFieldDefinition(
            key="reward_margin", label="Reward margin", metric="train/rewards/margins", required=True
        ),
        SummaryFieldDefinition(
            key="preference_accuracy", label="Pair ordering accuracy", metric="train/rewards/accuracies", unit="ratio"
        ),
        SummaryFieldDefinition(key="chosen_reward", label="Chosen reward", metric="train/rewards/chosen"),
        SummaryFieldDefinition(key="rejected_reward", label="Rejected reward", metric="train/rewards/rejected"),
        SummaryFieldDefinition(key="final_loss", label="DPO loss", metric="train/loss", required=True),
        SummaryFieldDefinition(key="chosen_logp", label="Chosen log probability", metric="train/logps/chosen"),
        SummaryFieldDefinition(key="rejected_logp", label="Rejected log probability", metric="train/logps/rejected"),
        SummaryFieldDefinition(key="entropy", label="Token entropy", metric="train/entropy"),
        SummaryFieldDefinition(
            key="token_accuracy", label="Chosen-token accuracy", metric="train/mean_token_accuracy", unit="ratio"
        ),
        SummaryFieldDefinition(key="grad_norm", label="Gradient norm", metric="train/grad_norm"),
        SummaryFieldDefinition(
            key="tokens_per_second",
            label="Non-padding tokens / second",
            metric="train/non_padding_tokens_per_second",
            unit="tokens/s",
        ),
        SummaryFieldDefinition(key="step_time", label="Step time", metric="train/step_time_seconds", unit="s"),
        SummaryFieldDefinition(key="preference_pairs", label="Preference pairs", metric="train/data/preference_pairs"),
        SummaryFieldDefinition(
            key="prompt_tokens", label="Mean prompt tokens", metric="train/data/prompt_tokens_mean", unit="tokens"
        ),
        SummaryFieldDefinition(
            key="chosen_tokens", label="Mean chosen tokens", metric="train/data/chosen_tokens_mean", unit="tokens"
        ),
        SummaryFieldDefinition(
            key="rejected_tokens", label="Mean rejected tokens", metric="train/data/rejected_tokens_mean", unit="tokens"
        ),
        SummaryFieldDefinition(
            key="prompt_tokens_p95", label="Prompt tokens p95", metric="train/data/prompt_tokens_p95", unit="tokens"
        ),
        SummaryFieldDefinition(
            key="chosen_tokens_p95", label="Chosen tokens p95", metric="train/data/chosen_tokens_p95", unit="tokens"
        ),
        SummaryFieldDefinition(
            key="rejected_tokens_p95",
            label="Rejected tokens p95",
            metric="train/data/rejected_tokens_p95",
            unit="tokens",
        ),
        SummaryFieldDefinition(
            key="max_length_headroom",
            label="Minimum length headroom",
            metric="train/data/max_length_headroom_min",
            unit="tokens",
        ),
        SummaryFieldDefinition(
            key="chosen_longer_fraction",
            label="Chosen longer than rejected",
            metric="train/data/chosen_longer_fraction",
            unit="ratio",
        ),
        SummaryFieldDefinition(
            key="score_coverage",
            label="Source score coverage",
            metric="train/data/preference_score_coverage",
            unit="ratio",
        ),
        SummaryFieldDefinition(
            key="score_margin", label="Mean source score margin", metric="train/data/preference_score_margin_mean"
        ),
        SummaryFieldDefinition(
            key="max_length_utilization",
            label="Max-length utilization",
            metric="train/data/max_length_utilization",
            unit="ratio",
        ),
    ),
    charts=(
        ChartDefinition(
            key="preferences",
            title="Pair ordering",
            question="Is the policy consistently ranking chosen completions above rejected ones?",
            metrics=(
                "train/rewards/margins",
                "train/rewards/accuracies",
                "train/rewards/chosen",
                "train/rewards/rejected",
            ),
        ),
        ChartDefinition(
            key="policy",
            title="Policy movement",
            question="Is separation coming from stronger chosen likelihood, weaker rejected likelihood, or both?",
            metrics=("train/logps/chosen", "train/logps/rejected", "train/mean_token_accuracy"),
        ),
        ChartDefinition(
            key="objective",
            title="Objective",
            question="Is the preference objective converging without a sharp loss of policy uncertainty?",
            metrics=("train/loss", "train/entropy"),
        ),
        ChartDefinition(
            key="stability",
            title="Stability",
            question="Are update magnitudes controlled by the learning-rate schedule and clipping threshold?",
            metrics=("train/grad_norm", "train/learning_rate", "train/gradient_clipped"),
        ),
        ChartDefinition(
            key="efficiency",
            title="Efficiency",
            question="Is effective token throughput stable as preference optimization proceeds?",
            metrics=("train/non_padding_tokens_per_second", "train/step_time_seconds"),
        ),
    ),
    metric_help=_help_for(
        "train/rewards/margins",
        "train/rewards/accuracies",
        "train/rewards/chosen",
        "train/rewards/rejected",
        "train/loss",
        "train/logps/chosen",
        "train/logps/rejected",
        "train/entropy",
        "train/mean_token_accuracy",
        "train/grad_norm",
        "train/learning_rate",
        "train/gradient_clipped",
        "train/non_padding_tokens_per_second",
        "train/step_time_seconds",
        "train/num_tokens",
        "train/data/preference_pairs",
        "train/data/prompt_tokens_mean",
        "train/data/chosen_tokens_mean",
        "train/data/rejected_tokens_mean",
        "train/data/prompt_tokens_p95",
        "train/data/chosen_tokens_p95",
        "train/data/rejected_tokens_p95",
        "train/data/max_length_headroom_min",
        "train/data/chosen_longer_fraction",
        "train/data/preference_score_coverage",
        "train/data/preference_score_margin_mean",
        "train/data/max_length_utilization",
        "train/validation/loss",
        "train/validation/rewards/margins",
        "train/validation/rewards/accuracies",
        "train/validation/rewards/chosen",
        "train/validation/rewards/rejected",
        "train/validation/logps/chosen",
        "train/validation/logps/rejected",
        "train/logits/chosen",
        "train/logits/rejected",
    ),
    health_rules=(
        HealthRuleDefinition(
            id="dpo-loss-non-finite",
            kind="non_finite",
            metric="train/loss",
            message="DPO loss contains a non-finite value.",
            severity="error",
        ),
        HealthRuleDefinition(
            id="dpo-negative-margin",
            kind="threshold",
            metric="train/rewards/margins",
            operator="lt",
            threshold=0.0,
            message="The latest preference reward margin is negative.",
        ),
        HealthRuleDefinition(
            id="dpo-grad-non-finite",
            kind="non_finite",
            metric="train/grad_norm",
            message="DPO gradient norm contains a non-finite value.",
            severity="error",
        ),
        HealthRuleDefinition(
            id="dpo-clipping-active",
            kind="threshold",
            metric="train/gradient_clipped",
            operator="gt",
            threshold=0.5,
            message="The latest DPO update exceeded the gradient clipping threshold.",
        ),
    ),
    comparison_keys=("reward_margin", "preference_accuracy", "chosen_reward", "chosen_logp", "final_loss", "grad_norm"),
    artifact_roles=_training_artifacts(),
    delta_tip_metrics=(
        "train/rewards/margins",
        "train/rewards/accuracies",
        "train/logps/chosen",
        "train/logps/rejected",
        "train/loss",
        "train/grad_norm",
    ),
    evidence_requirements=(
        EvidenceRequirementDefinition(
            key="objective",
            label="Preference objective",
            level="required",
            metrics=("train/loss", "train/rewards/margins", "train/rewards/accuracies"),
            reason="Proves that the DPO objective and pair ordering were measured during optimization.",
        ),
        EvidenceRequirementDefinition(
            key="implicit_rewards",
            label="Implicit reward decomposition",
            level="required",
            metrics=("train/rewards/chosen", "train/rewards/rejected"),
            reason="Explains whether separation comes from chosen improvement, rejected suppression, or both.",
        ),
        EvidenceRequirementDefinition(
            key="policy_movement",
            label="Policy movement",
            level="required",
            metrics=("train/logps/chosen", "train/logps/rejected"),
            reason="Provides direct policy-likelihood evidence rather than relying on the reward margin alone.",
        ),
        EvidenceRequirementDefinition(
            key="optimization",
            label="Optimization stability",
            level="required",
            metrics=("train/grad_norm", "train/learning_rate"),
            reason="Makes unstable or stalled parameter updates diagnosable.",
        ),
        EvidenceRequirementDefinition(
            key="rendered_pairs",
            label="Rendered pair population",
            level="required",
            metrics=(
                "train/data/preference_pairs",
                "train/data/prompt_tokens_mean",
                "train/data/chosen_tokens_mean",
                "train/data/rejected_tokens_mean",
                "train/data/prompt_tokens_p95",
                "train/data/chosen_tokens_p95",
                "train/data/rejected_tokens_p95",
                "train/data/max_length_headroom_min",
                "train/data/max_length_utilization",
                "train/data/preference_score_coverage",
            ),
            reason="Records the exact rendered population and length distribution used by the optimizer.",
        ),
        EvidenceRequirementDefinition(
            key="runtime",
            label="Effective runtime",
            level="required",
            metrics=("train/num_tokens", "train/non_padding_tokens_per_second", "train/step_time_seconds"),
            reason="Provides progress, effective throughput, and step-time evidence using logical training units.",
        ),
        EvidenceRequirementDefinition(
            key="gradient_clipping",
            label="Gradient clipping",
            level="conditional",
            condition="gradient_clipping_enabled",
            metrics=("train/gradient_clipped",),
            reason="Required when the optimizer selects a clipping threshold.",
        ),
        EvidenceRequirementDefinition(
            key="held_out_preferences",
            label="Held-out preference validation",
            level="conditional",
            condition="validation_configured",
            metrics=(
                "train/validation/loss",
                "train/validation/rewards/margins",
                "train/validation/rewards/accuracies",
                "train/validation/rewards/chosen",
                "train/validation/rewards/rejected",
                "train/validation/logps/chosen",
                "train/validation/logps/rejected",
            ),
            reason="Required when a validation selection is present and necessary for research-readiness.",
        ),
        EvidenceRequirementDefinition(
            key="source_scores",
            label="Source preference strength",
            level="conditional",
            condition="source_scores_available",
            metrics=("train/data/preference_score_margin_mean",),
            reason="Required when source scores exist so ambiguous and strong preferences can be distinguished.",
        ),
        EvidenceRequirementDefinition(
            key="policy_uncertainty",
            label="Policy uncertainty",
            level="diagnostic",
            metrics=("train/entropy",),
            reason="Useful for collapse diagnosis but potentially incompatible with fused loss paths that avoid materializing full logits.",
        ),
        EvidenceRequirementDefinition(
            key="token_prediction",
            label="Chosen-token prediction",
            level="diagnostic",
            metrics=("train/mean_token_accuracy",),
            reason="A secondary language-modeling diagnostic rather than direct preference evidence.",
        ),
        EvidenceRequirementDefinition(
            key="raw_logits",
            label="Raw token scores",
            level="diagnostic",
            metrics=("train/logits/chosen", "train/logits/rejected"),
            reason="Low-level backend diagnostics that are less comparable than normalized log-probabilities.",
        ),
    ),
)

GRPO_TELEMETRY = JobTelemetryDefinition(
    schema_version=3,
    job_kind="train.grpo",
    display_name="Group relative policy optimization",
    summary_fields=(
        SummaryFieldDefinition(key="reward_mean", label="Mean reward", metric="train/rl/reward_mean", required=True),
        SummaryFieldDefinition(key="reward_std", label="Reward standard deviation", metric="train/rl/reward_std"),
        SummaryFieldDefinition(
            key="zero_variance",
            label="Zero-variance groups",
            metric="train/rl/group_zero_variance_fraction",
            unit="ratio",
        ),
        SummaryFieldDefinition(key="policy_loss", label="Policy loss", metric="train/rl/policy_loss", required=True),
        SummaryFieldDefinition(key="entropy", label="Policy entropy", metric="train/rl/entropy"),
        SummaryFieldDefinition(key="grad_norm", label="Gradient norm", metric="train/grad_norm"),
        SummaryFieldDefinition(
            key="truncation_rate",
            label="Completion truncation rate",
            metric="train/rl/completion_truncation_rate",
            unit="ratio",
        ),
        SummaryFieldDefinition(
            key="importance_ratio_mean",
            label="Mean importance ratio",
            metric="train/rl/importance_sampling_ratio_mean",
        ),
        SummaryFieldDefinition(
            key="rollout_tps", label="Rollout throughput", metric="train/rl/rollout_tokens_per_second", unit="tokens/s"
        ),
        SummaryFieldDefinition(
            key="clip_fraction", label="Clip fraction", metric="train/rl/clip_fraction", unit="ratio"
        ),
        SummaryFieldDefinition(
            key="clip_fraction_low", label="Lower clip fraction", metric="train/rl/clip_fraction_low", unit="ratio"
        ),
        SummaryFieldDefinition(
            key="clip_fraction_high", label="Upper clip fraction", metric="train/rl/clip_fraction_high", unit="ratio"
        ),
        SummaryFieldDefinition(
            key="dynamic_candidate_batches",
            label="Candidate batches",
            metric="train/rl/dynamic_sampling_candidate_batches",
        ),
        SummaryFieldDefinition(
            key="dynamic_retained_fraction",
            label="Retained fraction",
            metric="train/rl/dynamic_sampling_retained_fraction",
            unit="ratio",
        ),
        SummaryFieldDefinition(
            key="active_retained_fraction",
            label="Active-sampling retained fraction",
            metric="train/rl/active_sampling_retained_fraction",
            unit="ratio",
        ),
        SummaryFieldDefinition(
            key="failed_rollouts", label="Failed rollouts", metric="train/rl/rollouts_failed", reducer="sum"
        ),
    ),
    charts=(
        ChartDefinition(
            key="optimization",
            title="Policy optimization",
            question="Does reward improve while groups retain relative signal and policy updates remain controlled?",
            metrics=(
                "train/rl/reward_mean",
                "train/rl/reward_std",
                "train/rl/group_zero_variance_fraction",
                "train/rl/policy_loss",
                "train/rl/entropy",
                "train/rl/kl",
                "train/rl/clip_fraction",
                "train/rl/clip_fraction_low",
                "train/rl/clip_fraction_high",
            ),
        ),
        ChartDefinition(
            key="stability",
            title="Update stability",
            question="Are gradient scale and learning rate behaving as configured?",
            metrics=("train/grad_norm", "train/learning_rate"),
        ),
        ChartDefinition(
            key="rollouts",
            title="Rollout population",
            question="How much requested evidence completed, failed, truncated, or became unscorable?",
            metrics=(
                "train/rl/rollouts_requested",
                "train/rl/rollouts_attempted",
                "train/rl/rollouts_completed",
                "train/rl/rollouts_failed",
                "train/rl/rollouts_truncated",
                "train/rl/rollouts_unscorable",
                "train/rl/rollouts_missing",
            ),
        ),
        ChartDefinition(
            key="efficiency",
            title="Runtime efficiency",
            question="Where does step time go, and what effective rollout throughput results?",
            metrics=(
                "train/rl/rollout_tokens_per_second",
                "train/step_time_seconds",
                "train/rl/time/rollout_seconds",
                "train/rl/time/reward_seconds",
                "train/rl/time/actor_update_seconds",
                "train/rl/time/weight_sync_seconds",
            ),
        ),
        ChartDefinition(
            key="freshness",
            title="Policy freshness",
            question="Do decoupled or asynchronous rollouts still represent the policy being optimized?",
            metrics=(
                "train/rl/sampling_logp_delta_mean",
                "train/rl/sampling_logp_delta_max",
                "train/rl/importance_sampling_ratio_mean",
                "train/rl/policy_staleness_mean",
                "train/rl/policy_staleness_max",
            ),
        ),
        ChartDefinition(
            key="acceleration",
            title="Rollout acceleration",
            question="Did the selected MTP and quantized-cache paths produce complete runtime evidence?",
            metrics=(
                "serve/backend/speculative_acceptance_rate",
                "serve/backend/speculative_accepted_length",
                "serve/backend/kv_cache_peak_usage_ratio",
            ),
        ),
        # Sampling modes are conditional on the selected algorithm. Keep them
        # after the core policy-health evidence so they do not interrupt the
        # common GRPO reading order.
        ChartDefinition(
            key="dynamic_sampling",
            title="Dynamic sampling",
            question="How much rollout population is discarded before the optimizer receives a usable group signal?",
            metrics=(
                "train/rl/dynamic_sampling_candidate_batches",
                "train/rl/dynamic_sampling_retained_fraction",
            ),
        ),
        ChartDefinition(
            key="active_sampling_yield",
            title="Active sampling yield",
            question="How many generation rounds were needed, and what share of generated candidates supplied usable reward variation?",
            metrics=(
                "train/rl/active_sampling_generation_rounds",
                "train/rl/active_sampling_retained_fraction",
            ),
        ),
        ChartDefinition(
            key="active_sampling_population",
            title="Active sampling population",
            question="How did the bounded candidate window divide into generated, retained, and unused rows?",
            metrics=(
                "train/rl/active_sampling_candidate_groups_reserved",
                "train/rl/active_sampling_candidate_groups_generated",
                "train/rl/active_sampling_candidate_groups_retained",
                "train/rl/active_sampling_candidate_groups_unused",
            ),
        ),
        ChartDefinition(
            key="tool_behavior",
            title="Tool behavior",
            question="Are multi-turn trajectories invoking tools successfully?",
            metrics=("train/rl/tool_call_frequency", "train/rl/tool_failure_frequency"),
        ),
    ),
    metric_help=_help_for(
        "train/rl/reward_mean",
        "train/rl/reward_std",
        "train/rl/group_zero_variance_fraction",
        "train/rl/policy_loss",
        "train/rl/kl",
        "train/rl/entropy",
        "train/rl/clip_fraction",
        "train/rl/clip_fraction_low",
        "train/rl/clip_fraction_high",
        "train/rl/dynamic_sampling_candidate_batches",
        "train/rl/dynamic_sampling_retained_fraction",
        "train/rl/active_sampling_generation_rounds",
        "train/rl/active_sampling_retained_fraction",
        "train/rl/active_sampling_generated_rows",
        "train/rl/active_sampling_candidate_groups_reserved",
        "train/rl/active_sampling_candidate_groups_generated",
        "train/rl/active_sampling_candidate_groups_retained",
        "train/rl/active_sampling_candidate_groups_unused",
        "train/rl/curriculum/candidate_groups",
        "train/rl/curriculum/unique_tasks",
        "train/rl/curriculum/new_tasks",
        "train/rl/curriculum/discovery_reserved",
        "train/rl/curriculum/discovery_fulfilled",
        "train/rl/curriculum/duplicate_fallbacks",
        "train/rl/curriculum/refill_round",
        "train/rl/curriculum/class_candidate_groups",
        "train/grad_norm",
        "train/learning_rate",
        "train/step_time_seconds",
        "train/rl/rollouts_requested",
        "train/rl/rollouts_attempted",
        "train/rl/rollouts_completed",
        "train/rl/rollouts_failed",
        "train/rl/rollouts_truncated",
        "train/rl/rollouts_unscorable",
        "train/rl/rollouts_missing",
        "train/rl/completion_tokens_mean",
        "train/rl/completion_tokens_max",
        "train/rl/completion_truncation_rate",
        "train/rl/rollout_tokens_per_second",
        "train/rl/sampling_logp_delta_mean",
        "train/rl/sampling_logp_delta_max",
        "train/rl/importance_sampling_ratio_mean",
        "train/rl/importance_sampling_ratio_min",
        "train/rl/importance_sampling_ratio_max",
        "train/rl/policy_staleness_mean",
        "train/rl/policy_staleness_max",
        "train/rl/trajectory_version_span_mean",
        "train/rl/tool_call_frequency",
        "train/rl/tool_failure_frequency",
        "train/rl/time/rollout_seconds",
        "train/rl/time/reward_seconds",
        "train/rl/time/actor_forward_seconds",
        "train/rl/time/actor_update_seconds",
        "train/rl/time/weight_sync_seconds",
        "train/rl/time/checkpoint_seconds",
        "serve/backend/speculative_draft_tokens",
        "serve/backend/speculative_accepted_tokens",
        "serve/backend/speculative_acceptance_rate",
        "serve/backend/speculative_accepted_length",
        "serve/backend/kv_cache_capacity_tokens",
        "serve/backend/kv_cache_peak_usage_ratio",
    ),
    projection_metrics=(
        "train/rl/curriculum/candidate_groups",
        "train/rl/curriculum/unique_tasks",
        "train/rl/curriculum/new_tasks",
        "train/rl/curriculum/discovery_reserved",
        "train/rl/curriculum/discovery_fulfilled",
        "train/rl/curriculum/duplicate_fallbacks",
        "train/rl/curriculum/refill_round",
        "train/rl/curriculum/class_candidate_groups",
    ),
    health_rules=(
        HealthRuleDefinition(
            id="grpo-reward-non-finite",
            kind="non_finite",
            metric="train/rl/reward_mean",
            message="GRPO reward contains a non-finite value.",
            severity="error",
        ),
        HealthRuleDefinition(
            id="grpo-policy-loss-non-finite",
            kind="non_finite",
            metric="train/rl/policy_loss",
            message="GRPO policy loss contains a non-finite value.",
            severity="error",
        ),
        HealthRuleDefinition(
            id="grpo-gradient-non-finite",
            kind="non_finite",
            metric="train/grad_norm",
            message="GRPO gradient norm contains a non-finite value.",
            severity="error",
        ),
        HealthRuleDefinition(
            id="grpo-zero-variance-groups",
            kind="threshold",
            metric="train/rl/group_zero_variance_fraction",
            operator="gte",
            threshold=1.0,
            message="Every observed rollout group has zero reward variance.",
            severity="error",
        ),
        HealthRuleDefinition(
            id="grpo-rollout-failures",
            kind="threshold",
            metric="train/rl/rollouts_failed",
            operator="gt",
            threshold=0,
            message="One or more rollout attempts failed.",
            severity="error",
        ),
        HealthRuleDefinition(
            id="grpo-unscorable-rollouts",
            kind="threshold",
            metric="train/rl/rollouts_unscorable",
            operator="gt",
            threshold=0,
            message="One or more rollouts did not produce a finite reward.",
            severity="error",
        ),
        HealthRuleDefinition(
            id="grpo-truncated-rollouts",
            kind="threshold",
            metric="train/rl/rollouts_truncated",
            operator="gt",
            threshold=0,
            message="One or more rollouts were truncated.",
            severity="warning",
        ),
        HealthRuleDefinition(
            id="grpo-high-truncation-rate",
            kind="threshold",
            metric="train/rl/completion_truncation_rate",
            operator="gt",
            threshold=0.05,
            message="Completion truncation exceeds the approximate five-percent promotion guard.",
            severity="warning",
        ),
    ),
    comparison_keys=(
        "reward_mean",
        "policy_loss",
        "entropy",
        "truncation_rate",
        "importance_ratio_mean",
        "rollout_tps",
        "clip_fraction_low",
        "clip_fraction_high",
        "dynamic_candidate_batches",
        "dynamic_retained_fraction",
        "active_retained_fraction",
        "failed_rollouts",
    ),
    trace_sections=(TraceSectionDefinition(trace_type="verifiers", label="Rollouts & rewards"),),
    artifact_roles=_training_artifacts(),
    delta_tip_metrics=(
        "train/rl/reward_mean",
        "train/rl/policy_loss",
        "train/rl/rollout_tokens_per_second",
        "train/rl/clip_fraction_low",
        "train/rl/clip_fraction_high",
        "train/rl/dynamic_sampling_candidate_batches",
        "train/rl/dynamic_sampling_retained_fraction",
        "train/rl/active_sampling_generation_rounds",
        "train/rl/active_sampling_retained_fraction",
        "serve/backend/speculative_acceptance_rate",
    ),
    evidence_requirements=(
        EvidenceRequirementDefinition(
            key="learning_signal",
            label="Relative learning signal",
            level="required",
            metrics=("train/rl/reward_mean", "train/rl/reward_std", "train/rl/group_zero_variance_fraction"),
            reason="Reward level and within-group variation are both required to interpret GRPO learning.",
        ),
        EvidenceRequirementDefinition(
            key="controlled_update",
            label="Controlled policy update",
            level="required",
            metrics=(
                "train/rl/policy_loss",
                "train/rl/entropy",
                "train/rl/clip_fraction",
                "train/grad_norm",
                "train/learning_rate",
            ),
            reason="The policy objective must be paired with exploration, clipping, and gradient-scale evidence.",
        ),
        EvidenceRequirementDefinition(
            key="dapo_sampling",
            label="DAPO dynamic sampling",
            level="conditional",
            condition="dapo_algorithm_enabled",
            metrics=(
                "train/rl/dynamic_sampling_candidate_batches",
                "train/rl/dynamic_sampling_retained_fraction",
            ),
            reason="DAPO must expose candidate-batch usage and retained population so sampler exhaustion is auditable.",
        ),
        EvidenceRequirementDefinition(
            key="dapo_asymmetric_clipping",
            label="DAPO asymmetric clipping",
            level="conditional",
            condition="dapo_algorithm_enabled",
            metrics=("train/rl/clip_fraction_low", "train/rl/clip_fraction_high"),
            reason="DAPO must expose lower- and upper-side clipping separately; one combined clip fraction is insufficient.",
        ),
        EvidenceRequirementDefinition(
            key="olmo3_active_sampling",
            label="OLMo3 active sampling",
            level="conditional",
            condition="olmo3_algorithm_enabled",
            metrics=(
                "train/rl/active_sampling_generation_rounds",
                "train/rl/active_sampling_retained_fraction",
                "train/rl/active_sampling_generated_rows",
                "train/rl/active_sampling_candidate_groups_reserved",
                "train/rl/active_sampling_candidate_groups_generated",
                "train/rl/active_sampling_candidate_groups_retained",
                "train/rl/active_sampling_candidate_groups_unused",
            ),
            reason="OLMo3 active sampling must expose bounded candidate generation and retained-population accounting so rollout cost and learning population remain distinguishable.",
        ),
        EvidenceRequirementDefinition(
            key="rollout_population",
            label="Rollout population",
            level="required",
            metrics=(
                "train/rl/rollouts_attempted",
                "train/rl/rollouts_completed",
                "train/rl/rollouts_failed",
                "train/rl/rollouts_truncated",
                "train/rl/rollouts_unscorable",
            ),
            reason="Every update needs an auditable population denominator and terminal outcomes.",
        ),
        EvidenceRequirementDefinition(
            key="completion_shape",
            label="Completion shape",
            level="required",
            metrics=(
                "train/rl/completion_tokens_mean",
                "train/rl/completion_tokens_max",
                "train/rl/completion_truncation_rate",
            ),
            reason="Length and truncation reveal whether the generation limit is shaping the observed reward.",
        ),
        EvidenceRequirementDefinition(
            key="runtime_efficiency",
            label="Runtime efficiency",
            level="required",
            metrics=("train/step_time_seconds", "train/rl/rollout_tokens_per_second"),
            reason="End-to-end step cost and effective rollout throughput are required for a useful runtime comparison.",
        ),
        EvidenceRequirementDefinition(
            key="reference_policy",
            label="Reference-policy drift",
            level="conditional",
            condition="reference_kl_enabled",
            metrics=("train/rl/kl",),
            reason="KL evidence is owed whenever a non-zero reference penalty is selected.",
        ),
        EvidenceRequirementDefinition(
            key="policy_freshness",
            label="Rollout-policy correction",
            level="conditional",
            condition="decoupled_rollout",
            metrics=(
                "train/rl/sampling_logp_delta_mean",
                "train/rl/sampling_logp_delta_max",
                "train/rl/importance_sampling_ratio_mean",
                "train/rl/importance_sampling_ratio_min",
                "train/rl/importance_sampling_ratio_max",
            ),
            reason="A decoupled rollout server must expose how its sampling probabilities differ from the actor update.",
        ),
        EvidenceRequirementDefinition(
            key="asynchronous_freshness",
            label="Asynchronous policy freshness",
            level="conditional",
            condition="asynchronous_rollout",
            metrics=(
                "train/rl/policy_staleness_mean",
                "train/rl/policy_staleness_max",
                "train/rl/trajectory_version_span_mean",
            ),
            reason="Asynchronous sampling owes explicit policy-version staleness evidence.",
        ),
        EvidenceRequirementDefinition(
            key="mtp_runtime",
            label="MTP runtime evidence",
            level="conditional",
            condition="mtp_rollout_enabled",
            metrics=(
                "serve/backend/speculative_draft_tokens",
                "serve/backend/speculative_accepted_tokens",
                "serve/backend/speculative_acceptance_rate",
                "serve/backend/speculative_accepted_length",
            ),
            reason="Selecting MTP does not prove acceleration; drafts and acceptances must come from runtime counters.",
        ),
        EvidenceRequirementDefinition(
            key="quantized_kv_runtime",
            label="Quantized KV-cache evidence",
            level="conditional",
            condition="quantized_kv_cache",
            metrics=("serve/backend/kv_cache_peak_usage_ratio",),
            reason="Selecting TurboQuant does not prove usable capacity or headroom; runtime usage must be observed.",
        ),
        EvidenceRequirementDefinition(
            key="tool_behavior",
            label="Tool-use behavior",
            level="conditional",
            condition="tool_environment",
            metrics=("train/rl/tool_call_frequency", "train/rl/tool_failure_frequency"),
            reason="Tool environments owe invocation and failure coverage in addition to reward.",
        ),
        EvidenceRequirementDefinition(
            key="phase_timing",
            label="Phase timing",
            level="diagnostic",
            metrics=(
                "train/rl/time/rollout_seconds",
                "train/rl/time/reward_seconds",
                "train/rl/time/actor_forward_seconds",
                "train/rl/time/actor_update_seconds",
                "train/rl/time/weight_sync_seconds",
                "train/rl/time/checkpoint_seconds",
            ),
            reason="Phase attribution explains end-to-end step time but is not required for the minimum learning view.",
        ),
        EvidenceRequirementDefinition(
            key="kv_capacity",
            label="KV-cache capacity",
            level="diagnostic",
            metrics=("serve/backend/kv_cache_capacity_tokens",),
            reason="Capacity complements peak usage when the serving runtime exposes it directly.",
        ),
    ),
)


def _group_policy_telemetry_variant(
    source: JobTelemetryDefinition,
    *,
    job_kind: str,
    display_name: str,
    acronym: str,
) -> JobTelemetryDefinition:
    """Reuse the common group-policy evidence contract without GRPO labels."""
    source_acronym = "GRPO"
    return source.model_copy(
        update={
            "job_kind": job_kind,
            "display_name": display_name,
            "health_rules": tuple(
                rule.model_copy(
                    update={
                        "id": rule.id.replace("grpo-", f"{acronym.lower()}-", 1),
                        "message": rule.message.replace(source_acronym, acronym),
                    }
                )
                for rule in source.health_rules
            ),
        }
    )


GDPO_TELEMETRY = _group_policy_telemetry_variant(
    GRPO_TELEMETRY,
    job_kind="train.gdpo",
    display_name="GDPO policy optimization",
    acronym="GDPO",
)

CAPO_TELEMETRY = _group_policy_telemetry_variant(
    GRPO_TELEMETRY,
    job_kind="train.capo",
    display_name="CAPO policy optimization",
    acronym="CAPO",
)

# Advantages are centred within their groups, so SAMPO's plain advantage means
# are zero by construction. The credit tab shows magnitudes and shares instead.
_SAMPO_CREDIT_CHART = ChartDefinition(
    key="hierarchical_credit",
    title="Hierarchical credit",
    question="How much of each turn's credit is its own, and did turns have a comparable attempt to be judged against?",
    metrics=(
        "train/rl/turn_credit_share",
        "train/rl/turn_advantage_informative_fraction",
        "train/rl/singleton_anchor_fraction",
        "train/rl/episode_advantage_abs_mean",
        "train/rl/turn_advantage_abs_mean",
        "train/rl/anchor_group_size_mean",
    ),
)

_SAMPO_CREDIT_SUMMARY = (
    SummaryFieldDefinition(
        key="turn_credit_share", label="Turn share of credit", metric="train/rl/turn_credit_share", unit="ratio"
    ),
    SummaryFieldDefinition(
        key="turn_credit_coverage",
        label="Turns with turn credit",
        metric="train/rl/turn_advantage_informative_fraction",
        unit="ratio",
    ),
    SummaryFieldDefinition(key="episode_credit", label="Episode credit", metric="train/rl/episode_advantage_abs_mean"),
    SummaryFieldDefinition(key="turn_credit", label="Turn credit", metric="train/rl/turn_advantage_abs_mean"),
    SummaryFieldDefinition(
        key="anchor_group_size", label="Anchor group size", metric="train/rl/anchor_group_size_mean"
    ),
    SummaryFieldDefinition(
        key="sparse_reward_projection",
        label="Sparse-reward projection",
        metric="train/rl/sparse_reward_projection_fraction",
        unit="ratio",
    ),
)


def _sampo_telemetry() -> JobTelemetryDefinition:
    """SAMPO reads like GRPO, plus the hierarchical credit it adds.

    It shares GRPO's population, stability, runtime, freshness, acceleration and
    active-sampling evidence (the TRL backend emits the same metrics) and adds
    one tab for episode- versus turn-level credit. Dynamic sampling is a GRPO
    algorithm option SAMPO does not have.
    """

    charts: list[ChartDefinition] = []
    for chart in GRPO_TELEMETRY.charts:
        if chart.key == "dynamic_sampling":
            continue
        charts.append(chart)
        if chart.key == "optimization":
            charts.append(_SAMPO_CREDIT_CHART)
    summary_fields = (*GRPO_TELEMETRY.summary_fields, *_SAMPO_CREDIT_SUMMARY)
    fields: dict[str, Any] = {
        "schema_version": 2,
        "job_kind": "train.sampo",
        "display_name": "Step-aware multi-turn policy optimization",
        "summary_fields": summary_fields,
        "charts": tuple(charts),
        "health_rules": SAMPO_HEALTH_RULES,
        "comparison_keys": ("reward_mean", "turn_credit_share", "policy_loss", "failed_rollouts"),
        "trace_sections": GRPO_TELEMETRY.trace_sections,
        "artifact_roles": _training_artifacts(),
        "delta_tip_metrics": (
            "train/rl/reward_mean",
            "train/rl/turn_credit_share",
            "train/rl/policy_loss",
            "train/rl/rollout_tokens_per_second",
        ),
        "projection_metrics": GRPO_TELEMETRY.projection_metrics,
        "evidence_requirements": SAMPO_EVIDENCE_REQUIREMENTS,
    }
    names = {
        *(field.metric for field in summary_fields),
        *(metric for chart in charts for metric in chart.metrics),
        *(rule.metric for rule in SAMPO_HEALTH_RULES),
        *(metric for requirement in SAMPO_EVIDENCE_REQUIREMENTS for metric in requirement.metrics),
        *GRPO_TELEMETRY.projection_metrics,
    }
    return JobTelemetryDefinition(**fields, metric_help=_help_for(*sorted(names)))


SAMPO_HEALTH_RULES: tuple[HealthRuleDefinition, ...] = (
    HealthRuleDefinition(
        id="sampo-reward-non-finite",
        kind="non_finite",
        metric="train/rl/reward_mean",
        message="SAMPO reward contains a non-finite value.",
        severity="error",
    ),
    HealthRuleDefinition(
        id="sampo-policy-loss-non-finite",
        kind="non_finite",
        metric="train/rl/policy_loss",
        message="SAMPO policy loss contains a non-finite value.",
        severity="error",
    ),
    HealthRuleDefinition(
        id="sampo-rollout-failures",
        kind="threshold",
        metric="train/rl/rollouts_failed",
        operator="gt",
        threshold=0,
        message="One or more SAMPO rollout attempts failed.",
        severity="error",
    ),
    HealthRuleDefinition(
        id="sampo-unscorable-rollouts",
        kind="threshold",
        metric="train/rl/rollouts_unscorable",
        operator="gt",
        threshold=0,
        message="One or more SAMPO rollouts did not produce a finite reward.",
        severity="error",
    ),
)

SAMPO_EVIDENCE_REQUIREMENTS: tuple[EvidenceRequirementDefinition, ...] = (
    EvidenceRequirementDefinition(
        key="hierarchical_credit",
        label="Hierarchical credit assignment",
        level="required",
        metrics=(
            "train/rl/episode_advantage_mean",
            "train/rl/turn_advantage_mean",
            "train/rl/anchor_group_size_mean",
            "train/rl/sparse_reward_projection_fraction",
        ),
        reason="SAMPO needs direct evidence that its episode and turn-level credit assignment executed.",
    ),
    EvidenceRequirementDefinition(
        key="learning_signal",
        label="Relative learning signal",
        level="required",
        metrics=("train/rl/reward_mean", "train/rl/reward_std", "train/rl/group_zero_variance_fraction"),
        reason="Reward level and within-group variation are both required to interpret policy learning.",
    ),
    EvidenceRequirementDefinition(
        key="controlled_update",
        label="Controlled policy update",
        level="required",
        metrics=(
            "train/rl/policy_loss",
            "train/rl/entropy",
            "train/rl/clip_fraction",
            "train/grad_norm",
            "train/learning_rate",
        ),
        reason="The sequence-clipped objective must be paired with exploration and gradient-scale evidence.",
    ),
    EvidenceRequirementDefinition(
        key="rollout_population",
        label="Rollout population",
        level="required",
        metrics=(
            "train/rl/rollouts_attempted",
            "train/rl/rollouts_completed",
            "train/rl/rollouts_failed",
            "train/rl/rollouts_truncated",
            "train/rl/rollouts_unscorable",
        ),
        reason="Every update needs an auditable population denominator and terminal outcomes.",
    ),
    EvidenceRequirementDefinition(
        key="runtime_efficiency",
        label="Runtime efficiency",
        level="required",
        metrics=("train/step_time_seconds", "train/rl/rollout_tokens_per_second"),
        reason="End-to-end step cost and rollout throughput are required for runtime comparison.",
    ),
    EvidenceRequirementDefinition(
        key="reference_policy",
        label="Reference-policy drift",
        level="conditional",
        condition="reference_kl_enabled",
        metrics=("train/rl/kl",),
        reason="KL evidence is owed whenever a non-zero reference penalty is selected.",
    ),
    EvidenceRequirementDefinition(
        key="tool_behavior",
        label="Tool-use behavior",
        level="conditional",
        condition="tool_environment",
        metrics=("train/rl/tool_call_frequency", "train/rl/tool_failure_frequency"),
        reason="Tool environments owe invocation and failure coverage in addition to reward.",
    ),
)

SAMPO_TELEMETRY = _sampo_telemetry()

DISTILL_TELEMETRY = JobTelemetryDefinition(
    job_kind="train.distill",
    display_name="On-policy distillation",
    summary_fields=(
        SummaryFieldDefinition(key="final_loss", label="Final loss", metric="train/distill/loss", required=True),
        SummaryFieldDefinition(key="reverse_kl", label="Reverse KL", metric="train/distill/reverse_kl", required=True),
        SummaryFieldDefinition(
            key="scored_tokens",
            label="Scored tokens",
            metric="train/distill/scored_tokens",
            reducer="sum",
            required=True,
        ),
        SummaryFieldDefinition(
            key="teacher_latency_ms",
            label="Teacher latency",
            metric="train/distill/teacher_latency_ms",
            reducer="mean",
            unit="ms",
        ),
        SummaryFieldDefinition(
            key="teacher_failures",
            label="Teacher failures",
            metric="train/distill/teacher_failures",
            reducer="sum",
            required=True,
        ),
    ),
    charts=(
        ChartDefinition(
            key="objective",
            title="Distillation objective",
            metrics=("train/distill/loss", "train/distill/reverse_kl"),
        ),
        ChartDefinition(
            key="teacher",
            title="Teacher scoring",
            metrics=(
                "train/distill/scored_tokens",
                "train/distill/teacher_latency_ms",
                "train/distill/teacher_failures",
            ),
        ),
    ),
    metric_help=_help_for(
        "train/distill/loss",
        "train/distill/reverse_kl",
        "train/distill/scored_tokens",
        "train/distill/teacher_latency_ms",
        "train/distill/teacher_failures",
    ),
    health_rules=(
        HealthRuleDefinition(
            id="distill-loss-non-finite",
            kind="non_finite",
            metric="train/distill/loss",
            message="Distillation loss contains a non-finite value.",
            severity="error",
        ),
        HealthRuleDefinition(
            id="distill-teacher-failures",
            kind="threshold",
            metric="train/distill/teacher_failures",
            operator="gt",
            threshold=0.0,
            message="Teacher scoring failed for at least one batch.",
            severity="error",
        ),
    ),
    comparison_keys=("final_loss", "reverse_kl", "scored_tokens", "teacher_failures"),
    trace_sections=(TraceSectionDefinition(trace_type="verifiers", label="Distillation rollouts"),),
    artifact_roles=_training_artifacts(),
    delta_tip_metrics=(
        "train/distill/loss",
        "train/distill/reverse_kl",
        "train/distill/scored_tokens",
        "train/distill/teacher_latency_ms",
        "train/distill/teacher_failures",
    ),
)


def _eval_definition(job_kind: Literal["eval.general", "eval.domain"], display_name: str) -> JobTelemetryDefinition:
    return JobTelemetryDefinition(
        job_kind=job_kind,
        display_name=display_name,
        summary_fields=(
            SummaryFieldDefinition(
                key="rollouts_complete",
                label="Completed rollouts",
                metric="eval/run/rollouts_complete",
                reducer="sum",
                required=True,
            ),
            SummaryFieldDefinition(
                key="rollouts_failed", label="Failed rollouts", metric="eval/run/rollouts_failed", reducer="sum"
            ),
            SummaryFieldDefinition(
                key="rollouts_truncated",
                label="Truncated rollouts",
                metric="eval/run/rollouts_truncated",
                reducer="sum",
            ),
            SummaryFieldDefinition(
                key="model_call_error_rollouts",
                label="Model-call errors",
                metric="eval/run/model_call_error_rollouts",
                reducer="sum",
            ),
            SummaryFieldDefinition(
                key="model_call_http_400_rollouts",
                label="Model HTTP 400 errors",
                metric="eval/run/model_call_http_400_rollouts",
                reducer="sum",
            ),
            SummaryFieldDefinition(
                key="context_overflow_rollouts",
                label="Context overflows",
                metric="eval/run/context_overflow_rollouts",
                reducer="sum",
            ),
            SummaryFieldDefinition(
                key="trace_sync_complete", label="Trace synchronization complete", metric="eval/trace_sync_complete"
            ),
        ),
        charts=(
            ChartDefinition(
                key="rollouts",
                title="Rollout completion",
                metrics=("eval/run/rollouts_complete", "eval/run/rollouts_failed", "eval/run/rollouts_truncated"),
            ),
        ),
        metric_help=_help_for(
            "eval/run/rollouts_complete",
            "eval/run/rollouts_failed",
            "eval/run/rollouts_truncated",
            "eval/run/model_call_error_rollouts",
            "eval/run/model_call_http_400_rollouts",
            "eval/run/context_overflow_rollouts",
            "eval/trace_sync_complete",
            "eval/trace_sync_schema_mismatch",
        ),
        health_rules=(
            HealthRuleDefinition(
                id=f"{job_kind}-context-overflow",
                kind="threshold",
                metric="eval/run/context_overflow_rollouts",
                operator="gt",
                threshold=0.0,
                message="Model context overflow invalidated evaluation attempts; check per-call output and prompt budgets.",
            ),
            HealthRuleDefinition(
                id=f"{job_kind}-model-call-errors",
                kind="threshold",
                metric="eval/run/model_call_error_rollouts",
                operator="gt",
                threshold=0.0,
                message="Model-call failures invalidated evaluation attempts; these are not semantic zero scores.",
            ),
            HealthRuleDefinition(
                id=f"{job_kind}-trace-schema-mismatch",
                kind="threshold",
                metric="eval/trace_sync_schema_mismatch",
                operator="gt",
                threshold=0.0,
                message="Trace schema mismatch: evaluation traces were not synchronized; use a compatible runtime.",
            ),
            HealthRuleDefinition(
                id=f"{job_kind}-trace-sync",
                kind="threshold",
                metric="eval/trace_sync_complete",
                operator="lt",
                threshold=1.0,
                message="Evaluation trace synchronization is incomplete.",
            ),
        ),
        comparison_keys=("rollouts_complete", "rollouts_failed", "rollouts_truncated"),
        trace_sections=(TraceSectionDefinition(trace_type="verifiers", label="Evaluation rollouts"),),
        artifact_roles=(
            ArtifactRoleDefinition(kind="verifiers-evaluation", label="Native evaluation bundle", direction="output"),
        ),
        delta_tip_metrics=("eval/run/rollouts_complete", "eval/run/rollouts_failed"),
    )


GENERAL_EVAL_TELEMETRY = _eval_definition("eval.general", "General evaluation")
DOMAIN_EVAL_TELEMETRY = _eval_definition("eval.domain", "Domain evaluation")

SERVE_SMOKE_TELEMETRY = JobTelemetryDefinition(
    job_kind="serve.smoke",
    display_name="Serving health smoke test",
    summary_fields=(
        SummaryFieldDefinition(
            key="healthy",
            label="Endpoint healthy",
            metric="serve/probe_healthy",
            unit="ratio",
            required=True,
        ),
        SummaryFieldDefinition(
            key="model_available",
            label="Model available",
            metric="serve/probe_model_available",
            unit="ratio",
            required=True,
        ),
        SummaryFieldDefinition(
            key="probe_latency",
            label="Probe latency",
            metric="serve/probe_latency_seconds",
            unit="s",
            required=True,
        ),
    ),
    charts=(
        ChartDefinition(
            key="probe",
            title="Health probe",
            question="Did the managed endpoint answer its health and model-discovery checks?",
            metrics=("serve/probe_healthy", "serve/probe_model_available", "serve/probe_latency_seconds"),
        ),
    ),
    metric_help=(
        *_help_for("serve/probe_healthy"),
        *_help_for("serve/probe_model_available"),
        *_help_for("serve/probe_latency_seconds"),
    ),
    health_rules=(
        HealthRuleDefinition(
            id="serve-smoke-unhealthy",
            kind="threshold",
            metric="serve/probe_healthy",
            operator="lt",
            threshold=1.0,
            message="The managed serving endpoint did not pass its health probe.",
            severity="error",
        ),
        HealthRuleDefinition(
            id="serve-smoke-model-unavailable",
            kind="threshold",
            metric="serve/probe_model_available",
            operator="lt",
            threshold=1.0,
            message="The selected model was not exposed by the managed serving endpoint.",
            severity="error",
        ),
    ),
    comparison_keys=("healthy", "model_available", "probe_latency"),
    artifact_roles=(ArtifactRoleDefinition(kind="serving-log", label="Serving log", direction="output"),),
    delta_tip_metrics=("serve/probe_healthy", "serve/probe_model_available", "serve/probe_latency_seconds"),
)

DATA_PREPARE_TELEMETRY = JobTelemetryDefinition(
    job_kind="data.prepare",
    display_name="Dataset preparation",
    summary_fields=(
        SummaryFieldDefinition(
            key="examples",
            label="Prepared examples",
            metric="data/examples",
            reducer="sum",
            required=True,
        ),
        SummaryFieldDefinition(
            key="bytes",
            label="Prepared bytes",
            metric="data/bytes",
            reducer="sum",
            unit="bytes",
            required=True,
        ),
    ),
    charts=(
        ChartDefinition(
            key="prepared_dataset",
            title="Prepared dataset",
            question="How much immutable dataset content did this job materialize?",
            metrics=("data/examples", "data/bytes"),
        ),
    ),
    metric_help=(
        *_help_for("data/examples"),
        *_help_for("data/bytes"),
    ),
    comparison_keys=("examples", "bytes"),
    artifact_roles=(ArtifactRoleDefinition(kind="dataset", label="Prepared dataset", direction="output"),),
    delta_tip_metrics=("data/examples", "data/bytes"),
)

SERVE_BENCHMARK_TELEMETRY = JobTelemetryDefinition(
    job_kind="serve.benchmark",
    display_name="Serving capacity benchmark",
    summary_fields=(
        SummaryFieldDefinition(
            key="output_tokens_measured",
            label="Measured output tokens",
            metric="serve/run/output_tokens_measured",
            unit="tokens",
            required=True,
        ),
        SummaryFieldDefinition(
            key="measurement_duration",
            label="Measurement duration",
            metric="serve/run/measurement_duration_s",
            unit="s",
            required=True,
        ),
        SummaryFieldDefinition(
            key="requests_measured",
            label="Measured requests",
            metric="serve/run/requests_measured",
            required=True,
        ),
        SummaryFieldDefinition(
            key="concurrency",
            label="Concurrency",
            metric="serve/run/concurrency",
            required=True,
        ),
        SummaryFieldDefinition(
            key="context_window",
            label="Context allocation",
            metric="serve/run/context_tokens",
            unit="tokens",
            required=True,
        ),
        SummaryFieldDefinition(
            key="peak_vram",
            label="Peak GPU memory",
            metric="serve/backend/peak_vram_bytes",
            unit="bytes",
        ),
    ),
    charts=(
        ChartDefinition(
            key="point-population",
            title="Measured point population",
            question="How many requests and output tokens were measured at each concurrency?",
            metrics=("serve/run/requests_measured", "serve/run/output_tokens_measured"),
        ),
        ChartDefinition(
            key="measurement-time",
            title="Measurement duration",
            question="How long was the timed inference window at each concurrency?",
            metrics=("serve/run/measurement_duration_s",),
        ),
    ),
    metric_help=(
        *_help_for("serve/run/output_tokens_measured"),
        *_help_for("serve/run/measurement_duration_s"),
        *_help_for("serve/run/requests_measured"),
        *_help_for("serve/run/concurrency"),
        *_help_for("serve/run/context_tokens"),
        *_help_for("serve/backend/peak_vram_bytes"),
    ),
    comparison_keys=(
        "output_tokens_measured",
        "measurement_duration",
        "requests_measured",
        "concurrency",
        "context_window",
        "peak_vram",
    ),
    trace_sections=(TraceSectionDefinition(trace_type="inference", label="Measured requests"),),
    artifact_roles=(
        ArtifactRoleDefinition(kind="serving-result", label="Serving result", direction="output"),
        ArtifactRoleDefinition(kind="serving-benchmark", label="Legacy serving benchmark", direction="output"),
    ),
    evidence_requirements=(
        EvidenceRequirementDefinition(
            key="capacity-point",
            label="Capacity point",
            level="required",
            metrics=(
                "serve/run/output_tokens_measured",
                "serve/run/measurement_duration_s",
                "serve/run/requests_measured",
                "serve/run/concurrency",
                "serve/run/context_tokens",
            ),
            reason="A serving decision needs point counters plus the complete measured request-trace population from which rates and latency are derived.",
        ),
    ),
)

DEFAULT_TELEMETRY_DEFINITIONS: Mapping[str, JobTelemetryDefinition] = MappingProxyType(
    {
        definition.job_kind: definition
        for definition in (
            SFT_TELEMETRY,
            DPO_TELEMETRY,
            GRPO_TELEMETRY,
            GDPO_TELEMETRY,
            CAPO_TELEMETRY,
            SAMPO_TELEMETRY,
            DISTILL_TELEMETRY,
            DATA_PREPARE_TELEMETRY,
            GENERAL_EVAL_TELEMETRY,
            DOMAIN_EVAL_TELEMETRY,
            SERVE_SMOKE_TELEMETRY,
            SERVE_BENCHMARK_TELEMETRY,
        )
    }
)


def telemetry_registry(
    definitions: Iterable[JobTelemetryDefinition] = DEFAULT_TELEMETRY_DEFINITIONS.values(),
) -> Mapping[str, JobTelemetryDefinition]:
    result: dict[str, JobTelemetryDefinition] = {}
    for definition in definitions:
        if definition.job_kind in result:
            raise ValueError(f"duplicate job telemetry definition: {definition.job_kind}")
        result[definition.job_kind] = definition
    return MappingProxyType(result)


__all__ = [
    "ArtifactRoleDefinition",
    "ChartDefinition",
    "CAPO_TELEMETRY",
    "DATA_PREPARE_TELEMETRY",
    "DEFAULT_TELEMETRY_DEFINITIONS",
    "DISTILL_TELEMETRY",
    "DOMAIN_EVAL_TELEMETRY",
    "DPO_TELEMETRY",
    "EvidenceCondition",
    "EvidenceRequirementDefinition",
    "GENERAL_EVAL_TELEMETRY",
    "GDPO_TELEMETRY",
    "GROUP_POLICY_JOB_KINDS",
    "GRPO_TELEMETRY",
    "HealthRuleDefinition",
    "JobTelemetryDefinition",
    "MetricHelp",
    "SAMPO_TELEMETRY",
    "SFT_TELEMETRY",
    "SERVE_BENCHMARK_TELEMETRY",
    "SERVE_SMOKE_TELEMETRY",
    "SummaryFieldDefinition",
    "TraceSectionDefinition",
    "telemetry_registry",
]
