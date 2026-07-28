"""Versioned job telemetry definitions owned by Observatory."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Literal

from pydantic import Field, model_validator

from .models import AlertSeverity, MetricHelp, ObservatoryModel

type Reducer = Literal["last", "min", "max", "mean", "sum"]
type HealthRuleKind = Literal["threshold", "non_finite"]
type ThresholdOperator = Literal["gt", "gte", "lt", "lte", "eq"]
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
        }


def _training_artifacts() -> tuple[ArtifactRoleDefinition, ...]:
    return (
        ArtifactRoleDefinition(kind="model-adapter", label="Trained adapter", direction="output"),
        ArtifactRoleDefinition(kind="model-weights", label="Trained weights", direction="output"),
        ArtifactRoleDefinition(kind="training-summary", label="Native training summary", direction="output"),
    )


def _metric(
    metric: str,
    label: str,
    description: str,
    interpretation: str,
    *,
    unit: str | None = None,
    caveat: str | None = None,
) -> MetricHelp:
    return MetricHelp(
        metric=metric,
        label=label,
        description=description,
        interpretation=interpretation,
        caveat=caveat,
        unit=unit,
    )


_METRIC_HELP = {
    item.metric: item
    for item in (
        _metric(
            "train/loss",
            "Training loss",
            "Mean supervised objective on the current optimization batch.",
            "A sustained decrease means the model is fitting the rendered training targets; compare it with held-out loss before judging generalization.",
            caveat="Masking, packing, batch composition, and sequence length affect its scale.",
        ),
        _metric(
            "train/validation/loss",
            "Validation loss",
            "The same supervised objective measured on the reserved validation split without updating weights.",
            "A flattening curve suggests diminishing learning benefit; a rise while training loss falls is evidence of overfitting.",
            caveat="Compare only runs using the same split, renderer, masking policy, and maximum sequence length.",
        ),
        _metric(
            "train/mean_token_accuracy",
            "Token accuracy",
            "Share of supervised tokens whose highest-probability prediction matches the target token.",
            "Higher is better, but use it with loss because it ignores how confident incorrect and correct predictions are.",
            unit="ratio",
            caveat="Only supervised, non-masked tokens should contribute.",
        ),
        _metric(
            "train/grad_norm",
            "Gradient norm",
            "Trainer-recorded global magnitude of gradients before the optimizer update.",
            "Spikes can indicate unstable batches; a persistently tiny value can indicate stalled learning.",
            caveat="The exact norm and whether it is measured before or after clipping depend on the trainer binding.",
        ),
        _metric(
            "train/learning_rate",
            "Learning rate",
            "Optimizer step size selected by the learning-rate schedule at this logical step.",
            "Read it beside loss and gradient norm to explain warmup, decay, and changes in update behavior.",
        ),
        _metric(
            "train/gradient_clipped",
            "Gradient clipped",
            "Whether, or what share of, the current gradient update exceeded the clipping threshold.",
            "Occasional clipping is protective; frequent clipping suggests the update scale or data batches deserve inspection.",
            unit="ratio",
            caveat="A per-step boolean is displayed as 0% or 100%; aggregated trainers may report a fraction.",
        ),
        _metric(
            "train/non_padding_tokens_per_second",
            "Non-padding tokens / second",
            "Effective supervised and context tokens processed per second, excluding padding.",
            "Higher is better for throughput and is more comparable than raw tokens when sequence lengths and padding differ.",
            unit="tokens/s",
            caveat="Hardware, world size, packing, checkpointing, and measurement window must match for fair comparisons.",
        ),
        _metric(
            "train/step_time_seconds",
            "Step time",
            "Wall-clock duration of one logical training step.",
            "Stable or falling values indicate healthy runtime performance; spikes should be correlated with system metrics and checkpoint activity.",
            unit="s",
        ),
        _metric(
            "train/num_tokens",
            "Processed tokens",
            "Cumulative number of non-padding training tokens reported by the trainer.",
            "Use it as the denominator for progress and effective throughput rather than comparing optimizer steps alone.",
            unit="tokens",
            caveat="The trainer binding determines whether prompt and completion tokens are both included.",
        ),
        _metric(
            "train/data/supervision_token_ratio",
            "Supervision-token ratio",
            "Fraction of rendered non-padding tokens that contribute to the supervised loss.",
            "Very low values mean most compute is spent on context rather than learning targets.",
            unit="ratio",
            caveat="The expected range depends on the conversation template and assistant-only masking policy.",
        ),
        _metric(
            "train/data/truncation_rate",
            "Truncated examples",
            "Fraction of rendered examples shortened by the configured maximum sequence length.",
            "Lower is usually safer; a high rate means targets or important context may be removed.",
            unit="ratio",
            caveat="Inspect which side of the example is truncated before deciding whether the signal is damaged.",
        ),
        _metric(
            "train/data/max_length_utilization",
            "Max-length utilization",
            "Average rendered sequence length as a fraction of the configured maximum length.",
            "High utilization improves dense batches but leaves less headroom for length variation and increases truncation risk.",
            unit="ratio",
        ),
        _metric(
            "train/rewards/margins",
            "Reward margin",
            "Average chosen-response reward minus rejected-response reward.",
            "Positive and increasing values indicate the policy is separating preferred responses from rejected ones.",
        ),
        _metric(
            "train/rewards/chosen",
            "Chosen reward",
            "Average implicit reward assigned to preferred responses.",
            "It should generally stay above rejected reward; its absolute scale is less useful than their margin.",
        ),
        _metric(
            "train/rewards/rejected",
            "Rejected reward",
            "Average implicit reward assigned to non-preferred responses.",
            "Read it with chosen reward to determine which side is driving the preference margin.",
        ),
        _metric(
            "train/rewards/accuracies",
            "Preference accuracy",
            "Fraction of pairs for which the chosen response receives the higher implicit reward.",
            "Higher values mean the learned preference ordering agrees with the dataset more often.",
            unit="ratio",
        ),
        _metric(
            "train/logps/chosen",
            "Chosen log probability",
            "Average sequence log probability the policy assigns to preferred completions.",
            "An increase means the chosen responses are becoming more likely under the policy; read it with rejected log probability to distinguish positive learning from rejection-only suppression.",
            caveat="Sequence log probability is length-sensitive, so compare runs only when rendering and length distributions match.",
        ),
        _metric(
            "train/logps/rejected",
            "Rejected log probability",
            "Average sequence log probability the policy assigns to rejected completions.",
            "A decrease means rejected responses are becoming less likely; if chosen log probability also falls, the margin may be widening through broad suppression.",
            caveat="Sequence log probability is length-sensitive, so compare runs only when rendering and length distributions match.",
        ),
        _metric(
            "train/logits/chosen",
            "Chosen mean logit",
            "Average raw model logit on tokens from preferred completions.",
            "Use it as a low-level diagnostic for changes in chosen-token scores, not as a standalone quality measure.",
        ),
        _metric(
            "train/logits/rejected",
            "Rejected mean logit",
            "Average raw model logit on tokens from rejected completions.",
            "Compare it with chosen mean logit only as a low-level optimization diagnostic.",
        ),
        _metric(
            "train/entropy",
            "Token entropy",
            "Average uncertainty of the policy distribution over non-masked completion tokens.",
            "A sharp fall can indicate the policy is becoming overly confident or collapsing; stable entropy alongside improving preference separation is generally healthier.",
        ),
        _metric(
            "train/validation/rewards/margins",
            "Validation reward margin",
            "Average implicit chosen-minus-rejected reward on reserved preference pairs without updating weights.",
            "A positive margin that improves with training is held-out evidence that preference separation generalizes beyond optimization pairs.",
        ),
        _metric(
            "train/validation/rewards/accuracies",
            "Validation preference accuracy",
            "Fraction of reserved pairs whose chosen response receives the higher implicit reward.",
            "Use it with validation margin and loss; saturated training accuracy with weak validation accuracy indicates memorization.",
            unit="ratio",
        ),
        _metric(
            "train/validation/rewards/chosen",
            "Validation chosen reward",
            "Average implicit reward for chosen responses in the reserved preference split.",
            "Read it with rejected reward to see which side drives held-out separation.",
        ),
        _metric(
            "train/validation/rewards/rejected",
            "Validation rejected reward",
            "Average implicit reward for rejected responses in the reserved preference split.",
            "Read it with chosen reward to distinguish improved chosen likelihood from rejection-only suppression.",
        ),
        _metric(
            "train/validation/logps/chosen",
            "Validation chosen log probability",
            "Average policy log-probability assigned to chosen completions in reserved pairs.",
            "It reveals whether held-out chosen responses become more likely under the policy.",
            caveat="Sequence log probability is length-sensitive.",
        ),
        _metric(
            "train/validation/logps/rejected",
            "Validation rejected log probability",
            "Average policy log-probability assigned to rejected completions in reserved pairs.",
            "Use it with chosen log probability to explain held-out margin changes.",
            caveat="Sequence log probability is length-sensitive.",
        ),
        _metric(
            "train/data/preference_pairs",
            "Preference pairs",
            "Number of rendered chosen-versus-rejected pairs in the training population.",
            "Very small populations can be memorized quickly, so training accuracy and reward margin should not be read as generalization evidence.",
        ),
        _metric(
            "train/data/prompt_tokens_mean",
            "Mean prompt tokens",
            "Average rendered prompt length across preference pairs.",
            "Use it to understand how much of each sequence is shared context and to compare dataset composition across runs.",
            unit="tokens",
        ),
        _metric(
            "train/data/chosen_tokens_mean",
            "Mean chosen tokens",
            "Average rendered length of preferred completions.",
            "Compare it with rejected length to detect a preference dataset that systematically favors longer or shorter answers.",
            unit="tokens",
        ),
        _metric(
            "train/data/rejected_tokens_mean",
            "Mean rejected tokens",
            "Average rendered length of rejected completions.",
            "Compare it with chosen length because length imbalance can become an unintended preference shortcut.",
            unit="tokens",
        ),
        _metric(
            "train/data/prompt_tokens_p95",
            "Prompt tokens p95",
            "Ninety-fifth percentile of rendered prompt length across preference pairs.",
            "A high tail value identifies prompts most likely to constrain batch density or approach the sequence limit.",
            unit="tokens",
        ),
        _metric(
            "train/data/chosen_tokens_p95",
            "Chosen tokens p95",
            "Ninety-fifth percentile of preferred-completion length.",
            "Compare it with rejected p95 to detect a length shortcut hidden by similar means.",
            unit="tokens",
        ),
        _metric(
            "train/data/rejected_tokens_p95",
            "Rejected tokens p95",
            "Ninety-fifth percentile of rejected-completion length.",
            "Compare it with chosen p95 to detect systematic tail imbalance between alternatives.",
            unit="tokens",
        ),
        _metric(
            "train/data/max_length_headroom_min",
            "Minimum length headroom",
            "Smallest remaining token capacity between a rendered preference pair and the configured maximum length.",
            "Low headroom means at least one pair is close to failing rendering when the template or data changes.",
            unit="tokens",
            caveat="Current DPO rendering rejects over-limit pairs instead of truncating them.",
        ),
        _metric(
            "train/data/chosen_longer_fraction",
            "Chosen longer than rejected",
            "Fraction of pairs whose preferred completion is longer than its rejected completion.",
            "Values far from a balanced population may indicate a systematic length preference that the policy can exploit.",
            unit="ratio",
            caveat="Some tasks legitimately prefer concise or detailed answers; interpret this with task intent.",
        ),
        _metric(
            "train/data/preference_score_coverage",
            "Source score coverage",
            "Fraction of preference pairs that retain explicit source scores for both alternatives.",
            "Higher coverage makes it possible to audit how strong or ambiguous the original preference signal was.",
            unit="ratio",
        ),
        _metric(
            "train/data/preference_score_margin_mean",
            "Mean source score margin",
            "Average source-score difference between chosen and rejected completions for scored pairs.",
            "Larger margins represent clearer source preferences; very small margins may be noisy or ambiguous.",
            caveat="Only comparable when the same scorer and score scale produced every pair.",
        ),
        _metric(
            "train/data/max_length_utilization",
            "Max-length utilization",
            "Average rendered prompt plus longer completion length as a fraction of the configured sequence limit.",
            "High utilization leaves little room for length variation and increases curation failures near the maximum length.",
            unit="ratio",
        ),
        _metric(
            "train/rl/reward_mean",
            "Mean reward",
            "Mean verifier reward across the sampled rollout group.",
            "An upward trend suggests policy improvement when the verifier and sampling policy are unchanged.",
        ),
        _metric(
            "train/rl/reward_std",
            "Reward standard deviation",
            "Spread of verifier rewards within sampled rollout groups.",
            "Some spread supplies a learning signal; collapse near zero can make relative advantages uninformative.",
        ),
        _metric(
            "train/rl/kl",
            "KL divergence",
            "Recorded divergence between the updated policy and its reference policy.",
            "Rising KL means the policy is moving farther from the reference; interpret it against reward improvement and the configured penalty.",
        ),
        _metric(
            "train/rl/entropy",
            "Policy entropy",
            "Uncertainty of the policy distribution over sampled tokens.",
            "A sharp decline can indicate premature policy collapse; a persistently high value can indicate weak optimization.",
        ),
        _metric(
            "train/rl/clip_fraction",
            "Clip fraction",
            "Fraction of policy updates constrained by the objective's clipping range.",
            "High values mean many updates are hitting the trust-region guardrail and may be too aggressive.",
            unit="ratio",
        ),
        _metric(
            "train/rl/group_zero_variance_fraction",
            "Zero-variance groups",
            "Fraction of rollout groups whose rewards contain no within-group variation.",
            "High values mean GRPO cannot rank alternatives inside many groups even when reward varies globally.",
            unit="ratio",
        ),
        _metric(
            "train/rl/policy_loss",
            "Policy loss",
            "Policy optimization objective after relative advantages and clipping are applied.",
            "Interpret its trend with reward, entropy, clipping, and gradient norm rather than as a standalone quality score.",
        ),
        _metric(
            "train/rl/rollouts_attempted",
            "Attempted rollouts",
            "Number of rollout attempts requested for the logical step.",
            "This is the population denominator for completion, failure, truncation, and scoring coverage.",
        ),
        _metric(
            "train/rl/rollouts_completed",
            "Completed rollouts",
            "Number of rollout attempts that returned a terminal result.",
            "Compare it with attempted rollouts; a gap means the policy update saw less evidence than configured.",
        ),
        _metric(
            "train/rl/rollouts_failed",
            "Failed rollouts",
            "Number of rollout attempts that ended in an execution or environment error.",
            "Any non-zero value requires trace-level inspection because failed work can bias the effective batch.",
        ),
        _metric(
            "train/rl/rollouts_truncated",
            "Truncated rollouts",
            "Number of completed rollouts stopped by a token, step, or time limit.",
            "Truncation can hide task completion and bias rewards toward shorter trajectories.",
        ),
        _metric(
            "train/rl/rollouts_unscorable",
            "Unscorable rollouts",
            "Number of returned rollouts without a finite verifier reward.",
            "Unscorable trajectories cannot contribute a valid relative learning signal and should be treated as evidence loss.",
        ),
        _metric(
            "train/rl/completion_tokens_mean",
            "Mean completion tokens",
            "Average generated completion length for the rollout population.",
            "Length drift can indicate changing task behavior, truncation pressure, or reward exploitation.",
            unit="tokens",
        ),
        _metric(
            "train/rl/completion_tokens_max",
            "Maximum completion tokens",
            "Longest generated completion in the observed rollout population.",
            "Repeated contact with the configured limit suggests generation is capacity constrained.",
            unit="tokens",
        ),
        _metric(
            "train/rl/completion_truncation_rate",
            "Completion truncation rate",
            "Fraction of completions stopped at the configured generation limit.",
            "A rising rate means reward and completion statistics increasingly describe partial trajectories.",
            unit="ratio",
        ),
        _metric(
            "train/rl/rollout_tokens_per_second",
            "Rollout tokens per second",
            "Effective completion-token throughput during rollout generation.",
            "Compare runs only when model, hardware, sequence distribution, and rollout topology are comparable.",
            unit="tokens/s",
        ),
        _metric(
            "train/rl/sampling_logp_delta_mean",
            "Mean sampling log-probability delta",
            "Mean difference between rollout-server sampling log-probabilities and actor recomputation.",
            "Drift away from zero can expose serving-versus-training numerical mismatch or stale policy weights.",
        ),
        _metric(
            "train/rl/sampling_logp_delta_max",
            "Maximum sampling log-probability delta",
            "Largest observed rollout-versus-actor log-probability difference.",
            "Outliers can destabilize importance correction even when the mean remains small.",
        ),
        _metric(
            "train/rl/importance_sampling_ratio_mean",
            "Mean importance ratio",
            "Mean actor-to-rollout probability ratio used to correct decoupled sampling.",
            "Values near one indicate close policies; interpret with the min and max because the mean hides tail instability.",
        ),
        _metric(
            "train/rl/importance_sampling_ratio_min",
            "Minimum importance ratio",
            "Smallest actor-to-rollout probability ratio in the observed batch.",
            "Very small values identify samples whose rollout probability substantially exceeds the current actor probability.",
        ),
        _metric(
            "train/rl/importance_sampling_ratio_max",
            "Maximum importance ratio",
            "Largest actor-to-rollout probability ratio in the observed batch.",
            "Large tail values can dominate updates unless correction or clipping is active.",
        ),
        _metric(
            "train/rl/policy_staleness_mean",
            "Mean policy staleness",
            "Average number of policy versions between trajectory generation and optimization.",
            "Higher staleness weakens on-policy assumptions and should be read with importance ratios and reward progress.",
            unit="versions",
        ),
        _metric(
            "train/rl/policy_staleness_max",
            "Maximum policy staleness",
            "Largest observed policy-version delay for trajectories consumed by the update.",
            "A high tail can reveal queue backlog that is hidden by an acceptable mean.",
            unit="versions",
        ),
        _metric(
            "train/rl/trajectory_version_span_mean",
            "Mean trajectory version span",
            "Average policy-version span represented within an optimization batch.",
            "Wide batches mix behavior from different policy states and make update interpretation less direct.",
            unit="versions",
        ),
        _metric(
            "train/rl/tool_call_frequency",
            "Tool-call frequency",
            "Fraction of rollouts that invoked at least one environment tool.",
            "A change can represent better task engagement or reward gaming; inspect linked traces to distinguish them.",
            unit="ratio",
        ),
        _metric(
            "train/rl/tool_failure_frequency",
            "Tool-failure frequency",
            "Fraction of rollouts with at least one failed tool invocation.",
            "Failures reduce usable evidence and often explain reward degradation or longer trajectories.",
            unit="ratio",
        ),
        _metric(
            "train/rl/time/rollout_seconds",
            "Rollout time",
            "Wall-clock time spent generating trajectories for a logical step.",
            "Use it with total step time to locate generation-bound runs.",
            unit="s",
        ),
        _metric(
            "train/rl/time/reward_seconds",
            "Reward time",
            "Wall-clock time spent evaluating verifier rewards.",
            "A large share of step time identifies verifier or environment evaluation as the bottleneck.",
            unit="s",
        ),
        _metric(
            "train/rl/time/actor_forward_seconds",
            "Actor forward time",
            "Wall-clock time spent recomputing actor probabilities for sampled trajectories.",
            "Compare it with actor update time to distinguish scoring cost from backpropagation cost.",
            unit="s",
        ),
        _metric(
            "train/rl/time/actor_update_seconds",
            "Actor update time",
            "Wall-clock time spent applying the policy optimization update.",
            "A rising value with stable token counts can reveal memory pressure or distributed synchronization overhead.",
            unit="s",
        ),
        _metric(
            "train/rl/time/weight_sync_seconds",
            "Weight synchronization time",
            "Wall-clock time spent moving updated policy weights to the rollout runtime.",
            "This is a direct tax of decoupled serving and should be judged against rollout throughput gains.",
            unit="s",
        ),
        _metric(
            "train/rl/time/checkpoint_seconds",
            "Checkpoint time",
            "Wall-clock time spent writing recovery state.",
            "Checkpoint spikes are operational overhead, not model-learning regressions.",
            unit="s",
        ),
        _metric(
            "serve/backend/speculative_draft_tokens",
            "Speculative draft tokens",
            "Number of tokens proposed by the speculative MTP path.",
            "This is opportunity volume, not acceleration by itself; pair it with accepted tokens and measured throughput.",
            unit="tokens",
        ),
        _metric(
            "serve/backend/speculative_accepted_tokens",
            "Accepted speculative tokens",
            "Number of drafted tokens accepted by target-model verification.",
            "More accepted tokens can reduce target decoding work, but only measured throughput establishes a speedup.",
            unit="tokens",
        ),
        _metric(
            "serve/backend/speculative_acceptance_rate",
            "Speculative acceptance rate",
            "Fraction of drafted MTP tokens accepted by target-model verification.",
            "Acceptance describes draft quality; it does not account for drafting overhead or prove end-to-end acceleration.",
            unit="ratio",
        ),
        _metric(
            "serve/backend/speculative_accepted_length",
            "Accepted speculative length",
            "Mean accepted speculative-token run length per verification cycle.",
            "Longer accepted runs usually reduce target decode iterations, subject to drafting and synchronization cost.",
            unit="tokens",
        ),
        _metric(
            "serve/backend/kv_cache_capacity_tokens",
            "KV-cache token capacity",
            "Runtime-reported token capacity for the selected KV-cache representation.",
            "Capacity is a configuration outcome and must not be inferred from the selected TurboQuant dtype alone.",
            unit="tokens",
        ),
        _metric(
            "serve/backend/kv_cache_peak_usage_ratio",
            "Peak KV-cache usage",
            "Highest observed share of the runtime KV-cache capacity used during the run.",
            "A high value signals limited headroom; a missing value means quantized-cache qualification is incomplete.",
            unit="ratio",
        ),
        _metric(
            "train/distill/loss",
            "Distillation loss",
            "Student training objective computed from teacher-scored tokens.",
            "A decrease means the student is better matching the teacher under the configured objective.",
        ),
        _metric(
            "train/distill/reverse_kl",
            "Reverse KL",
            "Reverse Kullback-Leibler divergence from student to teacher on scored tokens.",
            "Lower values indicate closer student alignment to the teacher distribution.",
        ),
        _metric(
            "train/distill/scored_tokens",
            "Scored tokens",
            "Number of tokens successfully scored by the teacher and included in distillation.",
            "Use it to confirm that effective supervision volume grows as expected.",
        ),
        _metric(
            "train/distill/teacher_latency_ms",
            "Teacher latency",
            "Mean time spent obtaining teacher scores for a batch.",
            "Higher values identify teacher inference as a throughput bottleneck.",
            unit="ms",
        ),
        _metric(
            "train/distill/teacher_failures",
            "Teacher failures",
            "Count of teacher-scoring failures excluded from training.",
            "Any non-zero value means the effective training population is incomplete and should be investigated.",
        ),
        _metric(
            "eval/run/rollouts_complete",
            "Completed rollouts",
            "Number of evaluation rollouts that reached a terminal result.",
            "Compare it with the expected population to judge evaluation completeness.",
        ),
        _metric(
            "eval/run/rollouts_failed",
            "Failed rollouts",
            "Number of rollouts that ended with an execution or verifier error.",
            "Non-zero values reduce aggregate coverage and should be inspected at trace level.",
        ),
        _metric(
            "eval/run/rollouts_truncated",
            "Truncated rollouts",
            "Number of rollouts stopped by a token, time, or step limit.",
            "A high count can bias aggregate scores because tasks did not receive a full attempt.",
        ),
        _metric(
            "eval/trace_sync_complete",
            "Trace synchronization complete",
            "Whether all native evaluation traces have been synchronized into the tracking reader.",
            "A value of 100% means aggregates can link back to their complete trace population.",
            unit="ratio",
        ),
    )
}


def _help_for(*metrics: str) -> tuple[MetricHelp, ...]:
    return tuple(_METRIC_HELP[metric] for metric in metrics)


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
    schema_version=2,
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
        SummaryFieldDefinition(
            key="rollout_tps", label="Rollout throughput", metric="train/rl/rollout_tokens_per_second", unit="tokens/s"
        ),
        SummaryFieldDefinition(
            key="failed_rollouts", label="Failed rollouts", metric="train/rl/rollouts_failed", reducer="sum"
        ),
    ),
    charts=(
        ChartDefinition(
            key="learning_signal",
            title="Learning signal",
            question="Is reward improving while rollout groups still provide relative signal?",
            metrics=("train/rl/reward_mean", "train/rl/reward_std", "train/rl/group_zero_variance_fraction"),
        ),
        ChartDefinition(
            key="optimization",
            title="Policy optimization",
            question="Are updates controlled without collapsing exploration?",
            metrics=("train/rl/policy_loss", "train/rl/entropy", "train/rl/kl", "train/rl/clip_fraction"),
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
                "train/rl/rollouts_attempted",
                "train/rl/rollouts_completed",
                "train/rl/rollouts_failed",
                "train/rl/rollouts_truncated",
                "train/rl/rollouts_unscorable",
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
    ),
    metric_help=_help_for(
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
        "train/rl/rollouts_attempted",
        "train/rl/rollouts_completed",
        "train/rl/rollouts_failed",
        "train/rl/rollouts_truncated",
        "train/rl/rollouts_unscorable",
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
    ),
    comparison_keys=("reward_mean", "policy_loss", "rollout_tps", "failed_rollouts"),
    trace_sections=(TraceSectionDefinition(trace_type="verifiers", label="Rollouts & rewards"),),
    artifact_roles=_training_artifacts(),
    delta_tip_metrics=(
        "train/rl/reward_mean",
        "train/rl/policy_loss",
        "train/rl/rollout_tokens_per_second",
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

SAMPO_TELEMETRY = JobTelemetryDefinition(
    job_kind="train.sampo",
    display_name="Step-aware multi-turn policy optimization",
    summary_fields=(
        SummaryFieldDefinition(key="reward_mean", label="Mean reward", metric="train/rl/reward_mean", required=True),
        SummaryFieldDefinition(key="reward_std", label="Reward standard deviation", metric="train/rl/reward_std"),
        SummaryFieldDefinition(
            key="episode_advantage",
            label="Episode advantage",
            metric="train/rl/episode_advantage_mean",
            required=True,
        ),
        SummaryFieldDefinition(
            key="turn_advantage",
            label="Turn advantage",
            metric="train/rl/turn_advantage_mean",
            required=True,
        ),
        SummaryFieldDefinition(
            key="anchor_group_size",
            label="Anchor group size",
            metric="train/rl/anchor_group_size_mean",
        ),
        SummaryFieldDefinition(
            key="sparse_reward_projection",
            label="Sparse-reward projection",
            metric="train/rl/sparse_reward_projection_fraction",
            unit="ratio",
        ),
        SummaryFieldDefinition(key="policy_loss", label="Policy loss", metric="train/rl/policy_loss", required=True),
        SummaryFieldDefinition(
            key="failed_rollouts",
            label="Failed rollouts",
            metric="train/rl/rollouts_failed",
            reducer="sum",
        ),
    ),
    charts=(
        ChartDefinition(
            key="hierarchical_advantages",
            title="Hierarchical advantages",
            question="Do episode and turn credit assignments remain informative across the sampled trajectories?",
            metrics=(
                "train/rl/episode_advantage_mean",
                "train/rl/turn_advantage_mean",
                "train/rl/anchor_group_size_mean",
                "train/rl/sparse_reward_projection_fraction",
            ),
        ),
        ChartDefinition(
            key="learning_signal",
            title="Learning signal",
            question="Is verifier reward improving while groups retain enough variation for policy learning?",
            metrics=("train/rl/reward_mean", "train/rl/reward_std", "train/rl/group_zero_variance_fraction"),
        ),
        ChartDefinition(
            key="optimization",
            title="Policy optimization",
            question="Are sequence-clipped updates controlled without collapsing exploration?",
            metrics=(
                "train/rl/policy_loss",
                "train/rl/entropy",
                "train/rl/kl",
                "train/rl/clip_fraction",
                "train/grad_norm",
                "train/learning_rate",
            ),
        ),
        ChartDefinition(
            key="rollouts",
            title="Rollout population",
            question="How much requested multi-turn evidence completed, failed, truncated, or became unscorable?",
            metrics=(
                "train/rl/rollouts_attempted",
                "train/rl/rollouts_completed",
                "train/rl/rollouts_failed",
                "train/rl/rollouts_truncated",
                "train/rl/rollouts_unscorable",
            ),
        ),
        ChartDefinition(
            key="runtime",
            title="Runtime efficiency",
            question="What end-to-end step cost and effective rollout throughput were observed?",
            metrics=("train/step_time_seconds", "train/rl/rollout_tokens_per_second"),
        ),
        ChartDefinition(
            key="tool_behavior",
            title="Tool behavior",
            question="Are multi-turn trajectories invoking tools successfully?",
            metrics=("train/rl/tool_call_frequency", "train/rl/tool_failure_frequency"),
        ),
    ),
    metric_help=(
        *_help_for(
            "train/rl/reward_mean",
            "train/rl/reward_std",
            "train/rl/group_zero_variance_fraction",
            "train/rl/policy_loss",
            "train/rl/entropy",
            "train/rl/kl",
            "train/rl/clip_fraction",
            "train/grad_norm",
            "train/learning_rate",
            "train/rl/rollouts_attempted",
            "train/rl/rollouts_completed",
            "train/rl/rollouts_failed",
            "train/rl/rollouts_truncated",
            "train/rl/rollouts_unscorable",
            "train/step_time_seconds",
            "train/rl/rollout_tokens_per_second",
            "train/rl/tool_call_frequency",
            "train/rl/tool_failure_frequency",
        ),
        _metric(
            "train/rl/episode_advantage_mean",
            "Episode advantage",
            "Mean trajectory-level advantage assigned from the complete episode return.",
            "Read it with turn advantage to distinguish whole-trajectory credit from local step credit.",
        ),
        _metric(
            "train/rl/turn_advantage_mean",
            "Turn advantage",
            "Mean step-aware advantage assigned to sampled assistant turns.",
            "A useful turn-level signal should vary with consequential intermediate actions rather than only the terminal outcome.",
        ),
        _metric(
            "train/rl/anchor_group_size_mean",
            "Anchor group size",
            "Mean number of comparable turns contributing to each step-aware advantage anchor.",
            "Small groups provide weak relative evidence; compare this value across runs using the same sampling policy.",
        ),
        _metric(
            "train/rl/sparse_reward_projection_fraction",
            "Sparse-reward projection",
            "Fraction of sampled trajectories whose terminal reward was projected back across intermediate turns.",
            "A high value means SAMPO is relying heavily on its sparse-reward credit-assignment path.",
            unit="ratio",
        ),
    ),
    health_rules=(
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
    ),
    comparison_keys=(
        "reward_mean",
        "episode_advantage",
        "turn_advantage",
        "policy_loss",
        "failed_rollouts",
    ),
    trace_sections=(TraceSectionDefinition(trace_type="verifiers", label="Multi-turn rollouts"),),
    artifact_roles=_training_artifacts(),
    delta_tip_metrics=(
        "train/rl/reward_mean",
        "train/rl/episode_advantage_mean",
        "train/rl/turn_advantage_mean",
        "train/rl/policy_loss",
        "train/rl/rollout_tokens_per_second",
    ),
    evidence_requirements=(
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
    ),
)

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
            "eval/trace_sync_complete",
        ),
        health_rules=(
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
        _metric(
            "serve/probe_healthy",
            "Endpoint healthy",
            "Whether the managed endpoint returned a successful health response.",
            "A value of 100% confirms the endpoint process answered its health check.",
            unit="ratio",
        ),
        _metric(
            "serve/probe_model_available",
            "Model available",
            "Whether model discovery exposed the exact model selected by the inference binding.",
            "A healthy server is not usable for this job unless the selected model is also available.",
            unit="ratio",
        ),
        _metric(
            "serve/probe_latency_seconds",
            "Probe latency",
            "Wall-clock time for the combined health and model-discovery probe.",
            "Use this as startup qualification evidence, not as inference latency.",
            unit="s",
        ),
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
        _metric(
            "data/examples",
            "Prepared examples",
            "Number of examples materialized into the immutable prepared dataset.",
            "Use this count to verify that the packaged dataset population matches the selected source.",
        ),
        _metric(
            "data/bytes",
            "Prepared bytes",
            "Byte size of the immutable prepared dataset content.",
            "Use it with example count and the retained dataset artifact to detect unexpected packaging changes.",
            unit="bytes",
        ),
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
        _metric(
            "serve/run/output_tokens_measured",
            "Measured output tokens",
            "Successfully generated output tokens in the finalized benchmark point.",
            "Observatory divides this counter by the matching measurement duration to calculate aggregate throughput.",
            unit="tokens",
        ),
        _metric(
            "serve/run/measurement_duration_s",
            "Measurement duration",
            "Wall-clock duration of the finalized measured inference window.",
            "This is the denominator for aggregate output-token throughput.",
            unit="s",
        ),
        _metric(
            "serve/run/requests_measured",
            "Measured requests",
            "Number of non-warmup requests successfully retained for the point.",
            "Decision-grade latency requires the same complete request population in inference traces.",
        ),
        _metric(
            "serve/run/concurrency",
            "Concurrency",
            "Number of requests allowed to execute concurrently at this measured operating point.",
            "Concurrency is a search variable used to characterize the fixed hardware target, not a product constraint.",
        ),
        _metric(
            "serve/run/context_tokens",
            "Context allocation",
            "Maximum context allocation configured for the serving runtime.",
            "It must be at least the context window required by the product.",
            unit="tokens",
        ),
        _metric(
            "serve/backend/peak_vram_bytes",
            "Peak GPU memory",
            "Highest observed GPU memory allocation during the benchmark.",
            "Use it to understand target headroom; it is not a substitute for KV-cache capacity evidence.",
            unit="bytes",
        ),
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
    "DATA_PREPARE_TELEMETRY",
    "DEFAULT_TELEMETRY_DEFINITIONS",
    "DISTILL_TELEMETRY",
    "DOMAIN_EVAL_TELEMETRY",
    "DPO_TELEMETRY",
    "EvidenceCondition",
    "EvidenceRequirementDefinition",
    "GENERAL_EVAL_TELEMETRY",
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
