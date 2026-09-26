"""The framework semantic model: entities, dimensions, measures and metrics per job kind.

Each measure names the one place its values come from, so `entropy` means
`train/rl/entropy` here and nowhere else. Setting sources are paths into a run's
resolved inputs; several paths separated by `|` are tried in order.
"""

from __future__ import annotations

from .model import ROLLOUT_AGGREGATIONS, Dimension, Entity, Measure, Metric, SemanticModel, Source

GROUP_POLICY_KINDS = ("train.grpo", "train.gdpo", "train.capo")
RL_KINDS = (*GROUP_POLICY_KINDS, "train.sampo")
SUPERVISED_KINDS = ("train.sft", "train.dpo")
TRAIN_KINDS = (*RL_KINDS, *SUPERVISED_KINDS, "train.distill")
EVAL_KINDS = ("eval.general", "eval.domain")
ROLLOUT_KINDS = (*RL_KINDS, *EVAL_KINDS)
SERVE_KINDS = ("serve.benchmark",)


def _field(name: str) -> Source:
    return Source(kind="run_field", name=name)


def _setting(path: str) -> Source:
    return Source(kind="setting", name=path)


def _series(name: str, *, transform: str = "identity", within_step: str = "last") -> Source:
    return Source(kind="metric_series", name=name, transform=transform, within_step=within_step)  # type: ignore[arg-type]


# Rollout evidence recorded once per active-sampling generation round.
_ROUND_SUMS = frozenset(
    {
        "train/rl/time/rollout_seconds",
        "train/rl/rollouts_requested",
        "train/rl/rollouts_attempted",
        "train/rl/rollouts_completed",
        "train/rl/rollouts_failed",
        "train/rl/rollouts_truncated",
        "train/rl/rollouts_unscorable",
        "train/rl/rollout_selected_tokens",
    }
)
_ROUND_MEANS = frozenset(
    {
        "train/rl/reward_std",
        "train/rl/group_zero_variance_fraction",
        "train/rl/rollout_tokens_per_second",
        "train/rl/tool_call_frequency",
        "train/rl/tool_failure_frequency",
        "train/rl/episode_advantage_mean",
        "train/rl/turn_advantage_mean",
        "train/rl/anchor_group_size_mean",
        "train/rl/sparse_reward_projection_fraction",
    }
)


def _within_step(metric: str) -> str:
    return "sum" if metric in _ROUND_SUMS else "mean" if metric in _ROUND_MEANS else "last"


def _update(
    name: str,
    label: str,
    description: str,
    metric: str,
    kinds: tuple[str, ...],
    *,
    unit: str | None = None,
    aggregation: str = "mean",
    transform: str = "identity",
    within_step: str | None = None,
) -> Measure:
    return Measure(
        name=name,
        entity="update",
        label=label,
        description=description,
        unit=unit,
        source=_series(metric, transform=transform, within_step=within_step or _within_step(metric)),
        aggregation=aggregation,  # type: ignore[arg-type]
        job_kinds=kinds,
    )


def _rollout(
    name: str, label: str, description: str, fact: str, *, unit: str | None = None, aggregation: str = "mean"
) -> Measure:
    return Measure(
        name=name,
        entity="rollout",
        label=label,
        description=description,
        unit=unit,
        source=Source(kind="trace_fact", name=fact),
        aggregation=aggregation,  # type: ignore[arg-type]
        allowed=ROLLOUT_AGGREGATIONS,
        job_kinds=ROLLOUT_KINDS,
    )


ENTITIES = (
    Entity(name="run", description="One run of one job."),
    Entity(
        name="update",
        description="One optimizer update of a training run, at a logged step.",
        order_dimension="update.step",
    ),
    Entity(
        name="rollout",
        description=(
            "One rollout episode recorded as a trace. Rollouts are aggregated by the tracking backend and "
            "never downloaded, so only count, sum, mean and stddev are available."
        ),
    ),
    Entity(name="eval_task", description="One task of an evaluation run, over its repetitions."),
    Entity(
        name="load_level",
        description="One concurrency level of a serving benchmark run.",
        order_dimension="load_level.concurrency",
    ),
)

DIMENSIONS = (
    Dimension(name="run.id", entity="run", type="string", description="Run id.", source=_field("run_id")),
    Dimension(name="run.name", entity="run", type="string", description="Display name.", source=_field("display_name")),
    Dimension(name="run.project", entity="run", type="string", description="Project id.", source=_field("project_id")),
    Dimension(
        name="run.work_package",
        entity="run",
        type="string",
        description="Work package id.",
        source=_field("work_package_id"),
    ),
    Dimension(
        name="run.job_kind",
        entity="run",
        type="string",
        description="Job kind, for example train.sampo.",
        source=_field("job_kind"),
    ),
    Dimension(name="run.stage", entity="run", type="string", description="Workflow stage.", source=_field("stage")),
    Dimension(
        name="run.status",
        entity="run",
        type="string",
        description="running, succeeded, failed or cancelled.",
        source=_field("status"),
    ),
    Dimension(
        name="run.started_at", entity="run", type="time", description="Start time (UTC).", source=_field("started_at")
    ),
    Dimension(
        name="run.finished_at",
        entity="run",
        type="time",
        description="Finish time (UTC).",
        source=_field("finished_at"),
    ),
    Dimension(
        name="run.error", entity="run", type="string", description="Error type of a failed run.", source=_field("error")
    ),
    Dimension(
        name="run.model",
        entity="run",
        type="string",
        description="Model selection id.",
        source=_setting("model.selection_id|model.id"),
    ),
    Dimension(
        name="run.parent_run",
        entity="run",
        type="string",
        description="Run whose checkpoint this run started from or evaluated.",
        source=_setting("model_source.source_run_id"),
    ),
    Dimension(
        name="run.parent_step",
        entity="run",
        type="integer",
        description="Checkpoint step of the parent run.",
        source=_setting("model_source.checkpoint_step"),
    ),
    Dimension(
        name="run.environment",
        entity="run",
        type="string",
        description="Environment selection id.",
        source=_setting("environment.selection_id"),
    ),
    Dimension(
        name="run.settings",
        entity="run",
        type="string",
        description="Settings selection id.",
        source=_setting("settings.selection_id"),
    ),
    Dimension(
        name="run.training_binding",
        entity="run",
        type="string",
        description="Training binding selection id.",
        source=_setting("training.selection_id"),
    ),
    Dimension(
        name="run.inference_binding",
        entity="run",
        type="string",
        description="Rollout or evaluation inference binding selection id.",
        source=_setting("rollout_inference.selection_id|evaluation_inference.selection_id|inference.selection_id"),
    ),
    Dimension(
        name="run.algorithm",
        entity="run",
        type="string",
        description="Policy-optimization recipe recorded in settings (for example olmo3).",
        source=_setting("settings.resolved.algorithm"),
    ),
    Dimension(
        name="run.learning_rate",
        entity="run",
        type="number",
        description="Configured learning rate.",
        source=_setting("settings.resolved.learning_rate"),
    ),
    Dimension(
        name="run.kl_beta",
        entity="run",
        type="number",
        description="Configured KL penalty.",
        source=_setting("settings.resolved.beta"),
    ),
    Dimension(
        name="run.max_updates",
        entity="run",
        type="integer",
        description="Configured number of updates (max_steps).",
        source=_setting("settings.resolved.max_steps"),
    ),
    Dimension(
        name="run.prompts_per_update",
        entity="run",
        type="integer",
        description="Prompt groups per update.",
        source=_setting("settings.resolved.num_prompts_per_step"),
    ),
    Dimension(
        name="run.rollouts_per_prompt",
        entity="run",
        type="integer",
        description="Rollouts per prompt group.",
        source=_setting("settings.resolved.num_generations"),
    ),
    Dimension(
        name="run.max_length",
        entity="run",
        type="integer",
        description="Configured maximum sequence length in tokens.",
        source=_setting("settings.resolved.max_length"),
    ),
    Dimension(
        name="run.max_completion_length",
        entity="run",
        type="integer",
        description="Configured maximum tokens per model reply.",
        source=_setting(
            "settings.resolved.max_completion_length|evaluation_inference.resolved.sampling.max_tokens"
            "|environment.resolved.sampling.max_tokens"
        ),
    ),
    Dimension(
        name="run.context_tokens",
        entity="run",
        type="integer",
        description="Context length the inference engine served (max_model_len).",
        source=_setting(
            "rollout_inference.resolved.engine.max_model_len|evaluation_inference.resolved.engine.max_model_len"
            "|inference.resolved.engine.max_model_len"
        ),
    ),
    Dimension(
        name="run.update_kind",
        entity="run",
        type="string",
        description="Parameter update kind (lora, qlora, full).",
        source=_setting("training.resolved.parameter_update.kind"),
    ),
    Dimension(
        name="run.lora_rank",
        entity="run",
        type="integer",
        description="LoRA rank.",
        source=_setting("training.resolved.parameter_update.rank"),
    ),
    Dimension(
        name="update.step",
        entity="update",
        type="integer",
        description="Logged step (optimizer update).",
        source=Source(kind="derived", name="step"),
    ),
    Dimension(
        name="update.time",
        entity="update",
        type="number",
        description="Seconds since the run started, when the update was logged.",
        source=Source(kind="derived", name="elapsed_seconds"),
    ),
    Dimension(
        name="rollout.task",
        entity="rollout",
        type="string",
        description="Task id.",
        source=Source(kind="trace_fact", name="task_id"),
    ),
    Dimension(
        name="rollout.task_type",
        entity="rollout",
        type="string",
        description="Task type.",
        source=Source(kind="trace_fact", name="task_type"),
    ),
    Dimension(
        name="rollout.prompt_group",
        entity="rollout",
        type="string",
        description="Prompt group id (rollouts of one prompt in one update).",
        source=Source(kind="trace_fact", name="prompt_group_id"),
    ),
    Dimension(
        name="rollout.step",
        entity="rollout",
        type="integer",
        description="Update step the rollout was collected for.",
        source=Source(kind="trace_fact", name="rollout_step"),
    ),
    Dimension(
        name="rollout.truncated",
        entity="rollout",
        type="boolean",
        description="The rollout hit a turn, output or context budget.",
        source=Source(kind="trace_fact", name="is_truncated"),
    ),
    Dimension(
        name="rollout.failed",
        entity="rollout",
        type="boolean",
        description="The rollout failed execution.",
        source=Source(kind="trace_fact", name="has_error"),
    ),
    Dimension(
        name="rollout.model",
        entity="rollout",
        type="string",
        description="Served model name.",
        source=Source(kind="trace_fact", name="model"),
    ),
    Dimension(
        name="eval_task.task",
        entity="eval_task",
        type="string",
        description="Evaluation task id (stable across runs of the same task set).",
        source=Source(kind="eval_task", name="key"),
    ),
    Dimension(
        name="eval_task.label",
        entity="eval_task",
        type="string",
        description="Evaluation task display label.",
        source=Source(kind="eval_task", name="label"),
    ),
    Dimension(
        name="load_level.concurrency",
        entity="load_level",
        type="integer",
        description="Concurrent requests.",
        source=Source(kind="load_level", name="concurrency"),
    ),
    Dimension(
        name="load_level.context_tokens",
        entity="load_level",
        type="integer",
        description="Context length of the workload.",
        source=Source(kind="load_level", name="context_tokens"),
    ),
)

MEASURES = (
    Measure(
        name="runs",
        entity="run",
        label="Runs",
        description="Number of runs.",
        source=Source(kind="derived", name="runs"),
        aggregation="count",
        allowed=("count",),
        job_kinds=("*",),
    ),
    Measure(
        name="duration_seconds",
        entity="run",
        label="Run duration",
        description="Seconds from start to finish (or to now for a running run).",
        unit="s",
        source=Source(kind="derived", name="duration_seconds"),
        aggregation="sum",
        job_kinds=("*",),
    ),
    _update(
        "update_seconds", "Update time", "Wall time of one update.", "train/step_time_seconds", TRAIN_KINDS, unit="s"
    ),
    _update("loss", "Loss", "Training loss.", "train/loss", (*SUPERVISED_KINDS, *RL_KINDS)),
    _update("grad_norm", "Gradient norm", "Gradient norm before clipping.", "train/grad_norm", TRAIN_KINDS),
    _update(
        "logged_learning_rate",
        "Learning rate (logged)",
        "Learning rate the scheduler applied.",
        "train/learning_rate",
        TRAIN_KINDS,
    ),
    _update("reward", "Reward", "Mean episode reward of the update's rollouts.", "train/rl/reward_mean", RL_KINDS),
    _update("reward_std", "Reward spread", "Standard deviation of episode reward.", "train/rl/reward_std", RL_KINDS),
    _update("entropy", "Entropy", "Mean token entropy of the policy on sampled tokens.", "train/rl/entropy", RL_KINDS),
    _update("kl", "KL to reference", "KL divergence from the reference policy.", "train/rl/kl", RL_KINDS),
    _update("policy_loss", "Policy loss", "Policy-gradient loss.", "train/rl/policy_loss", RL_KINDS),
    _update(
        "clip_fraction", "Clipped share", "Share of tokens whose ratio was clipped.", "train/rl/clip_fraction", RL_KINDS
    ),
    _update(
        "rollout_seconds",
        "Rollout time",
        "Time spent collecting rollouts.",
        "train/rl/time/rollout_seconds",
        RL_KINDS,
        unit="s",
    ),
    _update(
        "actor_seconds",
        "Actor update time",
        "Time spent in the policy update.",
        "train/rl/time/actor_update_seconds",
        RL_KINDS,
        unit="s",
    ),
    _update(
        "truncation_rate",
        "Truncation rate",
        "Share of rollouts cut off by a turn, reply or context budget.",
        "train/rl/completion_truncation_rate",
        RL_KINDS,
    ),
    _update(
        "zero_spread_share",
        "Zero-spread groups",
        "Share of prompt groups whose rollouts all scored the same (no learning signal).",
        "train/rl/group_zero_variance_fraction",
        RL_KINDS,
    ),
    _update(
        "rollouts_attempted",
        "Rollouts attempted",
        "Rollouts generated for the update, over all active-sampling rounds.",
        "train/rl/rollouts_attempted",
        RL_KINDS,
        aggregation="sum",
    ),
    _update(
        "rollouts_completed",
        "Rollouts completed",
        "Rollouts completed in the update.",
        "train/rl/rollouts_completed",
        RL_KINDS,
        aggregation="sum",
    ),
    _update(
        "rollouts_failed",
        "Rollouts failed",
        "Rollouts that failed in the update.",
        "train/rl/rollouts_failed",
        RL_KINDS,
        aggregation="sum",
    ),
    _update(
        "rollouts_truncated",
        "Rollouts truncated",
        "Rollouts truncated in the update.",
        "train/rl/rollouts_truncated",
        RL_KINDS,
        aggregation="sum",
    ),
    _update(
        "rollout_tokens_per_second",
        "Rollout throughput",
        "Generated tokens per second during collection.",
        "train/rl/rollout_tokens_per_second",
        RL_KINDS,
        unit="tokens/s",
    ),
    _update(
        "tool_call_rate",
        "Tool calls per turn",
        "Share of turns that called a tool.",
        "train/rl/tool_call_frequency",
        RL_KINDS,
    ),
    _update(
        "tool_failure_rate",
        "Tool failure rate",
        "Share of tool calls that failed.",
        "train/rl/tool_failure_frequency",
        RL_KINDS,
    ),
    _update(
        "generated_rows",
        "Rows generated",
        "Rollouts generated by active sampling to fill the batch (refills included).",
        "train/rl/active_sampling_generated_rows",
        RL_KINDS,
        aggregation="sum",
    ),
    _update(
        "generation_rounds",
        "Refill rounds",
        "Active-sampling generation rounds in the update.",
        "train/rl/active_sampling_generation_rounds",
        RL_KINDS,
    ),
    _update(
        "correction_ratio",
        "Sampler correction ratio",
        "Mean vLLM importance-sampling ratio between the sampler and the trainer.",
        "train/rl/importance_sampling_ratio_mean",
        RL_KINDS,
    ),
    _update(
        "correction_clamp_share",
        "Correction clamped",
        "Share of tokens whose sampler correction was clamped.",
        "train/rl/importance_sampling_ratio_clamped_fraction",
        RL_KINDS,
    ),
    _update(
        "speculative_acceptance",
        "Draft acceptance",
        "Share of speculative draft tokens accepted.",
        "serve/backend/speculative_acceptance_rate",
        RL_KINDS,
    ),
    _update(
        "kv_cache_peak",
        "KV cache peak use",
        "Peak share of the rollout engine's KV cache in use.",
        "serve/backend/kv_cache_peak_usage_ratio",
        RL_KINDS,
    ),
    _update(
        "episode_advantage",
        "Episode advantage",
        "Mean SAMPO episode advantage.",
        "train/rl/episode_advantage_mean",
        ("train.sampo",),
    ),
    _update(
        "turn_advantage",
        "Turn advantage",
        "Mean SAMPO turn advantage.",
        "train/rl/turn_advantage_mean",
        ("train.sampo",),
    ),
    _update(
        "anchor_group_size",
        "Anchor group size",
        "Rollouts sharing a turn's anchor state, on average.",
        "train/rl/anchor_group_size_mean",
        ("train.sampo",),
    ),
    _update(
        "step_reward_share",
        "Step-reward share",
        "Share of rollouts that carried per-turn rewards (not only the final reward).",
        "train/rl/sparse_reward_projection_fraction",
        ("train.sampo",),
        transform="one_minus",
    ),
    _update(
        "token_accuracy",
        "Token accuracy",
        "Next-token accuracy on supervised tokens.",
        "train/mean_token_accuracy",
        SUPERVISED_KINDS,
    ),
    _update(
        "tokens_per_second",
        "Training throughput",
        "Non-padding tokens trained per second.",
        "train/non_padding_tokens_per_second",
        SUPERVISED_KINDS,
        unit="tokens/s",
    ),
    _update(
        "validation_loss", "Validation loss", "Loss on the validation split.", "train/validation/loss", SUPERVISED_KINDS
    ),
    _update(
        "preference_accuracy",
        "Preference accuracy",
        "Share of pairs ranked correctly.",
        "train/rewards/accuracies",
        ("train.dpo",),
    ),
    _update(
        "reward_margin",
        "Reward margin",
        "Chosen minus rejected implicit reward.",
        "train/rewards/margins",
        ("train.dpo",),
    ),
    _update("distill_loss", "Distillation loss", "Distillation loss.", "train/distill/loss", ("train.distill",)),
    _update("reverse_kl", "Reverse KL", "Reverse KL to the teacher.", "train/distill/reverse_kl", ("train.distill",)),
    _rollout("rollouts", "Rollouts", "Number of rollouts.", "trace_count", aggregation="sum"),
    _rollout("rollout_reward", "Rollout reward", "Task reward of a rollout.", "task_reward"),
    _rollout("algorithm_reward", "Algorithm reward", "Reward after the algorithm's shaping.", "algorithm_reward"),
    _rollout(
        "input_tokens",
        "Input tokens",
        "Prompt tokens over all model calls of a rollout.",
        "model_input_tokens",
        unit="tokens",
    ),
    _rollout(
        "output_tokens",
        "Output tokens",
        "Generated tokens over all model calls of a rollout.",
        "model_output_tokens",
        unit="tokens",
    ),
    _rollout(
        "thinking_tokens",
        "Thinking tokens",
        "Reasoning tokens over all model calls of a rollout.",
        "thinking_tokens",
        unit="tokens",
    ),
    _rollout("tool_calls", "Tool calls", "Tool calls in a rollout.", "tool_calls"),
    _rollout("model_calls", "Model calls", "Model calls (turns) in a rollout.", "model_calls"),
    _rollout(
        "rollout_latency_ms", "Rollout latency", "Summed model latency of a rollout.", "trace_latency_ms", unit="ms"
    ),
    *(
        Measure(
            name=name,
            entity="eval_task",
            label=label,
            description=description,
            source=Source(kind="eval_task", name=field),
            aggregation=aggregation,  # type: ignore[arg-type]
            job_kinds=EVAL_KINDS,
        )
        for name, label, description, field, aggregation in (
            ("task_reward", "Task reward", "Mean reward over the task's valid repetitions.", "mean_reward", "mean"),
            (
                "success_rate",
                "Success rate",
                "Share of the task's valid repetitions that succeeded.",
                "success_frequency",
                "mean",
            ),
            ("valid_repetitions", "Valid repetitions", "Repetitions with a valid result.", "valid_repetitions", "sum"),
            (
                "execution_failures",
                "Execution failures",
                "Repetitions that failed execution.",
                "execution_failures",
                "sum",
            ),
            ("eval_truncations", "Truncations", "Repetitions that hit a budget.", "truncations", "sum"),
        )
    ),
    *(
        Measure(
            name=name,
            entity="load_level",
            label=label,
            description=description,
            unit=unit,
            source=Source(kind="load_level", name=field),
            aggregation=aggregation,  # type: ignore[arg-type]
            job_kinds=SERVE_KINDS,
        )
        for name, label, description, field, unit, aggregation in (
            (
                "throughput",
                "Output throughput",
                "Aggregate output tokens per second.",
                "aggregate_output_tps",
                "tokens/s",
                "mean",
            ),
            ("failure_rate", "Failure rate", "Share of requests that failed.", "failure_rate", None, "mean"),
            ("completed_requests", "Completed requests", "Requests completed.", "completed_requests", None, "sum"),
            ("failed_requests", "Failed requests", "Requests failed.", "failed_requests", None, "sum"),
            ("ttft_p50_ms", "TTFT p50", "Median time to first token.", "p50_ttft_ms", "ms", "mean"),
            ("ttft_p95_ms", "TTFT p95", "95th-percentile time to first token.", "p95_ttft_ms", "ms", "mean"),
            ("tpot_p50_ms", "TPOT p50", "Median time per output token.", "p50_tpot_ms", "ms", "mean"),
            ("tpot_p95_ms", "TPOT p95", "95th-percentile time per output token.", "p95_tpot_ms", "ms", "mean"),
            (
                "output_tokens_mean",
                "Output tokens (mean)",
                "Mean output tokens per request.",
                "output_tokens_mean",
                "tokens",
                "mean",
            ),
            ("peak_vram_bytes", "Peak VRAM", "Peak GPU memory.", "peak_vram_bytes", "bytes", "max"),
        )
    ),
)


def _run_metric(
    name: str,
    label: str,
    description: str,
    metric: str,
    kinds: tuple[str, ...],
    *,
    unit: str | None = None,
    aggregation: str = "sum",
) -> Measure:
    """A number logged once per run; a run's value is its last logged point."""
    return Measure(
        name=name,
        entity="run",
        label=label,
        description=description,
        unit=unit,
        source=_series(metric),
        aggregation=aggregation,  # type: ignore[arg-type]
        job_kinds=kinds,
    )


RUN_METRIC_MEASURES = (
    _run_metric(
        "eval_rollouts_attempted",
        "Rollouts attempted",
        "Evaluation rollouts attempted.",
        "eval/run/rollouts_attempted",
        EVAL_KINDS,
    ),
    _run_metric(
        "eval_rollouts_complete",
        "Rollouts complete",
        "Evaluation rollouts that completed.",
        "eval/run/rollouts_complete",
        EVAL_KINDS,
    ),
    _run_metric(
        "eval_rollouts_failed",
        "Rollouts failed",
        "Evaluation rollouts that failed execution.",
        "eval/run/rollouts_failed",
        EVAL_KINDS,
    ),
    _run_metric(
        "eval_rollouts_truncated",
        "Rollouts truncated",
        "Evaluation rollouts that hit a budget.",
        "eval/run/rollouts_truncated",
        EVAL_KINDS,
    ),
    _run_metric(
        "eval_context_overflow_rollouts",
        "Context overflows",
        "Rollouts whose model request exceeded the context.",
        "eval/run/context_overflow_rollouts",
        EVAL_KINDS,
    ),
    _run_metric(
        "eval_model_call_error_rollouts",
        "Model-call errors",
        "Rollouts with a model-call error.",
        "eval/run/model_call_error_rollouts",
        EVAL_KINDS,
    ),
    _run_metric(
        "eval_http_400_rollouts",
        "HTTP 400 rollouts",
        "Rollouts with an HTTP 400 model-call error.",
        "eval/run/model_call_http_400_rollouts",
        EVAL_KINDS,
    ),
    _run_metric(
        "eval_trace_sync_complete",
        "Trace sync complete",
        "1 when every trace synchronized to tracking.",
        "eval/trace_sync_complete",
        EVAL_KINDS,
        aggregation="min",
    ),
    _run_metric(
        "serve_peak_vram_bytes",
        "Peak VRAM",
        "Peak GPU memory of the serving run.",
        "serve/backend/peak_vram_bytes",
        SERVE_KINDS,
        unit="bytes",
        aggregation="max",
    ),
    _run_metric(
        "serve_concurrency",
        "Concurrency",
        "Concurrency of a single-point benchmark.",
        "serve/run/concurrency",
        SERVE_KINDS,
        aggregation="max",
    ),
    _run_metric(
        "serve_context_tokens",
        "Context tokens",
        "Context length of a single-point benchmark.",
        "serve/run/context_tokens",
        SERVE_KINDS,
        aggregation="max",
    ),
    _run_metric(
        "serve_measurement_seconds",
        "Measurement time",
        "Measured benchmark duration.",
        "serve/run/measurement_duration_s",
        SERVE_KINDS,
        unit="s",
    ),
    _run_metric(
        "serve_output_tokens",
        "Output tokens measured",
        "Output tokens in the measurement.",
        "serve/run/output_tokens_measured",
        SERVE_KINDS,
        unit="tokens",
    ),
    _run_metric(
        "serve_requests",
        "Requests measured",
        "Requests in the measurement.",
        "serve/run/requests_measured",
        SERVE_KINDS,
    ),
    _run_metric(
        "probe_healthy",
        "Probe healthy",
        "1 when the serving smoke probe was healthy.",
        "serve/probe_healthy",
        ("serve.smoke",),
        aggregation="min",
    ),
    _run_metric(
        "probe_latency_seconds",
        "Probe latency",
        "Smoke probe latency.",
        "serve/probe_latency_seconds",
        ("serve.smoke",),
        unit="s",
        aggregation="max",
    ),
    _run_metric(
        "probe_model_available",
        "Model available",
        "1 when the served model was listed.",
        "serve/probe_model_available",
        ("serve.smoke",),
        aggregation="min",
    ),
    _run_metric("data_bytes", "Data bytes", "Bytes of prepared data.", "data/bytes", ("data.prepare",), unit="bytes"),
    _run_metric("data_examples", "Examples", "Prepared examples.", "data/examples", ("data.prepare",)),
)

EXTRA_UPDATE_MEASURES = (
    _update(
        "retained_share",
        "Groups retained",
        "Share of candidate groups active sampling kept.",
        "train/rl/active_sampling_retained_fraction",
        RL_KINDS,
    ),
    _update(
        "clip_fraction_high",
        "Clipped high",
        "Share of tokens clipped at the upper bound.",
        "train/rl/clip_fraction_high",
        RL_KINDS,
    ),
    _update(
        "clip_fraction_low",
        "Clipped low",
        "Share of tokens clipped at the lower bound.",
        "train/rl/clip_fraction_low",
        RL_KINDS,
    ),
    _update(
        "dynamic_candidate_batches",
        "Dynamic-sampling batches",
        "Candidate batches dynamic sampling generated.",
        "train/rl/dynamic_sampling_candidate_batches",
        GROUP_POLICY_KINDS,
    ),
    _update(
        "dynamic_retained_share",
        "Dynamic-sampling retained",
        "Share of rows dynamic sampling kept.",
        "train/rl/dynamic_sampling_retained_fraction",
        GROUP_POLICY_KINDS,
    ),
    _update(
        "length_utilization",
        "Length utilization",
        "Share of the maximum length used.",
        "train/data/max_length_utilization",
        SUPERVISED_KINDS,
    ),
    _update(
        "supervision_share",
        "Supervised tokens",
        "Share of tokens that carry loss.",
        "train/data/supervision_token_ratio",
        ("train.sft",),
    ),
    _update(
        "data_truncation_rate",
        "Example truncation",
        "Share of examples truncated to the maximum length.",
        "train/data/truncation_rate",
        ("train.sft",),
    ),
    _update("dpo_entropy", "Entropy", "Policy entropy during preference training.", "train/entropy", ("train.dpo",)),
    _update(
        "chosen_logps", "Chosen log-prob", "Log-probability of chosen responses.", "train/logps/chosen", ("train.dpo",)
    ),
    _update(
        "rejected_logps",
        "Rejected log-prob",
        "Log-probability of rejected responses.",
        "train/logps/rejected",
        ("train.dpo",),
    ),
    _update(
        "chosen_reward", "Chosen reward", "Implicit reward of chosen responses.", "train/rewards/chosen", ("train.dpo",)
    ),
    _update(
        "rejected_reward",
        "Rejected reward",
        "Implicit reward of rejected responses.",
        "train/rewards/rejected",
        ("train.dpo",),
    ),
    *(
        _update(name, label, label + ".", metric, ("train.dpo",))
        for name, label, metric in (
            ("chosen_longer_share", "Chosen longer than rejected", "train/data/chosen_longer_fraction"),
            ("chosen_tokens_mean", "Chosen tokens (mean)", "train/data/chosen_tokens_mean"),
            ("chosen_tokens_p95", "Chosen tokens (p95)", "train/data/chosen_tokens_p95"),
            ("length_headroom_min", "Minimum length headroom", "train/data/max_length_headroom_min"),
            ("preference_pairs", "Preference pairs", "train/data/preference_pairs"),
            ("preference_score_coverage", "Preference score coverage", "train/data/preference_score_coverage"),
            ("preference_score_margin", "Preference score margin", "train/data/preference_score_margin_mean"),
            ("prompt_tokens_mean", "Prompt tokens (mean)", "train/data/prompt_tokens_mean"),
            ("prompt_tokens_p95", "Prompt tokens (p95)", "train/data/prompt_tokens_p95"),
            ("rejected_tokens_mean", "Rejected tokens (mean)", "train/data/rejected_tokens_mean"),
            ("rejected_tokens_p95", "Rejected tokens (p95)", "train/data/rejected_tokens_p95"),
        )
    ),
    _update(
        "scored_tokens",
        "Scored tokens",
        "Tokens the teacher scored.",
        "train/distill/scored_tokens",
        ("train.distill",),
        aggregation="sum",
    ),
    _update(
        "teacher_failures",
        "Teacher failures",
        "Teacher scoring failures.",
        "train/distill/teacher_failures",
        ("train.distill",),
        aggregation="sum",
    ),
    _update(
        "teacher_latency_ms",
        "Teacher latency",
        "Teacher scoring latency.",
        "train/distill/teacher_latency_ms",
        ("train.distill",),
        unit="ms",
    ),
)

METRICS = (
    Metric(
        name="rollout_share",
        entity="update",
        label="Rollout share of update time",
        description="Share of update wall time spent collecting rollouts.",
        formula="sum(rollout_seconds) / sum(update_seconds)",
    ),
    Metric(
        name="actor_share",
        entity="update",
        label="Actor share of update time",
        description="Share of update wall time spent in the policy update.",
        formula="sum(actor_seconds) / sum(update_seconds)",
    ),
    Metric(
        name="rows_per_update",
        entity="update",
        label="Rows generated per update",
        description="Rollouts generated per update, including refills.",
        formula="sum(generated_rows) / count(update_seconds)",
    ),
)

FRAMEWORK_MODEL = SemanticModel(
    entities=ENTITIES,
    dimensions=DIMENSIONS,
    measures=(*MEASURES, *RUN_METRIC_MEASURES, *EXTRA_UPDATE_MEASURES),
    metrics=METRICS,
)

__all__ = ["EVAL_KINDS", "FRAMEWORK_MODEL", "RL_KINDS", "ROLLOUT_KINDS", "SERVE_KINDS", "TRAIN_KINDS"]
