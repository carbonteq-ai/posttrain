"""The framework semantic model: entities, dimensions, measures and metrics per job kind.

Each measure names the one place its values come from. Metric measures are
built from the metric catalog (`metric_catalog.py`), so `entropy` means
`train/rl/entropy` there and nowhere else. Setting sources are paths into a run's
resolved inputs; several paths separated by `|` are tried in order.
"""

from __future__ import annotations

from posttrain.common import EPISODE_ENDING_ATTRIBUTE, EPISODE_ENDING_DESCRIPTIONS

from ..metric_catalog import METRIC_CATALOG, MetricEntry
from .model import AGGREGATIONS, Dimension, Entity, Measure, Metric, SemanticModel, Source

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


def _series(name: str, *, transform: str = "identity") -> Source:
    return Source(kind="metric_series", name=name, transform=transform)  # type: ignore[arg-type]


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
            "One rollout episode recorded as a trace, with its facts (task, reward, tokens, truncation). "
            "Evaluation tasks are rollouts grouped by task."
        ),
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
        name="run.failed_phase",
        entity="run",
        type="string",
        description="Runtime phase that failed first (for example rollout or actor_update).",
        source=Source(kind="event", name="runtime_phase_failed:phase"),
    ),
    Dimension(
        name="run.failed_step",
        entity="run",
        type="integer",
        description="Update during which the first phase failed.",
        source=Source(kind="event", name="runtime_phase_failed:logical_step"),
    ),
    Dimension(
        name="run.error_message",
        entity="run",
        type="string",
        description="Safe error message of a failed run.",
        source=_field("error_message"),
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
        name="rollout.ending",
        entity="rollout",
        type="string",
        description=(
            "How the rollout ended: "
            + "; ".join(f"{name} ({meaning})" for name, meaning in EPISODE_ENDING_DESCRIPTIONS.items())
            + ". Every ending but completed and error is truncated; empty for traces recorded before the label."
        ),
        source=Source(kind="trace_fact", name="episode_ending", fallback_attribute=EPISODE_ENDING_ATTRIBUTE),
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


def _catalog_measure(entry: MetricEntry) -> Measure:
    return Measure(
        name=entry.name,
        entity=entry.entity,  # type: ignore[arg-type]
        label=entry.label,
        description=entry.description,
        unit=entry.unit,
        source=_series(entry.metric, transform=entry.transform),
        aggregation=entry.aggregation,
        allowed=entry.allowed or AGGREGATIONS,
        job_kinds=entry.job_kinds,
    )


# Metric measures come from the metric catalog, which job views share.
CATALOG_MEASURES = tuple(_catalog_measure(entry) for entry in METRIC_CATALOG if entry.entity is not None)

FRAMEWORK_MODEL = SemanticModel(
    entities=ENTITIES,
    dimensions=DIMENSIONS,
    measures=(*MEASURES, *CATALOG_MEASURES),
    metrics=METRICS,
)

__all__ = ["EVAL_KINDS", "FRAMEWORK_MODEL", "RL_KINDS", "ROLLOUT_KINDS", "SERVE_KINDS", "TRAIN_KINDS"]
