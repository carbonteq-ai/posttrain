"""Isolated veRL launcher for backend-neutral online RL and distillation."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import time
from dataclasses import asdict
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from posttrain.common import (
    AppendOnlyJsonlTailer,
    HubModelRef,
    LocalArtifactRef,
    ModelVariant,
    ProducedArtifact,
    RunContext,
    TraceFactSet,
    TraceFactUpdateObservation,
    TraceObservation,
)

from ...backend_support import verl_grpo_settings_problem, verl_training_loop_problem, verl_warmup_steps
from ...bindings import FullParameterUpdate, LoRAUpdate
from ...grpo_observations import GRPOObservationFeatures, normalize_grpo_metrics
from ...kl_reference import kl_reference_problem, resolved_kl_reference
from ...precision import resolve_precision, training_precision, verl_rollout_dtype
from ...profiles import GRPOSettings, SAMPOSettings, TrainingRenderer
from ...rendering import renderer_config_spec
from ...requests import CAPORequest, GDPORequest, GRPORequest, OnPolicyDistillationRequest, SAMPORequest
from ...results import TrainingSummary
from ..common import BackendTrainingResult
from .contracts import (
    VerlEnvironment,
    VerlEnvironmentExample,
    VerlHubArtifact,
    VerlInference,
    VerlLaunchManifest,
    VerlLocalArtifact,
    VerlModel,
    VerlPayload,
    VerlRunContext,
    VerlTarget,
    VerlWorkerResult,
)
from .metrics import (
    VerlMetricRecord,
    VerlRolloutRewardRecord,
    read_verl_metric_records,
    read_verl_rollout_reward_records,
)

_SUPPORTED_MODEL_FAMILIES = frozenset({"lfm2.5", "qwen3.5"})
_RESULT_FILE = "posttrain-result.json"


if TYPE_CHECKING:
    from .curriculum import CurriculumJournalReplay

VerlLaunchPlan = VerlLaunchManifest
# Identifies veRL online-RL runs that reproduce TRL's GRPO/DAPO/OLMo 3/SAMPO semantics.
VERL_SEMANTICS = "trl-parity-v1"


def build_grpo_launch_plan(request: GRPORequest, output_dir: Path) -> VerlLaunchPlan:
    _validate_backend(request.training.backend)
    _validate_model(request.policy, "policy")
    # adaptive_curriculum and every other GRPO setting veRL does not receive are
    # rejected instead of silently ignored.
    unsupported = verl_grpo_settings_problem(request.settings)
    if unsupported is not None:
        raise ValueError(unsupported)
    problem = kl_reference_problem(
        request.training.backend, request.settings.beta, request.settings.kl_reference, request.policy.form
    )
    if problem is not None:
        raise ValueError(problem)
    _validate_adapter_continuation(request.policy, request.training.update)
    _validate_active_sampling_capacity(request)
    return _plan(
        request,
        output_dir,
        "grpo",
        {
            "policy": _model(request.policy),
            "reference": _model(request.reference) if request.reference is not None else None,
            "algorithm": grpo_algorithm_payload(request.settings),
            "rollout": _inference(request.inference),
            "environment": _environment(request, output_dir),
            "curriculum_from": request.curriculum_from.path if request.curriculum_from is not None else None,
        },
    )


def _validate_active_sampling_capacity(request: GRPORequest | SAMPORequest) -> None:
    """The oversampled first round must fit rollout concurrency, as on TRL (the fork checks it again)."""

    from ...rollout_execution import oversampled_round_capacity_error

    active = request.settings.active_sampling
    if active is None:
        return
    from .worker import fork_native_names

    options = request.training.backend_options
    revision = options.get("source_revision")
    if (
        isinstance(revision, str)
        and "active_sampling" not in fork_native_names(revision)
        and options.get("source_dirty") is not True
    ):
        raise ValueError(
            f"veRL source revision {revision} has no round-based active sampling, which OLMo 3 requires; "
            "select a CarbonTeq veRL revision that provides it (none is released yet)"
        )
    engine = request.inference.engine
    declared = engine.get("max_num_seqs")
    execution = request.training.backend_options.get("rollout_execution")
    worker_slots = None
    if isinstance(execution, dict):
        workers, episodes = execution.get("env_workers"), execution.get("episodes_per_worker")
        if isinstance(workers, int) and isinstance(episodes, int):
            worker_slots = (workers, episodes)
    error = oversampled_round_capacity_error(
        num_prompts_per_step=request.settings.num_prompts_per_step,
        num_generations=request.settings.num_generations,
        oversample=active.oversample,
        vllm_max_num_seqs=declared if isinstance(declared, int) else request.settings.num_generations,
        environment_max_concurrent=getattr(request.bridge, "max_concurrent", None),
        worker_slots=worker_slots,
    )
    if error is not None:
        raise ValueError(error)


def grpo_algorithm_payload(settings: GRPOSettings) -> dict[str, Any]:
    """Map GRPO, DAPO or OLMo 3 settings to the veRL algorithm contract.

    OLMo 3 fixes mean-only group advantages and token-level sampler correction
    capped at 2 (validated by GRPOSettings); veRL receives both explicitly. Other
    algorithms keep veRL's historical advantage and correction behavior.
    """

    payload: dict[str, Any] = {
        "advantage_estimator": "grpo",
        "policy_updates": settings.policy_updates,
        "beta": settings.beta,
        "num_prompts_per_step": settings.num_prompts_per_step,
        "num_generations": settings.num_generations,
        "max_prompt_length": settings.max_prompt_length,
        "max_completion_length": settings.max_completion_length,
        "online_rl_algorithm": settings.algorithm,
        "shuffle_prompts": settings.shuffle_prompts,
        "clip_epsilon_low": settings.clip_epsilon_low,
        "clip_epsilon_high": settings.resolved_clip_epsilon_high,
        "dynamic_sampling": settings.dynamic_sampling is not None,
        "dynamic_sampling_max_candidate_batches": (
            settings.dynamic_sampling.max_candidate_batches if settings.dynamic_sampling is not None else None
        ),
        "mask_truncated_completions": settings.mask_truncated_completions,
        "overlong_buffer_tokens": settings.overlong_buffer_tokens,
        "overlong_penalty_factor": settings.overlong_penalty_factor,
        "truncation_penalty": settings.truncation_penalty,
    }
    if settings.adaptive_curriculum is not None:
        payload["adaptive_curriculum"] = {
            item.name: getattr(settings.adaptive_curriculum, item.name)
            for item in dataclass_fields(settings.adaptive_curriculum)
        }
    if settings.active_sampling is not None:
        payload.update(
            active_sampling=True,
            active_sampling_max_candidate_batches=settings.active_sampling.max_candidate_batches,
            active_sampling_oversample=settings.active_sampling.oversample,
            active_sampling_oversample_refill=settings.active_sampling.oversample_refill,
        )
    payload.update(_rollout_importance_sampling(settings))
    if settings.advantage_scaling == "none":
        payload["normalize_advantage_by_std"] = False
    else:
        payload.update(normalize_advantage_by_std=True, advantage_std_scope=settings.advantage_scaling)
    payload["max_admission_attempts"] = settings.max_admission_attempts
    return payload


def build_sampo_launch_plan(request: SAMPORequest, output_dir: Path) -> VerlLaunchPlan:
    """SAMPO on veRL: the fork's SAMPO estimator, TRL's sequence-ratio objective,
    round-based active sampling and optionally the adaptive curriculum."""

    _validate_backend(request.training.backend)
    _validate_model(request.policy, "policy")
    settings = request.settings
    problem = kl_reference_problem(request.training.backend, settings.beta, settings.kl_reference, request.policy.form)
    if problem is not None:
        raise ValueError(problem)
    _validate_adapter_continuation(request.policy, request.training.update)
    _validate_active_sampling_capacity(request)
    from .worker import fork_native_names

    options = request.training.backend_options
    revision = options.get("source_revision")
    if (
        isinstance(revision, str)
        and "sampo_token_credit" not in fork_native_names(revision)
        and options.get("source_dirty") is not True
    ):
        raise ValueError(
            f"selected veRL source revision {revision} does not register sampo_token_credit, "
            "which SAMPO requires; select a source with the token-local sequence-ratio loss"
        )
    return _plan(
        request,
        output_dir,
        "sampo",
        {
            "policy": _model(request.policy),
            "reference": _model(request.reference) if request.reference is not None else None,
            "algorithm": sampo_algorithm_payload(settings),
            "rollout": _inference(request.inference),
            "environment": _environment(request, output_dir),
        },
    )


def sampo_algorithm_payload(settings: SAMPOSettings) -> dict[str, Any]:
    """Map SAMPOSettings to the veRL algorithm contract (TRL SAMPO semantics)."""

    payload: dict[str, Any] = {
        "advantage_estimator": "sampo",
        "policy_updates": settings.policy_updates,
        "online_rl_algorithm": "sampo",
        "beta": settings.beta,
        "num_prompts_per_step": settings.num_prompts_per_step,
        "num_generations": settings.num_generations,
        "max_prompt_length": settings.max_prompt_length,
        "max_completion_length": settings.max_completion_length,
        "shuffle_prompts": settings.shuffle_prompts,
        "clip_epsilon_low": settings.clip_epsilon_low,
        "clip_epsilon_high": settings.clip_epsilon_high,
        "dynamic_sampling": False,
        "mask_truncated_completions": settings.mask_truncated_completions,
        "overlong_penalty_factor": 1.0,
        "truncation_penalty": settings.truncation_penalty,
        # TRL runs one admission attempt under active sampling; a failed group is refilled.
        "max_admission_attempts": settings.max_admission_attempts,
        "discount_gamma": settings.discount_gamma,
        "step_advantage_weight": settings.step_advantage_weight,
        "advantage_normalization": settings.advantage_normalization,
        "active_sampling": True,
        "active_sampling_max_candidate_batches": settings.active_sampling.max_candidate_batches,
        "active_sampling_oversample": settings.active_sampling.oversample,
        "active_sampling_oversample_refill": settings.active_sampling.oversample_refill,
    }
    payload.update(_rollout_importance_sampling(settings))
    if settings.adaptive_curriculum is not None:
        payload["adaptive_curriculum"] = {
            item.name: getattr(settings.adaptive_curriculum, item.name)
            for item in dataclass_fields(settings.adaptive_curriculum)
        }
    return payload


def _rollout_importance_sampling(settings: GRPOSettings | SAMPOSettings) -> dict[str, Any]:
    """TRL's vLLM sampler correction in the veRL contract: level, truncate or mask, and its bounds."""

    mode = settings.importance_sampling_mode
    return {
        "rollout_importance_sampling": "token" if mode.startswith("token") else "sequence",
        "rollout_importance_sampling_mode": "truncate" if mode.endswith("truncate") else "mask",
        "rollout_importance_sampling_cap": settings.importance_sampling_clip_max,
        "rollout_importance_sampling_min": settings.importance_sampling_clip_min,
    }


def build_structured_launch_plan(request: GDPORequest | CAPORequest, output_dir: Path) -> VerlLaunchPlan:
    from ...reward_recovery import reward_contract_digest

    _validate_backend(request.training.backend)
    _validate_model(request.policy, "policy")
    _validate_adapter_continuation(request.policy, request.training.update)
    technique = "gdpo" if isinstance(request, GDPORequest) else "capo"
    settings = request.settings
    algorithm: dict[str, Any] = {
        "advantage_estimator": technique,
        "policy_updates": settings.policy_updates,
        "reward_contract_digest": reward_contract_digest(request),
        "online_rl_algorithm": technique,
        "shuffle_prompts": settings.shuffle_prompts,
        "beta": settings.beta,
        "num_prompts_per_step": settings.num_prompts_per_step,
        "num_generations": settings.num_generations,
        "max_prompt_length": settings.max_prompt_length,
        "max_completion_length": settings.max_completion_length,
        "clip_epsilon_low": settings.clip_epsilon_low,
        "clip_epsilon_high": settings.clip_epsilon_high,
        "dynamic_sampling": False,
        "mask_truncated_completions": False,
        "overlong_penalty_factor": 1.0,
        "normalization_epsilon": settings.epsilon,
        "max_admission_attempts": settings.max_admission_attempts,
    }
    if isinstance(request, GDPORequest):
        algorithm.update(
            component_names=request.settings.component_names, component_weights=request.settings.component_weights
        )
    else:
        algorithm.update(
            outcome_component=request.settings.outcome_component,
            outcome_weight=request.settings.outcome_weight,
            process_weight=request.settings.process_weight,
        )
    return _plan(
        request,
        output_dir,
        technique,
        {
            "policy": _model(request.policy),
            "reference": _model(request.reference) if request.reference else None,
            "algorithm": algorithm,
            "rollout": _inference(request.inference),
            "environment": _environment(request, output_dir),
        },
    )


def run_structured_rl(
    context: RunContext, request: GDPORequest | CAPORequest, output_dir: Path
) -> BackendTrainingResult:
    return _launch(context, request, build_structured_launch_plan(request, output_dir), output_dir)


def build_distillation_launch_plan(
    request: OnPolicyDistillationRequest,
    output_dir: Path,
) -> VerlLaunchPlan:
    _validate_backend(request.training.backend)
    _validate_model(request.student, "student")
    _validate_model(request.teacher, "teacher")
    return _plan(
        request,
        output_dir,
        "distill",
        {
            "student": _model(request.student),
            "teacher": _model(request.teacher),
            "algorithm": {
                "advantage_estimator": "grpo",
                "loss_mode": "k1",
                "use_policy_gradient": True,
                "use_task_rewards": False,
                "temperature": request.settings.temperature,
                "num_prompts_per_step": request.settings.num_prompts_per_step,
                "num_generations": request.settings.num_generations,
                "max_prompt_length": request.settings.max_prompt_length,
                "max_completion_length": request.settings.max_completion_length,
            },
            "rollout": _inference(request.rollout_inference),
            "teacher_scoring": _inference(request.teacher_inference),
            "environment": _environment(request, output_dir),
        },
    )


def run_grpo(context: RunContext, request: GRPORequest, output_dir: Path) -> BackendTrainingResult:
    return _launch(context, request, build_grpo_launch_plan(request, output_dir), output_dir)


def run_sampo(context: RunContext, request: SAMPORequest, output_dir: Path) -> BackendTrainingResult:
    return _launch(context, request, build_sampo_launch_plan(request, output_dir), output_dir)


def run_distillation(
    context: RunContext,
    request: OnPolicyDistillationRequest,
    output_dir: Path,
) -> BackendTrainingResult:
    return _launch(context, request, build_distillation_launch_plan(request, output_dir), output_dir)


def _plan(
    request: GRPORequest | SAMPORequest | GDPORequest | CAPORequest | OnPolicyDistillationRequest,
    output_dir: Path,
    operation: Literal["grpo", "sampo", "gdpo", "capo", "distill"],
    operation_payload: dict[str, object],
) -> VerlLaunchPlan:
    runtime = request.training.runtime
    backend_options = request.training.backend_options
    executable = backend_options.get("python_executable")
    if not isinstance(executable, str) or not executable.strip():
        raise ValueError("veRL training requires backend_options.python_executable")
    executable_path = Path(executable).expanduser()
    if not executable_path.is_absolute():
        raise ValueError("veRL backend_options.python_executable must be an absolute path")
    working_directory = backend_options.get("working_directory")
    if not isinstance(working_directory, str) or not working_directory.strip():
        raise ValueError("veRL training requires backend_options.working_directory")
    worktree = Path(working_directory).expanduser()
    if not worktree.is_absolute():
        raise ValueError("veRL backend_options.working_directory must be an absolute path")
    source_revision = backend_options.get("source_revision")
    if not isinstance(source_revision, str) or len(source_revision) != 40:
        raise ValueError("veRL training requires a 40-character backend_options.source_revision")
    update = request.training.update
    if not isinstance(update, FullParameterUpdate | LoRAUpdate):
        raise ValueError("the qualified veRL slice supports full-parameter and LoRA updates")
    update_payload: dict[str, Any] = {"kind": update.kind}
    if isinstance(update, LoRAUpdate):
        update_payload.update(
            {
                "rank": update.rank,
                "alpha": update.alpha,
                "dropout": update.dropout,
                "target_modules": update.target_modules,
            }
        )
    loop = request.settings.loop
    world_size = request.training.target.placement.get("world_size", 1)
    if not isinstance(world_size, int):
        raise ValueError("veRL training target world_size must be an integer")
    selected_updates = getattr(request.settings, "policy_updates", None)
    if selected_updates is None:
        loop_problem = verl_training_loop_problem(
            loop,
            rows_per_update=request.settings.num_prompts_per_step * request.settings.num_generations,
            world_size=world_size,
        )
        if loop_problem is not None:
            raise ValueError(loop_problem)
    else:
        # The resolved scheduler owns boundaries; legacy group-size equality
        # would incorrectly force a complete population into one optimizer step.
        # Worker admission still refuses the unqualified legacy execution path.
        selected_updates.validate_legacy_loop(
            max_steps=loop.max_steps,
            per_device_batch_size=loop.per_device_batch_size,
            gradient_accumulation_steps=loop.gradient_accumulation_steps,
        )
    payload = VerlPayload.model_validate(
        {
            **operation_payload,
            "resolved_settings": (
                {"kind": operation, "settings": request.settings} if selected_updates is not None else None
            ),
            "training": {
                "binding_id": request.training.id,
                "renderer": _renderer_payload(
                    request.student if isinstance(request, OnPolicyDistillationRequest) else request.policy,
                    request.training.renderer,
                ),
                "update": update_payload,
                "loop": {
                    "max_steps": loop.max_steps,
                    "per_device_batch_size": loop.per_device_batch_size,
                    "gradient_accumulation_steps": loop.gradient_accumulation_steps,
                    "learning_rate": loop.learning_rate,
                    "lr_scheduler_type": loop.lr_scheduler_type,
                    "warmup_steps": verl_warmup_steps(loop),
                    "max_grad_norm": loop.max_grad_norm,
                    "checkpoint_steps": loop.checkpoint_steps,
                    "checkpoint_limit": loop.checkpoint_limit,
                    "seed": loop.seed,
                    "gradient_checkpointing": loop.gradient_checkpointing,
                },
                "parallelism": {
                    "tensor_parallel_size": request.training.parallelism.tensor_parallel_size,
                    "context_parallel_size": request.training.parallelism.context_parallel_size,
                    "expert_parallel_size": request.training.parallelism.expert_parallel_size,
                },
                "target": {
                    "id": request.training.target.id,
                    "world_size": request.training.target.placement.get("world_size", 1),
                },
                "runtime": {key: value for key, value in asdict(runtime).items() if key != "timeout_seconds"},
                "backend_options": dict(backend_options),
            },
            "resume_from": request.resume_from.path if request.resume_from is not None else None,
        },
    )
    return VerlLaunchPlan(
        operation=operation,
        backend=request.training.backend,
        backend_source_revision=source_revision,
        python_executable=executable_path,
        working_directory=worktree,
        output_directory=output_dir.resolve(),
        result_file=(output_dir / _RESULT_FILE).resolve(),
        payload=payload,
    )


def _renderer_payload(model: ModelVariant, renderer: TrainingRenderer) -> dict[str, object]:
    """Resolve the policy renderer on the launcher, as the TRL backend does in-process."""

    config, config_kwargs = renderer_config_spec(model, renderer)
    protocol = model.conversation.tool_calls
    return {
        "id": renderer.id,
        "implementation": renderer.implementation,
        "reasoning_mode": renderer.reasoning_mode,
        "model_family": model.family,
        "config": config,
        "config_kwargs": config_kwargs,
        "chat_template": model.conversation.chat_template.text(),
        "tool_call_protocol": (
            None
            if protocol is None
            else {"id": protocol.id, "start_token": protocol.start_token, "end_token": protocol.end_token}
        ),
    }


def _launch(
    context: RunContext,
    request: GRPORequest | SAMPORequest | GDPORequest | CAPORequest | OnPolicyDistillationRequest,
    plan: VerlLaunchPlan,
    output_dir: Path,
) -> BackendTrainingResult:
    manifest = output_dir / "posttrain-verl-launch.json"
    # Detached planning has no runtime identity. Inject the actual host context
    # at launch rather than deriving fake identities from an output directory.
    plan = plan.model_copy(
        update={
            "run_context": VerlRunContext.model_validate(
                {
                    **context.identity_attributes,
                    "workspace": context.workspace,
                }
            )
        }
    )
    snapshot_path = plan.payload.environment.bridge_snapshot
    snapshot_writer = getattr(request.bridge, "write_portable_snapshot", None)
    if not callable(snapshot_writer):
        raise TypeError("veRL currently requires a portable Verifiers environment bridge")
    with context.phase("data_preparation", {"backend": "verl"}):
        snapshot_writer(snapshot_path)
        plan.write(manifest)
    log_file = output_dir / "posttrain-verl.log"
    context.event(
        "training_runtime_resolved",
        {
            "backend": plan.backend,
            "backend_source_revision": plan.backend_source_revision,
            "isolated_python": str(plan.python_executable),
            "supported_model_families": ",".join(sorted(_SUPPORTED_MODEL_FAMILIES)),
        },
    )
    if isinstance(request, GRPORequest | SAMPORequest | GDPORequest | CAPORequest):
        context.event("grpo_runtime_resolved", _grpo_runtime_attributes(request, plan))
    timeout = _runtime_timeout(request)
    trace_tailer = _verifiers_trace_tailer(context, request)
    resolved_tailer = None
    if plan.payload.resolved_settings is not None:
        from .policy_observer import JOURNAL_NAME, observation_tailer

        resolved_tailer = observation_tailer(context, output_dir / JOURNAL_NAME)
    try:
        with context.phase("backend_execution", {"backend": "verl", "operation": plan.operation}):
            with log_file.open("w", encoding="utf-8") as stream:
                process = _start_isolated_worker(
                    plan,
                    manifest=manifest,
                    stdout=stream,
                )
                try:
                    returncode = _wait_for_isolated_worker(
                        process,
                        timeout=timeout,
                        tailer=trace_tailer,
                        context=context,
                        observation_tailer=resolved_tailer,
                    )
                except subprocess.TimeoutExpired as error:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    _record_failure_artifacts_best_effort(context, plan, output_dir)
                    log_tail = "\n".join(log_file.read_text(encoding="utf-8", errors="replace").splitlines()[-40:])
                    deadline = f"{timeout:g}s" if timeout is not None else "configured"
                    raise RuntimeError(
                        f"isolated veRL {plan.operation} process exceeded its {deadline} runtime deadline; "
                        f"log tail follows:\n{log_tail}"
                    ) from error
                except BaseException:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                    _record_failure_artifacts_best_effort(context, plan, output_dir)
                    raise
    finally:
        if trace_tailer is not None:
            try:
                trace_tailer.poll()
                _record_trace_sync_receipt(context, plan, output_dir, trace_tailer)
            except Exception:
                # A receipt is diagnostic evidence. It must not replace the
                # worker's terminal error or suppress bridge finalization.
                pass
    if returncode != 0:
        _record_failure_artifacts_best_effort(context, plan, output_dir)
        log_tail = "\n".join(log_file.read_text(encoding="utf-8", errors="replace").splitlines()[-40:])
        raise RuntimeError(
            f"isolated veRL {plan.operation} process exited with code {returncode}; log tail follows:\n{log_tail}"
        )
    if resolved_tailer is not None:
        stats = resolved_tailer.poll()
        if not stats.complete or not stats.emitted_records:
            raise RuntimeError("veRL resolved worker observations were not synchronized to the host")
    result_path = plan.result_file
    if not result_path.is_file():
        _record_failure_artifacts_best_effort(context, plan, output_dir)
        raise RuntimeError(f"veRL process completed without its result contract: {result_path}")
    result = VerlWorkerResult.read(result_path)
    backend, records = _backend_result(
        result, output_dir, loss_scaling=training_precision(request.training.backend_options) == "fp16"
    )
    curriculum = None
    if isinstance(request, GRPORequest | SAMPORequest) and request.settings.adaptive_curriculum is not None:
        from .curriculum import CURRICULUM_JOURNAL_NAME, CurriculumJournalReplay, read_curriculum_journal

        curriculum = CurriculumJournalReplay(context, read_curriculum_journal(output_dir / CURRICULUM_JOURNAL_NAME))
    if isinstance(request, GRPORequest | SAMPORequest | GDPORequest | CAPORequest):
        _replay_grpo_metrics(context, request, records, curriculum)
        _replay_trace_fact_updates(
            context,
            read_verl_rollout_reward_records(output_dir / "verl-rollout-rewards.jsonl"),
        )
    if curriculum is not None:
        assert isinstance(request, GRPORequest | SAMPORequest)
        curriculum.rest()
        _publish_curriculum_state(context, request, output_dir)
    return backend


def _publish_curriculum_state(context: RunContext, request: GRPORequest | SAMPORequest, output_dir: Path) -> None:
    """Replay the selector's curriculum events and publish its state like the TRL path."""

    from ...adaptive_curriculum import CURRICULUM_SNAPSHOT_NAME, digest_curriculum_state
    from .curriculum import checkpoint_views
    from .worker import CURRICULUM_CHECKPOINT_VIEWS_DIR, CURRICULUM_STATE_DIR

    curriculum = request.settings.adaptive_curriculum
    assert curriculum is not None
    state_dir = (output_dir / CURRICULUM_STATE_DIR).resolve()
    snapshot = state_dir / CURRICULUM_SNAPSHOT_NAME
    if not snapshot.is_file():
        raise RuntimeError(f"veRL completed without its adaptive curriculum state: {snapshot}")
    technique = "sampo" if isinstance(request, SAMPORequest) else request.settings.algorithm
    for step, view in checkpoint_views(output_dir / CURRICULUM_CHECKPOINT_VIEWS_DIR):
        # The same per-checkpoint view the TRL path publishes, so a later run can
        # warm-start with --curriculum-checkpoint-step from this veRL run.
        view = view.resolve()
        context.artifact(
            ProducedArtifact(
                name=f"training/{request.policy.id}/{technique}/checkpoint-{step:08d}/curriculum",
                kind="adaptive-curriculum-state",
                reference=LocalArtifactRef(view, digest_curriculum_state(view)),
                metadata={
                    "technique": technique,
                    "model_variant_id": request.policy.id,
                    "training_settings_id": request.settings.id,
                    "training_settings_revision": request.settings.revision,
                    "parameter_update_kind": request.training.update.kind,
                    "global_step": step,
                    "checkpoint_step": step,
                    "checkpoint_snapshot_id": f"{context.run_id}/step-{step:08d}",
                    "checkpoint_view": "curriculum",
                    "interrupted": False,
                    "training_backend": "verl",
                },
                role="checkpoint-curriculum",
            )
        )
    decision_index = json.loads(snapshot.read_text(encoding="utf-8")).get("decision_index")
    context.artifact(
        ProducedArtifact(
            name=f"training/{request.policy.id}/{technique}/adaptive-curriculum-state",
            kind="adaptive-curriculum-state",
            reference=LocalArtifactRef(state_dir, digest_curriculum_state(state_dir)),
            metadata={
                "class_field": curriculum.class_field,
                "decision_count": decision_index if isinstance(decision_index, int) else None,
                "format": "queued-jsonl-with-snapshot",
                "snapshot": CURRICULUM_SNAPSHOT_NAME,
                "training_backend": "verl",
            },
            role="controller-state",
        )
    )


def _verifiers_trace_tailer(
    context: RunContext,
    request: GRPORequest | SAMPORequest | GDPORequest | CAPORequest | OnPolicyDistillationRequest,
) -> AppendOnlyJsonlTailer | None:
    """Tail isolated native traces from the trusted parent process only."""

    trace_path = getattr(request.bridge, "trace_path", None)
    trace_observation = getattr(request.bridge, "trace_observation", None)
    mark_live_observed = getattr(request.bridge, "mark_live_observed", None)
    if not isinstance(trace_path, Path) or not callable(trace_observation) or not callable(mark_live_observed):
        return None

    def emit(record: dict[str, Any]) -> None:
        observation = cast(TraceObservation, trace_observation(record))
        if not isinstance(observation, TraceObservation):
            raise TypeError("veRL bridge trace_observation must return TraceObservation")
        context.trace(observation)
        mark_live_observed(observation.external_id)

    return AppendOnlyJsonlTailer(trace_path, emit)


def _wait_for_isolated_worker(
    process: subprocess.Popen[str],
    *,
    timeout: float | None,
    tailer: AppendOnlyJsonlTailer | None,
    context: RunContext,
    observation_tailer: AppendOnlyJsonlTailer | None = None,
) -> int:
    """Poll the worker and its native journal without exposing credentials to Ray."""

    deadline = time.monotonic() + timeout if timeout is not None else None
    while True:
        context.cancellation.raise_if_cancelled()
        if tailer is not None:
            tailer.poll()
        if observation_tailer is not None:
            observation_tailer.poll()
        returncode = process.poll()
        if returncode is not None:
            return returncode
        if deadline is not None and time.monotonic() >= deadline:
            assert timeout is not None
            raise subprocess.TimeoutExpired(process.args, timeout)
        time.sleep(0.2)


def _record_trace_sync_receipt(
    context: RunContext,
    plan: VerlLaunchPlan,
    output_dir: Path,
    tailer: AppendOnlyJsonlTailer,
) -> None:
    """Persist bounded synchronization facts without copying trace payloads."""

    stats = tailer.stats
    path = output_dir / "posttrain-verl-trace-sync.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "observation_source": "verifiers-jsonl-parent-tailer",
                "observed_records": stats.observed_records,
                "emitted_records": stats.emitted_records,
                "duplicate_records": stats.duplicate_records,
                "invalid_records": stats.invalid_records,
                "failed_records": stats.failed_records,
                "unsynchronized_records": stats.unsynchronized_records,
                "complete": stats.complete,
                "acknowledged_offset": tailer.offset,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    context.artifact(
        ProducedArtifact(
            name=f"training/diagnostics/verl/{plan.operation}/trace-sync-receipt",
            kind="training-runtime-receipt",
            reference=LocalArtifactRef(path.resolve(), hashlib.sha256(path.read_bytes()).hexdigest()),
            required=False,
            metadata={
                "training_backend": plan.backend,
                "backend_source_revision": plan.backend_source_revision,
                "operation": plan.operation,
                "trace_sync_complete": stats.complete,
            },
        )
    )


def _record_failure_artifacts(
    context: RunContext,
    plan: VerlLaunchPlan,
    output_dir: Path,
) -> None:
    candidates = (
        ("launch-manifest", "training-runtime-manifest", output_dir / "posttrain-verl-launch.json"),
        ("worker-log", "training-runtime-log", output_dir / "posttrain-verl.log"),
        ("native-log", "training-runtime-log", output_dir / "verl-native.log"),
        ("native-metrics", "training-metrics", output_dir / "verl-metrics.jsonl"),
    )
    for suffix, kind, path in candidates:
        if not path.is_file():
            continue
        context.artifact(
            ProducedArtifact(
                name=f"training/diagnostics/verl/{plan.operation}/{suffix}",
                kind=kind,
                reference=LocalArtifactRef(
                    path.resolve(),
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                ),
                required=False,
                metadata={
                    "training_backend": plan.backend,
                    "backend_source_revision": plan.backend_source_revision,
                    "operation": plan.operation,
                    "terminal_state": "failed",
                },
            )
        )


def _record_failure_artifacts_best_effort(
    context: RunContext,
    plan: VerlLaunchPlan,
    output_dir: Path,
) -> None:
    """Publish optional worker diagnostics without replacing a terminal error.

    A saturated tracking artifact queue is itself diagnostic, but it must not
    hide the isolated worker's exit code, timeout, or Python exception.
    """

    try:
        _record_failure_artifacts(context, plan, output_dir)
    except Exception as error:
        try:
            context.event(
                "training_diagnostic_publication_failed",
                {
                    "backend": plan.backend,
                    "operation": plan.operation,
                    "error_type": type(error).__name__,
                },
            )
        except Exception:
            # Tracking is already unavailable or backpressured. The caller's
            # original execution failure remains the reliable terminal cause.
            pass


def _backend_result(
    payload: VerlWorkerResult,
    output_dir: Path,
    *,
    loss_scaling: bool = False,
) -> tuple[BackendTrainingResult, tuple[VerlMetricRecord, ...]]:
    summary = payload.summary
    training_summary = TrainingSummary(
        global_step=summary.global_step,
        train_loss=summary.train_loss,
        runtime_seconds=summary.runtime_seconds,
        samples_per_second=summary.samples_per_second,
        steps_per_second=summary.steps_per_second,
    )
    model_dir = _output_path(payload.model_dir, output_dir, "model_dir")
    checkpoint = (
        _output_path(payload.recovery_checkpoint, output_dir, "recovery_checkpoint")
        if payload.recovery_checkpoint is not None
        else None
    )
    metrics_path = _output_file(payload.metrics_file, output_dir, "metrics_file")
    retention_manifest = (
        _output_file(payload.retention_manifest, output_dir, "retention_manifest")
        if payload.retention_manifest is not None
        else None
    )
    records = read_verl_metric_records(metrics_path, loss_scaling=loss_scaling)
    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)
    if checkpoint is not None and not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    return (
        BackendTrainingResult(
            training_summary,
            model_dir,
            checkpoint,
            metrics_path,
            retention_manifest,
        ),
        records,
    )


def _output_file(value: Path, output_dir: Path, field: str) -> Path:
    path = _output_path(value, output_dir, field)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _output_path(value: Path, output_dir: Path, field: str) -> Path:
    root = output_dir.resolve()
    path = value.resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"veRL result {field} must remain inside the run output directory")
    return path


def _replay_grpo_metrics(
    context: RunContext,
    request: GRPORequest | SAMPORequest | GDPORequest | CAPORequest,
    records: tuple[VerlMetricRecord, ...],
    curriculum: CurriculumJournalReplay | None = None,
) -> None:
    environment_category = getattr(request.environment, "category", "")
    features = GRPOObservationFeatures.from_request(
        request,
        tool_environment=isinstance(environment_category, str) and "tool" in environment_category.split("-"),
    )
    for record in records:
        if curriculum is not None:
            # Curriculum records of earlier and equal steps first: tracking steps never decrease.
            curriculum.through(record.step)
        normalized = normalize_grpo_metrics(
            backend="verl",
            step=record.step,
            native=record.data,
            features=features,
        )
        if normalized.metrics:
            context.metrics(
                normalized.metrics,
                step=record.step,
                attributes=normalized.attributes,
            )


def _replay_trace_fact_updates(
    context: RunContext,
    records: tuple[VerlRolloutRewardRecord, ...],
) -> None:
    """Emit worker-side shaped rewards from the trusted parent process only.

    An enrichment carries only ``algorithm_reward``: the rollout step and the
    reward components belong to the Verifiers source projection, and tracking
    backends reject an enrichment that supplies them.
    """

    for record in records:
        context.trace_fact_update(
            TraceFactUpdateObservation(
                "verifiers",
                record.trace_id,
                TraceFactSet(
                    namespace="posttrain.train.reward",
                    calculator_version="verl-algorithm-reward.v1",
                    measures={"algorithm_reward": record.algorithm_reward},
                    provenance={"algorithm_reward": "verl_agent_loop_reward_shaping"},
                ),
                attributes={"optimizer_step": record.step, "backend": "verl"},
            )
        )


def _start_isolated_worker(
    plan: VerlLaunchPlan,
    *,
    manifest: Path,
    stdout: Any,
) -> subprocess.Popen[str]:
    environment = _isolated_environment(plan.python_executable)
    environment["POSTTRAIN_VERL_ROLLOUT_REWARDS_PATH"] = str(plan.output_directory / "verl-rollout-rewards.jsonl")
    environment["POSTTRAIN_VERL_MANIFEST"] = str(manifest)
    return subprocess.Popen(
        plan.command,
        cwd=str(plan.working_directory),
        env=environment,
        stdout=stdout,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )


def _isolated_environment(python_executable: Path) -> dict[str, str]:
    blocked_prefixes = ("WANDB_", "TRACKIO_")
    environment = {
        key: value
        for key, value in os.environ.items()
        if (
            not key.upper().startswith(blocked_prefixes)
            and key != "VIRTUAL_ENV"
            and not key.startswith("UV_")
            and not key.startswith("PYTHON")
        )
    }
    isolated_bin = str(python_executable.parent)
    inherited_path = environment.get("PATH")
    environment["PATH"] = f"{isolated_bin}{os.pathsep}{inherited_path}" if inherited_path else isolated_bin
    # The isolated interpreter already owns an exact environment. Ray must not
    # rediscover the parent host's `uv run` command and replace worker startup
    # with the host workspace environment.
    environment["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONSAFEPATH"] = "1"
    projection = environment.get("POSTTRAIN_VERL_PYTHONPATH")
    if projection:
        projection_path = Path(projection)
        if not projection_path.is_absolute() or not projection_path.is_dir():
            raise RuntimeError("POSTTRAIN_VERL_PYTHONPATH must name the packaged absolute veRL worker projection")
        environment["PYTHONPATH"] = projection
    return environment


def _runtime_timeout(
    request: GRPORequest | SAMPORequest | GDPORequest | CAPORequest | OnPolicyDistillationRequest,
) -> float | None:
    return request.training.runtime.timeout_seconds


def _grpo_runtime_attributes(
    request: GRPORequest | SAMPORequest | GDPORequest | CAPORequest, plan: VerlLaunchPlan
) -> dict[str, Any]:
    engine = request.inference.engine
    speculative = engine.get("speculative_config")
    attributes: dict[str, Any] = {
        "training_backend": "verl",
        "backend_source_revision": plan.backend_source_revision,
        "training_binding_id": request.training.id,
        "inference_binding_id": request.inference.id,
        "inference_backend": request.inference.backend,
        "rollout_mode": str(engine.get("mode", "async")),
        "update_kind": request.training.update.kind,
        "world_size": request.training.target.placement.get("world_size", 1),
        **{
            key: value
            for key, value in resolve_precision(
                request.training.backend_options, engine, request.policy.weight_precision, backend="verl"
            )
            .as_dict()
            .items()
            if key in {"training_precision", "loss_scaling"}
        },
        "rollout_precision": verl_rollout_dtype(engine)[0],
        "kv_cache_dtype": str(engine.get("kv_cache_dtype", "auto")),
        "max_model_len": engine.get(
            "max_model_len",
            request.settings.max_prompt_length + request.settings.max_completion_length,
        ),
        "online_rl_algorithm": request.settings.algorithm if isinstance(request, GRPORequest) else plan.operation,
        "clip_epsilon_low": request.settings.clip_epsilon_low,
        "clip_epsilon_high": (
            request.settings.resolved_clip_epsilon_high
            if isinstance(request, GRPORequest)
            else request.settings.clip_epsilon_high
        ),
        "mask_truncated_completions": request.settings.mask_truncated_completions,
        "shuffle_prompts": request.settings.shuffle_prompts,
    }
    if isinstance(request, GRPORequest | SAMPORequest):
        # Runs from this version on reproduce TRL's objective, correction, scaling, admission
        # and sampling (docs/plan/verl-vortex-port.md); earlier veRL runs did not.
        attributes["verl_semantics"] = VERL_SEMANTICS
    if isinstance(request, GRPORequest):
        attributes["kl_reference"] = resolved_kl_reference(
            request.settings.beta, request.settings.kl_reference, request.policy.form
        )
        attributes["kl_reference_setting"] = request.settings.kl_reference
        attributes["overlong_buffer_tokens"] = request.settings.overlong_buffer_tokens
        attributes["overlong_penalty_factor"] = request.settings.overlong_penalty_factor
        attributes["truncation_penalty"] = request.settings.truncation_penalty
        active = request.settings.active_sampling
        attributes["active_sampling"] = active is not None
        curriculum = request.settings.adaptive_curriculum
        attributes["adaptive_curriculum"] = curriculum is not None
        if curriculum is not None:
            attributes["adaptive_curriculum_policy"] = curriculum.policy
            attributes["adaptive_curriculum_class_field"] = curriculum.class_field
            attributes["adaptive_curriculum_sampling_mode"] = (
                "active_sampling_refill" if active is not None else "initial_batch"
            )
        if active is not None:
            attributes["active_sampling_max_candidate_batches"] = active.max_candidate_batches
            attributes["active_sampling_oversample"] = active.oversample
            attributes["active_sampling_oversample_refill"] = active.oversample_refill
        attributes["advantage_scaling"] = "none" if request.settings.algorithm == "olmo3" else "group"
    elif isinstance(request, SAMPORequest):
        attributes["discount_gamma"] = request.settings.discount_gamma
        attributes["step_advantage_weight"] = request.settings.step_advantage_weight
        attributes["advantage_normalization"] = request.settings.advantage_normalization
    if isinstance(speculative, dict):
        attributes["speculative_method"] = str(speculative.get("method"))
        attributes["num_speculative_tokens"] = speculative.get("num_speculative_tokens")
    return attributes


def _validate_backend(backend: str) -> None:
    if backend.split("@", 1)[0] != "verl":
        raise ValueError(f"veRL adapter received incompatible training backend {backend!r}")


def _validate_model(model: ModelVariant, role: str) -> None:
    if model.family not in _SUPPORTED_MODEL_FAMILIES:
        qualified = ", ".join(sorted(_SUPPORTED_MODEL_FAMILIES))
        raise ValueError(f"veRL currently qualifies only {qualified}; {role} uses {model.family!r}")


def _model(model: ModelVariant | None) -> VerlModel | None:
    if model is None:
        return None
    base = None
    if model.form in {"adapter", "peft-adapter"}:
        # The adapter is attached to its foundation weights inside veRL.
        base = VerlHubArtifact(repo_id=model.base.repo_id, revision=model.base.revision)
    if isinstance(model.artifact, HubModelRef):
        artifact = VerlHubArtifact(repo_id=model.artifact.repo_id, revision=model.artifact.revision)
    elif isinstance(model.artifact, LocalArtifactRef):
        artifact = VerlLocalArtifact(path=model.artifact.path.resolve(), digest=model.artifact.digest)
    else:
        raise ValueError("veRL requires a HubModelRef or materialized LocalArtifactRef")
    return VerlModel(
        id=model.id,
        family=model.family,
        form=model.form,
        artifact=artifact,
        tokenizer_fingerprint=model.tokenizer_fingerprint,
        renderer_contract=model.renderer_contract,
        base=base,
    )


def _validate_adapter_continuation(model: ModelVariant, update: object) -> None:
    """A PEFT-adapter starting model continues training that adapter, so the plan must match it."""

    # The request itself rejects a full-parameter update from an unmerged adapter.
    if model.form not in {"adapter", "peft-adapter"} or not isinstance(update, LoRAUpdate):
        return
    if not isinstance(model.artifact, LocalArtifactRef):
        raise ValueError("the host must materialize the starting adapter before veRL training")
    config_path = model.artifact.path / "adapter_config.json"
    try:
        config = json.loads(config_path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"starting adapter lacks a readable {config_path.name}") from error
    rank = config.get("r")
    if rank != update.rank:
        # vLLM sizes its LoRA slots from the binding's rank; the actor takes the adapter's.
        raise ValueError(
            f"the starting adapter has LoRA rank {rank} but the training binding selects rank {update.rank}; "
            "select a binding with the adapter's rank"
        )


def _inference(binding: Any) -> VerlInference:
    return VerlInference(
        id=binding.id,
        backend=binding.backend,
        engine=dict(binding.engine),
        sampling=dict(binding.sampling),
        target=VerlTarget(
            id=binding.target.id,
            world_size=binding.target.placement.get("world_size", 1),
        ),
    )


def _environment(
    request: GRPORequest | SAMPORequest | GDPORequest | CAPORequest | OnPolicyDistillationRequest,
    output_dir: Path,
) -> VerlEnvironment:
    return VerlEnvironment(
        id=request.environment.id,
        revision=request.environment.revision,
        dataset_id=request.bridge.dataset.id,
        dataset_revision=request.bridge.dataset.revision,
        bridge_snapshot=(output_dir / "verifiers-bridge.pkl").resolve(),
        max_concurrent=getattr(request.bridge, "max_concurrent", None),
        examples=tuple(
            VerlEnvironmentExample(id=example.id, prompt=example.prompt, metadata=dict(example.metadata))
            for example in request.bridge.dataset.examples
        ),
    )


__all__ = [
    "VerlLaunchPlan",
    "build_distillation_launch_plan",
    "build_grpo_launch_plan",
    "build_sampo_launch_plan",
    "grpo_algorithm_payload",
    "sampo_algorithm_payload",
    "run_distillation",
    "run_grpo",
    "run_sampo",
]
