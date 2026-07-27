"""Isolated veRL launcher for backend-neutral online RL and distillation."""

from __future__ import annotations

import hashlib
import math
import os
import signal
import subprocess
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

from posttrain.common import (
    HubModelRef,
    LocalArtifactRef,
    ModelVariant,
    ProducedArtifact,
    RunContext,
)

from ...bindings import FullParameterUpdate, LoRAUpdate
from ...grpo_observations import GRPOObservationFeatures, normalize_grpo_metrics
from ...requests import GRPORequest, OnPolicyDistillationRequest, SAMPORequest
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
    VerlTarget,
    VerlWorkerResult,
)
from .metrics import VerlMetricRecord, read_verl_metric_records

_SUPPORTED_MODEL_FAMILIES = frozenset({"qwen3.5"})
_RESULT_FILE = "posttrain-result.json"


VerlLaunchPlan = VerlLaunchManifest


def build_grpo_launch_plan(request: GRPORequest, output_dir: Path) -> VerlLaunchPlan:
    _validate_backend(request.training.backend)
    _validate_model(request.policy, "policy")
    return _plan(
        request,
        output_dir,
        "grpo",
        {
            "policy": _model(request.policy),
            "reference": _model(request.reference) if request.reference is not None else None,
            "algorithm": {
                "advantage_estimator": "grpo",
                "beta": request.settings.beta,
                "num_prompts_per_step": request.settings.num_prompts_per_step,
                "num_generations": request.settings.num_generations,
                "max_prompt_length": request.settings.max_prompt_length,
                "max_completion_length": request.settings.max_completion_length,
                "online_rl_algorithm": request.settings.algorithm,
                "clip_epsilon_low": request.settings.clip_epsilon_low,
                "clip_epsilon_high": request.settings.resolved_clip_epsilon_high,
                "dynamic_sampling": request.settings.dynamic_sampling is not None,
                "dynamic_sampling_max_candidate_batches": (
                    request.settings.dynamic_sampling.max_candidate_batches
                    if request.settings.dynamic_sampling is not None
                    else None
                ),
                "mask_truncated_completions": request.settings.mask_truncated_completions,
                "overlong_buffer_tokens": request.settings.overlong_buffer_tokens,
                "overlong_penalty_factor": request.settings.overlong_penalty_factor,
            },
            "rollout": _inference(request.inference),
            "environment": _environment(request, output_dir),
        },
    )


def build_sampo_launch_plan(request: SAMPORequest, output_dir: Path) -> VerlLaunchPlan:
    _validate_backend(request.training.backend)
    _validate_model(request.policy, "policy")
    return _plan(
        request,
        output_dir,
        "sampo",
        {
            "policy": _model(request.policy),
            "reference": _model(request.reference) if request.reference is not None else None,
            "algorithm": {
                "advantage_estimator": "sampo",
                "beta": request.settings.beta,
                "num_prompts_per_step": request.settings.num_prompts_per_step,
                "num_generations": request.settings.num_generations,
                "max_prompt_length": request.settings.max_prompt_length,
                "max_completion_length": request.settings.max_completion_length,
                "online_rl_algorithm": "sampo",
                "clip_epsilon_low": request.settings.clip_epsilon_low,
                "clip_epsilon_high": request.settings.clip_epsilon_high,
                "dynamic_sampling": True,
                "dynamic_sampling_max_candidate_batches": (request.settings.dynamic_sampling.max_candidate_batches),
                "mask_truncated_completions": request.settings.mask_truncated_completions,
                "overlong_penalty_factor": 1.0,
                "discount_gamma": request.settings.discount_gamma,
                "step_advantage_weight": request.settings.step_advantage_weight,
                "advantage_normalization": request.settings.advantage_normalization,
            },
            "rollout": _inference(request.inference),
            "environment": _environment(request, output_dir),
        },
    )


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
                "use_policy_gradient": False,
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
    request: GRPORequest | SAMPORequest | OnPolicyDistillationRequest,
    output_dir: Path,
    operation: Literal["grpo", "sampo", "distill"],
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
    recipe_worktree = None
    recipe_source_revision = None
    algorithm_payload = operation_payload.get("algorithm")
    uses_dynamic_recipe = (
        operation in {"grpo", "sampo"}
        and isinstance(algorithm_payload, Mapping)
        and algorithm_payload.get("online_rl_algorithm") in {"dapo", "sampo"}
        and bool(algorithm_payload.get("dynamic_sampling"))
    )
    if uses_dynamic_recipe:
        recipe_directory = backend_options.get("dynamic_sampling_recipe_working_directory")
        if recipe_directory is None and operation == "grpo":
            recipe_directory = backend_options.get("dapo_recipe_working_directory")
        if not isinstance(recipe_directory, str) or not recipe_directory.strip():
            raise ValueError("veRL dynamic sampling requires backend_options.dynamic_sampling_recipe_working_directory")
        recipe_worktree = Path(recipe_directory).expanduser()
        if not recipe_worktree.is_absolute():
            raise ValueError("veRL dynamic-sampling recipe working directory must be an absolute path")
        recipe_source_revision = backend_options.get("dynamic_sampling_recipe_source_revision")
        if recipe_source_revision is None and operation == "grpo":
            recipe_source_revision = backend_options.get("dapo_recipe_source_revision")
        if not isinstance(recipe_source_revision, str) or len(recipe_source_revision) != 40:
            raise ValueError("veRL dynamic sampling requires a 40-character recipe source revision")
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
    payload = VerlPayload.model_validate(
        {
            **operation_payload,
            "training": {
                "binding_id": request.training.id,
                "renderer": {
                    "id": request.training.renderer.id,
                    "implementation": request.training.renderer.implementation,
                    "reasoning_mode": request.training.renderer.reasoning_mode,
                },
                "update": update_payload,
                "loop": {
                    "max_steps": loop.max_steps,
                    "per_device_batch_size": loop.per_device_batch_size,
                    "gradient_accumulation_steps": loop.gradient_accumulation_steps,
                    "learning_rate": loop.learning_rate,
                    "warmup_steps": math.ceil(loop.max_steps * loop.warmup_ratio),
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
        recipe_source_revision=recipe_source_revision,
        python_executable=executable_path,
        working_directory=worktree,
        recipe_working_directory=recipe_worktree,
        output_directory=output_dir.resolve(),
        result_file=(output_dir / _RESULT_FILE).resolve(),
        payload=payload,
    )


def _launch(
    context: RunContext,
    request: GRPORequest | SAMPORequest | OnPolicyDistillationRequest,
    plan: VerlLaunchPlan,
    output_dir: Path,
) -> BackendTrainingResult:
    manifest = output_dir / "posttrain-verl-launch.json"
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
    if isinstance(request, GRPORequest | SAMPORequest):
        context.event("grpo_runtime_resolved", _grpo_runtime_attributes(request, plan))
    timeout = _runtime_timeout(request)
    with context.phase("backend_execution", {"backend": "verl", "operation": plan.operation}):
        with log_file.open("w", encoding="utf-8") as stream:
            process = _start_isolated_worker(
                plan,
                manifest=manifest,
                stdout=stream,
            )
            try:
                returncode = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired as error:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                _record_failure_artifacts(context, plan, output_dir)
                log_tail = "\n".join(log_file.read_text(encoding="utf-8", errors="replace").splitlines()[-40:])
                raise RuntimeError(
                    f"isolated veRL {plan.operation} process exceeded its {timeout:g}s runtime deadline; "
                    f"log tail follows:\n{log_tail}"
                ) from error
    if returncode != 0:
        _record_failure_artifacts(context, plan, output_dir)
        log_tail = "\n".join(log_file.read_text(encoding="utf-8", errors="replace").splitlines()[-40:])
        raise RuntimeError(
            f"isolated veRL {plan.operation} process exited with code {returncode}; log tail follows:\n{log_tail}"
        )
    result_path = plan.result_file
    if not result_path.is_file():
        _record_failure_artifacts(context, plan, output_dir)
        raise RuntimeError(f"veRL process completed without its result contract: {result_path}")
    result = VerlWorkerResult.read(result_path)
    backend, records = _backend_result(result, output_dir)
    if isinstance(request, GRPORequest | SAMPORequest):
        _replay_grpo_metrics(context, request, records)
    return backend


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


def _backend_result(
    payload: VerlWorkerResult,
    output_dir: Path,
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
    records = read_verl_metric_records(metrics_path)
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
    request: GRPORequest | SAMPORequest,
    records: tuple[VerlMetricRecord, ...],
) -> None:
    environment_category = getattr(request.environment, "category", "")
    features = GRPOObservationFeatures.from_request(
        request,
        tool_environment=isinstance(environment_category, str) and "tool" in environment_category.split("-"),
    )
    for record in records:
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


def _start_isolated_worker(
    plan: VerlLaunchPlan,
    *,
    manifest: Path,
    stdout: Any,
) -> subprocess.Popen[str]:
    environment = _isolated_environment(plan.python_executable)
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


def _runtime_timeout(request: GRPORequest | SAMPORequest | OnPolicyDistillationRequest) -> float | None:
    return request.training.runtime.timeout_seconds


def _grpo_runtime_attributes(request: GRPORequest | SAMPORequest, plan: VerlLaunchPlan) -> dict[str, Any]:
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
        "rollout_precision": str(engine.get("dtype", "bfloat16")),
        "kv_cache_dtype": str(engine.get("kv_cache_dtype", "auto")),
        "max_model_len": engine.get(
            "max_model_len",
            request.settings.max_prompt_length + request.settings.max_completion_length,
        ),
        "online_rl_algorithm": request.settings.algorithm if isinstance(request, GRPORequest) else "sampo",
        "clip_epsilon_low": request.settings.clip_epsilon_low,
        "clip_epsilon_high": (
            request.settings.resolved_clip_epsilon_high
            if isinstance(request, GRPORequest)
            else request.settings.clip_epsilon_high
        ),
        "mask_truncated_completions": request.settings.mask_truncated_completions,
    }
    if isinstance(request, GRPORequest):
        attributes["overlong_buffer_tokens"] = request.settings.overlong_buffer_tokens
        attributes["overlong_penalty_factor"] = request.settings.overlong_penalty_factor
    else:
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
    request: GRPORequest | SAMPORequest | OnPolicyDistillationRequest,
    output_dir: Path,
) -> VerlEnvironment:
    return VerlEnvironment(
        id=request.environment.id,
        revision=request.environment.revision,
        dataset_id=request.bridge.dataset.id,
        dataset_revision=request.bridge.dataset.revision,
        bridge_snapshot=(output_dir / "verifiers-bridge.pkl").resolve(),
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
    "run_distillation",
    "run_grpo",
    "run_sampo",
]
