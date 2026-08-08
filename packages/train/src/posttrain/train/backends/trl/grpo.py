"""TRL GRPO translation over task-neutral rollout prompts and rewards."""

from __future__ import annotations

import asyncio
import math
import os
import time
from collections.abc import Mapping
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, cast

from posttrain.common import JsonValue, RunContext, TraceObservation
from posttrain.common.cuda import TorchModule, activate_cuda_toolkit

from ...grpo_observations import GRPOObservationFeatures, normalize_grpo_metrics
from ...online_rl import RolloutBatch
from ...profiles import SAMPOSettings, shape_online_reward
from ...requests import GRPORequest, SAMPORequest
from ...sampo_advantages import compute_sampo_advantages
from .common import (
    BackendTrainingResult,
    callback_type,
    emit_parameter_counts,
    emit_runtime_versions,
    finish_training,
    framework_imports,
    load_tokenizer,
    load_trainable_model,
    preserve_recovery_checkpoint_after_error,
    trainer_arguments,
    trainer_lifecycle,
    vllm_rollout_options,
)

_TRACE_REPLAY_METRICS = frozenset(
    {
        "train/rl/reward_std",
        "train/rl/group_zero_variance_fraction",
    }
)


class _ActorUpdateTelemetry:
    """Own one bounded actor-update phase between rollout and optimizer step."""

    def __init__(self, context: RunContext) -> None:
        self._context = context
        self._phase: AbstractContextManager[str] | None = None
        self._started_at: float | None = None
        self._optimizer_step: int | None = None
        self._completed_durations: dict[int, float] = {}
        self._last_token_count = 0.0

    @property
    def active(self) -> bool:
        return self._phase is not None

    def start(self, optimizer_step: int) -> None:
        if optimizer_step <= 0:
            raise ValueError("actor optimizer step must be positive")
        if self._phase is not None:
            raise RuntimeError(
                f"actor update for step {self._optimizer_step} is still active; cannot start step {optimizer_step}"
            )
        phase = self._context.phase(
            "actor_update",
            {"backend": "trl", "logical_step": optimizer_step},
        )
        phase.__enter__()
        self._phase = phase
        self._started_at = time.perf_counter()
        self._optimizer_step = optimizer_step

    def complete(self, optimizer_step: int) -> None:
        if self._phase is None:
            return
        if optimizer_step != self._optimizer_step:
            raise RuntimeError(f"actor update for step {self._optimizer_step} cannot complete at step {optimizer_step}")
        phase, started_at = self._reset()
        phase.__exit__(None, None, None)
        if started_at is not None:
            duration = time.perf_counter() - started_at
            self._completed_durations[optimizer_step] = duration
            self._context.metric(
                "train/rl/time/actor_update_seconds",
                duration,
                step=optimizer_step,
            )

    def record_tokens(self, optimizer_step: int, cumulative_tokens: float) -> None:
        """Emit actor throughput once TRL publishes the step's cumulative token count."""

        completed_steps = [step for step in self._completed_durations if step <= optimizer_step]
        if not completed_steps:
            return
        duration = sum(self._completed_durations.pop(step) for step in completed_steps)
        token_delta = cumulative_tokens - self._last_token_count
        self._last_token_count = cumulative_tokens
        if duration <= 0 or token_delta < 0:
            return
        self._context.metrics(
            {
                "train/rl/actor_processed_tokens": token_delta,
                "train/rl/actor_tokens_per_second": token_delta / duration,
            },
            step=optimizer_step,
        )

    def fail(self, error: BaseException) -> None:
        if self._phase is None:
            return
        phase, _started_at = self._reset()
        phase.__exit__(type(error), error, error.__traceback__)

    def _reset(self) -> tuple[AbstractContextManager[str], float | None]:
        phase = self._phase
        if phase is None:
            raise RuntimeError("actor update telemetry is not active")
        started_at = self._started_at
        self._phase = None
        self._started_at = None
        self._optimizer_step = None
        return phase, started_at


def _configure_torch_compile(engine: Mapping[str, object]) -> None:
    """Apply compile policy before importing torch or constructing vLLM."""

    value = engine.get("disable_torch_compile")
    if value is not None and not isinstance(value, bool):
        raise ValueError("TRL rollout disable_torch_compile must be a boolean")
    if value:
        # torch._dynamo reads this flag at import; setting it after importing
        # torch is too late for compile-decorated MTP helper modules.
        os.environ["TORCH_COMPILE_DISABLE"] = "1"


def _actor_update_callback_type(imports: Mapping[str, Any], telemetry: _ActorUpdateTelemetry) -> type[Any]:
    parent = imports["TrainerCallback"]

    class ActorUpdateCallback(parent):
        def on_step_end(self, args: Any, state: Any, control: Any, **_: Any) -> Any:
            del args
            telemetry.complete(int(state.global_step))
            return control

        def on_log(
            self,
            args: Any,
            state: Any,
            control: Any,
            logs: Mapping[str, object] | None = None,
            **_: Any,
        ) -> Any:
            del args
            token_count = (logs or {}).get("num_tokens")
            if isinstance(token_count, int | float) and not isinstance(token_count, bool):
                telemetry.record_tokens(int(state.global_step), float(token_count))
            return control

    return ActorUpdateCallback


def run_grpo(
    context: RunContext,
    request: GRPORequest,
    output_dir: Path,
) -> BackendTrainingResult:
    return _run_online_rl(context, request, output_dir)


def run_sampo(
    context: RunContext,
    request: SAMPORequest,
    output_dir: Path,
) -> BackendTrainingResult:
    return _run_online_rl(context, request, output_dir)


def _run_online_rl(
    context: RunContext,
    request: GRPORequest | SAMPORequest,
    output_dir: Path,
) -> BackendTrainingResult:
    _configure_torch_compile(request.inference.engine)
    if request.inference.backend.split("@", 1)[0] == "vllm":
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("PyTorch is not installed; install posttrain-train[trl-vllm]") from error
        activate_cuda_toolkit(cast(TorchModule, torch))
    try:
        from trl.trainer.grpo_config import GRPOConfig  # pyright: ignore[reportMissingImports]
        from trl.trainer.grpo_trainer import GRPOTrainer  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise RuntimeError("install posttrain-train with the trl extra") from error

    imports = framework_imports()
    emit_runtime_versions(context, imports)
    with context.phase("model_loading", {"backend": "trl"}):
        tokenizer = load_tokenizer(request.policy, imports)
        model = load_trainable_model(request.policy, request.training.update, request.settings.loop, imports)
    rows = []
    template_kwargs = request.policy.conversation.reasoning_mode(request.training.renderer.reasoning_mode).kwargs()
    for example in request.bridge.dataset.examples:
        prompt = [{"role": "user", "content": example.prompt}]
        rendered = tokenizer.apply_chat_template(
            prompt,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
            **template_kwargs,
        )
        if not isinstance(rendered, list) or any(not isinstance(token_id, int) for token_id in rendered):
            raise TypeError("chat template must return one flat token-id list")
        if len(rendered) > request.settings.max_prompt_length:
            raise ValueError(
                f"rollout example {example.id!r} has {len(rendered)} prompt tokens; "
                f"settings permit {request.settings.max_prompt_length}"
            )
        rows.append({"prompt": prompt, "example_id": example.id, **dict(example.metadata)})
    dataset = imports["Dataset"].from_list(rows)
    emit_parameter_counts(context, model, request.training.update)
    arguments = _online_rl_arguments(request, output_dir, template_kwargs)
    observation_features = (
        GRPOObservationFeatures.from_request(request)
        if isinstance(request, GRPORequest)
        else GRPOObservationFeatures(
            reference_kl_enabled=request.settings.beta > 0,
            decoupled_rollout=request.inference.backend.split("@", 1)[0] == "vllm",
            tool_environment=True,
        )
    )
    technique = request.settings.algorithm if isinstance(request, GRPORequest) else "sampo"
    context.event("grpo_runtime_resolved", _online_rl_runtime_attributes(request))
    actor_update = _ActorUpdateTelemetry(context)

    def normalize_metrics(step: int, native: Mapping[str, object]) -> Mapping[str, float]:
        metrics = dict(
            normalize_grpo_metrics(
                backend="trl",
                step=step,
                native=native,
                features=observation_features,
            ).metrics
        )
        # Verifiers trace replay owns reward-population evidence. The common
        # callback owns wall-clock step duration. Native TRL records can
        # contain both, but emitting them again creates two logical points per
        # optimizer step.
        for name in (*_TRACE_REPLAY_METRICS, "train/step_time_seconds"):
            metrics.pop(name, None)
        return metrics

    with context.phase("runtime_initialization", {"backend": "trl"}):
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=cast(Any, _bridge_reward),
            rollout_func=cast(Any, _rollout_function(context, request, tokenizer, actor_update)),
            args=GRPOConfig(**arguments),
            train_dataset=dataset,
            processing_class=tokenizer,
            callbacks=[
                callback_type(context, imports, metric_normalizer=normalize_metrics)(),
                _actor_update_callback_type(imports, actor_update)(),
            ],
        )
        _configure_liger_loss(trainer, request)
    resume = str(request.resume_from.path) if request.resume_from is not None else None
    with trainer_lifecycle(trainer):
        try:
            train_output = trainer.train(resume_from_checkpoint=resume)
            if actor_update.active:
                raise RuntimeError("TRL training completed before the active actor update reached an optimizer step")
            with context.phase("artifact_export", {"backend": "trl"}):
                return finish_training(
                    context,
                    trainer,
                    train_output,
                    tokenizer,
                    output_dir.parent,
                    technique,
                    request.training.update,
                    imports,
                )
        except BaseException as error:
            actor_update.fail(error)
            preserve_recovery_checkpoint_after_error(
                context,
                trainer,
                error,
                technique=technique,
                model=request.policy,
                settings=request.settings,
                update=request.training.update,
                imports=imports,
            )
            raise


def _rollout_function(
    context: RunContext,
    request: GRPORequest | SAMPORequest,
    tokenizer: Any,
    actor_update: _ActorUpdateTelemetry | None = None,
) -> Any:
    """Translate TRL generation batches into the public environment-rollout bridge contract."""

    def run_rollouts(
        prompts: list[Any],
        trainer: Any,
        *,
        inputs: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        if inputs is None or len(inputs) != len(prompts):
            raise ValueError("TRL must provide dataset rows aligned with rollout prompts")
        try:
            example_ids = tuple(str(row["example_id"]) for row in inputs)
        except KeyError as error:
            raise ValueError("every online-RL dataset row requires an example_id") from error
        from .online_rl import TrlPolicyGenerator

        generator = TrlPolicyGenerator(trainer, tokenizer, request.policy, request.settings, request.training)
        step = int(trainer.state.global_step)
        started_at = time.perf_counter()
        with context.phase("rollout", {"backend": "trl", "logical_step": step}):
            rollouts = asyncio.run(
                request.bridge.run(
                    RolloutBatch(example_ids=example_ids, step=step, model_id=request.policy.id),
                    generator,
                )
            )
        elapsed = time.perf_counter() - started_at
        if len(rollouts) != len(inputs):
            raise ValueError("online-RL bridge returned a rollout count that does not match the trainer batch")
        completion_tokens = sum(len(rollout.completion_ids) for rollout in rollouts)
        context.metrics(
            {
                "train/rl/rollouts_attempted": len(inputs),
                "train/rl/rollouts_completed": len(rollouts),
                "train/rl/rollouts_failed": 0,
                "train/rl/rollouts_truncated": sum(rollout.is_truncated for rollout in rollouts),
                "train/rl/rollouts_unscorable": sum(not math.isfinite(rollout.reward) for rollout in rollouts),
                "train/rl/time/rollout_seconds": elapsed,
                "train/rl/rollout_tokens_per_second": completion_tokens / elapsed if elapsed > 0 else 0.0,
            },
            step=step,
        )
        attributes = {
            "technique": request.settings.algorithm if isinstance(request, GRPORequest) else "sampo",
            "model_variant_id": request.policy.id,
            "training_settings_id": request.settings.id,
        }
        for rollout in rollouts:
            trace = rollout.trace
            context.trace(
                TraceObservation(
                    trace_type=trace.trace_type,
                    external_id=trace.external_id,
                    payload=trace.payload,
                    attributes={**trace.attributes, **attributes},
                )
            )
        if isinstance(request, GRPORequest):
            shaped_rewards = [
                shape_online_reward(request.settings, rollout.reward, len(rollout.completion_ids))
                for rollout in rollouts
            ]
        else:
            shaped_rewards = [rollout.reward for rollout in rollouts]
        result = {
            "prompt_ids": [list(rollout.prompt_ids) for rollout in rollouts],
            "completion_ids": [list(rollout.completion_ids) for rollout in rollouts],
            "logprobs": [list(rollout.sampling_logprobs) for rollout in rollouts],
            "env_mask": [list(rollout.env_mask) for rollout in rollouts],
            "rollout_reward": shaped_rewards,
            "task_reward": [rollout.reward for rollout in rollouts],
            "algorithm_reward": shaped_rewards,
            "is_truncated": [rollout.is_truncated for rollout in rollouts],
            "rollout_trace_id": [rollout.trace.external_id for rollout in rollouts],
        }
        if isinstance(request, SAMPORequest):
            advantages = compute_sampo_advantages(request.settings, example_ids, rollouts)
            result["precomputed_advantages"] = [list(values) for values in advantages.token_advantages]
            flat_turn_advantages = [value for values in advantages.turn_advantages for value in values]
            flat_group_sizes = [value for values in advantages.anchor_group_sizes for value in values]
            context.metrics(
                {
                    "train/rl/episode_advantage_mean": (
                        sum(advantages.episode_advantages) / len(advantages.episode_advantages)
                    ),
                    "train/rl/turn_advantage_mean": (sum(flat_turn_advantages) / len(flat_turn_advantages)),
                    "train/rl/anchor_group_size_mean": sum(flat_group_sizes) / len(flat_group_sizes),
                    "train/rl/sparse_reward_projection_fraction": (
                        sum(advantages.used_sparse_rewards) / len(advantages.used_sparse_rewards)
                    ),
                },
                step=step,
            )
        if actor_update is not None:
            actor_update.start(step + 1)
        return result

    return run_rollouts


def _bridge_reward(rollout_reward: list[float], **_: Any) -> list[float]:
    """Return rewards already computed by the native online-RL environment."""

    return [float(value) for value in rollout_reward]


def _online_rl_arguments(
    request: GRPORequest | SAMPORequest,
    output_dir: Path,
    template_kwargs: dict[str, Any],
) -> dict[str, Any]:
    arguments = trainer_arguments(request.settings.loop, output_dir)
    arguments.pop("max_length")
    settings = request.settings
    if isinstance(settings, SAMPOSettings):
        is_sampo = True
        advantage_scaling = "group"
        loss_type = "grpo"
        epsilon_high = settings.clip_epsilon_high
        importance_sampling_mode = "sequence_truncate"
        importance_sampling_clip_min = 0.1
        importance_sampling_clip_max = 3.0
    else:
        is_sampo = False
        advantage_scaling = settings.advantage_scaling
        loss_type = settings.algorithm
        epsilon_high = settings.resolved_clip_epsilon_high
        importance_sampling_mode = settings.importance_sampling_mode
        importance_sampling_clip_min = settings.importance_sampling_clip_min
        importance_sampling_clip_max = settings.importance_sampling_clip_max
    use_liger_kernel = request.training.backend_options.get("use_liger_kernel", False)
    if not isinstance(use_liger_kernel, bool):
        raise ValueError("TRL GRPO use_liger_kernel must be a boolean")
    liger_loss_compiled = request.training.backend_options.get("liger_loss_compiled", True)
    if not isinstance(liger_loss_compiled, bool):
        raise ValueError("TRL GRPO liger_loss_compiled must be a boolean")
    if not use_liger_kernel and "liger_loss_compiled" in request.training.backend_options:
        raise ValueError("TRL GRPO liger_loss_compiled requires use_liger_kernel=true")
    logits_chunk_size = request.training.backend_options.get("logits_chunk_size")
    if logits_chunk_size is not None and (
        isinstance(logits_chunk_size, bool) or not isinstance(logits_chunk_size, int) or logits_chunk_size < 1
    ):
        raise ValueError("TRL GRPO logits_chunk_size must be a positive integer")
    arguments.update(
        {
            "remove_unused_columns": False,
            "shuffle_dataset": False,
            "num_generations": request.settings.num_generations,
            "generation_batch_size": (request.settings.num_prompts_per_step * request.settings.num_generations),
            "max_completion_length": request.settings.max_completion_length,
            "chat_template_kwargs": template_kwargs,
            "beta": request.settings.beta,
            "loss_type": loss_type,
            "epsilon": request.settings.clip_epsilon_low,
            "epsilon_high": epsilon_high,
            "scale_rewards": advantage_scaling,
            "dynamic_sampling": (request.settings.dynamic_sampling is not None if not is_sampo else True),
            "dynamic_sampling_max_batches": (
                request.settings.dynamic_sampling.max_candidate_batches
                if request.settings.dynamic_sampling is not None
                else 10
            ),
            "dynamic_sampling_reward_std_epsilon": (0.0),
            "mask_truncated_completions": request.settings.mask_truncated_completions,
            "importance_sampling_level": "sequence" if is_sampo else "token",
            "use_precomputed_advantages": is_sampo,
            "use_liger_kernel": use_liger_kernel,
            "logits_chunk_size": logits_chunk_size,
            "use_vllm": request.inference.backend.split("@", 1)[0] == "vllm",
            "temperature": _sampling_number(request, "temperature", 1.0),
            "top_p": _sampling_number(request, "top_p", 1.0),
        }
    )
    if request.inference.backend.split("@", 1)[0] == "vllm":
        rollout = request.inference.engine
        speculative_config, engine_kwargs = vllm_rollout_options(request.policy, rollout)
        arguments.update(
            {
                "vllm_mode": rollout.get("mode"),
                "vllm_enable_sleep_mode": rollout.get("sleep_during_optimization", False),
                "vllm_gpu_memory_utilization": rollout.get("gpu_memory_utilization"),
                "vllm_tensor_parallel_size": rollout.get("tensor_parallel_size", 1),
                "vllm_max_model_length": rollout.get("max_model_len"),
                "vllm_speculative_config": speculative_config,
                "vllm_engine_kwargs": engine_kwargs,
                "vllm_weight_name_prefix": rollout.get("weight_name_prefix"),
                "vllm_weight_sync_mode": rollout.get("weight_sync_mode", "full"),
                "vllm_model_impl": "vllm",
                "vllm_importance_sampling_correction": True,
                "vllm_importance_sampling_mode": importance_sampling_mode,
                "vllm_importance_sampling_clip_min": importance_sampling_clip_min,
                "vllm_importance_sampling_clip_max": importance_sampling_clip_max,
            }
        )
    return arguments


def _configure_liger_loss(
    trainer: Any,
    request: GRPORequest | SAMPORequest,
) -> None:
    if not request.training.backend_options.get("use_liger_kernel", False):
        return
    compiled = request.training.backend_options.get("liger_loss_compiled", True)
    loss = getattr(trainer, "liger_loss", None)
    if loss is None or not hasattr(loss, "compiled"):
        raise RuntimeError("the selected TRL Liger loss does not expose compiled mode")
    loss.compiled = compiled


def _grpo_arguments(request: GRPORequest, output_dir: Path, template_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Compatibility name for the GRPO-only argument translator."""

    return _online_rl_arguments(request, output_dir, template_kwargs)


def _sampling_number(request: GRPORequest | SAMPORequest, key: str, default: float) -> float:
    value = request.inference.sampling.get(key)
    return float(value) if isinstance(value, (int, float)) else default


def _online_rl_runtime_attributes(request: GRPORequest | SAMPORequest) -> dict[str, JsonValue]:
    """Describe selected GRPO runtime features without claiming observed performance."""

    engine = request.inference.engine
    speculative = engine.get("speculative_config")
    backend_product, separator, backend_revision = request.training.backend.partition("@")
    attributes: dict[str, JsonValue] = {
        "training_backend": backend_product,
        "backend_source_revision": backend_revision if separator else "unresolved",
        "training_binding_id": request.training.id,
        "inference_binding_id": request.inference.id,
        "inference_backend": request.inference.backend,
        "rollout_mode": engine.get("mode", "colocate"),
        "update_kind": request.training.update.kind,
        "world_size": request.training.target.placement.get("world_size", 1),
        "rollout_precision": engine.get("dtype", request.policy.weight_precision),
        "kv_cache_dtype": engine.get("kv_cache_dtype", "auto"),
        "max_model_len": engine.get(
            "max_model_len", request.settings.max_prompt_length + request.settings.max_completion_length
        ),
        "use_liger_kernel": request.training.backend_options.get("use_liger_kernel", False),
        "liger_loss_compiled": request.training.backend_options.get("liger_loss_compiled", True),
        "logits_chunk_size": request.training.backend_options.get("logits_chunk_size"),
        "online_rl_algorithm": request.settings.algorithm if isinstance(request, GRPORequest) else "sampo",
        "advantage_scaling": (request.settings.advantage_scaling if isinstance(request, GRPORequest) else "group"),
        "clip_epsilon_low": request.settings.clip_epsilon_low,
        "clip_epsilon_high": (
            request.settings.resolved_clip_epsilon_high
            if isinstance(request, GRPORequest)
            else request.settings.clip_epsilon_high
        ),
        "mask_truncated_completions": request.settings.mask_truncated_completions,
        "dynamic_sampling": request.settings.dynamic_sampling is not None,
        "dynamic_sampling_max_candidate_batches": (
            request.settings.dynamic_sampling.max_candidate_batches
            if request.settings.dynamic_sampling is not None
            else None
        ),
    }
    if isinstance(request, GRPORequest):
        attributes["overlong_buffer_tokens"] = request.settings.overlong_buffer_tokens
        attributes["overlong_penalty_factor"] = request.settings.overlong_penalty_factor
    else:
        attributes["discount_gamma"] = request.settings.discount_gamma
        attributes["step_advantage_weight"] = request.settings.step_advantage_weight
        attributes["advantage_normalization"] = request.settings.advantage_normalization
    if isinstance(speculative, Mapping):
        attributes["speculative_method"] = speculative.get("method")
        attributes["num_speculative_tokens"] = speculative.get("num_speculative_tokens")
    return attributes


def _grpo_runtime_attributes(request: GRPORequest) -> dict[str, JsonValue]:
    """Compatibility name for the GRPO-only runtime evidence translator."""

    return _online_rl_runtime_attributes(request)


__all__ = ["run_grpo", "run_sampo"]
