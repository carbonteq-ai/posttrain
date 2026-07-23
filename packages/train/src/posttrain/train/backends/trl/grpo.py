"""TRL GRPO translation over task-neutral rollout prompts and rewards."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from posttrain.common import JsonValue, RunContext, TraceObservation
from posttrain.common.cuda import TorchModule, activate_cuda_toolkit

from ...grpo_observations import GRPOObservationFeatures, normalize_grpo_metrics
from ...online_rl import RolloutBatch
from ...requests import GRPORequest
from .common import (
    BackendTrainingResult,
    callback_type,
    emit_parameter_counts,
    emit_runtime_versions,
    finish_training,
    framework_imports,
    load_tokenizer,
    load_trainable_model,
    trainer_arguments,
    trainer_lifecycle,
    vllm_rollout_options,
)


def run_grpo(
    context: RunContext,
    request: GRPORequest,
    output_dir: Path,
) -> BackendTrainingResult:
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
    _emit_parameter_counts(context, model, request.training.update.kind)
    arguments = _grpo_arguments(request, output_dir, template_kwargs)
    observation_features = GRPOObservationFeatures.from_request(request)
    context.event("grpo_runtime_resolved", _grpo_runtime_attributes(request))

    def normalize_metrics(step: int, native: Mapping[str, object]) -> Mapping[str, float]:
        return normalize_grpo_metrics(
            backend="trl",
            step=step,
            native=native,
            features=observation_features,
        ).metrics

    with context.phase("runtime_initialization", {"backend": "trl"}):
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=cast(Any, _bridge_reward),
            rollout_func=cast(Any, _rollout_function(context, request, tokenizer)),
            args=GRPOConfig(**arguments),
            train_dataset=dataset,
            processing_class=tokenizer,
            callbacks=[callback_type(context, imports, metric_normalizer=normalize_metrics)()],
        )
    resume = str(request.resume_from.path) if request.resume_from is not None else None
    with trainer_lifecycle(trainer):
        with context.phase("actor_update", {"backend": "trl"}):
            train_output = trainer.train(resume_from_checkpoint=resume)
        with context.phase("artifact_export", {"backend": "trl"}):
            return finish_training(
                context,
                trainer,
                train_output,
                tokenizer,
                output_dir.parent,
                "grpo",
                request.training.update,
                imports,
            )


def _rollout_function(
    context: RunContext,
    request: GRPORequest,
    tokenizer: Any,
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
        unscorable = sum(not math.isfinite(rollout.reward) for rollout in rollouts)
        context.metrics(
            {
                "train/rl/rollouts_attempted": len(inputs),
                "train/rl/rollouts_completed": len(rollouts),
                "train/rl/rollouts_failed": 0,
                "train/rl/rollouts_truncated": sum(rollout.is_truncated for rollout in rollouts),
                "train/rl/rollouts_unscorable": unscorable,
                "train/rl/time/rollout_seconds": elapsed,
                "train/rl/rollout_tokens_per_second": completion_tokens / elapsed if elapsed > 0 else 0.0,
            },
            step=step,
        )
        attributes = {
            "technique": "grpo",
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
        return {
            "prompt_ids": [list(rollout.prompt_ids) for rollout in rollouts],
            "completion_ids": [list(rollout.completion_ids) for rollout in rollouts],
            "logprobs": [list(rollout.sampling_logprobs) for rollout in rollouts],
            "env_mask": [list(rollout.env_mask) for rollout in rollouts],
            "rollout_reward": [rollout.reward for rollout in rollouts],
            "is_truncated": [rollout.is_truncated for rollout in rollouts],
            "rollout_trace_id": [rollout.trace.external_id for rollout in rollouts],
        }

    return run_rollouts


def _bridge_reward(rollout_reward: list[float], **_: Any) -> list[float]:
    """Return rewards already computed by the native online-RL environment."""

    return [float(value) for value in rollout_reward]


def _grpo_arguments(request: GRPORequest, output_dir: Path, template_kwargs: dict[str, Any]) -> dict[str, Any]:
    arguments = trainer_arguments(request.settings.loop, output_dir)
    arguments.pop("max_length")
    use_liger_kernel = request.training.runtime.get("use_liger_kernel", False)
    if not isinstance(use_liger_kernel, bool):
        raise ValueError("TRL GRPO use_liger_kernel must be a boolean")
    logits_chunk_size = request.training.runtime.get("logits_chunk_size")
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
            "loss_type": "grpo",
            "scale_rewards": "group",
            "mask_truncated_completions": True,
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
                "vllm_importance_sampling_mode": request.settings.importance_sampling_mode,
                "vllm_importance_sampling_clip_min": request.settings.importance_sampling_clip_min,
                "vllm_importance_sampling_clip_max": request.settings.importance_sampling_clip_max,
            }
        )
    return arguments


def _sampling_number(request: GRPORequest, key: str, default: float) -> float:
    value = request.inference.sampling.get(key)
    return float(value) if isinstance(value, (int, float)) else default


def _grpo_runtime_attributes(request: GRPORequest) -> dict[str, JsonValue]:
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
        "use_liger_kernel": request.training.runtime.get("use_liger_kernel", False),
        "logits_chunk_size": request.training.runtime.get("logits_chunk_size"),
    }
    if isinstance(speculative, Mapping):
        attributes["speculative_method"] = speculative.get("method")
        attributes["num_speculative_tokens"] = speculative.get("num_speculative_tokens")
    return attributes


def _emit_parameter_counts(
    context: RunContext,
    model: Any,
    update_kind: str,
) -> None:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable < 1 or (update_kind != "full" and trainable >= total):
        raise RuntimeError(f"invalid PEFT parameter selection: trainable={trainable}, total={total}")
    context.metrics(
        {
            "train/parameters_total": total,
            "train/parameters_trainable": trainable,
            "train/parameters_trainable_fraction": trainable / total,
        }
    )


__all__ = ["run_grpo"]
