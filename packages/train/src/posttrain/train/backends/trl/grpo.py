"""TRL GRPO translation over task-neutral rollout prompts and rewards."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

from posttrain.common import ExecutionContext, TraceObservation
from posttrain.common.cuda import TorchModule, activate_cuda_toolkit

from ...online_rl import RolloutBatch
from ...requests import GRPORequest
from .common import (
    BackendTrainingResult,
    callback_type,
    finish_training,
    framework_imports,
    load_tokenizer,
    load_trainable_model,
    trainer_arguments,
    trainer_lifecycle,
)


def run_grpo(context: ExecutionContext, request: GRPORequest, output_dir: Path) -> BackendTrainingResult:
    if request.profile.rollout.engine == "vllm":
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
    tokenizer = load_tokenizer(request.model, imports)
    rows = []
    template_kwargs = request.model.profile.conversation.reasoning_mode(
        request.profile.renderer.reasoning_mode
    ).kwargs()
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
        if len(rendered) > request.profile.max_prompt_length:
            raise ValueError(
                f"rollout example {example.id!r} has {len(rendered)} prompt tokens; "
                f"profile permits {request.profile.max_prompt_length}"
            )
        rows.append({"prompt": prompt, "example_id": example.id, **dict(example.metadata)})
    dataset = imports["Dataset"].from_list(rows)
    model = load_trainable_model(request.model, request.profile.qlora, request.profile.loop, imports)
    _emit_parameter_counts(context, model)
    arguments = _grpo_arguments(request, output_dir, template_kwargs)
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=cast(Any, _bridge_reward),
        rollout_func=cast(Any, _rollout_function(context, request, tokenizer)),
        args=GRPOConfig(**arguments),
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[callback_type(context, imports)()],
    )
    resume = str(request.resume_from.path) if request.resume_from is not None else None
    with trainer_lifecycle(trainer):
        train_output = trainer.train(resume_from_checkpoint=resume)
        return finish_training(trainer, train_output, tokenizer, output_dir.parent, "grpo", imports)


def _rollout_function(
    context: ExecutionContext,
    request: GRPORequest,
    tokenizer: Any,
) -> Any:
    """Translate TRL generation batches into the public online-RL bridge contract."""

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

        generator = TrlPolicyGenerator(trainer, tokenizer, request.model.profile, request.profile)
        rollouts = asyncio.run(
            request.bridge.run(
                RolloutBatch(
                    example_ids=example_ids,
                    step=int(trainer.state.global_step),
                    model_id=request.model.profile.id,
                ),
                generator,
            )
        )
        if len(rollouts) != len(inputs):
            raise ValueError("online-RL bridge returned a rollout count that does not match the trainer batch")
        attributes = {
            "technique": "grpo",
            "model_profile_id": request.model.profile.id,
            "training_profile_id": request.profile.id,
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
    arguments = trainer_arguments(request.profile.loop, output_dir)
    arguments.pop("max_length")
    arguments.update(
        {
            "remove_unused_columns": False,
            "shuffle_dataset": False,
            "num_generations": request.profile.num_generations,
            "generation_batch_size": request.profile.num_generations,
            "max_completion_length": request.profile.max_completion_length,
            "chat_template_kwargs": template_kwargs,
            "beta": request.profile.beta,
            "loss_type": "grpo",
            "scale_rewards": "group",
            "mask_truncated_completions": True,
            "use_vllm": request.profile.rollout.engine == "vllm",
            "temperature": request.profile.temperature,
            "top_p": request.profile.top_p,
        }
    )
    if request.profile.rollout.engine == "vllm":
        rollout = request.profile.rollout
        arguments.update(
            {
                "vllm_mode": rollout.vllm_mode,
                "vllm_enable_sleep_mode": rollout.sleep_during_optimization,
                "vllm_gpu_memory_utilization": rollout.gpu_memory_utilization,
                "vllm_tensor_parallel_size": rollout.tensor_parallel_size,
                "vllm_max_model_length": rollout.max_model_length,
                "vllm_speculative_config": rollout.speculative_config(),
                "vllm_engine_kwargs": _vllm_engine_kwargs(request),
                "vllm_weight_name_prefix": rollout.weight_name_prefix,
                "vllm_weight_sync_mode": rollout.weight_sync_mode,
                "vllm_model_impl": "vllm",
                "vllm_importance_sampling_correction": True,
                "vllm_importance_sampling_mode": rollout.importance_sampling_mode,
                "vllm_importance_sampling_clip_min": rollout.importance_sampling_clip_min,
                "vllm_importance_sampling_clip_max": rollout.importance_sampling_clip_max,
            }
        )
    return arguments


def _vllm_engine_kwargs(request: GRPORequest) -> dict[str, Any] | None:
    rollout = request.profile.rollout
    values: dict[str, Any] = {}
    if rollout.text_only:
        values["language_model_only"] = True
    if rollout.skip_multimodal_profiling:
        values["skip_mm_profiling"] = True
    if rollout.kv_cache_memory_bytes is not None:
        values["kv_cache_memory_bytes"] = rollout.kv_cache_memory_bytes
    return values or None


def _emit_parameter_counts(context: ExecutionContext, model: Any) -> None:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable < 1 or trainable >= total:
        raise RuntimeError(f"invalid PEFT parameter selection: trainable={trainable}, total={total}")
    context.metrics(
        {
            "train/parameters_total": total,
            "train/parameters_trainable": trainable,
            "train/parameters_trainable_fraction": trainable / total,
        }
    )


__all__ = ["run_grpo"]
