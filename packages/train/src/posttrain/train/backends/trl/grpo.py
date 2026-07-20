"""TRL GRPO translation over task-neutral rollout prompts and rewards."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

from posttrain.common import ExecutionContext, TraceObservation
from posttrain.common.cuda import TorchModule, activate_cuda_toolkit

from ...data import CompletedRollout
from ...requests import GRPORequest
from .common import (
    BackendTrainingResult,
    callback_type,
    finish_training,
    framework_imports,
    load_tokenizer,
    load_trainable_model,
    trainer_arguments,
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
    for example in request.environment.dataset.examples:
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
        reward_funcs=cast(
            Any,
            _reward_function(
                context,
                request,
                frozenset(
                    token_id for token_id in (tokenizer.eos_token_id, tokenizer.pad_token_id) if token_id is not None
                ),
            ),
        ),
        args=GRPOConfig(**arguments),
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[callback_type(context, imports)()],
    )
    resume = str(request.resume_from.path) if request.resume_from is not None else None
    train_output = trainer.train(resume_from_checkpoint=resume)
    return finish_training(trainer, train_output, tokenizer, output_dir.parent, "grpo", imports)


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if not isinstance(completion, list) or not completion:
        raise TypeError("conversational GRPO completions must be non-empty message lists")
    message = completion[-1]
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise TypeError("GRPO completion messages require string content")
    return cast(str, message["content"])


def _reward_function(
    context: ExecutionContext,
    request: GRPORequest,
    terminal_token_ids: frozenset[int],
) -> Any:
    """Translate TRL callback arguments into the public online-RL environment contract."""

    async def score_rollouts(
        *,
        completions: list[Any],
        completion_ids: list[list[int]],
        example_id: list[str],
        trainer_state: Any,
        **_: Any,
    ) -> list[float]:
        if len(completions) != len(example_id) or len(completions) != len(completion_ids):
            raise ValueError("completion and rollout identity counts differ")
        step = int(trainer_state.global_step)
        scores = await asyncio.gather(
            *(
                request.environment.score(
                    CompletedRollout(
                        example_id=identifier,
                        completion=_completion_text(completion),
                        token_ids=tuple(int(value) for value in token_ids),
                        step=step,
                        terminated=bool(token_ids) and token_ids[-1] in terminal_token_ids,
                        model_id=request.model.profile.id,
                    )
                )
                for completion, token_ids, identifier in zip(
                    completions,
                    completion_ids,
                    example_id,
                    strict=True,
                )
            )
        )
        attributes = {
            "technique": "grpo",
            "model_profile_id": request.model.profile.id,
            "training_profile_id": request.profile.id,
        }
        for score in scores:
            trace = score.trace
            context.trace(
                TraceObservation(
                    trace_type=trace.trace_type,
                    external_id=trace.external_id,
                    payload=trace.payload,
                    attributes={**trace.attributes, **attributes},
                )
            )
        return [score.reward for score in scores]

    return score_rollouts


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
            "temperature": 1.0,
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
