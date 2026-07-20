"""TRL GRPO translation over task-neutral rollout prompts and rewards."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from posttrain.common import ExecutionContext

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
    for example in request.dataset.examples:
        prompt = [{"role": "user", "content": example.prompt}]
        rendered = tokenizer.apply_chat_template(
            prompt,
            tokenize=True,
            add_generation_prompt=True,
            **template_kwargs,
        )
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
        reward_funcs=cast(Any, request.reward),
        args=GRPOConfig(**arguments),
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[callback_type(context, imports)()],
    )
    resume = str(request.resume_from.path) if request.resume_from is not None else None
    train_output = trainer.train(resume_from_checkpoint=resume)
    return finish_training(trainer, train_output, tokenizer, output_dir.parent, "grpo", imports)


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
            "use_vllm": False,
            "temperature": 1.0,
        }
    )
    return arguments


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
