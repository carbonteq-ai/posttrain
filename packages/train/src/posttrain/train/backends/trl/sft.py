"""TRL SFT translation over renderer-pretokenized examples."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from posttrain.common import ExecutionContext

from ...rendering import render_supervised
from ...requests import SFTRequest
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


def run_sft(context: ExecutionContext, request: SFTRequest, output_dir: Path) -> BackendTrainingResult:
    try:
        from trl.trainer.sft_config import SFTConfig  # pyright: ignore[reportMissingImports]
        from trl.trainer.sft_trainer import SFTTrainer  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise RuntimeError("install posttrain-train with the trl extra") from error

    imports = framework_imports()
    tokenizer = load_tokenizer(request.model, imports)
    samples = render_supervised(
        tokenizer,
        request.model.profile,
        request.dataset,
        request.profile.renderer,
        max_length=request.profile.loop.max_length,
    )
    dataset = imports["Dataset"].from_list(
        [
            {"example_id": sample.id, "input_ids": list(sample.input_ids), "labels": list(sample.labels)}
            for sample in samples
        ]
    )
    model = load_trainable_model(request.model, request.profile.qlora, request.profile.loop, imports)
    _emit_parameter_counts(context, model)
    arguments = trainer_arguments(request.profile.loop, output_dir)
    arguments.update(
        {
            "dataset_kwargs": {"skip_prepare_dataset": True},
            "completion_only_loss": False,
            "assistant_only_loss": False,
            "packing": False,
            "shuffle_dataset": False,
        }
    )
    callback = callback_type(context, imports)()
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(**arguments),
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[callback],
    )
    resume = str(request.resume_from.path) if request.resume_from is not None else None
    with trainer_lifecycle(trainer):
        train_output = trainer.train(resume_from_checkpoint=resume)
        return finish_training(trainer, train_output, tokenizer, output_dir.parent, "sft", imports)


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


__all__ = ["run_sft"]
