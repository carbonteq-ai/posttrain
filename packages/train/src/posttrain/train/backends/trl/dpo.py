"""TRL DPO translation over renderer-pretokenized preference pairs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from posttrain.common import ExecutionContext

from ...rendering import render_preferences
from ...requests import DPORequest
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


def run_dpo(context: ExecutionContext, request: DPORequest, output_dir: Path) -> BackendTrainingResult:
    try:
        from trl.trainer.dpo_config import DPOConfig  # pyright: ignore[reportMissingImports]
        from trl.trainer.dpo_trainer import DPOTrainer  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise RuntimeError("install posttrain-train with the trl extra") from error

    imports = framework_imports()
    tokenizer = load_tokenizer(request.model, imports)
    samples = render_preferences(
        tokenizer,
        request.model.profile,
        request.dataset,
        request.profile.renderer,
        max_length=request.profile.loop.max_length,
    )
    dataset = imports["Dataset"].from_list(
        [
            {
                "example_id": sample.id,
                "prompt_ids": list(sample.prompt_ids),
                "chosen_ids": list(sample.chosen_ids),
                "rejected_ids": list(sample.rejected_ids),
            }
            for sample in samples
        ]
    )
    model = load_trainable_model(request.model, request.profile.qlora, request.profile.loop, imports)
    _emit_parameter_counts(context, model)
    arguments = trainer_arguments(request.profile.loop, output_dir)
    arguments.update(
        {
            "beta": request.profile.beta,
            "loss_type": ["sigmoid"],
            "use_liger_kernel": request.profile.loss_kernel == "liger",
            "precompute_ref_log_probs": False,
            "padding_free": False,
        }
    )
    callback = callback_type(context, imports)()

    class PretokenizedDPOTrainer(DPOTrainer):
        """Version-pinned adapter for renderer-produced TRL collator columns."""

        def _prepare_dataset(self, dataset: Any, processing_class: Any, args: Any, dataset_name: str) -> Any:
            required = {"prompt_ids", "chosen_ids", "rejected_ids"}
            if required.issubset(set(dataset.column_names)):
                return dataset
            return super()._prepare_dataset(dataset, processing_class, args, dataset_name)

    trainer = PretokenizedDPOTrainer(
        model=model,
        ref_model=None,
        args=DPOConfig(**arguments),
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[callback],
    )
    resume = str(request.resume_from.path) if request.resume_from is not None else None
    with trainer_lifecycle(trainer):
        train_output = trainer.train(resume_from_checkpoint=resume)
        return finish_training(trainer, train_output, tokenizer, output_dir.parent, "dpo", imports)


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


__all__ = ["run_dpo"]
