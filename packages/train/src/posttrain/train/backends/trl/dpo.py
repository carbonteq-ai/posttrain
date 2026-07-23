"""TRL DPO translation over renderer-pretokenized preference pairs."""

from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Any

from posttrain.common import RunContext
from posttrain.data import PreferenceDataset

from ...rendering import RenderedPreferenceExample, render_preferences
from ...requests import DPORequest
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
)


def run_dpo(
    context: RunContext,
    request: DPORequest,
    dataset_snapshot: PreferenceDataset,
    output_dir: Path,
) -> BackendTrainingResult:
    try:
        from trl.trainer.dpo_config import DPOConfig  # pyright: ignore[reportMissingImports]
        from trl.trainer.dpo_trainer import DPOTrainer  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise RuntimeError("install posttrain-train with the trl extra") from error

    imports = framework_imports()
    emit_runtime_versions(context, imports)
    with context.phase("model_loading", {"backend": "trl"}):
        tokenizer = load_tokenizer(request.model, imports)
        model = load_trainable_model(request.model, request.training.update, request.settings.loop, imports)
    with context.phase("data_preparation", {"backend": "trl"}):
        samples = render_preferences(
            tokenizer,
            request.model,
            dataset_snapshot,
            request.training.renderer,
            max_length=request.settings.loop.max_length,
        )
        _emit_preference_profile(
            context,
            samples,
            dataset_snapshot,
            request.settings.loop.max_length,
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
    emit_parameter_counts(context, model, request.training.update)
    _emit_parameter_counts(context, model)
    arguments = trainer_arguments(request.settings.loop, output_dir)
    arguments.update(
        {
            "beta": request.settings.beta,
            "loss_type": ["sigmoid"],
            "use_liger_kernel": request.settings.loss_kernel == "liger",
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

    with context.phase("runtime_initialization", {"backend": "trl"}):
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
        with context.phase("actor_update", {"backend": "trl"}):
            train_output = trainer.train(resume_from_checkpoint=resume)
        with context.phase("artifact_export", {"backend": "trl"}):
            return finish_training(
                context,
                trainer,
                train_output,
                tokenizer,
                output_dir.parent,
                "dpo",
                request.training.update,
                imports,
            )


def _emit_parameter_counts(context: RunContext, model: Any) -> None:
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


def _emit_preference_profile(
    context: RunContext,
    rendered: tuple[RenderedPreferenceExample, ...],
    dataset: PreferenceDataset,
    max_length: int,
) -> None:
    """Record immutable rendered-pair evidence before optimization starts."""

    pair_count = len(rendered)
    prompt_lengths = tuple(len(sample.prompt_ids) for sample in rendered)
    chosen_lengths = tuple(len(sample.chosen_ids) for sample in rendered)
    rejected_lengths = tuple(len(sample.rejected_ids) for sample in rendered)
    prompt_tokens = sum(prompt_lengths)
    chosen_tokens = sum(chosen_lengths)
    rejected_tokens = sum(rejected_lengths)
    chosen_longer = sum(len(sample.chosen_ids) > len(sample.rejected_ids) for sample in rendered)
    utilized_tokens = sum(
        len(sample.prompt_ids) + max(len(sample.chosen_ids), len(sample.rejected_ids)) for sample in rendered
    )
    headroom = tuple(
        max_length - len(sample.prompt_ids) - max(len(sample.chosen_ids), len(sample.rejected_ids))
        for sample in rendered
    )
    score_margins = tuple(
        example.chosen_score - example.rejected_score
        for example in dataset.examples
        if example.chosen_score is not None and example.rejected_score is not None
    )
    values: dict[str, float] = {
        "train/data/preference_pairs": float(pair_count),
        "train/data/prompt_tokens_mean": prompt_tokens / pair_count,
        "train/data/chosen_tokens_mean": chosen_tokens / pair_count,
        "train/data/rejected_tokens_mean": rejected_tokens / pair_count,
        "train/data/prompt_tokens_p95": _nearest_rank(prompt_lengths, 0.95),
        "train/data/chosen_tokens_p95": _nearest_rank(chosen_lengths, 0.95),
        "train/data/rejected_tokens_p95": _nearest_rank(rejected_lengths, 0.95),
        "train/data/max_length_headroom_min": float(min(headroom)),
        "train/data/chosen_longer_fraction": chosen_longer / pair_count,
        "train/data/preference_score_coverage": len(score_margins) / pair_count,
        "train/data/max_length_utilization": utilized_tokens / (pair_count * max_length),
    }
    if score_margins:
        values["train/data/preference_score_margin_mean"] = sum(score_margins) / len(score_margins)
    context.metrics(values)


def _nearest_rank(values: tuple[int, ...], quantile: float) -> float:
    """Return a deterministic nearest-rank quantile for a non-empty population."""

    ordered = tuple(sorted(values))
    index = max(0, min(len(ordered) - 1, ceil(quantile * len(ordered)) - 1))
    return float(ordered[index])


__all__ = ["run_dpo"]
