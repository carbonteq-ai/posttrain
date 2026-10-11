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
    checkpoint_callback_type,
    emit_parameter_counts,
    emit_runtime_versions,
    finish_training,
    framework_imports,
    load_frozen_adapter_reference,
    load_tokenizer,
    load_trainable_model,
    preserve_recovery_checkpoint_after_error,
    trainer_arguments,
    trainer_lifecycle,
)
from .precision_runtime import require_default_precision


def masked_completion(
    completion_mask: Any, examples: list[dict[str, Any]], max_length: int | None, truncation_mode: str
) -> Any:
    """Rebuild TRL's padded completion mask (chosen rows, then rejected) from per-token masks.

    Rows follow TRL's layout: prompt tokens 0, completion tokens from the example's mask,
    truncated like the input ids and right-padded with 0."""

    import torch

    masks = [[0] * len(example["prompt_ids"]) + list(example["chosen_mask"]) for example in examples]
    masks += [[0] * len(example["prompt_ids"]) + list(example["rejected_mask"]) for example in examples]
    if max_length is not None:
        window = slice(None, max_length) if truncation_mode == "keep_start" else slice(-max_length, None)
        masks = [mask[window] for mask in masks]
    rebuilt = torch.zeros_like(completion_mask)
    for row, mask in enumerate(masks):
        rebuilt[row, : len(mask)] = torch.tensor(mask, dtype=rebuilt.dtype)
    return rebuilt


def run_dpo(
    context: RunContext,
    request: DPORequest,
    dataset_snapshot: PreferenceDataset,
    output_dir: Path,
) -> BackendTrainingResult:
    require_default_precision(request.training.backend_options, "DPO")
    try:
        from trl.trainer.dpo_config import DPOConfig  # pyright: ignore[reportMissingImports]
        from trl.trainer.dpo_trainer import (  # pyright: ignore[reportMissingImports]
            DataCollatorForPreference,
            DPOTrainer,
        )
    except ImportError as error:
        raise RuntimeError("install posttrain-train with the trl extra") from error

    imports = framework_imports()
    emit_runtime_versions(context, imports)
    with context.phase("model_loading", {"backend": "trl"}):
        tokenizer = load_tokenizer(request.model, imports)
        model = load_trainable_model(request.model, request.training.update, request.settings.loop, imports)
        # Continuing an adapter (DPO after SFT): the reference is that adapter's policy, frozen.
        reference = load_frozen_adapter_reference(request.model, imports)
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
                    "chosen_mask": list(sample.chosen_mask),
                    "rejected_mask": list(sample.rejected_mask),
                }
                for sample in samples
            ]
        )
    emit_parameter_counts(context, model, request.training.update)
    arguments = trainer_arguments(request.settings.loop, output_dir)
    arguments.update(
        {
            "beta": request.settings.beta,
            **(
                {"loss_type": ["sigmoid", "sft"], "loss_weights": [1.0, request.settings.sft_weight]}
                if request.settings.sft_weight
                else {"loss_type": ["sigmoid"]}
            ),
            "use_liger_kernel": request.settings.loss_kernel == "liger",
            "precompute_ref_log_probs": False,
            "padding_free": False,
        }
    )
    callback = callback_type(context, imports)()
    checkpoint_callback = checkpoint_callback_type(
        context,
        imports,
        model=request.model,
        technique="dpo",
        settings=request.settings,
        update=request.training.update,
        workspace=output_dir.parent,
    )()

    class MaskedPreferenceCollator(DataCollatorForPreference):
        """TRL's preference collator with the renderer's per-token completion masks.

        TRL trains every completion token; in a multi-turn completion only the assistant
        tokens are policy output, so tool results are masked out of the preference loss."""

        def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
            output = super().torch_call(examples)
            output["completion_mask"] = masked_completion(
                output["completion_mask"], examples, self.max_length, self.truncation_mode
            )
            return output

    class PretokenizedDPOTrainer(DPOTrainer):
        """Version-pinned adapter for renderer-produced TRL collator columns."""

        def _set_signature_columns_if_needed(self) -> None:
            super()._set_signature_columns_if_needed()
            self._signature_columns = [*self._signature_columns, "chosen_mask", "rejected_mask"]

        def _prepare_dataset(self, dataset: Any, processing_class: Any, args: Any, dataset_name: str) -> Any:
            required = {"prompt_ids", "chosen_ids", "rejected_ids"}
            if required.issubset(set(dataset.column_names)):
                return dataset
            return super()._prepare_dataset(dataset, processing_class, args, dataset_name)

    with context.phase("runtime_initialization", {"backend": "trl"}):
        config = DPOConfig(**arguments)
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        trainer = PretokenizedDPOTrainer(
            model=model,
            ref_model=reference,
            args=config,
            data_collator=MaskedPreferenceCollator(
                pad_token_id=tokenizer.pad_token_id,
                max_length=config.max_length,
                truncation_mode=config.truncation_mode,
                pad_to_multiple_of=config.pad_to_multiple_of,
            ),
            train_dataset=dataset,
            processing_class=tokenizer,
            callbacks=[callback, checkpoint_callback],
        )
    resume = str(request.resume_from.path) if request.resume_from is not None else None
    with trainer_lifecycle(trainer):
        try:
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
        except BaseException as error:
            preserve_recovery_checkpoint_after_error(
                context,
                trainer,
                error,
                technique="dpo",
                model=request.model,
                settings=request.settings,
                update=request.training.update,
                imports=imports,
            )
            raise


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
