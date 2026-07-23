"""TRL SFT translation over renderer-pretokenized examples."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from posttrain.common import RunContext
from posttrain.data import SupervisedDataset

from ...rendering import RenderedSFTExample, render_supervised
from ...requests import SFTRequest
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


def run_sft(
    context: RunContext,
    request: SFTRequest,
    dataset_snapshot: SupervisedDataset,
    validation_snapshot: SupervisedDataset | None,
    output_dir: Path,
) -> BackendTrainingResult:
    try:
        from trl.trainer.sft_config import SFTConfig  # pyright: ignore[reportMissingImports]
        from trl.trainer.sft_trainer import SFTTrainer  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise RuntimeError("install posttrain-train with the trl extra") from error

    imports = framework_imports()
    emit_runtime_versions(context, imports)
    with context.phase("model_loading", {"backend": "trl"}):
        tokenizer = load_tokenizer(request.model, imports)
        model = load_trainable_model(request.model, request.training.update, request.settings.loop, imports)
    with context.phase("data_preparation", {"backend": "trl"}):
        samples = render_supervised(
            tokenizer,
            request.model,
            dataset_snapshot,
            request.training.renderer,
            max_length=request.settings.loop.max_length,
        )
        _emit_rendered_profile(context, samples, request.settings.loop.max_length, "train/data")
        dataset = imports["Dataset"].from_list(
            [
                {
                    "example_id": sample.id,
                    "input_ids": list(sample.input_ids),
                    "labels": list(sample.labels),
                }
                for sample in samples
            ]
        )
        validation_dataset = None
        if validation_snapshot is not None:
            validation_samples = render_supervised(
                tokenizer,
                request.model,
                validation_snapshot,
                request.training.renderer,
                max_length=request.settings.loop.max_length,
            )
            _emit_rendered_profile(
                context,
                validation_samples,
                request.settings.loop.max_length,
                "train/validation/data",
            )
            validation_dataset = imports["Dataset"].from_list(
                [
                    {
                        "example_id": sample.id,
                        "input_ids": list(sample.input_ids),
                        "labels": list(sample.labels),
                    }
                    for sample in validation_samples
                ]
            )
    emit_parameter_counts(context, model, request.training.update)
    arguments = _sft_arguments(request, output_dir)
    validation = request.settings.validation
    callback = callback_type(context, imports)()

    ObservedSFTTrainer = _observed_sft_trainer_type(SFTTrainer, context)
    with context.phase("runtime_initialization", {"backend": "trl"}):
        trainer = ObservedSFTTrainer(
            model=model,
            args=SFTConfig(**arguments),
            train_dataset=dataset,
            eval_dataset=validation_dataset,
            processing_class=tokenizer,
            callbacks=[callback],
        )
    resume = str(request.resume_from.path) if request.resume_from is not None else None
    with trainer_lifecycle(trainer):
        with context.phase("actor_update", {"backend": "trl"}):
            train_output = trainer.train(resume_from_checkpoint=resume)
        if (
            validation is not None
            and validation.at_end
            and not _evaluated_at_step(
                trainer.state.log_history,
                int(trainer.state.global_step),
            )
        ):
            trainer.evaluate()
        with context.phase("artifact_export", {"backend": "trl"}):
            return finish_training(
                context,
                trainer,
                train_output,
                tokenizer,
                output_dir.parent,
                "sft",
                request.training.update,
                imports,
            )


def _observed_sft_trainer_type(parent: type[Any], context: RunContext) -> type[Any]:
    class ObservedSFTTrainer(parent):
        """Separate validation host evidence from the enclosing update loop."""

        def evaluate(self, *args: Any, **kwargs: Any) -> Any:
            with context.phase("evaluation", {"backend": "trl"}):
                return super().evaluate(*args, **kwargs)

    return ObservedSFTTrainer


def _runtime_bool(request: SFTRequest, name: str, default: bool) -> bool:
    value = request.training.runtime.get(name, default)
    if not isinstance(value, bool):
        raise ValueError(f"SFT training runtime {name!r} must be a boolean")
    return value


def _sft_arguments(request: SFTRequest, output_dir: Path) -> dict[str, Any]:
    arguments = trainer_arguments(request.settings.loop, output_dir)
    arguments.update(
        {
            "dataset_kwargs": {"skip_prepare_dataset": True},
            "completion_only_loss": False,
            "assistant_only_loss": False,
            "packing": False,
            "shuffle_dataset": False,
            "use_liger_kernel": _runtime_bool(request, "use_liger_kernel", False),
        }
    )
    validation = request.settings.validation
    if validation is not None:
        arguments.update(
            {
                "eval_strategy": "steps",
                "eval_steps": validation.steps,
                "eval_on_start": validation.on_start,
                "per_device_eval_batch_size": (
                    validation.per_device_batch_size or request.settings.loop.per_device_batch_size
                ),
                "prediction_loss_only": True,
            }
        )
    return arguments


def _evaluated_at_step(log_history: Sequence[Mapping[str, object]], step: int) -> bool:
    return any(entry.get("step") == step and "eval_loss" in entry for entry in log_history)


def _emit_rendered_profile(
    context: RunContext,
    samples: tuple[RenderedSFTExample, ...],
    max_length: int,
    prefix: str,
) -> None:
    rendered = tuple(samples)
    if not rendered:
        raise ValueError("rendered SFT population cannot be empty")
    lengths = sorted(len(sample.input_ids) for sample in rendered)
    input_tokens = sum(lengths)
    supervised_tokens = sum(sum(label != -100 for label in sample.labels) for sample in rendered)
    source_tokens = sum(sample.source_length for sample in rendered)
    source_supervised_tokens = sum(sample.source_supervised_tokens for sample in rendered)
    truncated_examples = sum(sample.source_length > len(sample.input_ids) for sample in rendered)
    context.metrics(
        {
            f"{prefix}/examples": len(rendered),
            f"{prefix}/input_tokens": input_tokens,
            f"{prefix}/supervised_tokens": supervised_tokens,
            f"{prefix}/supervision_token_ratio": supervised_tokens / input_tokens,
            f"{prefix}/truncated_examples": truncated_examples,
            f"{prefix}/truncated_tokens": source_tokens - input_tokens,
            f"{prefix}/truncated_supervised_tokens": source_supervised_tokens - supervised_tokens,
            f"{prefix}/truncation_rate": truncated_examples / len(rendered),
            f"{prefix}/max_length_utilization": input_tokens / (len(rendered) * max_length),
            f"{prefix}/sequence_length_p50": _nearest_rank(lengths, 0.50),
            f"{prefix}/sequence_length_p90": _nearest_rank(lengths, 0.90),
            f"{prefix}/sequence_length_p99": _nearest_rank(lengths, 0.99),
            f"{prefix}/sequence_length_max": lengths[-1],
        }
    )


def _nearest_rank(values: list[int], quantile: float) -> int:
    return values[max(0, math.ceil(quantile * len(values)) - 1)]


__all__ = ["run_sft"]
