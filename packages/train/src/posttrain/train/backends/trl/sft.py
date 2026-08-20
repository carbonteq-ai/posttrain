"""TRL SFT translation over text and vision-language examples."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from posttrain.common import RunContext
from posttrain.data import SupervisedDataset, SupervisedExample

from ...rendering import RenderedSFTExample, render_supervised
from ...requests import SFTRequest
from .common import (
    BackendTrainingResult,
    callback_type,
    checkpoint_callback_type,
    emit_parameter_counts,
    emit_runtime_versions,
    finish_training,
    framework_imports,
    load_processor,
    load_tokenizer,
    load_trainable_model,
    preserve_recovery_checkpoint_after_error,
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

    visual = _visual_modality(dataset_snapshot, validation_snapshot, request)
    imports = framework_imports()
    emit_runtime_versions(context, imports)
    with context.phase("model_loading", {"backend": "trl"}):
        processing_class = load_processor(request.model, imports) if visual else load_tokenizer(request.model, imports)
        model = load_trainable_model(request.model, request.training.update, request.settings.loop, imports)
    with context.phase("data_preparation", {"backend": "trl"}):
        if visual:
            dataset = _visual_dataset(
                context,
                dataset_snapshot,
                processing_class,
                imports,
                "train/data",
                truncation_disabled=request.settings.visual_no_truncation,
            )
        else:
            dataset = _text_dataset(
                context,
                request,
                dataset_snapshot,
                processing_class,
                imports,
                "train/data",
            )
        validation_dataset = None
        if validation_snapshot is not None:
            if visual:
                validation_dataset = _visual_dataset(
                    context,
                    validation_snapshot,
                    processing_class,
                    imports,
                    "train/validation/data",
                    truncation_disabled=request.settings.visual_no_truncation,
                )
            else:
                validation_dataset = _text_dataset(
                    context,
                    request,
                    validation_snapshot,
                    processing_class,
                    imports,
                    "train/validation/data",
                )
    emit_parameter_counts(context, model, request.training.update)
    arguments = _sft_arguments(request, output_dir, visual=visual)
    validation = request.settings.validation
    callback = callback_type(context, imports)()
    checkpoint_callback = checkpoint_callback_type(
        context,
        imports,
        model=request.model,
        technique="sft",
        settings=request.settings,
        update=request.training.update,
        workspace=output_dir.parent,
    )()

    ObservedSFTTrainer = _observed_sft_trainer_type(SFTTrainer, context)
    with context.phase("runtime_initialization", {"backend": "trl"}):
        trainer = ObservedSFTTrainer(
            model=model,
            args=SFTConfig(**arguments),
            train_dataset=dataset,
            eval_dataset=validation_dataset,
            processing_class=processing_class,
            callbacks=[callback, checkpoint_callback],
        )
    resume = str(request.resume_from.path) if request.resume_from is not None else None
    with trainer_lifecycle(trainer):
        try:
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
                    processing_class,
                    output_dir.parent,
                    "sft",
                    request.training.update,
                    imports,
                )
        except BaseException as error:
            preserve_recovery_checkpoint_after_error(
                context,
                trainer,
                error,
                technique="sft",
                model=request.model,
                settings=request.settings,
                update=request.training.update,
                imports=imports,
            )
            raise


def _observed_sft_trainer_type(parent: type[Any], context: RunContext) -> type[Any]:
    class ObservedSFTTrainer(parent):
        """Separate validation host evidence from the enclosing update loop."""

        def evaluate(self, *args: Any, **kwargs: Any) -> Any:
            with context.phase("evaluation", {"backend": "trl"}):
                return super().evaluate(*args, **kwargs)

    return ObservedSFTTrainer


def _backend_option_bool(request: SFTRequest, name: str, default: bool) -> bool:
    value = request.training.backend_options.get(name, default)
    if not isinstance(value, bool):
        raise ValueError(f"SFT backend option {name!r} must be a boolean")
    return value


def _sft_arguments(request: SFTRequest, output_dir: Path, *, visual: bool = False) -> dict[str, Any]:
    arguments = trainer_arguments(request.settings.loop, output_dir)
    arguments.update(
        {
            "dataset_kwargs": {"skip_prepare_dataset": True},
            "completion_only_loss": False,
            "assistant_only_loss": False,
            "packing": False,
            "shuffle_dataset": False,
            "use_liger_kernel": _backend_option_bool(request, "use_liger_kernel", False),
        }
    )
    if visual:
        arguments.pop("dataset_kwargs")
        arguments.update(
            {
                "completion_only_loss": True,
                "padding_free": False,
            }
        )
        if request.settings.visual_no_truncation:
            arguments["max_length"] = None
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


def _visual_modality(
    dataset: SupervisedDataset,
    validation: SupervisedDataset | None,
    request: SFTRequest,
) -> bool:
    visual = _dataset_is_visual(dataset, "training")
    if validation is not None and _dataset_is_visual(validation, "validation") != visual:
        raise ValueError("SFT training and validation datasets must use the same modality")
    if visual and "image" not in request.model.capabilities.modalities:
        raise ValueError(f"model variant {request.model.id!r} does not declare image input capability")
    return visual


def _dataset_is_visual(dataset: SupervisedDataset, label: str) -> bool:
    populated = tuple(bool(example.media) for example in dataset.examples)
    if any(populated) and not all(populated):
        raise ValueError(f"{label} SFT dataset cannot mix text-only and image-bearing examples")
    return all(populated)


def _text_dataset(
    context: RunContext,
    request: SFTRequest,
    snapshot: SupervisedDataset,
    tokenizer: Any,
    imports: dict[str, Any],
    prefix: str,
) -> Any:
    samples = render_supervised(
        tokenizer,
        request.model,
        snapshot,
        request.training.renderer,
        max_length=request.settings.loop.max_length,
    )
    _emit_rendered_profile(context, samples, request.settings.loop.max_length, prefix)
    return imports["Dataset"].from_list(
        [
            {
                "example_id": sample.id,
                "input_ids": list(sample.input_ids),
                "labels": list(sample.labels),
            }
            for sample in samples
        ]
    )


def _visual_dataset(
    context: RunContext,
    snapshot: SupervisedDataset,
    processor: Any,
    imports: dict[str, Any],
    prefix: str,
    *,
    truncation_disabled: bool,
) -> Any:
    materialized = snapshot.metadata.get("materialized_path")
    if not isinstance(materialized, str) or not materialized:
        raise ValueError("visual SFT datasets require a materialized_path")
    root = Path(materialized).resolve().parent
    rows = [_visual_row(example, root, imports) for example in snapshot.examples]
    image_counts = sorted(len(example.media) for example in snapshot.examples)
    image_bytes = sum(path.stat().st_size for row in rows for path in row.pop("_verified_paths"))
    prompt_tokens, completion_tokens = _visual_text_token_counts(snapshot, processor)
    context.metrics(
        {
            f"{prefix}/examples": len(rows),
            f"{prefix}/images": sum(image_counts),
            f"{prefix}/image_bytes": image_bytes,
            f"{prefix}/prompt_text_tokens": prompt_tokens,
            f"{prefix}/completion_text_tokens": completion_tokens,
            f"{prefix}/truncation_disabled": float(truncation_disabled),
            f"{prefix}/images_per_example_p50": _nearest_rank(image_counts, 0.50),
            f"{prefix}/images_per_example_p90": _nearest_rank(image_counts, 0.90),
            f"{prefix}/images_per_example_p99": _nearest_rank(image_counts, 0.99),
            f"{prefix}/images_per_example_max": image_counts[-1],
        }
    )
    return imports["Dataset"].from_list(rows)


def _visual_text_token_counts(snapshot: SupervisedDataset, processor: Any) -> tuple[int, int]:
    tokenizer = processor.tokenizer
    prompt_tokens = 0
    completion_tokens = 0
    for example in snapshot.examples:
        prompt_tokens += len(
            tokenizer.apply_chat_template(
                [dict(message) for message in example.messages[:-1]],
                tokenize=True,
                add_generation_prompt=True,
            )
        )
        encoded = tokenizer(str(example.messages[-1]["content"]), add_special_tokens=False)
        completion_tokens += len(encoded["input_ids"])
    return prompt_tokens, completion_tokens


def _visual_row(example: SupervisedExample, root: Path, imports: dict[str, Any]) -> dict[str, Any]:
    if example.tools:
        raise ValueError(f"visual SFT example {example.id!r} cannot declare tools")
    expected_target = (len(example.messages) - 1,)
    if example.trainable_message_indices != expected_target:
        raise ValueError(f"visual SFT example {example.id!r} must train only its final assistant message")
    completion = dict(example.messages[-1])
    if completion.get("role") != "assistant" or not isinstance(completion.get("content"), str):
        raise ValueError(f"visual SFT example {example.id!r} requires a textual assistant completion")
    prompt = [dict(message) for message in example.messages[:-1]]
    if not prompt:
        raise ValueError(f"visual SFT example {example.id!r} requires at least one prompt message")

    images: list[Any] = []
    verified_paths: list[Path] = []
    for media in example.media:
        path = _safe_visual_asset(root, media.path)
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != media.sha256:
            raise ValueError(f"visual SFT asset digest does not match its dataset record: {media.path}")
        with imports["Image"].open(path) as image:
            images.append(image.convert("RGB").copy())
        verified_paths.append(path)
    return {
        "example_id": example.id,
        "images": images,
        "prompt": prompt,
        "completion": [completion],
        "_verified_paths": verified_paths,
    }


def _safe_visual_asset(root: Path, relative: str) -> Path:
    candidate = root / relative
    current = root
    for part in Path(relative).parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"visual SFT asset cannot contain symlinks: {relative}")
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError(f"visual SFT asset is missing or unsafe: {relative}")
    return resolved


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
