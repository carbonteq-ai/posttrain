"""Shared TRL adapter mechanics with lazy framework imports."""

from __future__ import annotations

import json
import math
import os
import sys
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

from posttrain.common import JsonValue, LocalArtifactRef, ModelVariant, RunContext

from ...bindings import FullParameterUpdate, LoRAUpdate, ParameterUpdatePlan, QLoRAUpdate, QuantizationAwareUpdate
from ...profiles import TrainingLoop
from ...results import TrainingSummary
from ..common import BackendTrainingResult


def framework_imports() -> dict[str, Any]:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForMultimodalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainerCallback,
        )
        from transformers.trainer_utils import get_last_checkpoint
    except ImportError as error:
        raise RuntimeError("install posttrain-train with the trl extra") from error
    return {
        "torch": torch,
        "Dataset": Dataset,
        "LoraConfig": LoraConfig,
        "PeftModel": PeftModel,
        "get_peft_model": get_peft_model,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoModelForMultimodalLM": AutoModelForMultimodalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "TrainerCallback": TrainerCallback,
        "get_last_checkpoint": get_last_checkpoint,
    }


def vllm_rollout_options(
    model: ModelVariant, engine: Mapping[str, JsonValue]
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Validate and translate shared rollout-engine selections for colocated TRL."""
    speculative = engine.get("speculative_config")
    if speculative is not None:
        if engine.get("mode", "colocate") != "colocate":
            raise ValueError("TRL trainer-side speculative_config requires colocated vLLM mode")
        if not isinstance(speculative, Mapping):
            raise ValueError("TRL rollout speculative_config must be a mapping")
        if speculative.get("method") != "mtp":
            raise ValueError("TRL currently supports only native MTP speculative rollout")
        count = speculative.get("num_speculative_tokens")
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError("TRL MTP num_speculative_tokens must be a positive integer")
        if not model.capabilities.mtp:
            raise ValueError(f"model variant {model.id!r} does not declare a native MTP head")

    values: dict[str, Any] = {}
    if engine.get("text_only"):
        values["language_model_only"] = True
    if engine.get("skip_mm_profiling"):
        values["skip_mm_profiling"] = True
    if engine.get("enforce_eager") is not None:
        enforce_eager = engine["enforce_eager"]
        if not isinstance(enforce_eager, bool):
            raise ValueError("TRL rollout enforce_eager must be a boolean")
        values["enforce_eager"] = enforce_eager
    if engine.get("kv_cache_memory_bytes") is not None:
        values["kv_cache_memory_bytes"] = engine["kv_cache_memory_bytes"]
    kv_cache_dtype = engine.get("kv_cache_dtype")
    if kv_cache_dtype is not None:
        if not isinstance(kv_cache_dtype, str) or not kv_cache_dtype:
            raise ValueError("TRL rollout kv_cache_dtype must be a non-empty string")
        values["kv_cache_dtype"] = kv_cache_dtype
        if kv_cache_dtype.startswith("turboquant_"):
            values["dtype"] = "float16"
    if speculative is not None:
        values["disable_log_stats"] = False
    return dict(speculative) if isinstance(speculative, Mapping) else None, values or None


def load_tokenizer(model: ModelVariant, imports: dict[str, Any]) -> Any:
    tokenizer = imports["AutoTokenizer"].from_pretrained(
        model.base.repo_id,
        revision=model.base.revision,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def trainable_model_factory(model: ModelVariant, imports: dict[str, Any]) -> Any:
    """Select the Transformers factory required by a supported model family."""

    if model.family == "gemma4":
        return imports["AutoModelForMultimodalLM"]
    return imports["AutoModelForCausalLM"]


def load_trainable_model(
    model: ModelVariant,
    update: ParameterUpdatePlan,
    loop: TrainingLoop,
    imports: dict[str, Any],
) -> Any:
    torch = imports["torch"]
    if isinstance(update, QuantizationAwareUpdate):
        raise ValueError("the TRL adapter does not yet implement quantization-aware updates")
    load_options: dict[str, Any] = {
        "revision": model.base.revision,
        "device_map": {"": 0},
        "dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
        "trust_remote_code": False,
    }
    if isinstance(update, QLoRAUpdate):
        load_options["quantization_config"] = imports["BitsAndBytesConfig"](
            load_in_4bit=True,
            bnb_4bit_quant_type=update.quant_type,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=update.double_quant,
        )
    base = trainable_model_factory(model, imports).from_pretrained(model.base.repo_id, **load_options)
    base.config.use_cache = False
    if isinstance(update, QLoRAUpdate):
        base = imports["prepare_model_for_kbit_training"](
            base,
            use_gradient_checkpointing=loop.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    if model.form == "foundation":
        if update.kind == "full":
            return base
        assert isinstance(update, (LoRAUpdate, QLoRAUpdate))
        config = imports["LoraConfig"](
            r=update.rank,
            lora_alpha=update.alpha,
            lora_dropout=update.dropout,
            target_modules=update.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        return imports["get_peft_model"](base, config)
    if model.form not in {"adapter", "peft-adapter"}:
        raise ValueError("training currently accepts foundation or PEFT-adapter variants")
    if update.kind == "full":
        raise ValueError("full-parameter updates cannot resume from a PEFT-adapter variant")
    if not isinstance(model.artifact, LocalArtifactRef):
        raise ValueError("the host must materialize a remote adapter artifact before training")
    if not model.artifact.path.is_dir():
        raise FileNotFoundError(model.artifact.path)
    return imports["PeftModel"].from_pretrained(base, model.artifact.path, is_trainable=True)


def emit_parameter_counts(context: RunContext, model: Any, update: ParameterUpdatePlan) -> None:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable < 1:
        raise RuntimeError(f"training selected no parameters: trainable={trainable}, total={total}")
    if not isinstance(update, FullParameterUpdate) and trainable >= total:
        raise RuntimeError(f"invalid PEFT parameter selection: trainable={trainable}, total={total}")
    context.metrics(
        {
            "train/parameters_total": total,
            "train/parameters_trainable": trainable,
            "train/parameters_trainable_fraction": trainable / total,
        }
    )


def emit_runtime_versions(context: RunContext, imports: dict[str, Any]) -> None:
    torch = imports["torch"]
    context.event(
        "training_runtime_resolved",
        {
            "python": sys.version.split()[0],
            "torch": version("torch"),
            "transformers": version("transformers"),
            "trl": version("trl"),
            "peft": version("peft"),
            "bitsandbytes": version("bitsandbytes"),
            "datasets": version("datasets"),
            "cuda": str(torch.version.cuda),
        },
    )


type MetricNormalizer = Callable[[int, Mapping[str, object]], Mapping[str, float]]


def callback_type(
    context: RunContext,
    imports: dict[str, Any],
    *,
    metric_normalizer: MetricNormalizer | None = None,
) -> type[Any]:
    parent = imports["TrainerCallback"]

    class ObservationCallback(parent):
        def __init__(self) -> None:
            super().__init__()
            self._step_started_at: float | None = None
            self._last_token_count: float | None = None
            self._last_token_time: float | None = None

        def on_step_begin(self, args: Any, state: Any, control: Any, **_: Any) -> Any:
            del args, state
            self._step_started_at = time.perf_counter()
            return control

        def on_step_end(self, args: Any, state: Any, control: Any, **_: Any) -> Any:
            del args
            if self._step_started_at is not None:
                context.metric(
                    "train/step_time_seconds",
                    time.perf_counter() - self._step_started_at,
                    step=int(state.global_step),
                )
                self._step_started_at = None
            return control

        def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **_: Any) -> Any:
            values: dict[str, float] = {}
            records = logs or {}
            if metric_normalizer is not None:
                values.update(metric_normalizer(int(state.global_step), records))
            else:
                for name, value in records.items():
                    if isinstance(value, int | float):
                        numeric = float(value)
                        if not math.isfinite(numeric):
                            raise FloatingPointError(f"non-finite training metric {name}={numeric}")
                        metric_name = (
                            f"train/validation/{name.removeprefix('eval_')}"
                            if name.startswith("eval_")
                            else f"train/{name}"
                        )
                        values[metric_name] = numeric
            grad_norm = (logs or {}).get("grad_norm")
            max_grad_norm = getattr(args, "max_grad_norm", None)
            if isinstance(grad_norm, int | float) and isinstance(max_grad_norm, int | float):
                values["train/gradient_clipped"] = float(grad_norm >= max_grad_norm)
            token_count = (logs or {}).get("num_tokens")
            now = time.perf_counter()
            if isinstance(token_count, int | float):
                numeric_tokens = float(token_count)
                if self._last_token_count is not None and self._last_token_time is not None:
                    elapsed = now - self._last_token_time
                    delta = numeric_tokens - self._last_token_count
                    if elapsed > 0 and delta >= 0:
                        values["train/non_padding_tokens_per_second"] = delta / elapsed
                self._last_token_count = numeric_tokens
                self._last_token_time = now
            if values:
                context.metrics(values, step=int(state.global_step))
            return control

    return ObservationCallback


def trainer_arguments(loop: TrainingLoop, output_dir: Path) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "output_dir": str(output_dir),
        "max_steps": loop.max_steps,
        "per_device_train_batch_size": loop.per_device_batch_size,
        "gradient_accumulation_steps": loop.gradient_accumulation_steps,
        "learning_rate": loop.learning_rate,
        "warmup_steps": math.ceil(loop.max_steps * loop.warmup_ratio),
        "max_grad_norm": loop.max_grad_norm,
        "logging_strategy": "steps",
        "logging_steps": loop.logging_steps,
        "logging_first_step": True,
        "save_strategy": "no" if loop.checkpoint_steps == 0 else "steps",
        "save_total_limit": loop.checkpoint_limit,
        "seed": loop.seed,
        "data_seed": loop.seed,
        "gradient_checkpointing": loop.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "bf16": True,
        "fp16": False,
        "use_cache": False,
        "report_to": "none",
        "disable_tqdm": True,
        "dataloader_pin_memory": True,
        "remove_unused_columns": True,
        "max_length": loop.max_length,
    }
    if loop.checkpoint_steps > 0:
        arguments["save_steps"] = loop.checkpoint_steps
    return arguments


@contextmanager
def trainer_lifecycle(trainer: Any) -> Iterator[None]:
    """Close Accelerate's distributed runtime after success or failure."""
    try:
        yield
    finally:
        trainer.accelerator.end_training()


def finish_training(
    context: RunContext,
    trainer: Any,
    train_output: Any,
    tokenizer: Any,
    workspace: Path,
    technique: Literal["sft", "dpo", "grpo", "dapo", "sampo", "distill"],
    update: ParameterUpdatePlan,
    imports: dict[str, Any],
) -> BackendTrainingResult:
    model_dir = workspace / ("weights" if isinstance(update, FullParameterUpdate) else "adapter")
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    latest = imports["get_last_checkpoint"](trainer.args.output_dir)
    metrics = train_output.metrics
    summary = TrainingSummary(
        global_step=int(trainer.state.global_step),
        train_loss=float(metrics["train_loss"]),
        runtime_seconds=float(metrics["train_runtime"]),
        samples_per_second=float(metrics["train_samples_per_second"]),
        steps_per_second=float(metrics["train_steps_per_second"]),
    )
    summary_file = workspace / "training-summary.json"
    summary_file.write_text(
        json.dumps(
            {
                "technique": technique,
                "summary": {
                    "global_step": summary.global_step,
                    "train_loss": summary.train_loss,
                    "runtime_seconds": summary.runtime_seconds,
                    "samples_per_second": summary.samples_per_second,
                    "steps_per_second": summary.steps_per_second,
                },
                "log_history": trainer.state.log_history,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    torch = imports["torch"]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        context.metrics({"train/peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / (1024**3)})
    return BackendTrainingResult(
        summary,
        model_dir,
        Path(latest).resolve() if latest is not None else None,
        summary_file,
    )


__all__ = [
    "BackendTrainingResult",
    "callback_type",
    "emit_parameter_counts",
    "emit_runtime_versions",
    "finish_training",
    "framework_imports",
    "load_tokenizer",
    "load_trainable_model",
    "trainable_model_factory",
    "trainer_lifecycle",
    "trainer_arguments",
    "vllm_rollout_options",
]
