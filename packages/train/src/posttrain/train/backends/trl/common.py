"""Shared TRL adapter mechanics with lazy framework imports."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from posttrain.common import ExecutionContext, LocalArtifactRef, ModelVariant

from ...profiles import QLoRAProfile, TrainingLoop
from ...results import TrainingSummary


@dataclass(frozen=True, slots=True)
class BackendTrainingResult:
    summary: TrainingSummary
    adapter_dir: Path
    recovery_checkpoint: Path | None
    summary_file: Path


def framework_imports() -> dict[str, Any]:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
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
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "TrainerCallback": TrainerCallback,
        "get_last_checkpoint": get_last_checkpoint,
    }


def load_tokenizer(model: ModelVariant, imports: dict[str, Any]) -> Any:
    tokenizer = imports["AutoTokenizer"].from_pretrained(
        model.base_artifact.repo_id,
        revision=model.base_artifact.revision,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_trainable_model(
    model: ModelVariant,
    qlora: QLoRAProfile,
    loop: TrainingLoop,
    imports: dict[str, Any],
) -> Any:
    torch = imports["torch"]
    quantization = imports["BitsAndBytesConfig"](
        load_in_4bit=True,
        bnb_4bit_quant_type=qlora.quant_type,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=qlora.double_quant,
    )
    base = imports["AutoModelForCausalLM"].from_pretrained(
        model.base_artifact.repo_id,
        revision=model.base_artifact.revision,
        quantization_config=quantization,
        device_map={"": 0},
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=False,
    )
    base.config.use_cache = False
    base = imports["prepare_model_for_kbit_training"](
        base,
        use_gradient_checkpointing=loop.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    if model.format == "foundation":
        config = imports["LoraConfig"](
            r=qlora.lora_rank,
            lora_alpha=qlora.lora_alpha,
            lora_dropout=qlora.lora_dropout,
            target_modules=qlora.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        return imports["get_peft_model"](base, config)
    if model.format != "peft-adapter":
        raise ValueError("training currently accepts foundation or PEFT-adapter variants")
    if not isinstance(model.artifact, LocalArtifactRef):
        raise ValueError("the host must materialize a remote adapter artifact before training")
    if not model.artifact.path.is_dir():
        raise FileNotFoundError(model.artifact.path)
    return imports["PeftModel"].from_pretrained(base, model.artifact.path, is_trainable=True)


def callback_type(context: ExecutionContext, imports: dict[str, Any]) -> type[Any]:
    parent = imports["TrainerCallback"]

    class ObservationCallback(parent):
        def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **_: Any) -> Any:
            del args
            values: dict[str, float] = {}
            for name, value in (logs or {}).items():
                if isinstance(value, int | float):
                    numeric = float(value)
                    if not math.isfinite(numeric):
                        raise FloatingPointError(f"non-finite training metric {name}={numeric}")
                    values[f"train/{name}"] = numeric
            if values:
                context.metrics(values, step=int(state.global_step))
            return control

    return ObservationCallback


def trainer_arguments(loop: TrainingLoop, output_dir: Path) -> dict[str, Any]:
    return {
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
        "save_strategy": "steps",
        "save_steps": loop.checkpoint_steps,
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


@contextmanager
def trainer_lifecycle(trainer: Any) -> Iterator[None]:
    """Close Accelerate's distributed runtime after success or failure."""
    try:
        yield
    finally:
        trainer.accelerator.end_training()


def finish_training(
    trainer: Any,
    train_output: Any,
    tokenizer: Any,
    workspace: Path,
    technique: Literal["sft", "dpo", "grpo"],
    imports: dict[str, Any],
) -> BackendTrainingResult:
    adapter_dir = workspace / "adapter"
    trainer.save_model(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
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
    return BackendTrainingResult(
        summary,
        adapter_dir,
        Path(latest).resolve() if latest is not None else None,
        summary_file,
    )


__all__ = [
    "BackendTrainingResult",
    "callback_type",
    "finish_training",
    "framework_imports",
    "load_tokenizer",
    "load_trainable_model",
    "trainer_lifecycle",
    "trainer_arguments",
]
