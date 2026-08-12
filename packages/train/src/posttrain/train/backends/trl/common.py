"""Shared TRL adapter mechanics with lazy framework imports."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

from posttrain.common import JsonValue, LocalArtifactRef, ModelVariant, ProducedArtifact, RunContext

from ...bindings import FullParameterUpdate, LoRAUpdate, ParameterUpdatePlan, QLoRAUpdate, QuantizationAwareUpdate
from ...profiles import TrainingLoop
from ...results import TrainingSummary
from ..common import BackendTrainingResult
from ..retention import validate_adapter_only_directory

_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")


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
            raise ValueError(f"model variant {model.id!r} does not declare MTP capability")
        speculative = _resolve_speculative_assistant(model, speculative)

    values: dict[str, Any] = {}
    if engine.get("text_only"):
        values["language_model_only"] = True
    if engine.get("skip_mm_profiling"):
        values["skip_mm_profiling"] = True
    for key in ("max_num_seqs", "max_num_batched_tokens"):
        requested = engine.get(key)
        if requested is not None:
            if isinstance(requested, bool) or not isinstance(requested, int) or requested < 1:
                raise ValueError(f"TRL rollout {key} must be a positive integer")
            values[key] = requested
    if engine.get("enforce_eager") is not None:
        enforce_eager = engine["enforce_eager"]
        if not isinstance(enforce_eager, bool):
            raise ValueError("TRL rollout enforce_eager must be a boolean")
        values["enforce_eager"] = enforce_eager
    disable_torch_compile = engine.get("disable_torch_compile")
    if disable_torch_compile is not None:
        if not isinstance(disable_torch_compile, bool):
            raise ValueError("TRL rollout disable_torch_compile must be a boolean")
        # This is consumed by the job entrypoint before importing torch.  It
        # is intentionally not passed as a vLLM constructor kwarg: the
        # switch also disables compile-decorated MTP helper modules.
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


def _resolve_speculative_assistant(
    model: ModelVariant, speculative: Mapping[str, JsonValue]
) -> Mapping[str, JsonValue | str]:
    """Materialize a pinned paired assistant into the worker HF cache.

    vLLM's Gemma MTP config accepts a local/repository ``model`` path but does
    not accept a revision field. The framework keeps the immutable revision in
    the binding and resolves it before constructing the colocated TRL engine.
    Native MTP mappings remain unchanged.
    """

    assistant_model = speculative.get("assistant_model")
    assistant_revision = speculative.get("assistant_revision")
    if model.family == "gemma4" and assistant_model is None and assistant_revision is None:
        raise ValueError("Gemma MTP requires assistant_model and assistant_revision")
    if assistant_model is None and assistant_revision is None:
        return speculative
    if not isinstance(assistant_model, str) or not assistant_model.strip() or assistant_model.count("/") != 1:
        raise ValueError("TRL MTP assistant_model must be an owner/repository string")
    if not isinstance(assistant_revision, str) or _COMMIT_SHA.fullmatch(assistant_revision) is None:
        raise ValueError("TRL MTP assistant_revision must be a full 40-character commit SHA")

    expected_repo = model.provenance.get("mtp_assistant_repo_id")
    expected_revision = model.provenance.get("mtp_assistant_revision")
    if isinstance(expected_repo, str) and assistant_model != expected_repo:
        raise ValueError(
            f"MTP assistant_model {assistant_model!r} does not match model variant {model.id!r} provenance"
        )
    if isinstance(expected_revision, str) and assistant_revision != expected_revision:
        raise ValueError(f"MTP assistant_revision does not match model variant {model.id!r} provenance")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError("install posttrain-train with the trl extra for paired-assistant MTP") from error
    local_path = snapshot_download(repo_id=assistant_model, revision=assistant_revision)
    resolved = dict(speculative)
    resolved.pop("assistant_model", None)
    resolved.pop("assistant_revision", None)
    resolved["model"] = str(local_path)
    return resolved


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
    *,
    model_dtype: str = "bfloat16",
) -> Any:
    torch = imports["torch"]
    if isinstance(update, QuantizationAwareUpdate):
        raise ValueError("the TRL adapter does not yet implement quantization-aware updates")
    dtype_by_name = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    try:
        dtype = dtype_by_name[model_dtype]
    except KeyError as error:
        raise ValueError("TRL trainable model dtype must be 'bfloat16' or 'float32'") from error
    if isinstance(update, QLoRAUpdate) and model_dtype != "bfloat16":
        raise ValueError("QLoRA training requires bfloat16 compute dtype")
    load_options: dict[str, Any] = {
        "revision": model.base.revision,
        "device_map": {"": 0},
        "dtype": dtype,
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
            non_finite: tuple[str, float] | None = None
            records = logs or {}
            if metric_normalizer is not None:
                values.update(metric_normalizer(int(state.global_step), records))
            else:
                for name, value in records.items():
                    if isinstance(value, int | float):
                        numeric = float(value)
                        if not math.isfinite(numeric):
                            # Preserve all valid values from this report before
                            # failing.  A native trainer commonly reports loss
                            # and grad_norm together; dropping the loss makes a
                            # non-finite gradient impossible to diagnose from
                            # retained run evidence.
                            non_finite = (name, numeric)
                            continue
                        metric_name = (
                            f"train/validation/{name.removeprefix('eval_')}"
                            if name.startswith("eval_")
                            else f"train/{name}"
                        )
                        values[metric_name] = numeric
            grad_norm = (logs or {}).get("grad_norm")
            max_grad_norm = getattr(args, "max_grad_norm", None)
            if (
                isinstance(grad_norm, int | float)
                and math.isfinite(float(grad_norm))
                and isinstance(max_grad_norm, int | float)
            ):
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
            if non_finite is not None:
                name, numeric = non_finite
                context.event(
                    "training_non_finite_metric",
                    {
                        "metric": name,
                        "value_class": "nan" if math.isnan(numeric) else "infinite",
                        "global_step": int(state.global_step),
                    },
                )
                raise FloatingPointError(f"non-finite training metric {name}={numeric}")
            return control

    return ObservationCallback


def checkpoint_callback_type(
    context: RunContext,
    imports: Mapping[str, Any],
    *,
    model: ModelVariant,
    technique: Literal["sft", "dpo", "grpo", "dapo", "olmo3", "sampo", "distill"],
    settings: Any,
    update: ParameterUpdatePlan,
    workspace: Path,
) -> type[Any]:
    """Create a callback that publishes both views after a trainer save.

    The callback only projects a loadable model view for adapter updates. Full
    parameter checkpoints remain recovery-only until a backend explicitly
    attests that its checkpoint representation is safe to load as a model.
    """

    parent = imports["TrainerCallback"]

    class CheckpointPublicationCallback(parent):
        def __init__(self) -> None:
            super().__init__()
            self._published_steps: set[int] = set()

        def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            del kwargs
            step = int(getattr(state, "global_step", 0))
            if step < 1 or step in self._published_steps:
                return control
            output_dir = Path(str(args.output_dir)).resolve()
            latest = imports["get_last_checkpoint"](str(output_dir))
            if latest is None:
                context.event("checkpoint_publication_unavailable", {"technique": technique, "global_step": step})
                return control
            checkpoint = Path(latest).resolve()
            publish_checkpoint_views(
                context,
                checkpoint,
                model=model,
                technique=technique,
                settings=settings,
                update=update,
                workspace=workspace,
                interrupted=False,
            )
            self._published_steps.add(step)
            return control

    return CheckpointPublicationCallback


def trainer_arguments(loop: TrainingLoop, output_dir: Path) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "output_dir": str(output_dir),
        "max_steps": loop.max_steps,
        "per_device_train_batch_size": loop.per_device_batch_size,
        "gradient_accumulation_steps": loop.gradient_accumulation_steps,
        "learning_rate": loop.learning_rate,
        "warmup_steps": math.ceil(loop.max_steps * loop.warmup_ratio),
        "lr_scheduler_type": loop.lr_scheduler_type,
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


def _project_checkpoint_model_view(checkpoint: Path, destination: Path) -> Path:
    """Copy adapter/tokenizer files while excluding recovery-only state."""

    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True, exist_ok=False)
    excluded = {
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
        "scaler.pt",
        "training_args.bin",
    }
    for source in checkpoint.iterdir():
        if not source.is_file() or source.name in excluded or source.name.startswith("rng_state"):
            continue
        if source.name.startswith(("model-", "pytorch_model")) or source.name in {
            "model.safetensors",
            "pytorch_model.bin",
        }:
            raise RuntimeError(f"LoRA checkpoint contains full base-model weights: {source.name}")
        shutil.copy2(source, temporary / source.name)
    if destination.exists():
        shutil.rmtree(destination)
    temporary.rename(destination)
    validate_adapter_only_directory(destination)
    return destination.resolve()


def publish_checkpoint_views(
    context: RunContext,
    checkpoint: Path,
    *,
    model: ModelVariant,
    technique: Literal["sft", "dpo", "grpo", "dapo", "olmo3", "sampo", "distill"],
    settings: Any,
    update: ParameterUpdatePlan,
    workspace: Path,
    interrupted: bool,
) -> None:
    """Publish a recovery view and, for LoRA, a paired loadable model view."""

    checkpoint = checkpoint.resolve()
    if isinstance(update, LoRAUpdate | QLoRAUpdate):
        validate_adapter_only_directory(checkpoint, require_recovery_state=True)
    try:
        trainer_state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
        checkpoint_step = trainer_state["global_step"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise RuntimeError(f"recovery checkpoint has invalid trainer state: {checkpoint}") from error
    if isinstance(checkpoint_step, bool) or not isinstance(checkpoint_step, int) or checkpoint_step < 0:
        raise RuntimeError(f"recovery checkpoint has invalid global_step: {checkpoint}")
    snapshot_id = f"{context.run_id}/step-{checkpoint_step:08d}"
    metadata = {
        "technique": technique,
        "model_variant_id": model.id,
        "training_settings_id": settings.id,
        "training_settings_revision": settings.revision,
        "parameter_update_kind": update.kind,
        "global_step": checkpoint_step,
        "checkpoint_step": checkpoint_step,
        "checkpoint_snapshot_id": snapshot_id,
    }
    recovery_ref = LocalArtifactRef(checkpoint, _digest_path(checkpoint))
    context.artifact(
        ProducedArtifact(
            name=f"training/{model.id}/{technique}/checkpoint-{checkpoint_step:08d}/recovery",
            kind="training-checkpoint",
            reference=recovery_ref,
            metadata={**metadata, "checkpoint_view": "recovery", "interrupted": interrupted},
            role="recovery",
        )
    )
    if isinstance(update, LoRAUpdate | QLoRAUpdate):
        destination = workspace / "checkpoints" / f"step-{checkpoint_step:08d}" / "model"
        model_dir = _project_checkpoint_model_view(checkpoint, destination)
        model_ref = LocalArtifactRef(model_dir, _digest_path(model_dir))
        context.artifact(
            ProducedArtifact(
                name=f"training/{model.id}/{technique}/checkpoint-{checkpoint_step:08d}/model",
                kind="model-adapter",
                reference=model_ref,
                metadata={**metadata, "checkpoint_view": "model", "interrupted": interrupted},
                role="checkpoint-model",
            )
        )
    context.event(
        "checkpoint_saved",
        {
            "technique": technique,
            "global_step": checkpoint_step,
            "checkpoint_snapshot_id": snapshot_id,
            "recovery_only": not isinstance(update, LoRAUpdate | QLoRAUpdate),
            "model_view_published": isinstance(update, LoRAUpdate | QLoRAUpdate),
            "interrupted": interrupted,
        },
    )


@contextmanager
def trainer_lifecycle(trainer: Any) -> Iterator[None]:
    """Close Accelerate's distributed runtime after success or failure."""
    try:
        yield
    finally:
        trainer.accelerator.end_training()


def publish_interrupted_recovery_checkpoint(
    context: RunContext,
    trainer: Any,
    *,
    technique: Literal["sft", "dpo", "grpo", "dapo", "olmo3", "sampo", "distill"],
    model: ModelVariant,
    settings: Any,
    update: ParameterUpdatePlan,
    imports: Mapping[str, Any],
) -> Path | None:
    """Publish the latest complete TRL checkpoint while an interrupted run still owns its workspace."""

    latest = imports["get_last_checkpoint"](trainer.args.output_dir)
    if latest is None:
        context.event("recovery_checkpoint_unavailable", {"technique": technique})
        return None
    checkpoint = Path(latest).resolve()
    publish_checkpoint_views(
        context,
        checkpoint,
        model=model,
        technique=technique,
        settings=settings,
        update=update,
        workspace=trainer.args.output_dir and Path(trainer.args.output_dir).parent,
        interrupted=True,
    )
    return checkpoint


def preserve_recovery_checkpoint_after_error(
    context: RunContext,
    trainer: Any,
    error: BaseException,
    *,
    technique: Literal["sft", "dpo", "grpo", "dapo", "olmo3", "sampo", "distill"],
    model: ModelVariant,
    settings: Any,
    update: ParameterUpdatePlan,
    imports: Mapping[str, Any],
) -> None:
    """Best-effort retention that never replaces the original training failure."""

    try:
        publish_interrupted_recovery_checkpoint(
            context,
            trainer,
            technique=technique,
            model=model,
            settings=settings,
            update=update,
            imports=imports,
        )
    except BaseException as checkpoint_error:
        error.add_note(f"failed to retain the latest recovery checkpoint: {checkpoint_error!r}")


def _digest_path(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.read_bytes())
        return digest.hexdigest()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        with child.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def finish_training(
    context: RunContext,
    trainer: Any,
    train_output: Any,
    tokenizer: Any,
    workspace: Path,
    technique: Literal["sft", "dpo", "grpo", "dapo", "olmo3", "sampo", "distill"],
    update: ParameterUpdatePlan,
    imports: dict[str, Any],
) -> BackendTrainingResult:
    model_dir = workspace / ("weights" if isinstance(update, FullParameterUpdate) else "adapter")
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    latest = imports["get_last_checkpoint"](trainer.args.output_dir)
    latest_path = Path(latest).resolve() if latest is not None else None
    if isinstance(update, LoRAUpdate | QLoRAUpdate):
        validate_adapter_only_directory(model_dir)
        if latest_path is not None:
            validate_adapter_only_directory(latest_path, require_recovery_state=True)
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
        latest_path,
        summary_file,
    )


__all__ = [
    "BackendTrainingResult",
    "callback_type",
    "checkpoint_callback_type",
    "emit_parameter_counts",
    "emit_runtime_versions",
    "finish_training",
    "framework_imports",
    "load_tokenizer",
    "load_trainable_model",
    "preserve_recovery_checkpoint_after_error",
    "publish_checkpoint_views",
    "publish_interrupted_recovery_checkpoint",
    "trainable_model_factory",
    "trainer_lifecycle",
    "trainer_arguments",
    "vllm_rollout_options",
]
