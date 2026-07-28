"""TRL adapter for Verifiers-driven on-policy distillation."""

from __future__ import annotations

import asyncio
import hashlib
import math
import subprocess
import sys
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import httpx
from posttrain.common import HubModelRef, RunContext, TraceObservation

from ...bindings import LoRAUpdate
from ...distillation import DistillationBatch, DistillationBatchLedger
from ...online_rl import EnvironmentRollout, RolloutBatch
from ...requests import OnPolicyDistillationRequest
from .common import (
    BackendTrainingResult,
    callback_type,
    emit_parameter_counts,
    emit_runtime_versions,
    finish_training,
    framework_imports,
    load_frozen_model,
    load_tokenizer,
    load_trainable_model,
    trainer_arguments,
    trainer_lifecycle,
    vllm_rollout_options,
)


def run_distillation(
    context: RunContext,
    request: OnPolicyDistillationRequest,
    output_dir: Path,
) -> BackendTrainingResult:
    if request.rollout_inference.backend.split("@", 1)[0] != "vllm":
        raise ValueError("the first TRL distillation adapter requires a vLLM student rollout binding")
    teacher_product = request.teacher_inference.backend.split("@", 1)[0]
    if teacher_product not in {"transformers", "vllm"}:
        raise ValueError("TRL distillation requires a transformers or vLLM teacher-score binding")
    teacher_url = request.teacher_inference.engine.get("base_url")
    if teacher_product == "vllm" and (not isinstance(teacher_url, str) or not teacher_url.strip()):
        raise ValueError("the vLLM teacher-score binding must provide engine.base_url")
    if not isinstance(request.teacher.artifact, HubModelRef):
        raise ValueError("TRL distillation currently requires a Hugging Face teacher model")
    _validate_gemma4_distillation_topology(request)

    try:
        from trl.experimental.distillation import (  # pyright: ignore[reportMissingImports]
            DistillationConfig,
            DistillationTrainer,
        )
    except ImportError as error:
        raise RuntimeError("install posttrain-train with the trl-vllm extra") from error
    _patch_local_teacher_divergence_numerics(DistillationTrainer)

    imports = framework_imports()
    emit_runtime_versions(context, imports)
    with context.phase("model_loading", {"backend": "trl"}):
        tokenizer = load_tokenizer(request.student, imports)
        model = load_trainable_model(request.student, request.training.update, request.settings.loop, imports)
        teacher_model = (
            load_frozen_model(
                request.teacher,
                imports,
                dtype=str(request.teacher_inference.engine.get("dtype", "bfloat16")),
            )
            if teacher_product == "transformers"
            else None
        )
    rows = [
        {
            "messages": [{"role": "user", "content": example.prompt}],
            "example_id": example.id,
            **dict(example.metadata),
        }
        for example in request.bridge.dataset.examples
    ]
    dataset = imports["Dataset"].from_list(rows)
    emit_parameter_counts(context, model, request.training.update)
    policy_revision = request.student.digest or request.student.revision
    if policy_revision is None:
        raise AssertionError("validated student variants require an immutable revision")
    ledger = DistillationBatchLedger(policy_revision)
    with _teacher_server_lifecycle(context, request, output_dir):
        arguments = _distillation_arguments(
            request,
            output_dir,
            teacher_url if isinstance(teacher_url, str) else None,
        )

        ObservedDistillationTrainer = _observed_distillation_trainer_type(context, DistillationTrainer)

        with context.phase("runtime_initialization", {"backend": "trl"}):
            trainer = ObservedDistillationTrainer(
                model=model,
                teacher_model=cast(Any, teacher_model),
                args=DistillationConfig(**arguments),
                train_dataset=dataset,
                processing_class=tokenizer,
                callbacks=[callback_type(context, imports, metric_normalizer=_normalize_distillation_metrics)()],
                rollout_func=cast(Any, _rollout_function(context, request, tokenizer, ledger)),
            )
        if teacher_product == "vllm":
            if trainer.teacher_client is None:
                raise RuntimeError("TRL did not initialize the configured teacher scoring client")
            trainer.teacher_client = cast(
                Any,
                _ObservedTeacherClient(context, trainer.teacher_client, lambda: int(trainer.state.global_step)),
            )
        elif trainer.teacher_model is None:
            raise RuntimeError("TRL did not initialize the colocated teacher model")
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
                    "distill",
                    request.training.update,
                    imports,
                )


@contextmanager
def _teacher_server_lifecycle(
    context: RunContext,
    request: OnPolicyDistillationRequest,
    output_dir: Path,
) -> Iterator[None]:
    teacher_product = request.teacher_inference.backend.split("@", 1)[0]
    if teacher_product != "vllm":
        yield
        return
    teacher_url = request.teacher_inference.engine.get("base_url")
    if not isinstance(teacher_url, str) or not teacher_url.strip():
        raise ValueError("the vLLM teacher-score binding must provide engine.base_url")
    if not isinstance(request.teacher.artifact, HubModelRef):
        raise ValueError("TRL distillation currently requires a Hugging Face teacher model")

    parsed = urlparse(teacher_url)
    host = parsed.hostname or "127.0.0.1"
    if parsed.port is None:
        raise ValueError("the vLLM teacher-score base_url must include an explicit port")
    port = parsed.port
    health_url = f"{parsed.scheme or 'http'}://{host}:{port}/health/"
    log_path = output_dir / "teacher-vllm-server.log"
    command = _teacher_server_command(request, host=host, port=port)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_stream:
        process = subprocess.Popen(
            command,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
    context.event(
        "distillation_teacher_server_started",
        {
            "teacher_url": teacher_url,
            "health_url": health_url,
            "log_path": str(log_path),
        },
    )
    deadline = time.monotonic() + 600.0
    try:
        while time.monotonic() < deadline:
            returncode = process.poll()
            if returncode is not None:
                tail = log_path.read_text(encoding="utf-8", errors="replace")[-16_000:]
                raise RuntimeError(f"teacher vLLM server exited with code {returncode}:\n{tail}")
            try:
                response = httpx.get(health_url, timeout=1.0)
                if response.is_success:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        else:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-16_000:]
            raise TimeoutError(f"teacher vLLM server did not become healthy:\n{tail}")
        yield
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


def _validate_gemma4_distillation_topology(request: OnPolicyDistillationRequest) -> None:
    if request.student.family != "gemma4" and request.teacher.family != "gemma4":
        return
    if request.student.family != "gemma4" or request.teacher.family != "gemma4":
        raise ValueError("Gemma 4 distillation requires a Gemma 4 student and teacher")
    if request.student.weight_precision != "bf16" or request.teacher.weight_precision != "bf16":
        raise ValueError("Gemma 4 distillation requires unquantized BF16 student and teacher weights")
    if type(request.training.update) is not LoRAUpdate:
        raise ValueError("Gemma 4 distillation requires a BF16 LoRA student update")
    if request.quantization is not None:
        raise ValueError("Gemma 4 distillation forbids student and teacher weight quantization")


def _observed_distillation_trainer_type(context: RunContext, parent: type[Any]) -> type[Any]:
    class ObservedDistillationTrainer(parent):
        def _generate_with_rollout_func(
            self,
            slices: list[dict[str, Any]],
            on_policy_indices: list[int],
        ) -> None:
            _generate_with_expanded_rollout_func(self, slices, on_policy_indices)

        def _get_teacher_logits(self, inputs: dict[str, Any]) -> Any:
            started_at = time.perf_counter()
            step = int(self.state.global_step)
            with context.phase("teacher_scoring", {"backend": "trl", "logical_step": step}):
                try:
                    result = super()._get_teacher_logits(inputs)
                except Exception:
                    context.metric("train/distill/teacher_failures", 1, step=step)
                    raise
                else:
                    context.metric("train/distill/teacher_failures", 0, step=step)
                    return result
                finally:
                    context.metric(
                        "train/distill/teacher_latency_ms",
                        (time.perf_counter() - started_at) * 1_000,
                        step=step,
                    )

        def _save_checkpoint(self, model: Any, trial: Any) -> None:
            step = int(self.state.global_step)
            with context.phase("checkpointing", {"backend": "trl", "logical_step": step}):
                super()._save_checkpoint(model, trial)

    return ObservedDistillationTrainer


def _generate_with_expanded_rollout_func(
    trainer: Any,
    slices: list[dict[str, Any]],
    on_policy_indices: list[int],
) -> None:
    """Preserve one independently conditioned training row per environment branch."""

    import torch

    prompts: list[Any] = []
    rollout_inputs: list[dict[str, Any]] = []
    local_slice_indices: list[int] = []
    pad_token_id = trainer.processing_class.pad_token_id

    for slice_idx in on_policy_indices:
        slice_inputs = slices[slice_idx]
        messages = slice_inputs.get("messages")
        source_inputs = slice_inputs.get("rollout_inputs")
        prompt_mask = slice_inputs.get("prompt_attention_mask")
        for row_idx, prompt in enumerate(slice_inputs["prompts"]):
            if prompt_mask is not None:
                prompt = prompt[prompt_mask[row_idx].bool()]
            elif pad_token_id is not None:
                prompt = prompt[prompt != pad_token_id]
            prompt_ids = prompt.tolist()
            structured_prompt = messages[row_idx] if messages is not None else prompt_ids
            prompts.append(structured_prompt)
            rollout_input = dict(source_inputs[row_idx]) if source_inputs is not None else {}
            rollout_input.update({"prompt_ids": prompt_ids, "input": structured_prompt})
            rollout_inputs.append(rollout_input)
            local_slice_indices.append(slice_idx)

    output = trainer.rollout_func(prompts, trainer, inputs=rollout_inputs)
    required_keys = {
        "prompt_ids",
        "prompt_lengths",
        "completion_ids",
        "completion_loss_mask",
        "logprobs",
    }
    missing_keys = required_keys - output.keys()
    if missing_keys:
        raise ValueError(f"rollout_func must return keys {sorted(missing_keys)} in its output dict")

    item_count = len(output["prompt_ids"])
    for key in required_keys:
        if len(output[key]) != item_count:
            raise ValueError(f"rollout_func field {key!r} must contain {item_count} items")
    raw_source_indices = output.get("source_indices")
    if raw_source_indices is None:
        if item_count != len(prompts):
            raise ValueError("expanded rollout_func output requires source_indices")
        source_indices = list(range(item_count))
    else:
        source_indices = list(raw_source_indices)
    if len(source_indices) != item_count:
        raise ValueError("rollout_func source_indices must align with generated items")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0 or value >= len(prompts)
        for value in source_indices
    ):
        raise ValueError("rollout_func source_indices contain an invalid prompt position")
    if set(source_indices) != set(range(len(prompts))):
        raise ValueError("rollout_func source_indices must cover every prompt")
    expanded_slice_indices = [local_slice_indices[value] for value in source_indices]

    rollout_ids = output.get("rollout_ids")
    if rollout_ids is not None:
        if len(rollout_ids) != item_count:
            raise ValueError("rollout_func rollout_ids must align with generated items")
        if len(set(rollout_ids)) != len(rollout_ids):
            raise ValueError("rollout_func rollout_ids must be unique")

    normalized: list[dict[str, Any]] = []
    for idx in range(item_count):
        prompt_ids = list(output["prompt_ids"][idx])
        completion_ids = list(output["completion_ids"][idx])
        completion_loss_mask = [bool(value) for value in output["completion_loss_mask"][idx]]
        logprobs = list(output["logprobs"][idx])
        prompt_length = output["prompt_lengths"][idx]
        if not isinstance(prompt_length, int) or prompt_length <= 0:
            raise ValueError("rollout_func prompt_lengths must contain positive integers")
        if prompt_length != len(prompt_ids):
            raise ValueError("rollout_func prompt_lengths must equal the corresponding prompt_ids length")
        if not prompt_ids or not completion_ids:
            raise ValueError("rollout_func must return non-empty prompt_ids and completion_ids")
        if len(completion_ids) > trainer.generation_config.max_new_tokens:
            raise ValueError("rollout_func completion_ids exceed max_completion_length")
        if len(completion_loss_mask) != len(completion_ids):
            raise ValueError("rollout_func completion_loss_mask must align with completion_ids")
        if len(logprobs) != len(completion_ids):
            raise ValueError("rollout_func logprobs must align with completion_ids")
        if not any(completion_loss_mask):
            raise ValueError("rollout_func completion_loss_mask must select at least one token")
        normalized.append(
            {
                "prompt_ids": prompt_ids,
                "completion_ids": completion_ids,
                "completion_loss_mask": completion_loss_mask,
                "logprobs": logprobs,
            }
        )

    device = trainer.accelerator.device
    pad_id = pad_token_id if pad_token_id is not None else 0
    for slice_idx in on_policy_indices:
        items = [
            item
            for item, owner in zip(normalized, expanded_slice_indices, strict=True)
            if owner == slice_idx
        ]
        prompt_width = max(len(item["prompt_ids"]) for item in items)
        completion_width = max(len(item["completion_ids"]) for item in items)
        input_rows = []
        attention_rows = []
        label_rows = []
        prompt_mask_rows = []
        logprob_rows = []
        loss_mask_rows = []
        prompt_texts = []
        completion_texts = []

        for item in items:
            prompt_padding = prompt_width - len(item["prompt_ids"])
            completion_padding = completion_width - len(item["completion_ids"])
            prompt_row = [pad_id] * prompt_padding + item["prompt_ids"]
            completion_row = item["completion_ids"] + [pad_id] * completion_padding
            loss_mask = item["completion_loss_mask"] + [False] * completion_padding
            labels = [-100] * prompt_width + [
                token if keep else -100
                for token, keep in zip(completion_row, loss_mask, strict=True)
            ]
            input_rows.append(prompt_row + completion_row)
            attention_rows.append(
                [0] * prompt_padding
                + [1] * len(item["prompt_ids"])
                + [1] * len(item["completion_ids"])
                + [0] * completion_padding
            )
            label_rows.append(labels)
            prompt_mask_rows.append([0] * prompt_padding + [1] * len(item["prompt_ids"]))
            logprob_rows.append(item["logprobs"] + [0.0] * completion_padding)
            loss_mask_rows.append(loss_mask)
            prompt_texts.append(
                trainer.processing_class.decode(item["prompt_ids"], skip_special_tokens=False)
            )
            completion_texts.append(
                trainer.processing_class.decode(item["completion_ids"], skip_special_tokens=False)
            )

        updated = dict(slices[slice_idx])
        updated["input_ids"] = torch.tensor(input_rows, dtype=torch.long, device=device)
        updated["attention_mask"] = torch.tensor(attention_rows, dtype=torch.long, device=device)
        updated["labels"] = torch.tensor(label_rows, dtype=torch.long, device=device)
        updated["prompts"] = updated["input_ids"][:, :prompt_width]
        updated["prompt_attention_mask"] = torch.tensor(
            prompt_mask_rows,
            dtype=torch.long,
            device=device,
        )
        updated["prompt_length"] = prompt_width
        updated["sampling_logprobs"] = torch.tensor(logprob_rows, dtype=torch.float32, device=device)
        updated["completion_loss_mask"] = torch.tensor(loss_mask_rows, dtype=torch.bool, device=device)
        if rollout_ids is not None:
            updated["rollout_ids"] = [
                rollout_id
                for rollout_id, owner in zip(rollout_ids, expanded_slice_indices, strict=True)
                if owner == slice_idx
            ]
        trainer._buffered_inputs[slice_idx] = updated
        trainer._buffered_text_logs[slice_idx] = (prompt_texts, completion_texts)


def _teacher_server_command(
    request: OnPolicyDistillationRequest,
    *,
    host: str,
    port: int,
) -> tuple[str, ...]:
    artifact = request.teacher.artifact
    if not isinstance(artifact, HubModelRef):
        raise ValueError("TRL distillation currently requires a Hugging Face teacher model")
    engine = request.teacher_inference.engine
    command: list[str] = [
        str(Path(sys.executable).with_name("trl")),
        "vllm-serve",
        "--model",
        artifact.repo_id,
        "--revision",
        artifact.revision,
        "--host",
        host,
        "--port",
        str(port),
    ]
    for key, flag in (
        ("max_model_len", "--max_model_len"),
        ("gpu_memory_utilization", "--gpu_memory_utilization"),
        ("tensor_parallel_size", "--tensor_parallel_size"),
    ):
        value = engine.get(key)
        if isinstance(value, int | float):
            command.extend([flag, str(value)])
    if engine.get("gpu_memory_utilization") is None:
        command.extend(["--gpu_memory_utilization", "0.15"])
    if engine.get("enforce_eager") is not False:
        command.append("--enforce_eager")
    dtype = engine.get("dtype")
    if isinstance(dtype, str) and dtype:
        command.extend(["--dtype", dtype])
    return tuple(command)


def _patch_local_teacher_divergence_numerics(trainer_type: Any) -> None:
    """Stabilize the local-teacher sparse top-1 path for padded on-policy batches."""
    if getattr(trainer_type, "_posttrain_local_teacher_patch", False):
        return

    def _compute_local_sparse_top_1_divergence_loss(
        self: Any,
        student_logits: Any,
        teacher_logits: Any,
        completion_tokens: Any,
        labels: Any,
        num_items_in_batch=None,
    ) -> Any:
        import torch
        import torch.nn.functional as F

        student_log_probs = F.log_softmax(student_logits.float() / self.temperature, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits.float() / self.temperature, dim=-1)

        teacher_top1_token_ids = teacher_logits.argmax(dim=-1)
        teacher_top1_logprobs = teacher_log_probs.gather(
            dim=-1,
            index=teacher_top1_token_ids.unsqueeze(-1),
        ).squeeze(-1)
        reverse_token_ids = self._get_reverse_kl_top_1_tokens(student_logits, completion_tokens)
        reverse_teacher_logprobs = teacher_log_probs.gather(
            dim=-1,
            index=reverse_token_ids.unsqueeze(-1),
        ).squeeze(-1)
        if labels is not None:
            pad_mask = labels == -100
            neutral = torch.zeros((), dtype=reverse_teacher_logprobs.dtype, device=reverse_teacher_logprobs.device)
            teacher_top1_logprobs = torch.where(pad_mask, neutral, teacher_top1_logprobs)
            reverse_teacher_logprobs = torch.where(pad_mask, neutral, reverse_teacher_logprobs)

        return self._compute_sparse_top_1_divergence_loss(
            student_log_probs=student_log_probs,
            teacher_top1_token_ids=teacher_top1_token_ids,
            teacher_top1_logprobs=teacher_top1_logprobs,
            reverse_token_ids=reverse_token_ids,
            reverse_teacher_logprobs=reverse_teacher_logprobs,
            labels=labels,
            num_items_in_batch=num_items_in_batch,
        )

    trainer_type._compute_local_sparse_top_1_divergence_loss = _compute_local_sparse_top_1_divergence_loss
    trainer_type._posttrain_local_teacher_patch = True


def _rollout_function(
    context: RunContext,
    request: OnPolicyDistillationRequest,
    tokenizer: Any,
    ledger: DistillationBatchLedger,
) -> Any:
    """Translate a fresh Verifiers batch into the TRL exact-token hook."""

    def run_rollouts(
        prompts: list[Any],
        trainer: Any,
        *,
        inputs: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        if inputs is None or len(inputs) != len(prompts):
            raise ValueError("TRL must provide dataset rows aligned with distillation prompts")
        try:
            example_ids = tuple(str(row["example_id"]) for row in inputs)
        except KeyError as error:
            raise ValueError("every distillation dataset row requires an example_id") from error

        from .online_rl import TrlPolicyGenerator

        step = int(trainer.state.global_step)
        generator = TrlPolicyGenerator(
            trainer,
            tokenizer,
            request.student,
            request.settings,
            request.training,
        )
        with context.phase("rollout", {"backend": "trl", "logical_step": step}):
            rollouts = tuple(
                asyncio.run(
                    request.bridge.run(
                        RolloutBatch(example_ids=example_ids, step=step, model_id=request.student.id),
                        generator,
                    )
                )
            )
        source_indices: list[int] = []
        for rollout in rollouts:
            value = rollout.trace.attributes.get("source_batch_position")
            if value is None and len(rollouts) == len(inputs):
                value = len(source_indices)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value >= len(inputs):
                raise ValueError("environment rollout has no valid source batch position")
            source_indices.append(value)
        if set(source_indices) != set(range(len(inputs))):
            raise ValueError("environment rollouts do not cover every distillation trainer input")
        for rollout in rollouts:
            _validate_rollout_token_budget(request, rollout)
        trace_ids = tuple(rollout.trace.external_id for rollout in rollouts)
        digest = hashlib.sha256()
        digest.update(ledger.policy_revision.encode())
        digest.update(str(step).encode())
        for trace_id in trace_ids:
            digest.update(b"\0")
            digest.update(trace_id.encode())
        batch = DistillationBatch(
            batch_id=f"distill-{digest.hexdigest()[:20]}",
            policy_revision=ledger.policy_revision,
            step=step,
            rollouts=rollouts,
        )
        consumed = ledger.consume(batch)
        attributes = {
            "technique": "distill",
            "student_model_variant_id": request.student.id,
            "teacher_model_variant_id": request.teacher.id,
            "training_settings_id": request.settings.id,
            "distillation_batch_id": batch.batch_id,
            "policy_revision": batch.policy_revision,
        }
        for rollout in consumed:
            trace = rollout.trace
            context.trace(
                TraceObservation(
                    trace_type=trace.trace_type,
                    external_id=trace.external_id,
                    payload=trace.payload,
                    attributes={**trace.attributes, **attributes},
                )
            )
        scored_tokens = sum(sum(rollout.env_mask) for rollout in consumed)
        context.metric("train/distill/scored_tokens", scored_tokens, step=step, attributes=attributes)
        context.event(
            "distillation_batch_consumed",
            {
                **attributes,
                "step": step,
                "source_prompts": len(inputs),
                "rollouts": len(consumed),
                "scored_tokens": scored_tokens,
            },
        )
        return {
            "prompt_ids": [list(rollout.prompt_ids) for rollout in consumed],
            "prompt_lengths": [len(rollout.prompt_ids) for rollout in consumed],
            "completion_ids": [list(rollout.completion_ids) for rollout in consumed],
            "completion_loss_mask": [list(rollout.env_mask) for rollout in consumed],
            "logprobs": [list(rollout.sampling_logprobs) for rollout in consumed],
            "rollout_ids": list(trace_ids),
            "source_indices": source_indices,
        }

    return run_rollouts


def _validate_rollout_token_budget(
    request: OnPolicyDistillationRequest,
    rollout: EnvironmentRollout,
) -> None:
    prompt_tokens = len(rollout.prompt_ids)
    completion_tokens = len(rollout.completion_ids)
    total_tokens = prompt_tokens + completion_tokens
    prompt_limit = request.settings.max_prompt_length
    completion_limit = request.settings.max_completion_length
    total_limit = request.settings.loop.max_length
    if (
        prompt_tokens <= prompt_limit
        and completion_tokens <= completion_limit
        and total_tokens <= total_limit
    ):
        return
    raise ValueError(
        "distillation rollout exceeds the configured token budget before teacher scoring: "
        f"example_id={rollout.example_id!r}, tangent={_rollout_tangent(rollout)!r}, "
        f"prompt_tokens={prompt_tokens}/{prompt_limit}, "
        f"completion_tokens={completion_tokens}/{completion_limit}, "
        f"total_tokens={total_tokens}/{total_limit}"
    )


def _rollout_tangent(rollout: EnvironmentRollout) -> str:
    direct = rollout.trace.attributes.get("tangent")
    if isinstance(direct, str) and direct:
        return direct
    payload = rollout.trace.payload
    for container_name in ("info", "task"):
        container = payload.get(container_name)
        if not isinstance(container, Mapping):
            continue
        candidates = (container, container.get("data"))
        for candidate in candidates:
            if isinstance(candidate, Mapping):
                tangent = candidate.get("tangent") or candidate.get("tangent_id")
                if isinstance(tangent, str) and tangent:
                    return tangent
    return "unknown"


class _ObservedTeacherClient:
    """Record teacher scoring latency and failures without changing TRL's client contract."""

    def __init__(self, context: RunContext, client: Any, step: Any | None = None) -> None:
        self._context = context
        self._client = client
        self._step = step or (lambda: None)

    def get_sequence_logprobs(self, **kwargs: Any) -> Any:
        started = time.perf_counter()
        step = self._step()
        with self._context.phase("teacher_scoring", {"backend": "trl"}):
            try:
                result = self._client.get_sequence_logprobs(**kwargs)
            except Exception:
                self._context.metric("train/distill/teacher_failures", 1, step=step)
                raise
            else:
                self._context.metric("train/distill/teacher_failures", 0, step=step)
                return result
            finally:
                self._context.metric(
                    "train/distill/teacher_latency_ms",
                    (time.perf_counter() - started) * 1_000,
                    step=step,
                )


def _normalize_distillation_metrics(_step: int, records: Mapping[str, object]) -> Mapping[str, float]:
    values: dict[str, float] = {}
    raw_loss = records.get("on_policy_loss", records.get("loss"))
    if isinstance(raw_loss, int | float) and not isinstance(raw_loss, bool):
        loss = float(raw_loss)
        if not math.isfinite(loss):
            raise FloatingPointError(f"non-finite distillation loss={loss}")
        values["train/distill/loss"] = loss
        values["train/distill/reverse_kl"] = loss
    for raw_name, metric_name in (
        ("grad_norm", "train/grad_norm"),
        ("learning_rate", "train/learning_rate"),
        ("num_input_tokens_seen", "train/num_tokens"),
    ):
        raw = records.get(raw_name)
        if isinstance(raw, int | float) and not isinstance(raw, bool):
            numeric = float(raw)
            if not math.isfinite(numeric):
                raise FloatingPointError(f"non-finite training metric {raw_name}={numeric}")
            values[metric_name] = numeric
    return values


def _distillation_arguments(
    request: OnPolicyDistillationRequest,
    output_dir: Path,
    teacher_url: str | None,
) -> dict[str, Any]:
    teacher_artifact = request.teacher.artifact
    if not isinstance(teacher_artifact, HubModelRef):
        raise ValueError("TRL distillation currently requires a Hugging Face teacher model")
    arguments = trainer_arguments(request.settings.loop, output_dir)
    rollout = request.rollout_inference.engine
    speculative_config, engine_kwargs = vllm_rollout_options(request.student, rollout)
    teacher_product = request.teacher_inference.backend.split("@", 1)[0]
    use_liger_kernel = request.training.backend_options.get("use_liger_kernel", False)
    if not isinstance(use_liger_kernel, bool):
        raise ValueError("TRL distillation use_liger_kernel must be a boolean")
    use_bf16 = request.training.backend_options.get("bf16", True)
    if not isinstance(use_bf16, bool):
        raise ValueError("TRL distillation bf16 must be a boolean")
    if not use_bf16:
        arguments["bf16"] = False
        arguments["fp16"] = False
    arguments.update(
        {
            "remove_unused_columns": False,
            "include_num_input_tokens_seen": "non_padding",
            "use_liger_kernel": use_liger_kernel,
            "lmbda": 1.0,
            "beta": 1.0,
            "reverse_kl_top_1_mode": "sampled",
            "loss_top_k": 1,
            "temperature": request.settings.temperature,
            "num_generations": request.settings.num_generations,
            "generation_batch_size": request.settings.num_prompts_per_step,
            "max_prompt_length": request.settings.max_prompt_length,
            "max_completion_length": request.settings.max_completion_length,
            "use_teacher_server": teacher_product == "vllm",
            "teacher_model_server_url": teacher_url,
            "teacher_model_revision": teacher_artifact.revision,
            # The adapter loads a pinned local teacher object with the same
            # architecture resolver as the student. TRL must not auto-load it.
            "teacher_model_init_kwargs": None,
            "use_vllm": True,
            "vllm_mode": rollout.get("mode", "colocate"),
            "vllm_server_base_url": rollout.get("base_url"),
            "vllm_gpu_memory_utilization": rollout.get("gpu_memory_utilization", 0.3),
            "vllm_tensor_parallel_size": rollout.get("tensor_parallel_size", 1),
            "vllm_max_model_length": rollout.get("max_model_len"),
            "vllm_enable_sleep_mode": rollout.get("sleep_during_optimization", False),
            "vllm_weight_sync_mode": (
                request.training.backend_options.get("rollout_weight_sync_mode")
                if isinstance(request.training.backend_options.get("rollout_weight_sync_mode"), str)
                else rollout.get("weight_sync_mode", "full")
            ),
            "vllm_speculative_config": speculative_config,
            "vllm_engine_kwargs": engine_kwargs,
            "top_p": _sampling_number(request, "top_p", 1.0),
        }
    )
    return arguments


def _sampling_number(request: OnPolicyDistillationRequest, key: str, default: float) -> float:
    value = request.rollout_inference.sampling.get(key)
    return float(value) if isinstance(value, (int, float)) else default


__all__ = ["run_distillation"]
