"""TRL adapter for Verifiers-driven on-policy distillation."""

from __future__ import annotations

import asyncio
import hashlib
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import httpx
from posttrain.common import HubModelRef, RunContext, TraceObservation

from ...distillation import DistillationBatch, DistillationBatchLedger
from ...online_rl import RolloutBatch, policy_sampling_from_binding
from ...requests import OnPolicyDistillationRequest
from .common import (
    BackendTrainingResult,
    callback_type,
    checkpoint_callback_type,
    emit_parameter_counts,
    emit_runtime_versions,
    finish_training,
    framework_imports,
    load_tokenizer,
    load_trainable_model,
    preserve_recovery_checkpoint_after_error,
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

    try:
        from trl.experimental.iw_opd import (  # pyright: ignore[reportMissingImports]
            IWOPDConfig,
            IWOPDTrainer,
        )
    except ImportError as error:
        raise RuntimeError("install posttrain-train with the trl-vllm extra") from error
    _patch_local_teacher_divergence_numerics(IWOPDTrainer)

    imports = framework_imports()
    emit_runtime_versions(context, imports)
    with context.phase("model_loading", {"backend": "trl"}):
        tokenizer = load_tokenizer(request.student, imports)
        model = load_trainable_model(request.student, request.training.update, request.settings.loop, imports)
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

        class ObservedIWOPDTrainer(IWOPDTrainer):
            def _get_teacher_logits(self, inputs: dict[str, Any]) -> Any:
                started_at = time.perf_counter()
                with context.phase("teacher_scoring", {"backend": "trl"}):
                    try:
                        result = super()._get_teacher_logits(inputs)
                    except Exception:
                        context.metric("train/distill/teacher_failures", 1)
                        raise
                    else:
                        context.metric("train/distill/teacher_failures", 0)
                        return result
                    finally:
                        context.metric(
                            "train/distill/teacher_latency_ms",
                            (time.perf_counter() - started_at) * 1_000,
                        )

        checkpoint_callback = checkpoint_callback_type(
            context,
            imports,
            model=request.student,
            technique="distill",
            settings=request.settings,
            update=request.training.update,
            workspace=output_dir.parent,
        )()
        with context.phase("runtime_initialization", {"backend": "trl"}):
            trainer = ObservedIWOPDTrainer(
                model=model,
                teacher_model=cast(
                    Any,
                    (None if teacher_product == "vllm" else request.teacher.artifact.repo_id),
                ),
                args=IWOPDConfig(**arguments),
                train_dataset=dataset,
                processing_class=tokenizer,
                callbacks=[callback_type(context, imports)(), checkpoint_callback],
                rollout_func=cast(Any, _rollout_function(context, request, tokenizer, ledger)),
            )
        if teacher_product == "vllm":
            if trainer.teacher_client is None:
                raise RuntimeError("TRL did not initialize the configured teacher scoring client")
            trainer.teacher_client = cast(
                Any,
                _ObservedTeacherClient(context, trainer.teacher_client),
            )
        elif trainer.teacher_model is None:
            raise RuntimeError("TRL did not initialize the colocated teacher model")
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
                        "distill",
                        request.training.update,
                        imports,
                    )
            except BaseException as error:
                preserve_recovery_checkpoint_after_error(
                    context,
                    trainer,
                    error,
                    technique="distill",
                    model=request.student,
                    settings=request.settings,
                    update=request.training.update,
                    imports=imports,
                )
                raise


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
        if len(rollouts) != len(inputs):
            raise ValueError("environment rollout count does not match the distillation trainer batch")
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
        }

    return run_rollouts


class _ObservedTeacherClient:
    """Record teacher scoring latency and failures without changing TRL's client contract."""

    def __init__(self, context: RunContext, client: Any) -> None:
        self._context = context
        self._client = client

    def get_sequence_logprobs(self, **kwargs: Any) -> Any:
        started = time.perf_counter()
        with self._context.phase("teacher_scoring", {"backend": "trl"}):
            try:
                result = self._client.get_sequence_logprobs(**kwargs)
            except Exception:
                self._context.metric("train/distill/teacher_failures", 1)
                raise
            else:
                self._context.metric("train/distill/teacher_failures", 0)
                return result
            finally:
                self._context.metric(
                    "train/distill/teacher_latency_ms",
                    (time.perf_counter() - started) * 1_000,
                )


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
    teacher_engine = request.teacher_inference.engine
    use_liger_kernel = request.training.backend_options.get("use_liger_kernel", False)
    if not isinstance(use_liger_kernel, bool):
        raise ValueError("TRL distillation use_liger_kernel must be a boolean")
    use_bf16 = request.training.backend_options.get("bf16", True)
    if not isinstance(use_bf16, bool):
        raise ValueError("TRL distillation bf16 must be a boolean")
    if not use_bf16:
        arguments["bf16"] = False
        arguments["fp16"] = False
    sampling = policy_sampling_from_binding(
        request.rollout_inference,
        request.settings.max_completion_length,
        default_temperature=request.settings.temperature,
    )
    arguments.update(
        {
            "remove_unused_columns": False,
            "use_liger_kernel": use_liger_kernel,
            "lmbda": 1.0,
            "distillation_objective": "iw_opd",
            "beta": 1.0,
            "reverse_kl_top_1_mode": "sampled",
            "loss_top_k": 1,
            "temperature": sampling.temperature,
            "num_generations": request.settings.num_generations,
            "generation_batch_size": request.settings.num_prompts_per_step,
            "max_prompt_length": request.settings.max_prompt_length,
            "max_completion_length": request.settings.max_completion_length,
            "use_teacher_server": teacher_product == "vllm",
            "teacher_model_server_url": teacher_url,
            "teacher_model_revision": teacher_artifact.revision,
            "teacher_model_init_kwargs": (
                {
                    "revision": teacher_artifact.revision,
                    "dtype": teacher_engine.get("dtype", "bfloat16"),
                }
                if teacher_product == "transformers"
                else None
            ),
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
            "top_p": sampling.top_p,
            "top_k": sampling.top_k,
            "min_p": sampling.min_p,
            "repetition_penalty": sampling.repetition_penalty,
        }
    )
    if sampling.presence_penalty:
        arguments["generation_kwargs"] = {"presence_penalty": sampling.presence_penalty}
    return arguments


__all__ = ["run_distillation"]
