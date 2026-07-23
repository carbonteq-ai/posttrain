"""TRL adapter for Verifiers-driven on-policy distillation."""

from __future__ import annotations

import asyncio
import hashlib
import time
from pathlib import Path
from typing import Any, cast

from posttrain.common import RunContext, TraceObservation

from ...distillation import DistillationBatch, DistillationBatchLedger
from ...online_rl import RolloutBatch
from ...requests import OnPolicyDistillationRequest
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
    vllm_rollout_options,
)


def run_distillation(
    context: RunContext,
    request: OnPolicyDistillationRequest,
    output_dir: Path,
) -> BackendTrainingResult:
    if request.rollout_inference.backend.split("@", 1)[0] != "vllm":
        raise ValueError("the first TRL distillation adapter requires a vLLM student rollout binding")
    if request.teacher_inference.backend.split("@", 1)[0] != "vllm":
        raise ValueError("the first TRL distillation adapter requires a vLLM teacher-score binding")
    teacher_url = request.teacher_inference.engine.get("base_url")
    if not isinstance(teacher_url, str) or not teacher_url.strip():
        raise ValueError("the teacher-score binding must provide engine.base_url")

    try:
        from trl.experimental.distillation import (  # pyright: ignore[reportMissingImports]
            DistillationConfig,
            DistillationTrainer,
        )
    except ImportError as error:
        raise RuntimeError("install posttrain-train with the trl-vllm extra") from error

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
    arguments = _distillation_arguments(request, output_dir, teacher_url)
    with context.phase("runtime_initialization", {"backend": "trl"}):
        trainer = DistillationTrainer(
            model=model,
            teacher_model=cast(Any, None),
            args=DistillationConfig(**arguments),
            train_dataset=dataset,
            processing_class=tokenizer,
            callbacks=[callback_type(context, imports)()],
            rollout_func=cast(Any, _rollout_function(context, request, tokenizer, ledger)),
        )
    if trainer.teacher_client is None:
        raise RuntimeError("TRL did not initialize the configured teacher scoring client")
    trainer.teacher_client = cast(Any, _ObservedTeacherClient(context, trainer.teacher_client))
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
    teacher_url: str,
) -> dict[str, Any]:
    arguments = trainer_arguments(request.settings.loop, output_dir)
    rollout = request.rollout_inference.engine
    speculative_config, engine_kwargs = vllm_rollout_options(request.student, rollout)
    arguments.update(
        {
            "remove_unused_columns": False,
            "lmbda": 1.0,
            "beta": 1.0,
            "reverse_kl_top_1_mode": "sampled",
            "loss_top_k": 1,
            "temperature": request.settings.temperature,
            "num_generations": request.settings.num_generations,
            "generation_batch_size": request.settings.loop.per_device_batch_size
            * request.settings.loop.gradient_accumulation_steps,
            "max_prompt_length": request.settings.max_prompt_length,
            "max_completion_length": request.settings.max_completion_length,
            "use_teacher_server": True,
            "teacher_model_server_url": teacher_url,
            "use_vllm": True,
            "vllm_mode": rollout.get("mode", "colocate"),
            "vllm_server_base_url": rollout.get("base_url"),
            "vllm_gpu_memory_utilization": rollout.get("gpu_memory_utilization", 0.3),
            "vllm_tensor_parallel_size": rollout.get("tensor_parallel_size", 1),
            "vllm_max_model_length": rollout.get("max_model_len"),
            "vllm_enable_sleep_mode": rollout.get("sleep_during_optimization", False),
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
