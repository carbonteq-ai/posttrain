"""TRL adapter for Verifiers-driven on-policy distillation."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
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
from ...online_rl import RolloutBatch
from ...rendering import create_renderer
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
    restore_checkpoint_runtime_states,
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
    memory_safe_iw_opd = request.training.backend_options.get("memory_safe_iw_opd_loss", False)
    if not isinstance(memory_safe_iw_opd, bool):
        raise ValueError("memory_safe_iw_opd_loss must be a boolean")
    if memory_safe_iw_opd:
        _validate_memory_safe_iw_opd_request(request, teacher_product)
        _validate_iw_opd_private_contract(IWOPDTrainer)
    runtime_state_paths = _runtime_state_paths(request)

    imports = framework_imports()
    emit_runtime_versions(context, imports)
    with context.phase("model_loading", {"backend": "trl"}):
        tokenizer = load_tokenizer(request.student, imports)
        teacher_tokenizer = load_tokenizer(request.teacher, imports)
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
            def _generate_policy_turns(
                self,
                prompt_ids: list[list[int]],
                generation_overrides: list[dict[str, Any]],
            ) -> tuple[list[list[int]], list[list[float]]]:
                return _generate_heterogeneous_colocated_iw_opd_turns(
                    context,
                    self,
                    prompt_ids,
                    generation_overrides,
                )

            def compute_loss(
                self,
                model: Any,
                inputs: dict[str, Any],
                return_outputs: bool = False,
                num_items_in_batch: Any = None,
            ) -> Any:
                if not memory_safe_iw_opd:
                    return super().compute_loss(
                        model,
                        inputs,
                        return_outputs=return_outputs,
                        num_items_in_batch=num_items_in_batch,
                    )
                effective_items = num_items_in_batch
                trainer_scale = 1
                if effective_items is None:
                    effective_items = _buffered_selected_token_count(self)
                    # Transformers divides a custom loss by the accumulation
                    # window when its pre-generation batch has no labels. Undo
                    # that division after normalizing over the generated window.
                    trainer_scale = int(
                        getattr(
                            self,
                            "current_gradient_accumulation_steps",
                            self.args.gradient_accumulation_steps,
                        )
                    )
                result = _memory_safe_server_iw_opd_loss(
                    self,
                    model,
                    inputs,
                    return_outputs=return_outputs,
                    num_items_in_batch=effective_items,
                    chunk_size=_positive_backend_integer(
                        request,
                        "logit_chunk_size",
                        default=16,
                    ),
                )
                if return_outputs:
                    loss, outputs = result
                    return loss * trainer_scale, outputs
                return result * trainer_scale

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

            def _get_local_constrained_teacher_logprobs(
                self,
                inputs: dict[str, Any],
                aligned_prompt_length: int,
            ) -> dict[str, Any]:
                started_at = time.perf_counter()
                with context.phase("teacher_scoring", {"backend": "transformers"}):
                    try:
                        result = _local_constrained_teacher_logprobs(
                            self,
                            inputs,
                            aligned_prompt_length,
                        )
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
            runtime_state_paths=runtime_state_paths,
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
                rollout_func=cast(
                    Any,
                    _rollout_function(
                        context,
                        request,
                        tokenizer,
                        ledger,
                        teacher_tokenizer=teacher_tokenizer,
                    ),
                ),
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
        if request.resume_from is not None and runtime_state_paths:
            restore_checkpoint_runtime_states(request.resume_from.path, runtime_state_paths)
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
    kv_cache_dtype = engine.get("kv_cache_dtype")
    if isinstance(kv_cache_dtype, str) and kv_cache_dtype:
        command.extend(["--kv_cache_dtype", kv_cache_dtype])
    enable_prefix_caching = engine.get("enable_prefix_caching")
    if enable_prefix_caching is not None:
        if not isinstance(enable_prefix_caching, bool):
            raise ValueError("teacher enable_prefix_caching must be a boolean")
        command.extend(["--enable_prefix_caching", str(enable_prefix_caching).lower()])
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
    *,
    teacher_tokenizer: Any | None = None,
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
        constrained = request.settings.probability_space == "generation_constrained"
        if constrained and teacher_tokenizer is None:
            raise ValueError("generation-constrained OPD requires the teacher tokenizer")
        teacher_renderer = (
            create_renderer(teacher_tokenizer, request.teacher, request.training.renderer) if constrained else None
        )
        teacher_prompt_ids: list[list[int]] = []
        teacher_completion_ids: list[list[int]] = []
        structured_output_schemas: list[dict[str, Any]] = []
        schema_digests: list[str] = []
        constrained_request_ids: list[str] = []
        allowed_set_digests: list[list[str]] = []
        grammar_prefix_ids: list[list[int]] = []
        if constrained:
            if teacher_renderer is None:
                raise ValueError("generation-constrained OPD requires the teacher tokenizer")
            for rollout in consumed:
                marker = _opd_marker(rollout.trace.payload)
                contract = generator.selected_turn_contract(
                    str(marker["selected_prompt_sha256"]),
                    str(marker["selected_completion_sha256"]),
                )
                messages = cast(list[dict[str, Any]], contract["messages"])
                tools = cast(list[dict[str, Any]], contract["tools"])
                rendered = teacher_renderer.render(
                    messages,
                    tools=tools or None,
                    add_generation_prompt=True,
                )
                structured = contract.get("structured_outputs")
                schema = structured.get("json") if isinstance(structured, dict) else None
                if not isinstance(schema, dict):
                    raise ValueError("generation-constrained OPD requires one JSON schema per selected turn")
                schema_digest = _json_sha256(schema)
                completion = list(rollout.completion_ids)
                generated = [int(value) for value in cast(list[int], contract["completion_ids"])]
                if len(generated) < len(completion) or generated[-len(completion) :] != completion:
                    raise ValueError("selected OPD completion is not a suffix of the generated assistant turn")
                grammar_prefix = generated[: len(generated) - len(completion)]
                teacher_prompt_ids.append([int(value) for value in rendered.token_ids] + grammar_prefix)
                teacher_completion_ids.append(completion)
                grammar_prefix_ids.append(grammar_prefix)
                structured_output_schemas.append(schema)
                schema_digests.append(schema_digest)
                constrained_request_ids.append(rollout.trace.external_id)
                allowed_set_digests.append(
                    [
                        _allowed_set_digest(
                            schema_digest,
                            completion,
                            position,
                            grammar_prefix_ids=grammar_prefix,
                        )
                        for position in range(len(completion))
                    ]
                )
            generator.clear_turn_contracts()
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
        result = {
            "prompt_ids": [list(rollout.prompt_ids) for rollout in consumed],
            "prompt_lengths": [len(rollout.prompt_ids) for rollout in consumed],
            "completion_ids": [list(rollout.completion_ids) for rollout in consumed],
            "completion_loss_mask": [list(rollout.env_mask) for rollout in consumed],
            "logprobs": [list(rollout.sampling_logprobs) for rollout in consumed],
            "rollout_ids": list(trace_ids),
        }
        if constrained:
            result.update(
                {
                    "teacher_prompt_ids": teacher_prompt_ids,
                    "teacher_completion_ids": teacher_completion_ids,
                    "teacher_completion_offsets": [0] * len(consumed),
                    "structured_output_schemas": structured_output_schemas,
                    "schema_digests": schema_digests,
                    "constrained_request_ids": constrained_request_ids,
                    "allowed_set_digests": allowed_set_digests,
                    "grammar_prefix_ids": grammar_prefix_ids,
                }
            )
        return result

    return run_rollouts


def _opd_marker(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise TypeError("OPD trace payload must be an object")
    info = record.get("info")
    policy = info.get("policy_prism") if isinstance(info, dict) else None
    marker = policy.get("opd") if isinstance(policy, dict) else None
    if not isinstance(marker, dict):
        raise ValueError("generation-constrained OPD trace is missing its selected-turn marker")
    return marker


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _allowed_set_digest(
    schema_digest: str,
    completion_ids: list[int],
    position: int,
    *,
    grammar_prefix_ids: list[int] | None = None,
) -> str:
    prefix = [] if grammar_prefix_ids is None else grammar_prefix_ids
    return _json_sha256(
        {
            "completion_prefix": prefix + completion_ids[:position],
            "position": position,
            "schema_digest": schema_digest,
            "xgrammar_contract": "0.2.3-json-schema-any-whitespace-false",
        }
    )


class _ObservedTeacherClient:
    """Record teacher scoring latency and failures without changing TRL's client contract."""

    def __init__(self, context: RunContext, client: Any) -> None:
        self._context = context
        self._client = client

    def get_sequence_logprobs(self, **kwargs: Any) -> Any:
        return self._observe("get_sequence_logprobs", kwargs)

    def get_constrained_sequence_logprobs(self, **kwargs: Any) -> Any:
        return self._observe("get_constrained_sequence_logprobs", kwargs)

    def _observe(self, method: str, kwargs: dict[str, Any]) -> Any:
        started = time.perf_counter()
        with self._context.phase("teacher_scoring", {"backend": "trl"}):
            try:
                result = getattr(self._client, method)(**kwargs)
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
    arguments.update(
        {
            "remove_unused_columns": False,
            "use_liger_kernel": use_liger_kernel,
            "lmbda": 1.0,
            "distillation_objective": "iw_opd",
            "iw_opd_gamma": _backend_number(request, "iw_opd_gamma", 0.5),
            "iw_opd_epsilon": _backend_number(request, "iw_opd_epsilon", 1e-8),
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
            "top_p": _sampling_number(request, "top_p", 1.0),
        }
    )
    return arguments


def _validate_iw_opd_private_contract(trainer_type: Any) -> None:
    """Fail closed when the pinned IW-OPD seams used by the bounded loss move."""

    expected = {
        "_compute_prompt_length": ("self", "inputs"),
        "_get_teacher_token_logprobs_from_server": (
            "self",
            "inputs",
            "aligned_prompt_length",
        ),
    }
    for name, parameters in expected.items():
        method = getattr(trainer_type, name, None)
        if method is None:
            raise RuntimeError(f"pinned IWOPDTrainer is missing required method {name}")
        actual = tuple(inspect.signature(method).parameters)
        if actual != parameters:
            raise RuntimeError(f"pinned IWOPDTrainer method {name} has parameters {actual}, expected {parameters}")


def _generate_heterogeneous_colocated_iw_opd_turns(
    context: RunContext,
    trainer: Any,
    prompt_ids: list[list[int]],
    generation_overrides: list[dict[str, Any]],
) -> tuple[list[list[int]], list[list[float]]]:
    """Generate a bounded colocated-vLLM batch with per-request schemas and limits."""

    if len(prompt_ids) != len(generation_overrides) or not prompt_ids:
        raise ValueError("policy prompts and generation overrides must be non-empty and aligned")
    if not trainer.use_vllm:
        raise RuntimeError("heterogeneous Policy rollouts require colocated vLLM")
    generation = trainer.vllm_generation
    if generation.mode != "colocate" or generation.tensor_parallel_size != 1:
        raise RuntimeError("heterogeneous Policy rollouts require single-GPU colocated vLLM")

    if (
        trainer.state.global_step != trainer._last_vllm_sync_step
        and trainer.state.global_step % trainer.vllm_sync_frequency == 0
    ):
        generation.sync_weights()
        trainer._last_vllm_sync_step = trainer.state.global_step

    try:
        from trl.experimental.iw_opd.iw_opd_trainer import (  # pyright: ignore[reportMissingImports]
            _accumulate_spec_decode_metrics,
        )
        from trl.generation.vllm_generation import extract_logprobs  # pyright: ignore[reportMissingImports]
        from vllm import SamplingParams  # pyright: ignore[reportMissingImports]
        from vllm.sampling_params import StructuredOutputsParams  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise RuntimeError("install posttrain-train with the trl-vllm extra") from error

    common_kwargs = {
        "n": 1,
        "repetition_penalty": generation.repetition_penalty,
        "temperature": generation.temperature,
        "top_p": generation.top_p,
        "top_k": generation.top_k,
        "min_p": 0.0 if generation.min_p is None else generation.min_p,
        "logprobs": generation.logprobs,
        **dict(generation.generation_kwargs),
    }
    common_kwargs.pop("max_tokens", None)
    common_kwargs.pop("structured_outputs", None)
    sampling_params = []
    for override in generation_overrides:
        max_tokens = override.get("max_tokens")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("each Policy rollout requires a positive max_tokens override")
        structured = override.get("structured_outputs")
        if structured is not None and not isinstance(structured, dict):
            raise TypeError("structured_outputs must be a mapping when supplied")
        parameters = {**common_kwargs, "max_tokens": max_tokens}
        if structured is not None:
            parameters["structured_outputs"] = StructuredOutputsParams(**structured)
        sampling_params.append(SamplingParams(**parameters))

    generation._wake_weights_for_generation()
    if generation.enable_sleep_mode:
        generation.llm.wake_up(tags=["kv_cache"])
    prompts = [{"prompt_token_ids": ids} for ids in prompt_ids]
    wave_size = generation.max_num_seqs or len(prompts)
    outputs = []
    if generation._kv_cache_peak_tracker is not None:
        generation._kv_cache_peak_tracker.reset()
    for start in range(0, len(prompts), wave_size):
        outputs.extend(
            generation.llm.generate(
                prompts[start : start + wave_size],
                sampling_params=sampling_params[start : start + wave_size],
                use_tqdm=False,
                lora_request=generation._lora_request,
            )
        )
    completion_ids = [output.token_ids for request in outputs for output in request.outputs]
    logprobs, _logprob_token_ids = extract_logprobs(outputs)
    generation._collect_generation_metrics()
    generation._sleep_colocated_engine()
    mode = "train" if trainer.model.training else "eval"
    _accumulate_spec_decode_metrics(trainer._metrics[mode], generation.last_generation_metrics)
    if logprobs is None:
        raise RuntimeError("vLLM must return sampled-token logprobs for Policy IW-OPD rollouts")
    context.metric("train/rollout/request_batch_size", len(prompts))
    context.metric("train/rollout/resident_wave_size", min(wave_size, len(prompts)))
    return completion_ids, [[float(token_logprobs[0]) for token_logprobs in row] for row in logprobs]


def _validate_memory_safe_iw_opd_request(
    request: OnPolicyDistillationRequest,
    teacher_product: str,
) -> None:
    if teacher_product not in {"transformers", "vllm"}:
        raise ValueError("memory-safe IW-OPD requires a transformers or vLLM teacher")
    student_artifact = request.student.artifact
    if (
        request.student.family != "gemma4"
        or not isinstance(student_artifact, HubModelRef)
        or student_artifact.repo_id != "google/gemma-4-E2B-it"
    ):
        raise ValueError("memory-safe IW-OPD is qualified only for gemma4-e2b-it")
    if request.settings.num_generations != 1:
        raise ValueError("memory-safe IW-OPD requires one generation per prompt")
    if request.settings.probability_space != "generation_constrained":
        raise ValueError("memory-safe Policy IW-OPD requires generation-constrained probabilities")
    if request.settings.teacher_prompt_alignment != "model_native_prefix_exact_completion":
        raise ValueError("memory-safe Policy IW-OPD requires model-native teacher prompt alignment")
    if request.settings.loop.per_device_batch_size != 1:
        raise ValueError("generation-constrained memory-safe IW-OPD is qualified for physical batch one")
    loop = request.settings.loop
    expected_logical_batch = loop.per_device_batch_size * loop.gradient_accumulation_steps
    if request.settings.num_prompts_per_step != expected_logical_batch:
        raise ValueError(
            "memory-safe IW-OPD requires num_prompts_per_step to equal physical batch times gradient accumulation"
        )
    if request.training.backend_options.get("use_liger_kernel", False):
        raise ValueError("memory-safe IW-OPD is incompatible with the Liger full-logits path")
    gamma = _backend_number(request, "iw_opd_gamma", 0.5)
    epsilon = _backend_number(request, "iw_opd_epsilon", 1e-8)
    if gamma < 0:
        raise ValueError("iw_opd_gamma must be non-negative")
    if epsilon <= 0:
        raise ValueError("iw_opd_epsilon must be positive")
    _positive_backend_integer(request, "logit_chunk_size", default=16)


def _positive_backend_integer(
    request: OnPolicyDistillationRequest,
    key: str,
    *,
    default: int,
) -> int:
    value = request.training.backend_options.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _runtime_state_paths(request: OnPolicyDistillationRequest) -> tuple[Path, ...]:
    raw = request.training.backend_options.get("checkpoint_runtime_state_paths", [])
    if not isinstance(raw, list) or not all(isinstance(value, str) for value in raw):
        raise ValueError("checkpoint_runtime_state_paths must be a list of relative paths")
    paths = tuple(Path(value) for value in raw)
    for path in paths:
        if path.is_absolute() or not path.parts or ".." in path.parts:
            raise ValueError("checkpoint_runtime_state_paths must contain normalized relative paths")
    if len(set(paths)) != len(paths):
        raise ValueError("checkpoint_runtime_state_paths cannot contain duplicates")
    return paths


def _memory_safe_server_iw_opd_loss(
    trainer: Any,
    model: Any,
    inputs: dict[str, Any],
    *,
    return_outputs: bool,
    num_items_in_batch: Any,
    chunk_size: int,
) -> Any:
    """Compute exact sampled-token IW-OPD without retaining sequence-wide logits."""

    import torch

    if (
        (not trainer.use_teacher_server and trainer.teacher_model is None)
        or trainer.distillation_objective != "iw_opd"
    ):
        raise ValueError("memory-safe IW-OPD received an unsupported trainer configuration")
    if trainer.accelerator.num_processes != 1:
        raise ValueError("memory-safe IW-OPD is qualified only for one trainer process")

    prompt_length = trainer._compute_prompt_length(inputs)  # noqa: SLF001 - pinned fork contract
    if trainer.use_teacher_server:
        teacher_result = trainer._get_teacher_token_logprobs_from_server(  # noqa: SLF001
            inputs,
            prompt_length,
        )
    else:
        teacher_result = trainer._get_local_constrained_teacher_logprobs(  # noqa: SLF001
            inputs,
            prompt_length,
        )
    completion_length = int(teacher_result["actual_logprobs"].shape[1])
    all_labels = inputs["labels"][:, prompt_length:]
    if (all_labels[:, completion_length:] != -100).any():
        raise ValueError("teacher server returned fewer completion logprobs than the selected IW-OPD tokens")
    labels = all_labels[:, :completion_length]
    completion_tokens = inputs["input_ids"][:, prompt_length : prompt_length + completion_length]
    valid_mask = labels != -100
    valid_count = valid_mask.sum()
    if int(valid_count.item()) == 0:
        raise ValueError("memory-safe IW-OPD requires at least one selected completion token")

    teacher_actual_logprobs = teacher_result["actual_logprobs"]
    missing_teacher = valid_mask & ~torch.isfinite(teacher_actual_logprobs)
    if missing_teacher.any():
        raise ValueError(
            "teacher logprobs are missing for "
            f"{int(missing_teacher.sum().item())}/{int(valid_count.item())} selected IW-OPD tokens"
        )

    rollout_logprobs = inputs.get("rollout_logprobs")
    if rollout_logprobs is None:
        raise ValueError("memory-safe IW-OPD requires the exact rollout logprobs")
    rollout_logprobs = rollout_logprobs[:, prompt_length : prompt_length + completion_length]

    unwrapped = trainer.accelerator.unwrap_model(model)
    student_outputs = _gemma_hidden_forward(unwrapped, inputs)
    hidden = student_outputs.last_hidden_state[:, prompt_length - 1 : prompt_length - 1 + completion_length, :]
    if hidden.shape[:2] != labels.shape:
        raise RuntimeError("student hidden states do not align with IW-OPD completion positions")
    base = _gemma_base_model(unwrapped)
    head = base.get_output_embeddings()
    softcap = base.config.get_text_config().final_logit_softcapping
    schemas = inputs.get("structured_output_schemas")
    grammar_prefixes = inputs.get("grammar_prefix_ids")
    expected_allowed_digests = inputs.get("allowed_set_digests")
    teacher_allowed_counts = teacher_result.get("allowed_counts")
    if (
        not isinstance(schemas, list)
        or len(schemas) != 1
        or not isinstance(grammar_prefixes, list)
        or len(grammar_prefixes) != 1
        or not isinstance(expected_allowed_digests, list)
        or len(expected_allowed_digests) != 1
        or not isinstance(teacher_allowed_counts, list)
        or len(teacher_allowed_counts) != 1
    ):
        raise ValueError("generation-constrained IW-OPD metadata must align with physical batch one")
    if not bool(valid_mask[0].all()):
        raise ValueError("selected OPD completion tokens must form one contiguous constrained stage")
    matcher, xgrammar = _xgrammar_matcher(trainer.processing_class, schemas[0], head.weight.shape[0])
    for token_id in grammar_prefixes[0]:
        if not matcher.accept_token(int(token_id)):
            raise ValueError("current-student grammar prefix is disallowed by XGrammar")
    student_actual_chunks: list[Any] = []
    for start in range(0, completion_length, chunk_size):
        end = min(start + chunk_size, completion_length)
        logits = head(hidden[:, start:end, :])
        if softcap is not None:
            logits = torch.tanh(logits / softcap) * softcap
        scaled = logits.float() / trainer.temperature
        bitmask = xgrammar.allocate_token_bitmask(end - start, scaled.shape[-1])
        for local_position, position in enumerate(range(start, end)):
            matcher.fill_next_token_bitmask(bitmask, local_position)
            selected_token = int(completion_tokens[0, position].item())
            if not matcher.accept_token(selected_token):
                raise ValueError(f"current-student token is disallowed by XGrammar at position {position}")
            expected_digest = _allowed_set_digest(
                schema_digest=inputs["schema_digests"][0],
                completion_ids=completion_tokens[0].tolist(),
                position=position,
                grammar_prefix_ids=grammar_prefixes[0],
            )
            if expected_digest != expected_allowed_digests[0][position]:
                raise ValueError("current-student allowed-token digest differs from rollout evidence")
        xgrammar.apply_token_bitmask_inplace(
            scaled.reshape(end - start, -1),
            bitmask.to(scaled.device),
        )
        for local_position, position in enumerate(range(start, end)):
            count = int(torch.isfinite(scaled[0, local_position]).sum().item())
            if count != int(teacher_allowed_counts[0][position]):
                raise ValueError("teacher and current-student allowed-token counts differ")
        selected = scaled.gather(
            dim=-1,
            index=completion_tokens[:, start:end].unsqueeze(-1),
        ).squeeze(-1)
        student_actual_chunks.append(selected - torch.logsumexp(scaled, dim=-1))
    student_actual_logprobs = torch.cat(student_actual_chunks, dim=1)

    safe_teacher = torch.where(valid_mask, teacher_actual_logprobs, 0.0)
    safe_rollout = torch.where(valid_mask, rollout_logprobs, 0.0)
    advantages_base = (safe_teacher - safe_rollout).detach()
    advantages_base = torch.where(valid_mask, advantages_base, 0.0)
    absolute = advantages_base.abs()
    prefix_absolute = absolute.cumsum(dim=1) - absolute
    total_absolute = absolute.sum(dim=1, keepdim=True).clamp_min(trainer.iw_opd_epsilon)
    weights = (1.0 + trainer.iw_opd_gamma * (1.0 - prefix_absolute / total_absolute)).detach()
    advantages = weights * advantages_base
    token_loss = torch.where(
        valid_mask,
        -student_actual_logprobs * advantages,
        torch.zeros_like(student_actual_logprobs),
    )

    with torch.no_grad():
        mode = "train" if model.training else "eval"
        valid_advantages = advantages[valid_mask]
        valid_absolute = absolute[valid_mask]
        valid_weights = weights[valid_mask]
        trainer._metrics[mode]["iw_opd/mean_advantage"].append(valid_advantages.mean().item())
        trainer._metrics[mode]["iw_opd/mean_abs_advantage"].append(valid_absolute.mean().item())
        trainer._metrics[mode]["iw_opd/mean_weight"].append(valid_weights.mean().item())
        trainer._metrics[mode]["iw_opd/min_weight"].append(valid_weights.min().item())
        trainer._metrics[mode]["iw_opd/max_weight"].append(valid_weights.max().item())

    denominator = valid_count if num_items_in_batch is None else num_items_in_batch
    if isinstance(denominator, torch.Tensor):
        denominator = denominator.to(token_loss.device)
        if not torch.isfinite(denominator).all() or (denominator <= 0).any():
            raise ValueError("IW-OPD accumulation denominator must be finite and positive")
    elif not isinstance(denominator, int | float) or denominator <= 0:
        raise ValueError("IW-OPD accumulation denominator must be positive")
    loss = token_loss.sum() / denominator
    return (loss, student_outputs) if return_outputs else loss


def _local_constrained_teacher_logprobs(
    trainer: Any,
    inputs: dict[str, Any],
    aligned_prompt_length: int,
) -> dict[str, Any]:
    """Score an exact completion in parallel with a frozen local teacher.

    The teacher receives its model-native prompt, while the exact student
    completion remains token-aligned.  Only hidden states are materialized for
    the sequence; the vocabulary projection is chunked and normalized over the
    same XGrammar-allowed set used by rollout and current-student scoring.
    """

    import torch

    teacher = trainer.accelerator.unwrap_model(trainer.teacher_model)
    prompts = inputs.get("teacher_prompt_ids")
    completions = inputs.get("teacher_completion_ids")
    offsets = inputs.get("teacher_completion_offsets")
    schemas = inputs.get("structured_output_schemas")
    schema_digests = inputs.get("schema_digests")
    expected_allowed_digests = inputs.get("allowed_set_digests")
    grammar_prefixes = inputs.get("grammar_prefix_ids")
    request_ids = inputs.get("constrained_request_ids")
    values = (
        prompts,
        completions,
        offsets,
        schemas,
        schema_digests,
        expected_allowed_digests,
        request_ids,
        grammar_prefixes,
    )
    if any(not isinstance(value, list) or len(value) != 1 for value in values):
        raise ValueError("local constrained teacher metadata must align with physical batch one")
    prompt_ids = prompts[0]
    completion_ids = completions[0]
    offset = offsets[0]
    grammar_prefix = grammar_prefixes[0]
    if (
        not isinstance(prompt_ids, list)
        or not isinstance(completion_ids, list)
        or not completion_ids
        or not isinstance(offset, int)
        or offset < 0
        or not isinstance(grammar_prefix, list)
    ):
        raise ValueError("local constrained teacher token alignment is invalid")

    device = next(teacher.parameters()).device
    sequence = torch.tensor([prompt_ids + completion_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(sequence)
    teacher.eval()
    with torch.no_grad():
        outputs = teacher.model(
            input_ids=sequence,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state[:, len(prompt_ids) - 1 : -1, :]
        if hidden.shape[1] != len(completion_ids):
            raise RuntimeError("local teacher hidden states do not align with the exact completion")
        head = teacher.get_output_embeddings()
        softcap = teacher.config.get_text_config().final_logit_softcapping
        matcher, xgrammar = _xgrammar_matcher(trainer.processing_class, schemas[0], head.weight.shape[0])
        for token_id in grammar_prefix:
            if not matcher.accept_token(int(token_id)):
                raise ValueError("teacher grammar prefix is disallowed by XGrammar")
        actual_chunks: list[Any] = []
        allowed_counts: list[int] = []
        observed_digests: list[str] = []
        chunk_size = 16
        for start in range(0, len(completion_ids), chunk_size):
            end = min(start + chunk_size, len(completion_ids))
            logits = head(hidden[:, start:end, :])
            if softcap is not None:
                logits = torch.tanh(logits / softcap) * softcap
            scaled = logits.float() / trainer.temperature
            bitmask = xgrammar.allocate_token_bitmask(end - start, scaled.shape[-1])
            for local_position, position in enumerate(range(start, end)):
                matcher.fill_next_token_bitmask(bitmask, local_position)
                selected = int(completion_ids[position])
                if not matcher.accept_token(selected):
                    raise ValueError(f"teacher completion token is disallowed by XGrammar at position {position}")
                digest = _allowed_set_digest(
                    schema_digests[0],
                    completion_ids,
                    position,
                    grammar_prefix_ids=grammar_prefix,
                )
                if digest != expected_allowed_digests[0][position]:
                    raise ValueError("local teacher allowed-token digest differs from rollout evidence")
                observed_digests.append(digest)
            xgrammar.apply_token_bitmask_inplace(
                scaled.reshape(end - start, -1),
                bitmask.to(scaled.device),
            )
            allowed_counts.extend(
                int(torch.isfinite(scaled[0, local_position]).sum().item())
                for local_position in range(end - start)
            )
            selected = scaled.gather(
                dim=-1,
                index=torch.tensor(
                    [completion_ids[start:end]], dtype=torch.long, device=scaled.device
                ).unsqueeze(-1),
            ).squeeze(-1)
            actual_chunks.append(selected - torch.logsumexp(scaled, dim=-1))
        actual_row = torch.cat(actual_chunks, dim=1)

    completion_mask = inputs["labels"][:, aligned_prompt_length:] != -100
    valid_positions = torch.nonzero(completion_mask[0], as_tuple=False).flatten()
    if valid_positions.numel() == 0:
        raise ValueError("local constrained teacher received no selected student tokens")
    student_completion_length = int(valid_positions[-1].item()) + 1
    if offset + len(completion_ids) > student_completion_length:
        raise ValueError("local teacher completion offset exceeds selected student completion")
    actual = torch.full(
        (1, student_completion_length),
        float("-inf"),
        dtype=torch.float32,
        device=inputs["input_ids"].device,
    )
    actual[0, offset : offset + len(completion_ids)] = actual_row[0].to(actual.device)
    return {
        "actual_logprobs": actual,
        "topk_logprobs": actual.unsqueeze(-1),
        "topk_token_ids": torch.tensor(
            [[[token_id] for token_id in completion_ids]],
            dtype=torch.long,
            device=actual.device,
        ),
        "allowed_counts": [allowed_counts],
        "allowed_set_digests": [observed_digests],
    }


def _xgrammar_matcher(tokenizer: Any, schema: dict[str, Any], vocab_size: int) -> tuple[Any, Any]:
    try:
        import xgrammar
    except ImportError as error:
        raise RuntimeError("generation-constrained IW-OPD requires xgrammar") from error
    info = xgrammar.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size)
    compiler = xgrammar.GrammarCompiler(info, max_threads=8, cache_enabled=True)
    # XGrammar's JSON-schema compiler makes object-property order observable in
    # the token FSM.  The colocated vLLM path emits required properties in the
    # schema's declared ``required`` order; alphabetically sorting the wire
    # schema here therefore constructs a different probability space.  Keep
    # both scorers on the exact generation ordering while leaving the canonical
    # validation schema and its digest unchanged.
    schema_text = json.dumps(_xgrammar_generation_schema(schema), separators=(",", ":"))
    context = compiler.compile_json_schema(schema_text, any_whitespace=False)
    return xgrammar.GrammarMatcher(context), xgrammar


def _xgrammar_generation_schema(value: Any) -> Any:
    """Order object properties exactly as vLLM's XGrammar generation FSM."""

    if isinstance(value, list):
        return [_xgrammar_generation_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {
        key: _xgrammar_generation_schema(item)
        for key, item in value.items()
        if key != "properties"
    }
    properties = value.get("properties")
    if isinstance(properties, dict):
        required = value.get("required")
        required_names = [item for item in required if isinstance(item, str)] if isinstance(required, list) else []
        ordered_names = [
            *[name for name in required_names if name in properties],
            *[name for name in properties if name not in required_names],
        ]
        result["properties"] = {
            name: _xgrammar_generation_schema(properties[name]) for name in ordered_names
        }
    return result


def _buffered_selected_token_count(trainer: Any) -> Any:
    import torch

    buffered = getattr(trainer, "_buffered_inputs", None)
    if not isinstance(buffered, list) or not buffered or any(item is None for item in buffered):
        raise RuntimeError("IW-OPD accumulation buffer is unavailable after rollout generation")
    counts = [item["labels"].ne(-100).sum() for item in buffered]
    total = torch.stack(counts).sum()
    if int(total.item()) <= 0:
        raise ValueError("IW-OPD accumulation window contains no selected tokens")
    return total


def _gemma_hidden_forward(unwrapped: Any, inputs: dict[str, Any]) -> Any:
    base = _gemma_base_model(unwrapped)
    return base.model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        use_cache=False,
        return_dict=True,
    )


def _gemma_base_model(model: Any) -> Any:
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    architecture = type(base).__name__
    if architecture != "Gemma4ForConditionalGeneration":
        raise TypeError(f"memory-safe IW-OPD requires Gemma4ForConditionalGeneration, got {architecture}")
    return base


def _sampling_number(request: OnPolicyDistillationRequest, key: str, default: float) -> float:
    value = request.rollout_inference.sampling.get(key)
    return float(value) if isinstance(value, (int, float)) else default


def _backend_number(request: OnPolicyDistillationRequest, key: str, default: float) -> float:
    value = request.training.backend_options.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{key} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{key} must be finite")
    return numeric


__all__ = ["run_distillation"]
