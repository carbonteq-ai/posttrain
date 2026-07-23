"""Environment-only GRPO composition for lab qualification scenarios."""

from __future__ import annotations

from dataclasses import replace
from typing import cast

from posttrain.common import InferenceBinding, ModelVariant, RunContext
from posttrain.eval import EnvironmentBinding
from posttrain.train import (
    GRPORequest,
    LoRAUpdate,
    QLoRAUpdate,
    TrainingResult,
    grpo,
    parameter_update_digest,
)

VerifiersGRPOJobRequest = GRPORequest


def run_grpo_materialized(
    context: RunContext,
    request: GRPORequest,
    *,
    input_name: str | None = None,
) -> TrainingResult:
    local_model = _materialized_model(context, request.policy, input_name)
    return grpo(
        context,
        GRPORequest(
            policy=local_model,
            bridge=request.bridge,
            settings=request.settings,
            environment=request.environment,
            training=request.training,
            inference=replace(request.inference, model=local_model),
            quantization=request.quantization,
            reference=request.reference,
            resume_from=request.resume_from,
        ),
    )


def grpo_job_inputs(request: GRPORequest) -> dict[str, str | int | float | bool]:
    environment = cast(EnvironmentBinding, request.environment)
    result: dict[str, str | int | float | bool] = {
        "model_variant_id": request.policy.id,
        "input_model_form": request.policy.form,
        "base_model_revision": request.policy.base.revision,
        "training_settings_id": request.settings.id,
        "training_binding_id": request.training.id,
        "training_target_id": request.training.target.id,
        "training_renderer_id": request.training.renderer.id,
        "reasoning_mode": request.training.renderer.reasoning_mode,
        "environment_id": environment.id,
        "environment_package": environment.source.package,
        "environment_revision": environment.revision,
        "environment_category": environment.category,
        "environment_num_tasks": environment.num_tasks,
        "environment_num_rollouts": environment.num_rollouts,
        "max_steps": request.settings.loop.max_steps,
        "learning_rate": request.settings.loop.learning_rate,
        "parameter_update_kind": request.training.update.kind,
        "parameter_update_digest": parameter_update_digest(request.training.update),
        "grpo_beta": request.settings.beta,
        "grpo_num_generations": request.settings.num_generations,
        "grpo_max_prompt_length": request.settings.max_prompt_length,
        "grpo_max_completion_length": request.settings.max_completion_length,
        "grpo_temperature": _float_setting(request.inference, "temperature", 1.0),
        "grpo_top_p": _float_setting(request.inference, "top_p", 1.0),
        "rollout_binding_id": request.inference.id,
        "rollout_target_id": request.inference.target.id,
        "rollout_engine": request.inference.backend.split("@", 1)[0],
        "rollout_sleep_during_optimization": bool(request.inference.engine.get("sleep_during_optimization", False)),
    }
    domains = environment.parameters.get("domains")
    if isinstance(domains, (list, tuple)):
        domain_names = tuple(item for item in domains if isinstance(item, str))
        if len(domain_names) == len(domains):
            result["environment_domains"] = ",".join(domain_names)
    for key in ("sampling_seed", "toolset", "search_top_k", "max_turns", "max_total_tokens"):
        value = environment.parameters.get(key)
        if isinstance(value, (str, int, float, bool)):
            result[f"environment_{key}"] = value
    if environment.reward_components:
        result["environment_reward_components"] = ",".join(environment.reward_components)
    if isinstance(request.training.update, LoRAUpdate | QLoRAUpdate):
        result["peft_rank"] = request.training.update.rank
        result["peft_alpha"] = request.training.update.alpha
        result["peft_target_modules"] = request.training.update.target_modules
    if isinstance(request.training.update, QLoRAUpdate):
        result["qlora_quant_type"] = request.training.update.quant_type
        result["qlora_compute_dtype"] = request.training.update.compute_dtype
        result["qlora_double_quant"] = request.training.update.double_quant
    if request.inference.backend.split("@", 1)[0] == "vllm":
        _add_vllm_inputs(result, request)
    return result


def _add_vllm_inputs(
    result: dict[str, str | int | float | bool],
    request: GRPORequest,
) -> None:
    rollout = request.inference.engine
    result["rollout_vllm_mode"] = str(rollout.get("mode", ""))
    result["rollout_vllm_gpu_memory_utilization"] = _number(rollout.get("gpu_memory_utilization"))
    result["rollout_vllm_tensor_parallel_size"] = int(_number(rollout.get("tensor_parallel_size"), 1))
    result["rollout_vllm_max_model_length"] = int(_number(rollout.get("max_model_len")))
    result["rollout_text_only"] = bool(rollout.get("text_only", False))
    result["rollout_vllm_weight_name_prefix"] = str(rollout.get("weight_name_prefix", ""))
    result["rollout_vllm_weight_sync_mode"] = str(rollout.get("weight_sync_mode", "full"))
    result["rollout_vllm_importance_sampling_mode"] = request.settings.importance_sampling_mode
    result["rollout_vllm_importance_sampling_clip_min"] = request.settings.importance_sampling_clip_min or 0.0
    result["rollout_vllm_importance_sampling_clip_max"] = request.settings.importance_sampling_clip_max or 0.0
    result["rollout_skip_multimodal_profiling"] = bool(rollout.get("skip_mm_profiling", False))
    result["rollout_kv_cache_memory_bytes"] = int(_number(rollout.get("kv_cache_memory_bytes")))
    speculative = rollout.get("speculative_config")
    if isinstance(speculative, dict):
        result["rollout_speculative_method"] = str(speculative.get("method", ""))
        result["rollout_num_speculative_tokens"] = int(_number(speculative.get("num_speculative_tokens")))


def _materialized_model(
    context: RunContext,
    model: ModelVariant,
    input_name: str | None,
) -> ModelVariant:
    if model.form not in {"adapter", "peft-adapter"}:
        return model
    return replace(model, artifact=context.input_artifact(input_name or "model_adapter"))


def _number(value: object, default: float = 0.0) -> float:
    return float(value) if isinstance(value, (int, float)) else default


def _float_setting(binding: InferenceBinding, key: str, default: float) -> float:
    return _number(binding.sampling.get(key), default)


__all__ = ["VerifiersGRPOJobRequest", "grpo_job_inputs", "run_grpo_materialized"]
