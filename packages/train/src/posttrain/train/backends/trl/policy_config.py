"""Configuration translation for the TRL policy-optimization runtime."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from posttrain.common import JsonValue

from ...online_rl import policy_sampling_from_binding
from ...profiles import CAPOSettings, GDPOSettings, GRPOSettings, SAMPOSettings
from ...requests import CAPORequest, GDPORequest, GRPORequest, SAMPORequest
from ...rollout_execution import RolloutExecutionConfig, validate_execution_config
from .common import trainer_arguments, vllm_rollout_options
from .policy_rollouts import technique as _technique


def _configure_torch_compile(engine: Mapping[str, object]) -> None:
    """Apply compile policy before importing torch or constructing vLLM."""

    value = engine.get("disable_torch_compile")
    if value is not None and not isinstance(value, bool):
        raise ValueError("TRL rollout disable_torch_compile must be a boolean")
    if value:
        # torch._dynamo reads this flag at import; setting it after importing
        # torch is too late for compile-decorated MTP helper modules.
        os.environ["TORCH_COMPILE_DISABLE"] = "1"


def _online_rl_arguments(
    request: GRPORequest | SAMPORequest | GDPORequest | CAPORequest,
    output_dir: Path,
    template_kwargs: dict[str, Any],
) -> dict[str, Any]:
    _rollout_execution_config(request)
    arguments = trainer_arguments(request.settings.loop, output_dir)
    arguments.pop("max_length")
    settings = request.settings
    if isinstance(settings, GDPOSettings | CAPOSettings):
        is_sampo = False
        advantage_scaling = "group"
        loss_type = "grpo"
        epsilon_high = settings.clip_epsilon_high
        importance_sampling_mode = "token_truncate"
        importance_sampling_clip_min = None
        # Structured rewards precompute advantages, but a decoupled vLLM
        # sampler still needs bounded token-level correction against the actor.
        # Retain TRL's native upper bound instead of selecting an invalid
        # truncation mode with both bounds disabled.
        importance_sampling_clip_max = 3.0
    elif isinstance(settings, SAMPOSettings):
        is_sampo = True
        advantage_scaling = "group"
        loss_type = "grpo"
        epsilon_high = settings.clip_epsilon_high
        importance_sampling_mode = "sequence_truncate"
        importance_sampling_clip_min = 0.1
        importance_sampling_clip_max = 3.0
    else:
        is_sampo = False
        advantage_scaling = settings.advantage_scaling
        loss_type = settings.algorithm
        epsilon_high = settings.resolved_clip_epsilon_high
        importance_sampling_mode = settings.importance_sampling_mode
        importance_sampling_clip_min = settings.importance_sampling_clip_min
        importance_sampling_clip_max = settings.importance_sampling_clip_max
    is_olmo3 = isinstance(settings, GRPOSettings) and settings.algorithm == "olmo3"
    if is_olmo3 and request.inference.backend.split("@", 1)[0] != "vllm":
        raise ValueError("the OLMo 3 GRPO recipe requires a vLLM rollout binding")
    use_liger_kernel = request.training.backend_options.get("use_liger_kernel", False)
    if not isinstance(use_liger_kernel, bool):
        raise ValueError("TRL GRPO use_liger_kernel must be a boolean")
    if use_liger_kernel and isinstance(settings, GDPOSettings | CAPOSettings):
        raise ValueError("GDPO/CAPO Liger loss is not qualified")
    liger_loss_compiled = request.training.backend_options.get("liger_loss_compiled", True)
    if not isinstance(liger_loss_compiled, bool):
        raise ValueError("TRL GRPO liger_loss_compiled must be a boolean")
    if not use_liger_kernel and "liger_loss_compiled" in request.training.backend_options:
        raise ValueError("TRL GRPO liger_loss_compiled requires use_liger_kernel=true")
    logits_chunk_size = request.training.backend_options.get("logits_chunk_size")
    if logits_chunk_size is not None and (
        isinstance(logits_chunk_size, bool) or not isinstance(logits_chunk_size, int) or logits_chunk_size < 1
    ):
        raise ValueError("TRL GRPO logits_chunk_size must be a positive integer")
    parity_limit = request.training.backend_options.get("vllm_policy_parity_max_mean_logp_delta")
    if parity_limit is not None and (
        isinstance(parity_limit, bool)
        or not isinstance(parity_limit, int | float)
        or not math.isfinite(parity_limit)
        or parity_limit <= 0
    ):
        raise ValueError("TRL GRPO vLLM policy parity limit must be a finite positive number")
    parity_sequence_limit = request.training.backend_options.get("vllm_policy_parity_max_sequence_tokens")
    if parity_sequence_limit is not None and (
        isinstance(parity_sequence_limit, bool)
        or not isinstance(parity_sequence_limit, int)
        or parity_sequence_limit < 2
    ):
        raise ValueError("TRL GRPO vLLM policy parity sequence limit must be an integer greater than one")
    sampling = policy_sampling_from_binding(request.inference, request.settings.max_completion_length)
    if isinstance(settings, GDPOSettings):
        arguments["reward_weights"] = list(settings.component_weights)
        arguments["multi_objective_aggregation"] = "normalize_then_sum"
    arguments.update(
        {
            "remove_unused_columns": False,
            # Prompt order is part of a reproducible RL population. Historic
            # selections preserve their fixed order; new campaigns can opt
            # into a seed-recorded permutation for each data epoch.
            "shuffle_dataset": settings.shuffle_prompts,
            "num_generations": request.settings.num_generations,
            "generation_batch_size": (request.settings.num_prompts_per_step * request.settings.num_generations),
            "max_completion_length": request.settings.max_completion_length,
            "chat_template_kwargs": template_kwargs,
            "beta": request.settings.beta,
            # Pin the existing sampled k3 KL gradient. Upstream 1.10+ changes
            # this default; adopting corrected KL is a separate recipe change.
            "use_bias_correction_kl": False,
            "loss_type": loss_type,
            "epsilon": request.settings.clip_epsilon_low,
            "epsilon_high": epsilon_high,
            "scale_rewards": advantage_scaling,
            "dynamic_sampling": (request.settings.dynamic_sampling is not None if not is_sampo else True),
            "dynamic_sampling_max_batches": (
                request.settings.dynamic_sampling.max_candidate_batches
                if request.settings.dynamic_sampling is not None
                else 10
            ),
            "dynamic_sampling_reward_std_epsilon": (0.0),
            "mask_truncated_completions": request.settings.mask_truncated_completions,
            "importance_sampling_level": "sequence" if is_sampo else "token",
            "use_precomputed_advantages": is_sampo or isinstance(settings, GDPOSettings | CAPOSettings),
            "use_liger_kernel": use_liger_kernel,
            "logits_chunk_size": logits_chunk_size,
            "use_vllm": request.inference.backend.split("@", 1)[0] == "vllm",
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "top_k": sampling.top_k,
            "min_p": sampling.min_p,
            "repetition_penalty": sampling.repetition_penalty,
        }
    )
    if sampling.presence_penalty:
        if request.inference.backend.split("@", 1)[0] != "vllm":
            raise ValueError("TRL presence_penalty requires a vLLM rollout binding")
        arguments["generation_kwargs"] = {"presence_penalty": sampling.presence_penalty}
    if is_olmo3:
        # Olmo3GRPOConfig owns these recipe-defining fields as init=False
        # invariants. Posttrain only supplies workload and capacity controls.
        for name in (
            "beta",
            "loss_type",
            "epsilon",
            "epsilon_high",
            "scale_rewards",
            "dynamic_sampling",
            "dynamic_sampling_max_batches",
            "dynamic_sampling_reward_std_epsilon",
            "importance_sampling_level",
            "use_vllm",
        ):
            arguments.pop(name)
        olmo3_settings = cast(GRPOSettings, settings)
        assert olmo3_settings.active_sampling is not None
        arguments["active_sampling_max_batches"] = olmo3_settings.active_sampling.max_candidate_batches
    if request.inference.backend.split("@", 1)[0] == "vllm":
        rollout = request.inference.engine
        speculative_config, engine_kwargs = vllm_rollout_options(request.policy, rollout)
        arguments.update(
            {
                "vllm_mode": rollout.get("mode"),
                "vllm_request_mode": rollout.get("request_mode", "batch"),
                "vllm_enable_sleep_mode": rollout.get("sleep_during_optimization", False),
                "vllm_gpu_memory_utilization": rollout.get("gpu_memory_utilization"),
                "vllm_tensor_parallel_size": rollout.get("tensor_parallel_size", 1),
                "vllm_max_model_length": rollout.get("max_model_len"),
                "vllm_speculative_config": speculative_config,
                "vllm_engine_kwargs": engine_kwargs,
                "vllm_weight_name_prefix": rollout.get("weight_name_prefix"),
                "vllm_weight_sync_mode": rollout.get("weight_sync_mode", "full"),
                "vllm_model_impl": "vllm",
                "vllm_importance_sampling_correction": True,
                "vllm_importance_sampling_mode": importance_sampling_mode,
                "vllm_importance_sampling_clip_min": importance_sampling_clip_min,
                "vllm_importance_sampling_clip_max": importance_sampling_clip_max,
            }
        )
        if parity_limit is not None:
            arguments["vllm_policy_parity_max_mean_logp_delta"] = float(parity_limit)
        if parity_sequence_limit is not None:
            arguments["vllm_policy_parity_max_sequence_tokens"] = parity_sequence_limit
        if is_olmo3:
            for name in (
                "vllm_importance_sampling_correction",
                "vllm_importance_sampling_mode",
                "vllm_importance_sampling_clip_min",
                "vllm_importance_sampling_clip_max",
            ):
                arguments.pop(name)
    return arguments


def _rollout_execution_config(
    request: GRPORequest | SAMPORequest | GDPORequest | CAPORequest,
) -> RolloutExecutionConfig | None:
    """Validate the opt-in native TRL worker topology without changing direct mode."""

    raw = request.training.backend_options.get("rollout_execution")
    request_mode = request.inference.engine.get("request_mode", "batch")
    if request_mode not in {"batch", "async"}:
        raise ValueError("TRL inference engine request_mode must be either 'batch' or 'async'")
    if raw is None:
        if request_mode == "async":
            raise ValueError("TRL request_mode=async requires backend_options.rollout_execution")
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("TRL backend_options.rollout_execution must be a mapping")
    expected = {"env_workers", "episodes_per_worker", "worker_native_threads"}
    unknown = set(raw).difference(expected)
    missing = expected.difference(raw)
    if unknown or missing:
        details = []
        if missing:
            details.append(f"missing {', '.join(sorted(missing))}")
        if unknown:
            details.append(f"unknown {', '.join(sorted(unknown))}")
        raise ValueError(f"invalid TRL rollout_execution mapping: {'; '.join(details)}")
    values: dict[str, int] = {}
    for name in expected:
        value = raw[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"TRL rollout_execution.{name} must be an integer")
        values[name] = value
    if request.inference.backend.split("@", 1)[0] != "vllm":
        raise ValueError("TRL rollout_execution requires a vLLM rollout inference binding")
    if request.inference.engine.get("mode") != "colocate":
        raise ValueError("TRL rollout_execution currently requires colocated vLLM")
    if request_mode != "async":
        raise ValueError("TRL rollout_execution requires inference request_mode=async")
    if request.inference.engine.get("sleep_during_optimization") is not True:
        raise ValueError("TRL rollout_execution requires inference sleep_during_optimization=true")
    if request.training.update.kind not in {"lora", "qlora"}:
        raise ValueError("TRL asynchronous colocated rollout execution currently requires a LoRA update")
    devices_per_node = request.training.runtime.devices_per_node or 1
    if request.training.runtime.nodes * devices_per_node != 1:
        raise ValueError("TRL asynchronous colocated rollout execution currently requires one trainer process")
    global_limit = getattr(request.bridge, "max_concurrent", None)
    if not isinstance(global_limit, int) or isinstance(global_limit, bool):
        raise ValueError("TRL rollout_execution requires the environment bridge to declare max_concurrent")
    execution = RolloutExecutionConfig(**values)
    validate_execution_config(execution, global_limit=global_limit)
    return execution


def _configure_liger_loss(
    trainer: Any,
    request: GRPORequest | SAMPORequest | GDPORequest | CAPORequest,
) -> None:
    if not request.training.backend_options.get("use_liger_kernel", False):
        return
    compiled = request.training.backend_options.get("liger_loss_compiled", True)
    loss = getattr(trainer, "liger_loss", None)
    if loss is None or not hasattr(loss, "compiled"):
        raise RuntimeError("the selected TRL Liger loss does not expose compiled mode")
    loss.compiled = compiled


def _grpo_arguments(request: GRPORequest, output_dir: Path, template_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Compatibility name for the GRPO-only argument translator."""

    return _online_rl_arguments(request, output_dir, template_kwargs)


def _online_rl_runtime_attributes(
    request: GRPORequest | SAMPORequest | GDPORequest | CAPORequest,
) -> dict[str, JsonValue]:
    """Describe selected GRPO runtime features without claiming observed performance."""

    engine = request.inference.engine
    sampling = policy_sampling_from_binding(request.inference, request.settings.max_completion_length)
    speculative = engine.get("speculative_config")
    backend_product, separator, backend_revision = request.training.backend.partition("@")
    attributes: dict[str, JsonValue] = {
        "training_backend": backend_product,
        "backend_source_revision": backend_revision if separator else "unresolved",
        "training_binding_id": request.training.id,
        "inference_binding_id": request.inference.id,
        "inference_backend": request.inference.backend,
        "rollout_mode": engine.get("mode", "colocate"),
        "rollout_request_mode": engine.get("request_mode", "batch"),
        "rollout_sleep_during_optimization": engine.get("sleep_during_optimization", False),
        "rollout_gpu_memory_utilization": engine.get("gpu_memory_utilization"),
        "update_kind": request.training.update.kind,
        "world_size": request.training.target.placement.get("world_size", 1),
        "rollout_precision": engine.get("dtype", request.policy.weight_precision),
        "rollout_reasoning_mode": request.training.renderer.reasoning_mode,
        "rollout_temperature": sampling.temperature,
        "rollout_top_p": sampling.top_p,
        "rollout_top_k": sampling.top_k,
        "rollout_min_p": sampling.min_p,
        "rollout_repetition_penalty": sampling.repetition_penalty,
        "rollout_presence_penalty": sampling.presence_penalty,
        "kv_cache_dtype": engine.get("kv_cache_dtype", "auto"),
        "max_model_len": engine.get(
            "max_model_len", request.settings.max_prompt_length + request.settings.max_completion_length
        ),
        "use_liger_kernel": request.training.backend_options.get("use_liger_kernel", False),
        "liger_loss_compiled": request.training.backend_options.get("liger_loss_compiled", True),
        "logits_chunk_size": request.training.backend_options.get("logits_chunk_size"),
        "vllm_policy_parity_max_mean_logp_delta": request.training.backend_options.get(
            "vllm_policy_parity_max_mean_logp_delta"
        ),
        "vllm_policy_parity_max_sequence_tokens": request.training.backend_options.get(
            "vllm_policy_parity_max_sequence_tokens"
        ),
        "online_rl_algorithm": _technique(request),
        "advantage_scaling": (request.settings.advantage_scaling if isinstance(request, GRPORequest) else "group"),
        "clip_epsilon_low": request.settings.clip_epsilon_low,
        "clip_epsilon_high": (
            request.settings.resolved_clip_epsilon_high
            if isinstance(request, GRPORequest)
            else request.settings.clip_epsilon_high
        ),
        "mask_truncated_completions": request.settings.mask_truncated_completions,
        "shuffle_prompts": request.settings.shuffle_prompts,
        "dynamic_sampling": request.settings.dynamic_sampling is not None,
        "dynamic_sampling_max_candidate_batches": (
            request.settings.dynamic_sampling.max_candidate_batches
            if request.settings.dynamic_sampling is not None
            else None
        ),
        "active_sampling": (isinstance(request, GRPORequest) and request.settings.active_sampling is not None),
        "active_sampling_max_candidate_batches": (
            request.settings.active_sampling.max_candidate_batches
            if isinstance(request, GRPORequest) and request.settings.active_sampling is not None
            else None
        ),
    }
    execution = _rollout_execution_config(request)
    if execution is not None:
        attributes.update(
            rollout_env_workers=execution.env_workers,
            rollout_episodes_per_worker=execution.episodes_per_worker,
            rollout_worker_native_threads=execution.worker_native_threads,
            rollout_global_concurrency=execution.episode_capacity,
        )
    if isinstance(request, GRPORequest):
        attributes["overlong_buffer_tokens"] = request.settings.overlong_buffer_tokens
        attributes["overlong_penalty_factor"] = request.settings.overlong_penalty_factor
    elif isinstance(request, SAMPORequest):
        attributes["discount_gamma"] = request.settings.discount_gamma
        attributes["step_advantage_weight"] = request.settings.step_advantage_weight
        attributes["advantage_normalization"] = request.settings.advantage_normalization
    if isinstance(speculative, Mapping):
        attributes["speculative_method"] = speculative.get("method")
        attributes["num_speculative_tokens"] = speculative.get("num_speculative_tokens")
    return attributes


def _grpo_runtime_attributes(request: GRPORequest) -> dict[str, JsonValue]:
    """Compatibility name for the GRPO-only runtime evidence translator."""

    return _online_rl_runtime_attributes(request)


__all__ = [
    "_configure_liger_loss",
    "_configure_torch_compile",
    "_grpo_arguments",
    "_grpo_runtime_attributes",
    "_online_rl_arguments",
    "_online_rl_runtime_attributes",
]
