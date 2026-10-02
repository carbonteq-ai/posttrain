"""Admission for explicit native physical layouts and their numerical profiles."""

from __future__ import annotations

from typing import Any, Literal, cast

from ...precision import training_precision
from ...update_plan import ExecutionCapabilities
from ...update_records import InvalidPolicyUpdate
from .contracts import VerlLaunchManifest


def native_context_layout(manifest: VerlLaunchManifest) -> Literal["ragged", "dense-population"]:
    selected = manifest.payload.resolved_settings
    if selected is None or selected.settings.policy_updates is None:
        raise InvalidPolicyUpdate("native layout requires resolved policy update settings")
    layout = manifest.payload.training.backend_options.get("resolved_context_layout", "ragged")
    if layout not in ("ragged", "dense-population"):
        raise InvalidPolicyUpdate("native physical context layout has not been qualified")
    if layout == "ragged" and selected.settings.policy_updates.execution.records != 1:
        raise InvalidPolicyUpdate("native multi-record execution requires explicit dense-population layout")
    return cast(Literal["ragged", "dense-population"], layout)


def validate_native_execution_profile(
    manifest: VerlLaunchManifest, model_config: Any, engine_config: Any, *, actor_config: Any = None,
) -> None:
    """Check composed config before dispatch and effective config before allocation.

    The worker keeps its native one-record fallback; explicit resolved sizes
    override it per update. Layout selection never changes optimizer boundaries.
    Legacy ragged one-record execution retains its existing admission checks.
    """
    if native_context_layout(manifest) == "ragged":
        return
    update = manifest.payload.training.update
    actor = engine_config if actor_config is None else actor_config
    mixed = engine_config.get("mixed_precision") or {}
    expected = "bf16" if training_precision(manifest.payload.training.backend_options) == "bf16" else "fp16"
    param_names = {"bf16": ("bf16", "bfloat16"), "fp16": ("fp16", "float16")}
    if (update.kind != "lora" or update.dropout != 0
            or model_config.get("lora_rank") != update.rank or model_config.get("lora_alpha") != update.alpha
            or model_config.get("lora_dropout", 0.) != 0
            or model_config.get("use_remove_padding", False)
            or model_config.get("use_fused_kernels", False)
            or model_config.get("bitsandbytes", {}).get("enable", False)
            or model_config.get("override_config", {}).get("attn_implementation") != "sdpa"
            or actor.get("strategy") != "fsdp" or engine_config.get("strategy", "fsdp") != "fsdp"
            or actor.get("use_torch_compile", True) or engine_config.get("use_torch_compile", True)
            or actor.get("use_dynamic_bsz", False) or actor.get("use_fused_kernels", False)
            or engine_config.get("ulysses_sequence_parallel_size", 1) != 1
            or engine_config.get("model_dtype") not in ("fp32", "float32")
            or engine_config.get("qat", {}).get("enable", False)
            or mixed.get("param_dtype") not in param_names[expected]
            or any(mixed.get(name) not in ("fp32", "float32") for name in ("reduce_dtype", "buffer_dtype"))
            or any(engine_config.get(name) is not True for name in (
                "use_orig_params", "full_determinism", "lora_fp32_compute", "lora_rowwise_compute",
                "contiguous_linear_output_gradients", "full_precision_matmul", "math_sdpa"))):
        raise InvalidPolicyUpdate("native dense-population layout requires the qualified explicit LoRA arithmetic profile")


def native_execution_capabilities(manifest: VerlLaunchManifest, engine: Any) -> ExecutionCapabilities:
    """Advertise the checked layout at the actual initialized native actor."""
    validate_native_execution_profile(manifest, engine.model_config, engine.engine_config)
    selected = manifest.payload.resolved_settings
    assert selected is not None
    settings = selected.settings
    return ExecutionCapabilities(("grpo@1", "dapo@1", "sampo@1", "sampo-turns@1", "sampo-spans@1", "gdpo@1", "capo@1"),
        ("sampled-logp", "old-logp", "reference-logp"),
        settings.max_prompt_length + settings.max_completion_length, True,
        context_layout=native_context_layout(manifest))
