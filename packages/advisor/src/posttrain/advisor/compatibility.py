"""Compatibility and qualified-optimization rules over a run's recorded selections.

Each rule names a setting, the runtime or hardware fact it conflicts with, and
the measurement or failure behind it. Evidence documents:

- ``docs/plan/agentic-workload-inference-optimization.md`` (agentic rollouts)
- ``docs/plan/vllm-inference-optimization-platform.md`` (attention, MTP, DSpark, TurboQuant)
- ``docs/architecture/vllm-inference-optimization.md`` (batch invariance costs and defects)
- ``docs/tooling/trl/README.md`` and ``docs/plan/trl-lora-policy-parity.md`` (TRL rollouts)
- ``docs/plan/trl-mtp-turboquant.md`` (Qwen3.5 MTP and TurboQuant quality)

Errors are for settings that silently break correctness or undo a measured
optimization above 10%; the binding can acknowledge them with a reason like
the performance rules. Warnings are measured but smaller or conditional costs.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from typing import Any

from posttrain.common import ConfigurationIssue
from posttrain.common.validation import ConfigurationSeverity

from .snapshot import (
    Seat,
    Snapshot,
    backend_name,
    environment_seat,
    inference_seats,
    integer,
    items,
    mapping,
    number,
    served_model,
    target,
    training_seat,
)

_ACKNOWLEDGE = "If this is deliberate, acknowledge {code} in the binding's performance_acknowledgements with the reason."
_GENERATION = frozenset({"rollout", "eval", "screen", "teacher-score", "smoke", "judge"})
_SM120 = frozenset({"RTXPRO4500", "RTXPRO6000"})
# The CarbonTeq fork release that keeps hybrid models' multi-turn prefixes reusable
# and carries the SM120 batch-invariance kernels.
_CONTINUATION_FIX = (0, 29, 1, 3)
_HYBRID_FAMILIES = frozenset({"lfm2.5", "qwen3.5", "gemma4"})
_INVARIANCE_UNSUPPORTED = frozenset({"olmo-hybrid", "bailing"})
_PINNED_DSPARK_BACKEND = "vllm@62f6de733d7ae63b759329993bc209e67afdf431"
_PARITY_LIMIT = 0.05
# Qualified draft models by served base checkpoint: (method, draft repo, evidence).
QUALIFIED_DRAFTERS: Mapping[str, tuple[tuple[str, str, str], ...]] = {
    "google/gemma-4-12B-it": (
        ("mtp", "google/gemma-4-12B-it-assistant", "MTP-2 raised Gemma 4 12B decode from 181.4 to 253.9 tok/s (+40%) at concurrency 4"),
        ("dspark", "deepseek-ai/dspark_gemma4_12b_block7", "a DSpark block drafter is qualified for Gemma 4 12B"),
    ),
    "google/gemma-4-31B-it": (("mtp", "google/gemma-4-31B-it-assistant", "a paired MTP assistant is published for this checkpoint"),),
    "google/gemma-4-E4B-it": (("mtp", "google/gemma-4-E4B-it-assistant", "a paired MTP assistant is published for this checkpoint"),),
    "google/gemma-4-E2B-it": (("mtp", "google/gemma-4-E2B-it-assistant", "a paired MTP assistant is published for this checkpoint"),),
    "Nanbeige/Nanbeige4.2-3B": (("dspark", "Nanbeige/Nanbeige4.2-3B-DSpark", "a DSpark drafter is qualified for Nanbeige 4.2 3B"),),
}
_NATIVE_MTP_EVIDENCE = "Qwen3.5 native MTP heads accepted 86% of drafts on the 0.8B rollout qualification"


def _issue(
    seat: Seat,
    code: str,
    severity: ConfigurationSeverity,
    path: str,
    message: str,
    hint: str,
    related: tuple[str, ...] = (),
) -> ConfigurationIssue:
    acknowledgements = mapping(seat.resolved.get("performance_acknowledgements"))
    reason = acknowledgements.get(code)
    if severity in {"error", "warning"} and isinstance(reason, str) and reason.strip():
        return ConfigurationIssue(code, "info", "static", seat.role, path, f"{message} Acknowledged by {seat.label}: {reason}", None, related)
    if severity == "error":
        hint = f"{hint} {_ACKNOWLEDGE.format(code=code)}"
    return ConfigurationIssue(code, severity, "static", seat.role, path, message, hint, related)


def backend_version(backend: object) -> tuple[int, ...] | None:
    """``vllm@0.29.1.dev3`` -> (0, 29, 1, 3); release versions sort after their dev builds."""

    if not isinstance(backend, str) or "@" not in backend:
        return None
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:\.dev(\d+))?(?:\+.*)?", backend.split("@", 1)[1].strip())
    if match is None:
        return None
    major, minor, patch, dev = match.groups()
    return (int(major), int(minor), int(patch), int(dev) if dev is not None else 10_000)


def _speculative(engine: Mapping[str, Any]) -> Mapping[str, Any] | None:
    value = engine.get("speculative_config", engine.get("speculative"))
    return value if isinstance(value, Mapping) else None


def compatibility_findings(snapshot: Snapshot) -> tuple[ConfigurationIssue, ...]:
    issues: list[ConfigurationIssue] = []
    for seat in inference_seats(snapshot):
        if backend_name(seat.resolved.get("backend")) != "vllm":
            continue
        issues.extend(_binding_findings(snapshot, seat))
    issues.extend(_training_findings(snapshot))
    return tuple(issues)


def _binding_findings(snapshot: Snapshot, seat: Seat) -> Iterator[ConfigurationIssue]:
    role = seat.role
    engine: Mapping[str, Any] = seat.resolved["engine"]
    purposes = set(items(seat.resolved.get("purpose")))
    generates = bool(purposes & _GENERATION)
    rollout = "rollout" in purposes
    model = served_model(snapshot, seat)
    family = model.get("family")
    capabilities = mapping(model.get("capabilities"))
    target_entry = mapping(target(snapshot, seat.resolved.get("target_id")))
    hardware = mapping(target_entry.get("hardware"))
    accelerator = hardware.get("accelerator_model")
    architecture = hardware.get("gpu_architecture")
    backend = seat.resolved.get("backend")
    version = backend_version(backend)
    speculative = _speculative(engine)
    method = speculative.get("method") if speculative is not None else None
    kv_cache_dtype = engine.get("kv_cache_dtype")
    turboquant = isinstance(kv_cache_dtype, str) and kv_cache_dtype.startswith("turboquant_")
    flash = integer(engine.get("flash_attn_version"))
    priority = [value for value in items(engine.get("attention_backend_priority")) if isinstance(value, str)]
    max_num_seqs = integer(engine.get("max_num_seqs"))
    max_model_len = integer(engine.get("max_model_len"))
    invariant = engine.get("batch_invariant") is True

    # Attention backends.
    if accelerator in _SM120 and flash == 4:
        yield _issue(
            seat, "VLLM_NATIVE_FA4_UNSUPPORTED_ON_SM120", "error", f"{role}.engine.flash_attn_version",
            f"native vLLM FlashAttention 4 does not support SM120 target {accelerator} (it rejects paged KV); the "
            "separately qualified extracted SM120 kernel is not the native FLASH_ATTN backend",
            "Set flash_attn_version to 2 or select a runtime with a qualified SM120 attention backend.",
            (f"{role}.target_id",),
        )
    elif flash == 3 and architecture not in {None, "hopper"}:
        yield _issue(
            seat, "FLASH_ATTN_VERSION_UNSUPPORTED_ARCH", "warning", f"{role}.engine.flash_attn_version",
            f"FlashAttention 3 is Hopper-only; on {architecture} vLLM falls back to FlashAttention 2 with a warning",
            "Set flash_attn_version to 2 so the recorded setting matches what runs.",
        )
    elif flash == 4 and architecture == "ampere":
        yield _issue(
            seat, "FLASH_ATTN_VERSION_UNSUPPORTED_ARCH", "warning", f"{role}.engine.flash_attn_version",
            "FlashAttention 4 does not support Ampere; vLLM falls back to FlashAttention 2 with a warning",
            "Set flash_attn_version to 2 so the recorded setting matches what runs.",
        )
    if turboquant and flash is not None and flash >= 3:
        yield _issue(
            seat, "VLLM_TURBOQUANT_FLASH_ATTN_INCOMPATIBLE", "error", f"{role}.engine.flash_attn_version",
            f"TurboQuant KV cache cannot be compiled with FlashAttention {flash}; the runtime requires FlashAttention 2 "
            "for TurboQuant boundary layers",
            "Set flash_attn_version to 2 or select a non-TurboQuant KV-cache dtype.",
            (f"{role}.engine.kv_cache_dtype",),
        )
    if family == "gemma4" and "SM120_FA4" in priority:
        yield _issue(
            seat, "VLLM_SM120_FA4_ON_GEMMA", "error", f"{role}.engine.attention_backend_priority",
            "the extracted SM120 FA4 kernel cannot hold Gemma 4's 512-wide global attention heads in shared memory; "
            "both attempted extensions produced about 8 tok/s and corrupted text",
            "Use TRITON_ATTN for Gemma 4, as the qualified Gemma bindings do.",
        )
    if len(priority) > 1:
        yield _issue(
            seat, "ATTENTION_BACKEND_PRIORITY_TRUNCATED", "warning", f"{role}.engine.attention_backend_priority",
            f"only the first attention backend ({priority[0]}) is compiled into the engine; {', '.join(priority[1:])} "
            "are never used as fallbacks",
            "List one backend.",
        )

    # Hybrid-model prefix reuse across agentic turns.
    if (
        generates
        and environment_seat(snapshot) is not None  # multi-turn episodes continue earlier turns
        and family in _HYBRID_FAMILIES
        and version is not None
        and version < _CONTINUATION_FIX
        and engine.get("enable_prefix_caching") is not False
    ):
        busy = max_num_seqs is None or max_num_seqs >= 32
        yield _issue(
            seat, "VLLM_HYBRID_CONTINUATION_REUSE_MISSING", "error" if busy else "warning", f"{role}.backend",
            f"{backend} predates the CarbonTeq continuation fix, so each multi-turn {family} episode re-prefills the "
            "previous turn's generated tokens: at 32 concurrent episodes the fix made collections 1.17x faster, "
            "prefilled 78% fewer tokens and cut recomputed reusable context from 49.4% to 0.1%",
            "Select a binding on vllm@0.29.1.dev3 or later.",
        )

    # Prefix caching left to each vLLM version's default.
    if generates and "enable_prefix_caching" not in engine:
        yield _issue(
            seat, "VLLM_PREFIX_CACHING_IMPLICIT", "warning", f"{role}.engine.enable_prefix_caching",
            "prefix caching is not set, so it depends on the path's default (the offline serving profile turns it "
            "off, the managed server and TRL follow the vLLM version); the LFM2.5 replay served 80-89% of prompt "
            "tokens from the cache",
            "Set enable_prefix_caching: true.",
        )

    # TurboQuant.
    if turboquant and hardware.get("supports_turboquant") is False:
        yield _issue(
            seat, "TURBOQUANT_TARGET_UNSUPPORTED", "error", f"{role}.engine.kv_cache_dtype",
            f"target {target_entry.get('selection_id')} declares TurboQuant unsupported",
            "Use the native KV-cache dtype on this target.",
        )
    long_qwen = family == "qwen3.5" and (max_model_len or 0) >= 8_192
    if turboquant and long_qwen:
        yield _issue(
            seat, "TURBOQUANT_QUALITY_UNQUALIFIED", "warning", f"{role}.engine.kv_cache_dtype",
            "TurboQuant K8V4 gives about 2.67x KV capacity but missed beginning-of-context recall on Qwen3.5 at 8K, "
            "16K, 24K and 32.7K contexts",
            "Use the native KV cache for long Qwen3.5 contexts, or qualify recall for this workload.",
        )
    if not turboquant and hardware.get("supports_turboquant") is True and generates and not long_qwen:
        yield _issue(
            seat, "TURBOQUANT_AVAILABLE", "recommendation", f"{role}.engine.kv_cache_dtype",
            "the target supports TurboQuant KV cache (about 2.67x KV capacity) but this binding uses the native dtype",
            "Enable TurboQuant only after qualifying the model/backend/operation combination.",
        )
    if (
        backend == _PINNED_DSPARK_BACKEND
        and family == "nanbeige4.2"
        and turboquant
        and method == "dspark"
    ):
        yield _issue(
            seat, "VLLM_DSPARK_TURBOQUANT_INCOMPATIBLE", "error", f"{role}.engine.speculative_config",
            "DSpark cannot be compiled with TurboQuant in vLLM 62f6de733: its non-causal draft attention is "
            "unsupported by the TurboQuant backend",
            "Use the native KV cache for DSpark.",
            (f"{role}.engine.kv_cache_dtype",),
        )

    # Precision and speculative decoding against declared capabilities.
    if model.get("weight_precision") == "bf16" and hardware.get("supports_bf16") is False:
        yield _issue(
            seat, "BF16_TARGET_UNSUPPORTED", "error", f"{role}.target_id",
            f"target {target_entry.get('selection_id')} declares BF16 unsupported for BF16 model weights",
            "Select a BF16-capable target.",
        )
    if method == "mtp" and capabilities.get("mtp") is False and speculative is not None and "draft_model" not in speculative:
        yield _issue(
            seat, "MTP_MODEL_CAPABILITY_MISSING", "error", f"{role}.engine.speculative_config",
            f"{seat.resolved.get('model_variant_id')} does not declare native MTP heads and no MTP draft model is set",
            "Select an MTP-capable variant, add its paired assistant as draft_model, or remove speculative decoding.",
        )
    if method == "mtp" and hardware.get("supports_mtp") is False:
        yield _issue(
            seat, "MTP_TARGET_UNSUPPORTED", "error", f"{role}.engine.speculative_config",
            f"target {target_entry.get('selection_id')} declares MTP unsupported",
            "Select an MTP-capable target or remove speculative decoding.",
        )
    if purposes & {"rollout", "eval", "screen", "judge", "smoke"} and speculative is None and hardware.get("supports_mtp") is not False:
        drafters = QUALIFIED_DRAFTERS.get(str(mapping(model.get("base")).get("repo_id")), ())
        if capabilities.get("mtp") is True and not drafters:
            drafters = (("mtp", "native MTP heads", _NATIVE_MTP_EVIDENCE),)
        if drafters:
            choices = "; ".join(f"{name} ({draft}): {evidence}" for name, draft, evidence in drafters)
            yield _issue(
                seat, "SPECULATIVE_DECODING_AVAILABLE", "warning", f"{role}.engine.speculative_config",
                f"this model has qualified speculative decoding that the binding does not use: {choices}"
                + ("; in RL rollouts acceptance falls as the policy drifts from the drafter" if rollout else ""),
                f"Add speculative_config with method {drafters[0][0]} and num_speculative_tokens 2"
                + (f" and draft_model {drafters[0][1]}" if drafters[0][1] != "native MTP heads" else "")
                + ", after checking acceptance at this concurrency.",
            )

    # Batch invariance.
    evaluates = bool(purposes & {"eval", "screen", "judge"}) and not rollout
    if evaluates and not invariant and max_num_seqs != 1:
        yield _issue(
            seat, "BATCH_INVARIANCE_OFF_FOR_EVALUATION", "warning", f"{role}.engine.batch_invariant",
            "batch invariance is off, so each request's logits depend on which other requests share its batch; "
            "repeated or compared evaluations are not reproducible even with fixed seeds. On the CarbonTeq fork "
            "invariance costs 1-7% decode throughput for Gemma 4 and LFM2.5 (about 10% for sub-1B models)",
            "Set engine batch_invariant: true on a vllm@0.29.1.dev3 or later binding for evaluations that are compared.",
        )
    if invariant and (version is None or version < _CONTINUATION_FIX):
        yield _issue(
            seat, "BATCH_INVARIANCE_UNTUNED_RUNTIME", "error", f"{role}.engine.batch_invariant",
            f"{backend} lacks the tuned SM120 batch-invariant kernels; untuned invariant kernels cost 46-73% "
            "decode throughput against 1-10% on the fork",
            "Use a vllm@0.29.1.dev3 or later binding.",
            (f"{role}.backend",),
        )
    if invariant and family in _INVARIANCE_UNSUPPORTED:
        yield _issue(
            seat, "BATCH_INVARIANCE_UNVALIDATED_ARCH", "error", f"{role}.engine.batch_invariant",
            f"{family} is gated out of invariant mode; its kernels are not batch-invariant",
            "Turn batch_invariant off for this model.",
        )
    elif invariant and family == "qwen3.5" and engine.get("enable_prefix_caching") is True:
        yield _issue(
            seat, "BATCH_INVARIANCE_UNVALIDATED_ARCH", "warning", f"{role}.engine.batch_invariant",
            "Qwen3.5 gated-delta-net layers with prefix caching are not validated as batch-invariant",
            "Verify reproducibility for this binding, or turn prefix caching off where bit-exact results matter.",
        )
    if invariant and (flash == 4 or "SM120_FA4" in priority):
        yield _issue(
            seat, "BATCH_INVARIANCE_WITH_SM120_FA4", "error", f"{role}.engine.batch_invariant",
            "the SM120 FA4 kernel ignores batch-invariant mode and its launch-plan cache omits the tile shape, so "
            "results still vary with batch composition",
            "Use FlashAttention 2 or Triton attention with batch_invariant.",
        )


def _training_findings(snapshot: Snapshot) -> Iterator[ConfigurationIssue]:
    training = training_seat(snapshot)
    if training is None:
        return
    backend = backend_name(training.resolved.get("backend"))
    options = mapping(training.resolved.get("backend_options"))
    update = mapping(training.resolved.get("parameter_update"))
    rollout = next((seat for seat in inference_seats(snapshot) if "rollout" in items(seat.resolved.get("purpose"))), None)
    engine: Mapping[str, Any] = rollout.resolved["engine"] if rollout is not None else {}
    role = rollout.role if rollout is not None else training.role
    colocated = engine.get("mode") == "colocate"
    family = served_model(snapshot, rollout).get("family") if rollout is not None else None

    def issue(code: str, severity: ConfigurationSeverity, path: str, message: str, hint: str) -> ConfigurationIssue:
        return _issue(rollout or training, code, severity, path, message, hint)

    if backend == "trl" and rollout is not None:
        if colocated and "dtype" in engine and not str(engine.get("kv_cache_dtype", "")).startswith("turboquant_"):
            yield issue(
                "TRL_ROLLOUT_DTYPE_NOT_APPLIED", "warning", f"{role}.engine.dtype",
                "the colocated TRL rollout engine does not forward dtype (it follows the checkpoint precision); the "
                "value is only reported as the rollout precision, so the record can misstate what ran",
                "Remove dtype from colocated TRL rollout bindings.",
            )
        if colocated and engine.get("sleep_during_optimization") is not True:
            yield issue(
                "TRL_COLOCATED_ROLLOUT_WITHOUT_SLEEP", "warning", f"{role}.engine.sleep_during_optimization",
                "the colocated rollout engine keeps its weights and KV cache resident during optimization, so the "
                "trainer and vLLM memory coexist",
                "Set sleep_during_optimization: true.",
            )
        execution = mapping(options.get("rollout_execution"))
        if execution:
            problems = []
            if engine.get("request_mode") != "async":
                problems.append("request_mode must be async")
            if not colocated:
                problems.append("mode must be colocate")
            if engine.get("sleep_during_optimization") is not True:
                problems.append("sleep_during_optimization must be true")
            workers, episodes = integer(execution.get("env_workers")), integer(execution.get("episodes_per_worker"))
            environment = environment_seat(snapshot)
            limit = integer(environment.resolved.get("max_concurrent")) if environment is not None else None
            if workers and episodes and limit is not None and limit < workers * episodes:
                problems.append(f"the environment's max_concurrent {limit} is below {workers} x {episodes} worker slots")
            if problems:
                yield issue(
                    "TRL_ROLLOUT_EXECUTION_INVALID", "error", f"{training.role}.backend_options.rollout_execution",
                    "native rollout workers are configured but " + "; ".join(problems) + "; the job fails when training starts",
                    "Fix the listed settings.",
                )
        elif colocated and environment_seat(snapshot) is not None and engine.get("request_mode") != "async":
            yield issue(
                "TRL_ROLLOUT_BATCH_REQUEST_MODE", "recommendation", f"{role}.engine.request_mode",
                "multi-turn tool episodes wait for the slowest episode of each batch; independent request admission "
                "was 2.49x faster at concurrency 32 than at 1",
                "Set request_mode: async with backend_options.rollout_execution worker slots.",
            )
        speculative = _speculative(engine)
        if speculative is not None and speculative.get("method") == "uno" and (
            update.get("kind") not in {"lora", "qlora"} or engine.get("weight_sync_mode") != "lora"
        ):
            yield issue(
                "TRL_UNO_REQUIRES_LORA_SYNC", "error", f"{role}.engine.weight_sync_mode",
                "Uno speculative rollouts need a LoRA policy update synced as LoRA weights; the job fails when training starts",
                "Use a LoRA update and weight_sync_mode: lora.",
            )
        job = mapping(snapshot.get("job_definition")).get("kind")
        if options.get("use_liger_kernel") is True and job in {"train.gdpo", "train.capo"}:
            yield issue(
                "TRL_LIGER_UNQUALIFIED_FOR_ALGORITHM", "error", f"{training.role}.backend_options.use_liger_kernel",
                f"the Liger loss is not qualified with {job}; the job fails when training starts",
                "Set use_liger_kernel: false.",
            )
        if update.get("kind") == "qlora" and engine.get("weight_sync_mode") != "lora":
            yield issue(
                "QLORA_REQUIRES_LORA_WEIGHT_SYNC", "error", f"{role}.engine.weight_sync_mode",
                "a QLoRA policy synced as merged dense weights treats packed 4-bit storage as dense tensors",
                "Set weight_sync_mode: lora.",
            )
        prefix = engine.get("weight_name_prefix")
        if engine.get("weight_sync_mode") == "lora" and (
            (family == "qwen3.5" and prefix != "language_model.") or (family == "lfm2.5" and prefix is not None)
        ):
            yield issue(
                "WEIGHT_NAME_PREFIX_FAMILY_MISMATCH", "error", f"{role}.engine.weight_name_prefix",
                f"the LoRA weight names for {family} need prefix {'language_model.' if family == 'qwen3.5' else 'none'}, "
                f"got {prefix!r}; the adapter would not attach, and the first-rollout parity check cannot see it because "
                "fresh LoRA B matrices are zero",
                "Set weight_name_prefix to language_model. for Qwen3.5 and omit it for LFM2.5.",
            )
        limit = number(options.get("vllm_policy_parity_max_mean_logp_delta"))
        if limit is not None and limit > _PARITY_LIMIT:
            yield ConfigurationIssue(
                "POLICY_PARITY_LIMIT_RELAXED", "warning", "static", training.role,
                f"{training.role}.backend_options.vllm_policy_parity_max_mean_logp_delta",
                f"the first-rollout policy parity limit is {limit:g}, above the default {_PARITY_LIMIT:g}; healthy runs "
                "measured 0.006-0.014 and a broken actor measured 0.253",
                f"Keep the limit at {_PARITY_LIMIT:g} unless a measured mismatch is understood.",
            )
    if backend == "verl" and rollout is not None:
        if "enforce_eager" not in engine:
            yield issue(
                "VERL_ROLLOUT_EAGER_BY_DEFAULT", "error", f"{role}.engine.enforce_eager",
                "the veRL worker defaults enforce_eager to true when the binding omits it, disabling CUDA graphs; "
                "eager decode was 2.32x slower on the LFM2.5 rollout replay",
                "Set enforce_eager: false explicitly.",
            )
        yield issue(
            "VERL_PREFIX_CACHING_FORCED_OFF", "warning", f"{role}.engine.enable_prefix_caching",
            "the veRL worker hard-codes enable_prefix_caching=False regardless of the binding, so grouped and "
            "multi-turn prompts re-prefill shared context",
            "Fix the veRL worker to honour the binding; until then this cost applies to every veRL rollout.",
        )
        if "max_num_seqs" not in engine:
            yield issue(
                "VERL_ROLLOUT_SEQS_DEFAULT_TO_GROUP", "warning", f"{role}.engine.max_num_seqs",
                "the veRL worker defaults max_num_seqs to num_generations, serializing a step's groups",
                "Set max_num_seqs to the step's rows (prompts x generations) or what fits.",
            )


__all__ = ["QUALIFIED_DRAFTERS", "backend_version", "compatibility_findings"]
