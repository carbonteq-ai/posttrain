"""Compatibility, invariance and qualified-optimization rules on recorded snapshots."""

from typing import Any

from posttrain.advisor import rule_findings
from posttrain.advisor.compatibility import backend_version


def _served(family: str, repo: str, *, mtp: bool | None = None) -> dict[str, Any]:
    facts: dict[str, Any] = {"family": family, "weight_precision": "bf16", "base": {"repo_id": repo, "revision": "r"}}
    if mtp is not None:
        facts["capabilities"] = {"mtp": mtp, "native_context_window": 32_768}
    return facts


def _seat(snap, engine: dict[str, Any], *, purpose=("eval",), backend="vllm@0.29.1.dev3", model=None) -> dict[str, Any]:  # noqa: ANN001
    seat = snap.rollout(engine, purpose=purpose, backend=backend)
    if model is not None:
        seat["resolved"]["model"] = model
    return seat


def _codes(snapshot: dict[str, Any]) -> dict[str, str]:
    return {issue.code: issue.severity for issue in rule_findings(snapshot)}


def test_backend_versions_order_dev_builds_before_releases() -> None:
    assert backend_version("vllm@0.29.1.dev3") == (0, 29, 1, 3)
    assert backend_version("vllm@0.25.1") < backend_version("vllm@0.29.1.dev3")  # type: ignore[operator]
    assert backend_version("vllm@62f6de733d7ae63b759329993bc209e67afdf431") is None


def test_old_runtime_misses_hybrid_prefix_reuse_for_multi_turn_episodes(snap) -> None:
    lfm = _served("lfm2.5", "LiquidAI/LFM2.5-2.6B")
    seat = _seat(
        snap,
        {"enable_prefix_caching": True, "max_num_seqs": 32, "batch_invariant": True},
        backend="vllm@0.25.1",
        model=lfm,
    )
    codes = _codes({"environment": snap.environment(), "evaluation_inference": seat})
    assert codes["VLLM_HYBRID_CONTINUATION_REUSE_MISSING"] == "error"
    # Single-turn jobs (no environment) do not continue earlier turns.
    assert "VLLM_HYBRID_CONTINUATION_REUSE_MISSING" not in _codes({"evaluation_inference": seat})


def test_evaluations_without_batch_invariance_warn_and_invariance_needs_the_tuned_runtime(snap) -> None:
    base = {"enable_prefix_caching": True, "max_num_seqs": 20}
    assert _codes({"evaluation_inference": _seat(snap, base)})["BATCH_INVARIANCE_OFF_FOR_EVALUATION"] == "warning"
    invariant = {**base, "batch_invariant": True}
    assert "BATCH_INVARIANCE_OFF_FOR_EVALUATION" not in _codes({"evaluation_inference": _seat(snap, invariant)})
    assert (
        _codes({"evaluation_inference": _seat(snap, invariant, backend="vllm@0.25.1")})[
            "BATCH_INVARIANCE_UNTUNED_RUNTIME"
        ]
        == "error"
    )
    fa4 = _codes({"evaluation_inference": _seat(snap, {**invariant, "attention_backend_priority": ["SM120_FA4"]})})
    assert fa4["BATCH_INVARIANCE_WITH_SM120_FA4"] == "error"


def test_models_with_qualified_drafters_warn_when_speculation_is_off(snap) -> None:
    gemma = _served("gemma4", "google/gemma-4-12B-it")
    engine = {"enable_prefix_caching": True, "batch_invariant": True}
    (issue,) = [
        issue
        for issue in rule_findings({"judge": _seat(snap, engine, purpose=("judge",), model=gemma)})
        if issue.code == "SPECULATIVE_DECODING_AVAILABLE"
    ]
    assert issue.severity == "warning" and "+40%" in issue.message and "gemma-4-12B-it-assistant" in (issue.hint or "")
    qwen = _served("qwen3.5", "Qwen/Qwen3.5-2B", mtp=True)
    assert (
        _codes({"judge": _seat(snap, engine, purpose=("judge",), model=qwen)})["SPECULATIVE_DECODING_AVAILABLE"]
        == "warning"
    )
    with_mtp = {**engine, "speculative_config": {"method": "mtp", "num_speculative_tokens": 2}}
    assert "SPECULATIVE_DECODING_AVAILABLE" not in _codes(
        {"judge": _seat(snap, with_mtp, purpose=("judge",), model=qwen)}
    )
    lfm = _served("lfm2.5", "LiquidAI/LFM2.5-2.6B", mtp=False)
    assert "SPECULATIVE_DECODING_AVAILABLE" not in _codes({"judge": _seat(snap, engine, purpose=("judge",), model=lfm)})


def test_attention_backend_rules(snap) -> None:
    gemma = _served("gemma4", "google/gemma-4-12B-it")
    engine = {
        "enable_prefix_caching": True,
        "batch_invariant": True,
        "speculative_config": {"method": "mtp"},
        "attention_backend_priority": ["SM120_FA4", "TRITON_ATTN"],
    }
    codes = _codes({"judge": _seat(snap, engine, purpose=("judge",), model=gemma)})
    assert codes["VLLM_SM120_FA4_ON_GEMMA"] == "error"
    assert codes["ATTENTION_BACKEND_PRIORITY_TRUNCATED"] == "warning"
    fa4 = {
        "enable_prefix_caching": True,
        "flash_attn_version": 4,
        "kv_cache_dtype": "turboquant_k8v4",
        "batch_invariant": True,
    }
    codes = _codes({"execution_targets": snap.targets(), "judge": _seat(snap, fa4, purpose=("judge",))})
    assert codes["VLLM_NATIVE_FA4_UNSUPPORTED_ON_SM120"] == "error"
    assert codes["VLLM_TURBOQUANT_FLASH_ATTN_INCOMPATIBLE"] == "error"


def test_trl_rollout_rules(snap) -> None:
    engine = {
        **snap.TUNED,
        "request_mode": "batch",
        "dtype": "bfloat16",
        "weight_sync_mode": "lora",
        "weight_name_prefix": "model.",
    }
    rollout = _seat(snap, engine, purpose=("rollout",), model=_served("lfm2.5", "LiquidAI/LFM2.5-2.6B"))
    training = snap.training(kind="qlora")
    training["resolved"]["backend_options"] = {
        "rollout_execution": {"env_workers": 5, "episodes_per_worker": 8, "worker_native_threads": 1},
        "vllm_policy_parity_max_mean_logp_delta": 0.2,
    }
    codes = _codes(
        {"training": training, "environment": snap.environment(max_concurrent=32), "rollout_inference": rollout}
    )
    assert codes["TRL_ROLLOUT_EXECUTION_INVALID"] == "error"
    # The colocated TRL rollout forwards dtype to vLLM, so an explicit dtype is no longer ignored.
    assert "TRL_ROLLOUT_DTYPE_NOT_APPLIED" not in codes
    assert codes["WEIGHT_NAME_PREFIX_FAMILY_MISMATCH"] == "error"
    assert codes["POLICY_PARITY_LIMIT_RELAXED"] == "warning"


def test_verl_overrides_are_reported(snap) -> None:
    training = snap.training()
    training["resolved"]["backend"] = "verl@0.6"
    rollout = _seat(snap, {}, purpose=("rollout",))
    codes = _codes({"training": training, "rollout_inference": rollout})
    assert codes["VERL_ROLLOUT_EAGER_BY_DEFAULT"] == "error"
    assert codes["VERL_PREFIX_CACHING_OFF_BY_DEFAULT"] == "warning"
    assert codes["VERL_ROLLOUT_SEQS_DEFAULT_TO_GROUP"] == "warning"
    # The worker honours an explicit choice, so only an omitted key is reported.
    explicit = _seat(snap, {"enable_prefix_caching": True}, purpose=("rollout",))
    assert "VERL_PREFIX_CACHING_OFF_BY_DEFAULT" not in _codes({"training": training, "rollout_inference": explicit})


def test_acknowledged_compatibility_findings_become_information(snap) -> None:
    seat = _seat(snap, {"enable_prefix_caching": True, "max_num_seqs": 20})
    seat["resolved"]["performance_acknowledgements"] = {
        "BATCH_INVARIANCE_OFF_FOR_EVALUATION": "scores are compared only within one run"
    }
    assert _codes({"evaluation_inference": seat})["BATCH_INVARIANCE_OFF_FOR_EVALUATION"] == "info"


def test_float32_is_rejected_for_gated_deltanet_models(snap) -> None:
    qwen = _served("qwen3.5", "Qwen/Qwen3.5-0.8B")
    for dtype in ("float32", "float"):
        codes = _codes(
            {"rollout_inference": _seat(snap, {**snap.TUNED, "dtype": dtype}, purpose=("rollout",), model=qwen)}
        )
        assert codes["VLLM_FLOAT32_UNSUPPORTED_FOR_GATED_DELTANET"] == "error"
    # Float16 and bfloat16 run, and LFM2.5 (short convolutions, no Gated-DeltaNet) accepts float32.
    for dtype in ("float16", "bfloat16"):
        codes = _codes(
            {"rollout_inference": _seat(snap, {**snap.TUNED, "dtype": dtype}, purpose=("rollout",), model=qwen)}
        )
        assert "VLLM_FLOAT32_UNSUPPORTED_FOR_GATED_DELTANET" not in codes
    lfm = _served("lfm2.5", "LiquidAI/LFM2.5-1.2B-Thinking")
    codes = _codes(
        {"rollout_inference": _seat(snap, {**snap.TUNED, "dtype": "float32"}, purpose=("rollout",), model=lfm)}
    )
    assert "VLLM_FLOAT32_UNSUPPORTED_FOR_GATED_DELTANET" not in codes


def test_trl_training_precision_rules(snap) -> None:
    lfm = _served("lfm2.5", "LiquidAI/LFM2.5-1.2B-Thinking")
    training = snap.training()
    training["resolved"]["backend_options"] = {"training_precision": "fp16"}
    grpo = {"kind": "train.grpo"}

    def codes_for(engine: dict[str, Any], job: dict[str, str]) -> dict[str, str]:
        rollout = _seat(snap, {**snap.TUNED, **engine}, purpose=("rollout",), model=lfm)
        return _codes({"job_definition": job, "training": training, "rollout_inference": rollout})

    # An fp16 trainer with a bf16 (or checkpoint-dtype) sampler keeps the sampler's rounding.
    assert codes_for({}, grpo)["TRL_FP16_TRAINER_WITH_NON_FP16_ROLLOUT"] == "warning"
    assert codes_for({"dtype": "bfloat16"}, grpo)["TRL_FP16_TRAINER_WITH_NON_FP16_ROLLOUT"] == "warning"
    unified = codes_for({"dtype": "float16"}, grpo)
    assert "TRL_FP16_TRAINER_WITH_NON_FP16_ROLLOUT" not in unified
    assert "TRL_PRECISION_UNQUALIFIED_FOR_JOB" not in unified
    # Only online RL implements the precision options.
    assert codes_for({"dtype": "float16"}, {"kind": "train.sft"})["TRL_PRECISION_UNQUALIFIED_FOR_JOB"] == "error"
    training["resolved"]["backend_options"] = {"logits_float32": True}
    assert codes_for({}, {"kind": "train.dpo"})["TRL_PRECISION_UNQUALIFIED_FOR_JOB"] == "error"
    training["resolved"]["backend_options"] = {}
    assert "TRL_PRECISION_UNQUALIFIED_FOR_JOB" not in codes_for({}, {"kind": "train.sft"})


def test_float32_rejects_half_only_attention_and_lora_kernels(snap) -> None:
    lfm = _served("lfm2.5", "LiquidAI/LFM2.5-2.6B")

    def codes(engine: dict[str, Any]) -> dict[str, str]:
        return _codes({"rollout_inference": _seat(snap, {**snap.TUNED, **engine}, purpose=("rollout",), model=lfm)})

    assert codes({"dtype": "float32", "flash_attn_version": 2})["VLLM_FLOAT32_KERNEL_UNSUPPORTED"] == "error"
    assert (
        codes({"dtype": "float32", "attention_backend_priority": ["SM120_FA4"]})["VLLM_FLOAT32_KERNEL_UNSUPPORTED"]
        == "error"
    )
    assert codes({"dtype": "float32", "weight_sync_mode": "lora"})["VLLM_FLOAT32_KERNEL_UNSUPPORTED"] == "error"
    # A Triton-attention engine without LoRA can compute a float32 reference.
    assert "VLLM_FLOAT32_KERNEL_UNSUPPORTED" not in codes({"dtype": "float32"})
    assert "VLLM_FLOAT32_KERNEL_UNSUPPORTED" not in codes({"dtype": "bfloat16", "flash_attn_version": 2})


def test_float16_reports_bf16_only_kernels(snap) -> None:
    lfm = _served("lfm2.5", "LiquidAI/LFM2.5-2.6B")
    qwen = _served("qwen3.5", "Qwen/Qwen3.5-0.8B")
    invariant = {**snap.TUNED, "dtype": "float16", "batch_invariant": True, "max_num_seqs": 16}
    seat = _seat(snap, invariant, purpose=("rollout",), model=lfm)
    on_sm120 = _codes({"execution_targets": snap.targets(), "rollout_inference": seat})
    assert on_sm120["VLLM_FLOAT16_BF16_ONLY_KERNEL"] == "warning"
    # Without batch invariance LFM2.5 keeps every kernel it uses in bfloat16 or float16.
    plain = _seat(snap, {**snap.TUNED, "dtype": "float16"}, purpose=("rollout",), model=lfm)
    assert "VLLM_FLOAT16_BF16_ONLY_KERNEL" not in _codes(
        {"execution_targets": snap.targets(), "rollout_inference": plain}
    )
    gdn = _seat(snap, {**snap.TUNED, "dtype": "float16"}, purpose=("rollout",), model=qwen)
    assert _codes({"rollout_inference": gdn})["VLLM_FLOAT16_BF16_ONLY_KERNEL"] == "warning"
    bf16 = _seat(snap, {**invariant, "dtype": "bfloat16"}, purpose=("rollout",), model=qwen)
    assert "VLLM_FLOAT16_BF16_ONLY_KERNEL" not in _codes(
        {"execution_targets": snap.targets(), "rollout_inference": bf16}
    )
