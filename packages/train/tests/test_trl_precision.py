"""Trainer and rollout precision selections (training_precision, logits_float32, engine.dtype)."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from posttrain.common.variants import QWEN_35_2B
from posttrain.train import FullParameterUpdate, LoRAUpdate, QLoRAUpdate, TrainingLoop
from posttrain.train.backends.trl.common import load_trainable_model, trainer_arguments, vllm_rollout_options
from posttrain.train.backends.trl.policy_telemetry import SamplerGapAccumulator
from posttrain.train.backends.trl.precision_runtime import (
    LossScaleMonitor,
    require_default_precision,
    require_float32_trainable_parameters,
    upcast_logits_to_float32,
)
from posttrain.train.bindings import _validate_precision
from posttrain.train.precision import resolve_precision, rollout_dtype


def test_defaults_resolve_to_the_existing_bf16_behaviour() -> None:
    resolved = resolve_precision({}, {}, "bf16")
    assert resolved.as_dict() == {
        "training_precision": "bf16",
        "model_load_dtype": "bfloat16",
        "loss_scaling": "none",
        "logits_float32": False,
        "rollout_dtype": "bfloat16",
        "rollout_dtype_source": "checkpoint",
    }
    arguments = trainer_arguments(TrainingLoop(max_steps=2), Path("out"))
    assert (arguments["bf16"], arguments["fp16"]) == (True, False)
    _speculative, engine_kwargs = vllm_rollout_options(QWEN_35_2B, {"max_num_seqs": 4})
    assert engine_kwargs == {"max_num_seqs": 4}


def test_unified_fp16_resolves_trainer_scaling_and_rollout_dtype() -> None:
    resolved = resolve_precision({"training_precision": "fp16"}, {"dtype": "float16"}, "bf16")
    assert resolved.model_load_dtype == "float16"
    assert resolved.loss_scaling == "dynamic"
    assert (resolved.rollout_dtype, resolved.rollout_dtype_source) == ("float16", "binding")
    assert resolved.summary() == (
        "trainer fp16 (base weights float16, dynamic loss scaling; log-probs from float16 logits); "
        "rollout vLLM float16 (binding)"
    )
    arguments = trainer_arguments(TrainingLoop(max_steps=2), Path("out"), precision="fp16")
    assert (arguments["bf16"], arguments["fp16"]) == (False, True)
    with pytest.raises(ValueError, match="bf16.*fp16"):
        trainer_arguments(TrainingLoop(max_steps=2), Path("out"), precision=cast(Any, "fp32"))


@pytest.mark.parametrize("dtype", ["bfloat16", "float16", "float32"])
def test_rollout_engine_dtype_is_forwarded_to_vllm(dtype: str) -> None:
    _speculative, engine_kwargs = vllm_rollout_options(QWEN_35_2B, {"dtype": dtype})
    assert engine_kwargs == {"dtype": dtype}


def test_rollout_dtype_keeps_the_turboquant_float16_requirement() -> None:
    assert rollout_dtype({"kv_cache_dtype": "turboquant_k8v4"}) == ("float16", "turboquant")
    assert rollout_dtype({"kv_cache_dtype": "turboquant_k8v4", "dtype": "float16"}) == ("float16", "binding")
    with pytest.raises(ValueError, match="TurboQuant KV cache requires"):
        vllm_rollout_options(QWEN_35_2B, {"kv_cache_dtype": "turboquant_k8v4", "dtype": "bfloat16"})
    with pytest.raises(ValueError, match="bfloat16, float16, float32"):
        rollout_dtype({"dtype": "half"})
    assert rollout_dtype({"kv_cache_dtype": "fp8"}) == (None, "checkpoint")


def test_training_binding_rejects_precision_no_backend_implements() -> None:
    _validate_precision("trl@1.12.0.post10", LoRAUpdate(), {"training_precision": "fp16", "logits_float32": True})
    _validate_precision("verl@0.6", FullParameterUpdate(), {"training_precision": "bf16"})
    with pytest.raises(ValueError, match="requires a LoRA update"):
        _validate_precision("trl@1.12.0.post10", FullParameterUpdate(), {"training_precision": "fp16"})
    with pytest.raises(ValueError, match="requires a LoRA update"):
        _validate_precision("trl@1.12.0.post10", QLoRAUpdate(), {"training_precision": "fp16"})
    with pytest.raises(ValueError, match="TRL backend only"):
        _validate_precision("verl@0.6", LoRAUpdate(), {"training_precision": "fp16"})
    with pytest.raises(ValueError, match="one of bf16, fp16"):
        _validate_precision("trl@1.12.0.post10", LoRAUpdate(), {"training_precision": "float16"})
    with pytest.raises(ValueError, match="must be a boolean"):
        _validate_precision("trl@1.12.0.post10", LoRAUpdate(), {"logits_float32": "yes"})
    with pytest.raises(ValueError, match="online RL only, not SFT"):
        require_default_precision({"training_precision": "fp16"}, "SFT")
    require_default_precision({"training_precision": "bf16"}, "SFT")


def test_float16_loading_is_limited_to_lora_updates() -> None:
    calls: list[dict[str, object]] = []

    class Factory:
        @staticmethod
        def from_pretrained(_repo: str, **kwargs: object) -> Any:
            calls.append(kwargs)
            return SimpleNamespace(config=SimpleNamespace(use_cache=True))

    imports = {
        "torch": SimpleNamespace(bfloat16="bf16", float16="fp16", float32="fp32"),
        "AutoModelForCausalLM": Factory,
        "AutoModelForMultimodalLM": Factory,
        "get_peft_model": lambda model, _config: model,
        "LoraConfig": lambda **_kwargs: object(),
    }
    loop = cast(TrainingLoop, SimpleNamespace(gradient_checkpointing=False))
    load_trainable_model(QWEN_35_2B, LoRAUpdate(), loop, imports, model_dtype="float16")
    assert calls[-1]["dtype"] == "fp16"
    with pytest.raises(ValueError, match="requires a LoRA update"):
        load_trainable_model(QWEN_35_2B, FullParameterUpdate(), loop, imports, model_dtype="float16")


class _Scaler:
    def __init__(self, scale: float) -> None:
        self.scale = scale

    def get_scale(self) -> float:
        return self.scale


class _Context:
    def __init__(self) -> None:
        self.metrics_seen: list[tuple[dict[str, float], int | None]] = []
        self.events: list[tuple[str, dict[str, object]]] = []

    def metrics(self, values: dict[str, float], step: int | None = None) -> None:
        self.metrics_seen.append((dict(values), step))

    def event(self, name: str, attributes: dict[str, object]) -> None:
        self.events.append((name, dict(attributes)))


def test_loss_scale_monitor_records_scale_and_skipped_steps() -> None:
    context = _Context()
    monitor = LossScaleMonitor(cast(Any, context))
    monitor.observe(SimpleNamespace(scaler=None), 1)  # bf16: no scaler, nothing recorded
    assert context.metrics_seen == []
    monitor.observe(SimpleNamespace(scaler=_Scaler(65536.0), step_was_skipped=True), 1)
    monitor.observe(SimpleNamespace(scaler=_Scaler(32768.0), step_was_skipped=False), 2)
    assert context.metrics_seen == [
        (
            {"train/loss_scale": 65536.0, "train/optimizer_step_skipped": 1.0, "train/optimizer_steps_skipped": 1.0},
            1,
        ),
        (
            {"train/loss_scale": 32768.0, "train/optimizer_step_skipped": 0.0, "train/optimizer_steps_skipped": 1.0},
            2,
        ),
    ]
    assert context.events == [
        ("optimizer_step_skipped", {"global_step": 1, "loss_scale": 65536.0, "reason": "non-finite gradients"})
    ]


def test_infinite_grad_norm_is_dropped_only_for_a_skipped_step() -> None:
    monitor = LossScaleMonitor(cast(Any, _Context()))
    seen: list[dict[str, object]] = []

    def normalizer(_step: int, native: Any) -> dict[str, float]:
        seen.append(dict(native))
        return {}

    normalize = monitor.finite_grad_norm(normalizer)
    monitor.last_step_skipped = True
    normalize(1, {"grad_norm": math.inf, "loss": 0.5})
    monitor.last_step_skipped = False
    normalize(2, {"grad_norm": math.inf, "loss": 0.5})
    normalize(3, {"grad_norm": 1.5})
    assert seen == [{"loss": 0.5}, {"grad_norm": math.inf, "loss": 0.5}, {"grad_norm": 1.5}]


def test_sampler_gap_is_pooled_over_the_whole_update() -> None:
    torch = pytest.importorskip("torch")
    gap = SamplerGapAccumulator()
    # Two calls, as when the ratio is computed per micro-batch from the training forward.
    gap.add(
        torch.tensor([[-0.1, -0.2, -0.3, 0.0]]),
        torch.tensor([[-0.1, -0.1, float("nan"), 0.0]]),
        torch.tensor([[1.0, 1.0, 1.0, 0.0]]),
    )
    gap.add(torch.tensor([[-1.0, -1.0, 0.0]]), torch.tensor([[-1.5, -1.0, 0.0]]), torch.tensor([[1.0, 1.0, 0.0]]))
    gap.add(torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1, 2))  # no scored tokens: ignored
    stats = gap.flush()
    # Pooled tokens |gap| = 0.0, 0.1, 0.5, 0.0; the nearest-rank 99th percentile is the largest.
    assert stats["sampling/sampling_logp_difference/p99"] == pytest.approx(0.5)
    # Per-sequence sums of (trainer - sampler): -0.1 (the NaN token counts as zero) and +0.5.
    assert stats["sampling/sequence_logp_difference/abs_mean"] == pytest.approx(0.3)
    assert gap.flush() == {}


def _tiny_float16_lora(tmp_path: Path) -> tuple[Any, Any, Any]:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    config = transformers.Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
    )
    torch.manual_seed(0)
    lora = peft.LoraConfig(r=4, lora_alpha=8, target_modules="all-linear", task_type="CAUSAL_LM")
    fresh = peft.get_peft_model(transformers.AutoModelForCausalLM.from_config(config, dtype=torch.float16), lora)
    fresh.save_pretrained(tmp_path)
    resumed = peft.PeftModel.from_pretrained(
        transformers.AutoModelForCausalLM.from_config(config, dtype=torch.float16), tmp_path, is_trainable=True
    )
    return torch, fresh, resumed


def test_float16_lora_keeps_float32_adapters_and_steps_with_loss_scaling(tmp_path: Path) -> None:
    torch, fresh, resumed = _tiny_float16_lora(tmp_path)
    # Both load paths of load_trainable_model: a fresh adapter and one resumed with is_trainable=True.
    require_float32_trainable_parameters(fresh)
    require_float32_trainable_parameters(resumed)
    # TRL adds a frozen "ref" copy of a resumed adapter for the KL term; it is float32 and not trained.
    resumed.add_adapter("ref", resumed.peft_config["default"])
    ref = [parameter for name, parameter in resumed.named_parameters() if ".ref." in name]
    assert ref and all(p.dtype == torch.float32 and not p.requires_grad for p in ref)
    require_float32_trainable_parameters(resumed)

    handle = upcast_logits_to_float32(resumed)
    ids = torch.randint(0, 64, (1, 8))
    with torch.autocast("cpu", dtype=torch.float16):
        logits = resumed(input_ids=ids).logits
        with resumed.disable_adapter():
            reference = resumed(input_ids=ids).logits
    assert logits.dtype == torch.float32 and reference.dtype == torch.float32
    handle.remove()

    trainable = [parameter for parameter in resumed.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)
    scaler = torch.amp.GradScaler("cpu")
    with torch.autocast("cpu", dtype=torch.float16):
        loss = resumed(input_ids=ids, labels=ids).loss
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)  # refuses float16 gradients: "Attempting to unscale FP16 gradients"
    scaler.step(optimizer)
    scaler.update()
    assert all(parameter.grad is not None and parameter.grad.dtype == torch.float32 for parameter in trainable)


def test_float16_trainable_parameters_are_rejected(tmp_path: Path) -> None:
    torch, fresh, _resumed = _tiny_float16_lora(tmp_path)
    for parameter in fresh.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.to(torch.float16)
    with pytest.raises(ValueError, match="requires float32 trainable parameters"):
        require_float32_trainable_parameters(fresh)
