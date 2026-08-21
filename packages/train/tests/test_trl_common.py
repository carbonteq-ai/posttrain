"""Focused tests for family-aware TRL model loading."""

import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest
from posttrain.common import LocalArtifactRef, ProducedArtifact, RunContext
from posttrain.common.variants import GEMMA_4_12B_IT, GEMMA_4_E4B_IT, LFM_25_12B_THINKING, QWEN_35_2B
from posttrain.train import LoRAUpdate, TrainingLoop
from posttrain.train.backends.trl.common import (
    checkpoint_callback_type,
    emit_parameter_counts,
    finish_training,
    load_processor,
    load_trainable_model,
    preserve_recovery_checkpoint_after_error,
    publish_interrupted_recovery_checkpoint,
    trainable_model_factory,
    vllm_rollout_options,
)
from posttrain.train.backends.trl.grpo import _configure_torch_compile


class CausalFactory:
    pass


class MultimodalFactory:
    pass


IMPORTS = {
    "AutoModelForCausalLM": CausalFactory,
    "AutoModelForMultimodalLM": MultimodalFactory,
}


def test_load_processor_uses_pinned_model_revision_and_normalizes_tokenizer() -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    tokenizer = SimpleNamespace(pad_token_id=None, pad_token=None, eos_token="<eos>", padding_side="left")
    processor = SimpleNamespace(tokenizer=tokenizer)

    class ProcessorFactory:
        @staticmethod
        def from_pretrained(repo_id: str, **kwargs: object) -> object:
            calls.append((repo_id, kwargs))
            return processor

    loaded = load_processor(GEMMA_4_12B_IT, {"AutoProcessor": ProcessorFactory})

    assert loaded is processor
    assert calls == [
        (
            GEMMA_4_12B_IT.base.repo_id,
            {"revision": GEMMA_4_12B_IT.base.revision, "trust_remote_code": False},
        )
    ]
    assert tokenizer.pad_token == "<eos>"
    assert tokenizer.padding_side == "right"


@dataclass
class CaptureContext:
    run_id: str = "run/example"
    events: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    artifacts: list[ProducedArtifact] = field(default_factory=list)

    def event(self, name: str, attributes: dict[str, Any]) -> None:
        self.events.append((name, attributes))

    def artifact(self, artifact: ProducedArtifact) -> None:
        self.artifacts.append(artifact)


def _write_lora_recovery_checkpoint(path: Path) -> None:
    path.mkdir()
    for name in (
        "adapter_config.json",
        "adapter_model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
    ):
        (path / name).write_bytes(name.encode())
    (path / "trainer_state.json").write_text('{"global_step": 4}\n', encoding="utf-8")


def test_trainable_model_factory_uses_multimodal_loader_for_gemma4() -> None:
    assert trainable_model_factory(QWEN_35_2B, IMPORTS) is CausalFactory
    assert trainable_model_factory(LFM_25_12B_THINKING, IMPORTS) is CausalFactory
    assert trainable_model_factory(GEMMA_4_12B_IT, IMPORTS) is MultimodalFactory


def test_trainable_model_loader_honors_requested_dtype() -> None:
    calls: list[dict[str, object]] = []

    class Factory:
        @staticmethod
        def from_pretrained(_repo: str, **kwargs: object) -> Any:
            calls.append(kwargs)
            return SimpleNamespace(config=SimpleNamespace(use_cache=True))

    imports = {
        "torch": SimpleNamespace(bfloat16="bf16", float32="fp32"),
        "AutoModelForCausalLM": Factory,
        "AutoModelForMultimodalLM": Factory,
        "get_peft_model": lambda model, _config: model,
        "LoraConfig": lambda **_kwargs: object(),
    }
    loop = SimpleNamespace(gradient_checkpointing=False)

    model = load_trainable_model(
        QWEN_35_2B,
        LoRAUpdate(),
        cast(TrainingLoop, loop),
        imports,
        model_dtype="float32",
    )

    assert model.config.use_cache is False
    assert calls == [
        {
            "revision": QWEN_35_2B.base.revision,
            "device_map": {"": 0},
            "dtype": "fp32",
            "attn_implementation": "sdpa",
            "trust_remote_code": False,
        }
    ]
    with pytest.raises(ValueError, match="bfloat16.*float32"):
        load_trainable_model(QWEN_35_2B, LoRAUpdate(), cast(TrainingLoop, loop), imports, model_dtype="float16")


def test_gemma_e4b_adapter_reload_uses_the_multimodal_base(tmp_path: Path) -> None:
    calls: list[tuple[object, Path, bool]] = []
    adapter = tmp_path / "adapter"
    adapter.mkdir()

    class Factory:
        @staticmethod
        def from_pretrained(_repo: str, **_kwargs: object) -> Any:
            return SimpleNamespace(config=SimpleNamespace(use_cache=True))

    class PeftFactory:
        @staticmethod
        def from_pretrained(base: object, path: Path, *, is_trainable: bool) -> object:
            calls.append((base, path, is_trainable))
            return "reloaded-adapter"

    model = replace(
        GEMMA_4_E4B_IT,
        id="gemma4-e4b-it-qualification-adapter",
        artifact=LocalArtifactRef(adapter.resolve(), "a" * 64),
        form="peft-adapter",
        parent=GEMMA_4_E4B_IT.id,
        revision=None,
        digest=None,
    )
    imports = {
        "torch": SimpleNamespace(bfloat16="bf16", float32="fp32"),
        "AutoModelForCausalLM": CausalFactory,
        "AutoModelForMultimodalLM": Factory,
        "PeftModel": PeftFactory,
    }

    loaded = load_trainable_model(
        model,
        LoRAUpdate(),
        cast(TrainingLoop, SimpleNamespace(gradient_checkpointing=False)),
        imports,
    )

    assert loaded == "reloaded-adapter"
    assert len(calls) == 1
    assert calls[0][1:] == (adapter, True)


def test_parameter_evidence_rejects_lora_outside_the_language_target() -> None:
    class Parameter:
        def __init__(self, size: int, *, trainable: bool) -> None:
            self._size = size
            self.requires_grad = trainable

        def numel(self) -> int:
            return self._size

    parameters = {
        "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.default.weight": Parameter(
            4, trainable=True
        ),
        "base_model.model.model.vision_tower.layers.0.self_attn.q_proj.lora_A.default.weight": Parameter(
            4, trainable=True
        ),
        "base_model.model.model.language_model.embed_tokens.weight": Parameter(100, trainable=False),
    }
    model = SimpleNamespace(
        parameters=lambda: iter(parameters.values()),
        named_parameters=lambda: iter(parameters.items()),
    )
    update = LoRAUpdate(
        target_modules=(
            r"^model[.]language_model[.]layers[.]\d+[.]"
            r"(self_attn[.](q_proj|k_proj|v_proj|o_proj)|mlp[.](gate_proj|up_proj|down_proj))$"
        )
    )

    with pytest.raises(RuntimeError, match="outside target_modules.*vision_tower"):
        emit_parameter_counts(cast(RunContext, SimpleNamespace(metrics=lambda _values: None)), model, update)


def test_finish_training_retains_the_processor_with_the_adapter(tmp_path: Path) -> None:
    class Trainer:
        args = SimpleNamespace(output_dir=str(tmp_path / "checkpoints"))
        state = SimpleNamespace(global_step=2, log_history=[{"loss": 0.5, "step": 2}])

        @staticmethod
        def save_model(path: Path) -> None:
            path.mkdir(parents=True)
            (path / "adapter_config.json").write_text("{}\n", encoding="utf-8")
            (path / "adapter_model.safetensors").write_bytes(b"adapter")

    class Processor:
        @staticmethod
        def save_pretrained(path: Path) -> None:
            (path / "processor_config.json").write_text("{}\n", encoding="utf-8")

    imports = {
        "get_last_checkpoint": lambda _path: None,
        "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    }
    train_output = SimpleNamespace(
        metrics={
            "train_loss": 0.5,
            "train_runtime": 1.0,
            "train_samples_per_second": 1.0,
            "train_steps_per_second": 2.0,
        }
    )

    result = finish_training(
        cast(RunContext, SimpleNamespace(metrics=lambda _values: None)),
        Trainer(),
        train_output,
        Processor(),
        tmp_path,
        "sft",
        LoRAUpdate(),
        imports,
    )

    assert result.model_dir == tmp_path / "adapter"
    assert (result.model_dir / "processor_config.json").is_file()


def test_gemma_mtp_materializes_the_pinned_assistant_before_trl(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []
    hub = ModuleType("huggingface_hub")

    def snapshot_download(*, repo_id: str, revision: str) -> str:
        calls.append((repo_id, revision))
        return "/var/lib/posttrain/cache/huggingface/models--gemma-assistant/snapshots/pinned"

    hub.snapshot_download = snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    speculative, kwargs = vllm_rollout_options(
        GEMMA_4_12B_IT,
        {
            "mode": "colocate",
            "speculative_config": {
                "method": "mtp",
                "num_speculative_tokens": 1,
                "assistant_model": "google/gemma-4-12B-it-assistant",
                "assistant_revision": "364bd03c9952e5b7da73665ee30c9eccfc408345",
            },
        },
    )
    assert speculative == {
        "method": "mtp",
        "num_speculative_tokens": 1,
        "model": "/var/lib/posttrain/cache/huggingface/models--gemma-assistant/snapshots/pinned",
    }
    assert kwargs == {"disable_log_stats": False}
    assert calls == [("google/gemma-4-12B-it-assistant", "364bd03c9952e5b7da73665ee30c9eccfc408345")]


def test_gemma_mtp_rejects_an_unpinned_or_incomplete_assistant() -> None:
    with pytest.raises(ValueError, match="assistant_model and assistant_revision"):
        vllm_rollout_options(
            GEMMA_4_12B_IT,
            {"mode": "colocate", "speculative_config": {"method": "mtp", "num_speculative_tokens": 1}},
        )


def test_vllm_rollout_options_preserves_bounded_large_batch_wave_settings() -> None:
    speculative, kwargs = vllm_rollout_options(
        QWEN_35_2B,
        {
            "mode": "colocate",
            "max_num_seqs": 32,
            "max_num_batched_tokens": 32768,
        },
    )

    assert speculative is None
    assert kwargs == {"max_num_seqs": 32, "max_num_batched_tokens": 32768}
    with pytest.raises(ValueError, match="full 40-character commit SHA"):
        vllm_rollout_options(
            GEMMA_4_12B_IT,
            {
                "mode": "colocate",
                "speculative_config": {
                    "method": "mtp",
                    "num_speculative_tokens": 1,
                    "assistant_model": "google/gemma-4-12B-it-assistant",
                    "assistant_revision": "main",
                },
            },
        )


def test_mtp_compile_disable_is_applied_before_runtime_import(monkeypatch) -> None:
    monkeypatch.delenv("TORCH_COMPILE_DISABLE", raising=False)
    _configure_torch_compile({"disable_torch_compile": True})
    assert __import__("os").environ["TORCH_COMPILE_DISABLE"] == "1"


def test_mtp_compile_disable_requires_a_boolean() -> None:
    with pytest.raises(ValueError, match="disable_torch_compile must be a boolean"):
        _configure_torch_compile({"disable_torch_compile": "yes"})


def test_interrupted_lora_run_publishes_recovery_and_model_views(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-4"
    _write_lora_recovery_checkpoint(checkpoint)
    context = CaptureContext()
    trainer = SimpleNamespace(
        args=SimpleNamespace(output_dir=str(tmp_path)),
        state=SimpleNamespace(global_step=5),
    )

    retained = publish_interrupted_recovery_checkpoint(
        context,  # type: ignore[arg-type]
        trainer,
        technique="sft",
        model=QWEN_35_2B,
        settings=SimpleNamespace(id="training-settings/test", revision="1"),
        update=LoRAUpdate(),
        imports={"get_last_checkpoint": lambda _: str(checkpoint)},
    )

    assert retained == checkpoint
    assert len(context.artifacts) == 2
    recovery, model = context.artifacts
    assert recovery.kind == "training-checkpoint"
    assert recovery.role == "recovery"
    assert recovery.reference.path == checkpoint.resolve()  # type: ignore[union-attr]
    assert recovery.metadata["global_step"] == 4
    assert model.kind == "model-adapter"
    assert model.role == "checkpoint-model"
    assert model.metadata["checkpoint_snapshot_id"] == "run/example/step-00000004"
    assert model.reference.path.is_dir()  # type: ignore[union-attr]
    assert not tuple(checkpoint.glob("model*.safetensors"))
    assert context.events[-1][0] == "checkpoint_saved"
    assert context.events[-1][1]["global_step"] == 4


def test_checkpoint_callback_publishes_paired_views_on_save(tmp_path: Path) -> None:
    checkpoint = tmp_path / "trainer" / "checkpoint-4"
    checkpoint.parent.mkdir()
    _write_lora_recovery_checkpoint(checkpoint)

    class TrainerCallback:
        pass

    context = CaptureContext()
    callback = checkpoint_callback_type(
        context,  # type: ignore[arg-type]
        {
            "TrainerCallback": TrainerCallback,
            "get_last_checkpoint": lambda _: str(checkpoint),
        },
        model=QWEN_35_2B,
        technique="grpo",
        settings=SimpleNamespace(id="training-settings/test", revision="1"),
        update=LoRAUpdate(),
        workspace=tmp_path,
    )()

    result = callback.on_save(
        SimpleNamespace(output_dir=str(tmp_path / "trainer")),
        SimpleNamespace(global_step=4),
        "control",
    )

    assert result == "control"
    assert [artifact.kind for artifact in context.artifacts] == ["training-checkpoint", "model-adapter"]
    model_path = context.artifacts[1].reference.path  # type: ignore[union-attr]
    assert model_path == (tmp_path / "checkpoints" / "step-00000004" / "model").resolve()
    assert (model_path / "adapter_model.safetensors").is_file()
    assert not (model_path / "optimizer.pt").exists()


def test_interrupted_run_without_a_complete_checkpoint_is_not_published(tmp_path: Path) -> None:
    context = CaptureContext()
    trainer = SimpleNamespace(
        args=SimpleNamespace(output_dir=str(tmp_path)),
        state=SimpleNamespace(global_step=0),
    )

    retained = publish_interrupted_recovery_checkpoint(
        context,  # type: ignore[arg-type]
        trainer,
        technique="dpo",
        model=QWEN_35_2B,
        settings=SimpleNamespace(id="training-settings/test", revision="1"),
        update=LoRAUpdate(),
        imports={"get_last_checkpoint": lambda _: None},
    )

    assert retained is None
    assert context.artifacts == []
    assert context.events == [("recovery_checkpoint_unavailable", {"technique": "dpo"})]


def test_recovery_publication_failure_does_not_replace_training_error(tmp_path: Path) -> None:
    context = CaptureContext()
    trainer = SimpleNamespace(
        args=SimpleNamespace(output_dir=str(tmp_path)),
        state=SimpleNamespace(global_step=1),
    )
    error = RuntimeError("training failed")

    preserve_recovery_checkpoint_after_error(
        context,  # type: ignore[arg-type]
        trainer,
        error,
        technique="grpo",
        model=QWEN_35_2B,
        settings=SimpleNamespace(id="training-settings/test", revision="1"),
        update=LoRAUpdate(),
        imports={"get_last_checkpoint": lambda _: str(tmp_path / "missing")},
    )

    assert str(error) == "training failed"
    assert len(error.__notes__) == 1
    assert error.__notes__[0].startswith("failed to retain the latest recovery checkpoint: FileNotFoundError(")
    assert "missing" in error.__notes__[0]
