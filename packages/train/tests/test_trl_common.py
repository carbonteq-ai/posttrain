"""Focused tests for family-aware TRL model loading."""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest
from posttrain.common import ProducedArtifact
from posttrain.common.variants import GEMMA_4_12B_IT, LFM_25_12B_THINKING, QWEN_35_2B
from posttrain.train import LoRAUpdate, TrainingLoop
from posttrain.train.backends.trl.common import (
    checkpoint_callback_type,
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
