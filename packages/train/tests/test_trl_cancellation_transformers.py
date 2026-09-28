"""Cancellation checkpoints through the pinned transformers ``Trainer`` on CPU.

A tiny randomly initialized causal model with a rank-4 LoRA adapter trains for
a few updates. A real SIGTERM is raised inside an actor micro-step or inside
``optimizer.step``; the resulting checkpoint must match the in-memory adapter,
pass the publication checks, and resume in a new trainer.
"""

from __future__ import annotations

import json
import signal
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import FrameType, SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

from posttrain.common import HostCancellation, ProducedArtifact  # noqa: E402
from posttrain.common.variants import QWEN_35_2B  # noqa: E402
from posttrain.train import LoRAUpdate  # noqa: E402
from posttrain.train.backends.trl.cancellation import (  # noqa: E402
    UpdateBoundary,
    retain_checkpoint_after_interruption,
    update_boundary_callback_type,
    update_boundary_trainer_type,
)
from posttrain.train.backends.trl.common import CheckpointPublisher, checkpoint_callback_type  # noqa: E402

IMPORTS = {
    "TrainerCallback": transformers.TrainerCallback,
    "get_last_checkpoint": transformers.trainer_utils.get_last_checkpoint,
}


class _Context:
    run_id = "run/example"

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.artifacts: list[ProducedArtifact] = []

    def event(self, name: str, attributes: dict[str, Any]) -> None:
        self.events.append((name, attributes))

    def metric(self, name: str, value: float, *, step: int | None = None) -> None:
        del name, value, step

    def artifact(self, artifact: ProducedArtifact) -> None:
        self.artifacts.append(artifact)


class _Rows(torch.utils.data.Dataset):
    def __len__(self) -> int:
        return 64

    def __getitem__(self, index: int) -> dict[str, Any]:
        ids = torch.tensor([(index + offset) % 32 for offset in range(8)])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids}


def _model() -> Any:
    torch.manual_seed(0)
    config = transformers.GPT2Config(n_layer=1, n_embd=16, n_head=2, vocab_size=32, n_positions=16)
    base = transformers.GPT2LMHeadModel(config)
    lora = peft.LoraConfig(r=4, lora_alpha=8, target_modules=["c_attn"], lora_dropout=0.0)
    return peft.get_peft_model(base, lora)


def _arguments(output_dir: Path, max_steps: int) -> Any:
    return transformers.TrainingArguments(
        output_dir=str(output_dir),
        max_steps=max_steps,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=1e-2,
        save_strategy="steps",
        save_steps=2,
        save_total_limit=1,
        logging_steps=1,
        report_to="none",
        disable_tqdm=True,
        use_cpu=True,
        seed=0,
        data_seed=0,
    )


@contextmanager
def _host(gate: HostCancellation) -> Iterator[None]:
    previous = signal.getsignal(signal.SIGTERM)

    def handler(signum: int, frame: FrameType | None) -> None:
        del frame
        gate.request(signum)

    gate.arm()
    signal.signal(signal.SIGTERM, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)
        gate.disarm()


def _adapter(model: Any) -> dict[str, Any]:
    return {name: value.detach().clone() for name, value in peft.get_peft_model_state_dict(model).items()}


@pytest.mark.parametrize(
    ("phase", "expected_step"),
    [("actor", 3), ("optimizer", 5)],
    ids=["mid-actor", "inside-optimizer-step"],
)
def test_transformers_trainer_cancel_checkpoint_resumes(tmp_path: Path, phase: str, expected_step: int) -> None:
    gate = HostCancellation()
    context = _Context()
    boundary = UpdateBoundary(gate=gate)
    output_dir = tmp_path / "trainer"
    publisher = CheckpointPublisher(
        context,  # type: ignore[arg-type]
        model=QWEN_35_2B,
        technique="grpo",
        settings=SimpleNamespace(id="training-settings/test", revision="1"),
        update=LoRAUpdate(),
        workspace=tmp_path,
    )
    cancel_update = expected_step + 1 if phase == "actor" else expected_step

    class CancellingTrainer(update_boundary_trainer_type(transformers.Trainer, boundary)):
        def training_step(self, *args: Any, **kwargs: Any) -> Any:
            loss = super().training_step(*args, **kwargs)
            if phase == "actor" and self.state.global_step + 1 == cancel_update:
                signal.raise_signal(signal.SIGTERM)
            return loss

        def create_optimizer(self, *args: Any, **kwargs: Any) -> Any:
            optimizer = super().create_optimizer(*args, **kwargs)
            if phase == "optimizer":

                def cancel_inside_step(optimizer: Any, args: Any, kwargs: Any) -> None:
                    del optimizer, args, kwargs
                    if self.state.global_step + 1 == cancel_update:
                        signal.raise_signal(signal.SIGTERM)

                optimizer.register_step_post_hook(cancel_inside_step)
            return optimizer

    model = _model()
    trainer = CancellingTrainer(
        model=model,
        args=_arguments(output_dir, max_steps=8),
        train_dataset=_Rows(),
        callbacks=[
            update_boundary_callback_type(IMPORTS, boundary)(),
            checkpoint_callback_type(
                context,  # type: ignore[arg-type]
                IMPORTS,
                model=QWEN_35_2B,
                technique="grpo",
                settings=publisher.settings,
                update=publisher.update,
                workspace=tmp_path,
                publisher=publisher,
            )(),
        ],
    )
    with _host(gate), pytest.raises(SystemExit) as captured:
        try:
            trainer.train()
        except BaseException as error:
            retain_checkpoint_after_interruption(
                context,  # type: ignore[arg-type]
                trainer,
                error,
                boundary=boundary,
                publisher=publisher,
                imports=IMPORTS,
            )
            raise

    assert captured.value.code == 128 + signal.SIGTERM
    assert trainer.state.global_step == expected_step
    outcome = dict(context.events)["cancel_checkpoint"]
    assert outcome["outcome"] == "saved"
    assert outcome["previous_checkpoint_step"] == expected_step - 1
    checkpoint = output_dir / f"checkpoint-{expected_step}"
    # The periodic checkpoint survives: retention did not run before this one was committed.
    assert sorted(path.name for path in output_dir.glob("checkpoint-*")) == [
        f"checkpoint-{expected_step - 1}",
        f"checkpoint-{expected_step}",
    ]
    state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    assert state["global_step"] == expected_step
    saved = peft.utils.load_peft_weights(str(checkpoint))
    current = _adapter(trainer.model)
    # No optimizer step ran after the saved update, so the live adapter is that update.
    assert saved.keys() == current.keys()
    for name, value in current.items():
        assert torch.equal(saved[name], value), name
    assert [artifact.kind for artifact in context.artifacts[-2:]] == ["training-checkpoint", "model-adapter"]

    resumed = transformers.Trainer(
        model=_model(),
        args=_arguments(tmp_path / "resumed", max_steps=expected_step + 1),
        train_dataset=_Rows(),
    )
    resumed.train(resume_from_checkpoint=str(checkpoint))
    assert resumed.state.global_step == expected_step + 1
