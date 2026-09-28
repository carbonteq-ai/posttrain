"""Cancellation checkpoints over a small fake of the transformers update loop.

The fake keeps the order of one transformers ``Trainer`` update: rollout,
gradient-accumulation micro-steps, ``on_pre_optimizer_step``, a multi-parameter
``optimizer.step``, the scheduler step, ``global_step += 1``, ``on_step_end``,
and ``_maybe_log_save_evaluate`` with periodic saves and rotation. A real
SIGTERM is raised at a chosen phase, through the same handler shape
``posttrain-runtime`` installs.
"""

from __future__ import annotations

import json
import random
import signal
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType, SimpleNamespace
from typing import Any

import pytest
from posttrain.common import HostCancellation, ProducedArtifact
from posttrain.common.variants import QWEN_35_2B
from posttrain.train import LoRAUpdate
from posttrain.train.adaptive_curriculum import CURRICULUM_SNAPSHOT_NAME
from posttrain.train.backends.trl.cancellation import (
    CANCEL_CHECKPOINT_STAGING,
    UpdateBoundary,
    retain_checkpoint_after_interruption,
    update_boundary_callback_type,
    update_boundary_trainer_type,
)
from posttrain.train.backends.trl.common import CheckpointPublisher, checkpoint_callback_type

PARAMETERS = 3


@dataclass
class CaptureContext:
    run_id: str = "run/example"
    events: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    metrics: list[tuple[str, float, int | None]] = field(default_factory=list)
    artifacts: list[ProducedArtifact] = field(default_factory=list)

    def event(self, name: str, attributes: dict[str, Any]) -> None:
        self.events.append((name, attributes))

    def metric(self, name: str, value: float, *, step: int | None = None) -> None:
        # The tracker rejects a logical step below one already logged.
        logged = [logged_step for _, _, logged_step in self.metrics if logged_step is not None]
        if step is not None and logged and step < max(logged):
            raise ValueError("logical metric steps must be nondecreasing")
        self.metrics.append((name, value, step))

    def artifact(self, artifact: ProducedArtifact) -> None:
        self.artifacts.append(artifact)

    def cancel_outcome(self) -> dict[str, Any]:
        outcomes = [attributes for name, attributes in self.events if name == "cancel_checkpoint"]
        assert len(outcomes) == 1
        return outcomes[0]


class TrainerCallback:
    pass


def get_last_checkpoint(folder: str) -> str | None:
    steps = [
        int(path.name.removeprefix("checkpoint-"))
        for path in Path(folder).iterdir()
        if path.is_dir() and path.name.startswith("checkpoint-") and path.name.removeprefix("checkpoint-").isdigit()
    ]
    return str(Path(folder) / f"checkpoint-{max(steps)}") if steps else None


IMPORTS = {"TrainerCallback": TrainerCallback, "get_last_checkpoint": get_last_checkpoint}


class FakeTrainer:
    """The update order of ``transformers.Trainer._inner_training_loop``."""

    def __init__(
        self,
        *,
        output_dir: Path,
        callbacks: list[Any],
        checkpoint_steps: int,
        checkpoint_limit: int,
        max_steps: int,
        accumulation_steps: int = 2,
        phase_hook: Callable[[str, int], None] = lambda phase, update: None,
    ) -> None:
        self.args = SimpleNamespace(
            output_dir=str(output_dir),
            save_strategy="steps",
            save_total_limit=checkpoint_limit,
        )
        self.state = SimpleNamespace(global_step=0, total_flos=0.0)
        self.callbacks = callbacks
        self.checkpoint_steps = checkpoint_steps
        self.max_steps = max_steps
        self.accumulation_steps = accumulation_steps
        self.phase_hook = phase_hook
        self.weights = [0] * PARAMETERS
        self.moments = [0] * PARAMETERS
        self.scheduler_steps = 0
        self.grads = [0] * PARAMETERS
        self.current_flos = 0.0
        self.entropy_coef = 0.0
        self.model = self

    def _call(self, event: str) -> None:
        for callback in self.callbacks:
            handler = getattr(callback, event, None)
            if handler is not None:
                handler(self.args, self.state, "control")

    def train(self) -> None:
        while self.state.global_step < self.max_steps:
            update = self.state.global_step + 1
            # Rollout sampling consumes random numbers before the actor update.
            random.random()
            self.phase_hook("rollout", update)
            for micro_step in range(self.accumulation_steps):
                self.grads = [grad + 1 for grad in self.grads]
                self.current_flos += 10.0
                if micro_step == self.accumulation_steps - 1:
                    # TRL's adaptive entropy controller advances in the final micro-batch.
                    self.entropy_coef += 0.5
                self.phase_hook("actor", update)
            self._call("on_pre_optimizer_step")
            for index in range(PARAMETERS):
                self.moments[index] += 1
                self.weights[index] += 1
                if index == 0:
                    self.phase_hook("optimizer", update)
            self._call("on_optimizer_step")
            self.scheduler_steps += 1
            self.grads = [0] * PARAMETERS
            self.state.global_step += 1
            self._call("on_step_end")
            self._maybe_log_save_evaluate()

    def _maybe_log_save_evaluate(self) -> None:
        if self.state.global_step % self.checkpoint_steps == 0:
            self._save_checkpoint(self.model, None)
            self._call("on_save")

    def _get_output_dir(self, trial: object = None) -> str:
        del trial
        return str(self.args.output_dir)

    def _save_checkpoint(self, model: object, trial: object) -> None:
        del model
        self.state.total_flos += self.current_flos
        self.current_flos = 0.0
        run_dir = Path(self._get_output_dir(trial=trial))
        checkpoint = run_dir / f"checkpoint-{self.state.global_step}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "adapter_config.json").write_text("{}\n", encoding="utf-8")
        (checkpoint / "adapter_model.safetensors").write_text(json.dumps(self.weights), encoding="utf-8")
        (checkpoint / "optimizer.pt").write_text(json.dumps(self.moments), encoding="utf-8")
        (checkpoint / "scheduler.pt").write_text(json.dumps(self.scheduler_steps), encoding="utf-8")
        (checkpoint / "rng_state.pth").write_text(json.dumps(random.getstate()[1][:4]), encoding="utf-8")
        (checkpoint / "trainer_state.json").write_text(
            json.dumps(
                {
                    "global_step": self.state.global_step,
                    "total_flos": self.state.total_flos,
                    "entropy_coef": self.entropy_coef,
                }
            ),
            encoding="utf-8",
        )
        existing = sorted(
            (path for path in run_dir.iterdir() if path.name.startswith("checkpoint-")),
            key=lambda path: int(path.name.removeprefix("checkpoint-")),
        )
        for stale in existing[: max(0, len(existing) - self.args.save_total_limit)]:
            for child in stale.iterdir():
                child.unlink()
            stale.rmdir()


@contextmanager
def _host(gate: HostCancellation, *, max_deferral_seconds: float = 60.0) -> Iterator[None]:
    previous = signal.getsignal(signal.SIGTERM)

    def handler(signum: int, frame: FrameType | None) -> None:
        del frame
        gate.request(signum)

    gate.arm(max_deferral_seconds=max_deferral_seconds)
    signal.signal(signal.SIGTERM, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)
        gate.disarm()


@dataclass
class Outcome:
    context: CaptureContext
    trainer: Any
    output_dir: Path
    exit: SystemExit
    boundary_rng: Any = None
    controller_writes: list[tuple[str, Mapping[str, object]]] = field(default_factory=list)

    def checkpoint_steps(self) -> list[int]:
        return sorted(
            int(path.name.removeprefix("checkpoint-"))
            for path in self.output_dir.iterdir()
            if path.name.startswith("checkpoint-")
        )

    def checkpoint(self, step: int) -> dict[str, Any]:
        root = self.output_dir / f"checkpoint-{step}"
        return {
            "weights": json.loads((root / "adapter_model.safetensors").read_text(encoding="utf-8")),
            "moments": json.loads((root / "optimizer.pt").read_text(encoding="utf-8")),
            "scheduler": json.loads((root / "scheduler.pt").read_text(encoding="utf-8")),
            "rng": json.loads((root / "rng_state.pth").read_text(encoding="utf-8")),
            "trainer_state": json.loads((root / "trainer_state.json").read_text(encoding="utf-8")),
        }


def _run(
    tmp_path: Path,
    *,
    cancel_at: tuple[str, int],
    checkpoint_steps: int = 10,
    checkpoint_limit: int = 1,
    max_deferral_seconds: float = 60.0,
    timeout_seconds: float = 30.0,
    phase_delay: float = 0.0,
    save_delay: float = 0.0,
    with_controller: bool = False,
) -> Outcome:
    gate = HostCancellation()
    context = CaptureContext()
    controller = {"decisions": 0}
    controller_writes: list[tuple[str, Mapping[str, object]]] = []

    def capture_controller() -> Mapping[str, object]:
        return dict(controller)

    def write_controller(checkpoint: Path, state: Mapping[str, object] | None = None) -> None:
        recorded = dict(state) if state is not None else dict(controller)
        controller_writes.append((checkpoint.name, recorded))
        (checkpoint / CURRICULUM_SNAPSHOT_NAME).write_text(json.dumps(recorded), encoding="utf-8")

    boundary = UpdateBoundary(gate=gate, controller_state=capture_controller if with_controller else None)
    output_dir = tmp_path / "trainer"
    output_dir.mkdir()
    publisher = CheckpointPublisher(
        context,  # type: ignore[arg-type]
        model=QWEN_35_2B,
        technique="sampo",
        settings=SimpleNamespace(id="training-settings/test", revision="1"),
        update=LoRAUpdate(),
        workspace=tmp_path,
        checkpoint_state_writer=write_controller if with_controller else None,
    )
    boundary_rng: list[Any] = []

    def phase_hook(phase: str, update: int) -> None:
        if phase == "rollout":
            # The controller selects tasks for the next update during its rollout.
            controller["decisions"] += 1
            # A finished rollout batch logs its metrics at the update it feeds.
            context.metric("train/rl/rollout_batch_seconds", 1.0, step=update)
        if (phase, update) == cancel_at:
            boundary_rng.append(boundary.snapshot.rng["python"] if boundary.snapshot is not None else None)
            signal.raise_signal(signal.SIGTERM)
            random.random()
            time.sleep(phase_delay)

    trainer_type = update_boundary_trainer_type(FakeTrainer, boundary)
    if save_delay:
        base_save = trainer_type._save_checkpoint

        def slow_save(self: FakeTrainer, model: object, trial: object) -> None:
            if Path(self._get_output_dir(trial=trial)).name == CANCEL_CHECKPOINT_STAGING:
                time.sleep(save_delay)
            base_save(self, model, trial)

        trainer_type = type("SlowSaveTrainer", (trainer_type,), {"_save_checkpoint": slow_save})
    callbacks = [
        update_boundary_callback_type(IMPORTS, boundary)(),
        checkpoint_callback_type(
            context,  # type: ignore[arg-type]
            IMPORTS,
            model=QWEN_35_2B,
            technique="sampo",
            settings=publisher.settings,
            update=publisher.update,
            workspace=tmp_path,
            publisher=publisher,
        )(),
    ]
    trainer = trainer_type(
        output_dir=output_dir,
        callbacks=callbacks,
        checkpoint_steps=checkpoint_steps,
        checkpoint_limit=checkpoint_limit,
        max_steps=50,
        phase_hook=phase_hook,
    )
    random.seed(7)
    with _host(gate, max_deferral_seconds=max_deferral_seconds):
        with pytest.raises(SystemExit) as captured:
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
                    controller_state_writer=write_controller if with_controller else None,
                    timeout_seconds=timeout_seconds,
                )
                raise
    return Outcome(
        context,
        trainer,
        output_dir,
        captured.value,
        boundary_rng[0] if boundary_rng else None,
        controller_writes,
    )


def _completed_update(step: int) -> dict[str, Any]:
    return {"weights": [step] * PARAMETERS, "moments": [step] * PARAMETERS, "scheduler": step}


@pytest.mark.parametrize("phase", ["rollout", "actor"])
def test_cancel_between_updates_saves_the_last_completed_update(tmp_path: Path, phase: str) -> None:
    outcome = _run(tmp_path, cancel_at=(phase, 44), checkpoint_steps=10, checkpoint_limit=1)

    assert outcome.exit.code == 128 + signal.SIGTERM
    # Update 44 never reached its optimizer step, so updates 41-43 are the new progress.
    assert outcome.trainer.state.global_step == 43
    assert outcome.checkpoint_steps() == [40, 43]
    saved = outcome.checkpoint(43)
    assert {key: saved[key] for key in ("weights", "moments", "scheduler")} == _completed_update(43)
    assert saved["trainer_state"]["global_step"] == 43
    record = outcome.context.cancel_outcome()
    assert record["outcome"] == "saved"
    assert record["global_step"] == 43
    assert record["previous_checkpoint_step"] == 40
    assert record["deferred_by"] is None
    # Recorded at the cancelled update, which already logged its rollout metrics.
    assert ("train/cancel_checkpoint_step", 43, 44) in outcome.context.metrics
    assert record["cancelled_update"] == 44
    assert "metric_error" not in record
    assert not (outcome.output_dir / CANCEL_CHECKPOINT_STAGING).exists()
    cancel_views = [artifact for artifact in outcome.context.artifacts if artifact.metadata["global_step"] == 43]
    assert [artifact.kind for artifact in cancel_views] == ["training-checkpoint", "model-adapter"]
    assert all(artifact.metadata["interrupted"] is True for artifact in cancel_views)
    assert cancel_views[0].reference.path == (outcome.output_dir / "checkpoint-43").resolve()  # type: ignore[union-attr]
    assert cancel_views[0].reference.digest  # type: ignore[union-attr]
    # The periodic step 40 is not republished because the newer update was saved.
    assert [artifact.metadata["global_step"] for artifact in outcome.context.artifacts] == [
        10,
        10,
        20,
        20,
        30,
        30,
        40,
        40,
        43,
        43,
    ]


def test_cancel_checkpoint_restores_the_boundary_state_the_rollout_advanced(tmp_path: Path) -> None:
    outcome = _run(tmp_path, cancel_at=("actor", 44), with_controller=True)

    saved = outcome.checkpoint(43)
    # The rollout and final micro-batch of update 44 advanced these; the checkpoint holds update 43's values.
    assert saved["trainer_state"]["entropy_coef"] == 43 * 0.5
    assert saved["trainer_state"]["total_flos"] == 43 * 20.0
    assert saved["rng"] == list(outcome.boundary_rng[1][:4])
    assert outcome.controller_writes[-1] == ("checkpoint-43", {"decisions": 43})
    stored = json.loads((outcome.output_dir / "checkpoint-43" / CURRICULUM_SNAPSHOT_NAME).read_text(encoding="utf-8"))
    assert stored == {"decisions": 43}
    assert [artifact.kind for artifact in outcome.context.artifacts[-3:]] == [
        "training-checkpoint",
        "model-adapter",
        "adaptive-curriculum-state",
    ]


def test_cancel_during_optimizer_step_is_deferred_until_the_update_completes(tmp_path: Path) -> None:
    outcome = _run(tmp_path, cancel_at=("optimizer", 44))

    assert outcome.exit.code == 128 + signal.SIGTERM
    # The signal landed after the first parameter moved; the update still finished.
    assert outcome.trainer.state.global_step == 44
    assert outcome.checkpoint_steps() == [40, 44]
    saved = outcome.checkpoint(44)
    assert {key: saved[key] for key in ("weights", "moments", "scheduler")} == _completed_update(44)
    record = outcome.context.cancel_outcome()
    assert record["outcome"] == "saved"
    assert record["deferred_by"] == "optimizer_update"
    assert record["forced"] is False


def test_cancel_during_the_periodic_save_update_needs_no_second_checkpoint(tmp_path: Path) -> None:
    outcome = _run(tmp_path, cancel_at=("optimizer", 40), checkpoint_limit=3)

    assert outcome.trainer.state.global_step == 40
    assert outcome.checkpoint_steps() == [20, 30, 40]
    assert outcome.context.cancel_outcome()["reason"] == "already_checkpointed"
    assert {key: outcome.checkpoint(40)[key] for key in ("weights", "moments", "scheduler")} == _completed_update(40)


def test_a_forced_exit_inside_the_optimizer_step_writes_no_checkpoint(tmp_path: Path) -> None:
    outcome = _run(tmp_path, cancel_at=("optimizer", 44), max_deferral_seconds=0.2, phase_delay=5.0)

    assert outcome.exit.code == 128 + signal.SIGTERM
    # Weights moved for part of update 44 while global_step still says 43: never persist that.
    assert outcome.trainer.weights[0] == 44
    assert outcome.trainer.weights[1] == 43
    assert outcome.checkpoint_steps() == [40]
    record = outcome.context.cancel_outcome()
    assert record["outcome"] == "skipped"
    assert record["reason"] == "cancelled_inside_optimizer_update"
    assert record["forced"] is True


def test_a_slow_cancel_checkpoint_is_abandoned_within_its_bound(tmp_path: Path) -> None:
    started = time.monotonic()
    outcome = _run(tmp_path, cancel_at=("rollout", 44), timeout_seconds=0.2, save_delay=2.0)

    assert time.monotonic() - started < 2.0
    assert outcome.exit.code == 128 + signal.SIGTERM
    record = outcome.context.cancel_outcome()
    assert record["outcome"] == "failed"
    assert record["reason"] == "timed_out"
    # The incomplete staging directory is never mistaken for a checkpoint.
    assert outcome.checkpoint_steps() == [40]
    assert get_last_checkpoint(str(outcome.output_dir)) == str(outcome.output_dir / "checkpoint-40")


def test_cancel_before_any_update_in_this_process_keeps_existing_recovery(tmp_path: Path) -> None:
    outcome = _run(tmp_path, cancel_at=("rollout", 1))

    assert outcome.checkpoint_steps() == []
    assert outcome.context.cancel_outcome()["reason"] == "no_update_completed"
    assert ("recovery_checkpoint_unavailable", {"technique": "sampo"}) in outcome.context.events


def test_training_failure_is_not_treated_as_cancellation(tmp_path: Path) -> None:
    gate = HostCancellation()
    context = CaptureContext()
    boundary = UpdateBoundary(gate=gate)
    trainer = SimpleNamespace(
        args=SimpleNamespace(output_dir=str(tmp_path), save_strategy="steps"),
        state=SimpleNamespace(global_step=3),
    )
    error = RuntimeError("CUDA error")
    publisher = CheckpointPublisher(
        context,  # type: ignore[arg-type]
        model=QWEN_35_2B,
        technique="grpo",
        settings=SimpleNamespace(id="training-settings/test", revision="1"),
        update=LoRAUpdate(),
        workspace=tmp_path,
    )

    retain_checkpoint_after_interruption(
        context,  # type: ignore[arg-type]
        trainer,
        error,
        boundary=boundary,
        publisher=publisher,
        imports=IMPORTS,
    )

    assert not [name for name, _ in context.events if name == "cancel_checkpoint"]
    assert context.events == [("recovery_checkpoint_unavailable", {"technique": "grpo"})]


def test_fallback_republication_keeps_the_saved_controller_snapshot(tmp_path: Path) -> None:
    outcome = _run(tmp_path, cancel_at=("rollout", 41), with_controller=True)

    # Update 40 is already saved, so the existing path republishes it marked interrupted.
    assert outcome.context.cancel_outcome()["reason"] == "already_checkpointed"
    assert outcome.context.artifacts[-1].metadata["interrupted"] is True
    stored = json.loads((outcome.output_dir / "checkpoint-40" / CURRICULUM_SNAPSHOT_NAME).read_text(encoding="utf-8"))
    # The live controller already selected tasks for update 41; the periodic snapshot stays at 40.
    assert stored == {"decisions": 40}
    assert [write for write in outcome.controller_writes if write[0] == "checkpoint-40"] == [
        ("checkpoint-40", {"decisions": 40})
    ]


def test_a_rejected_cancel_checkpoint_metric_is_named_in_the_saved_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record_metric = CaptureContext.metric

    def reject_cancel_metric(self: CaptureContext, name: str, value: float, *, step: int | None = None) -> None:
        if name == "train/cancel_checkpoint_step":
            raise RuntimeError("tracking rejected the metric")
        record_metric(self, name, value, step=step)

    monkeypatch.setattr(CaptureContext, "metric", reject_cancel_metric)
    outcome = _run(tmp_path, cancel_at=("rollout", 44))

    record = outcome.context.cancel_outcome()
    assert record["outcome"] == "saved"
    assert "tracking rejected the metric" in record["metric_error"]
    assert any("cancellation checkpoint metric" in note for note in outcome.exit.__notes__)
