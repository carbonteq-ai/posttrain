"""Keep the last completed TRL update when the host cancels a training run.

One TRL update is: rollout generation, forward and backward passes over the
gradient-accumulation micro-batches, ``optimizer.step``, ``lr_scheduler.step``,
``global_step += 1``, and then logging plus any periodic save. Trainable
parameters and optimizer state change only inside ``optimizer.step``, so outside
the span from that call to the end of the periodic save the trainer holds
exactly the state of update ``global_step``; the partially accumulated gradients
of the next update are not part of a checkpoint.

:class:`UpdateBoundary` declares that span as a ``posttrain.common`` critical
section, so a host cancellation signal that lands inside it is delivered when the
update is complete, and records the small mutable state a periodic save at the
boundary would have captured (random-number generators, trainer counters, the
adaptive entropy controller, and the adaptive-curriculum controller).
:func:`retain_checkpoint_after_interruption` then saves that update through the
trainer's own ``_save_checkpoint``, restoring the boundary state first, into a
staging directory that is renamed into place only when complete. The save runs
on a worker thread with a bounded wait so cancellation never hangs on it.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import random
import shutil
import sys
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from posttrain.common import HostCancellation, JsonValue, RunContext, host_cancellation

from ...adaptive_curriculum import CURRICULUM_SNAPSHOT_NAME
from .common import CheckpointPublisher, preserve_recovery_checkpoint_after_error

CANCEL_CHECKPOINT_TIMEOUT_SECONDS = 60.0
CANCEL_CHECKPOINT_STAGING = ".cancel-checkpoint"
_LOGGER = logging.getLogger(__name__)
# Trainer attributes a checkpoint records that the next update advances before
# its optimizer step: FLOP accounting and the TRL adaptive entropy controller.
_TRAINER_FIELDS = ("current_flos", "entropy_coef", "_last_world_entropy")
_STATE_FIELDS = ("total_flos", "num_input_tokens_seen")
_MISSING = object()

ControllerStateWriter = Callable[[Path, Mapping[str, object]], None]


@dataclass(frozen=True, slots=True)
class UpdateBoundarySnapshot:
    """The mutable state a periodic save at ``global_step`` would have written."""

    global_step: int
    rng: Mapping[str, Any]
    trainer_fields: Mapping[str, Any] = field(default_factory=dict)
    state_fields: Mapping[str, Any] = field(default_factory=dict)
    controller_state: Mapping[str, object] | None = None

    def restore(self, trainer: Any) -> None:
        for name, value in self.trainer_fields.items():
            setattr(trainer, name, value)
        for name, value in self.state_fields.items():
            setattr(trainer.state, name, value)
        _restore_rng(self.rng)


class UpdateBoundary:
    """Own the atomic span of one optimizer update and its last boundary state."""

    def __init__(
        self,
        *,
        gate: HostCancellation | None = None,
        controller_state: Callable[[], Mapping[str, object]] | None = None,
    ) -> None:
        self.gate = gate or host_cancellation()
        self._section = self.gate.critical("optimizer_update")
        self._controller_state = controller_state
        self._in_update = False
        self._torn = False
        self._snapshot: UpdateBoundarySnapshot | None = None

    @property
    def snapshot(self) -> UpdateBoundarySnapshot | None:
        return self._snapshot

    @property
    def torn(self) -> bool:
        """Whether an update was interrupted between its optimizer step and its end."""

        return self._torn

    @property
    def in_update(self) -> bool:
        return self._in_update

    def begin(self) -> None:
        """Enter the atomic span immediately before ``optimizer.step``."""

        self._in_update = True
        self._section.open()

    def commit(self, trainer: Any) -> None:
        """Record the completed update, then deliver any cancellation it deferred."""

        if not self._in_update:
            return
        self._snapshot = self.capture(trainer)
        self._in_update = False
        self._section.close()

    def abandon(self) -> None:
        """Release the span on an error path without delivering a deferred exit."""

        if self._in_update:
            self._torn = True
            self._in_update = False
        self._section.close(deliver=False)

    def capture(self, trainer: Any) -> UpdateBoundarySnapshot:
        state = trainer.state
        return UpdateBoundarySnapshot(
            global_step=int(state.global_step),
            rng=_capture_rng(),
            trainer_fields=_present_fields(trainer, _TRAINER_FIELDS),
            state_fields=_present_fields(state, _STATE_FIELDS),
            controller_state=self._controller_state() if self._controller_state is not None else None,
        )

    def guard[**P, R](self, name: str, function: Callable[P, R]) -> Callable[P, R]:
        """Defer cancellation for the whole of ``function``."""

        gate = self.gate

        @functools.wraps(function)
        def guarded(*args: P.args, **kwargs: P.kwargs) -> R:
            with gate.critical(name):
                return function(*args, **kwargs)

        return guarded


def update_boundary_callback_type(imports: Mapping[str, Any], boundary: UpdateBoundary) -> type[Any]:
    """Open the atomic update span at the trainer's pre-optimizer-step event."""

    parent = imports["TrainerCallback"]

    class UpdateBoundaryCallback(parent):
        def on_pre_optimizer_step(self, args: Any, state: Any, control: Any, **_: Any) -> Any:
            del args, state
            boundary.begin()
            return control

    return UpdateBoundaryCallback


def update_boundary_trainer_type(parent: type[Any], boundary: UpdateBoundary) -> type[Any]:
    """Close the update span after logging and any periodic save of that update.

    The rollout engine's weight synchronization is also made atomic so a cancel
    never leaves a colocated engine half-loaded while the checkpoint is written.
    """

    class UpdateBoundaryTrainer(parent):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            generation = getattr(self, "vllm_generation", None)
            sync = getattr(generation, "sync_weights", None)
            if generation is not None and callable(sync):
                generation.sync_weights = boundary.guard("rollout_weight_sync", sync)

        def _maybe_log_save_evaluate(self, *args: Any, **kwargs: Any) -> Any:
            try:
                result = super()._maybe_log_save_evaluate(*args, **kwargs)
            except BaseException:
                boundary.abandon()
                raise
            boundary.commit(self)
            return result

    return UpdateBoundaryTrainer


def retain_checkpoint_after_interruption(
    context: RunContext,
    trainer: Any,
    error: BaseException,
    *,
    boundary: UpdateBoundary,
    publisher: CheckpointPublisher,
    imports: Mapping[str, Any],
    controller_state_writer: ControllerStateWriter | None = None,
    timeout_seconds: float = CANCEL_CHECKPOINT_TIMEOUT_SECONDS,
) -> None:
    """Save the last completed update on cancel, else republish the latest checkpoint.

    This never raises; a failure becomes a note on ``error`` and a
    ``cancel_checkpoint`` event, and the run still finalizes from ``error``.
    """

    boundary.abandon()
    try:
        saved = save_cancellation_checkpoint(
            context,
            trainer,
            error,
            boundary=boundary,
            publisher=publisher,
            imports=imports,
            controller_state_writer=controller_state_writer,
            timeout_seconds=timeout_seconds,
        )
    except BaseException as failure:
        error.add_note(f"failed to save a cancellation checkpoint: {failure!r}")
        saved = False
    if saved:
        return
    preserve_recovery_checkpoint_after_error(
        context,
        trainer,
        error,
        technique=publisher.technique,
        model=publisher.model,
        settings=publisher.settings,
        update=publisher.update,
        imports=imports,
        checkpoint_state_writer=_write_missing_controller_state(publisher.checkpoint_state_writer),
    )


def save_cancellation_checkpoint(
    context: RunContext,
    trainer: Any,
    error: BaseException,
    *,
    boundary: UpdateBoundary,
    publisher: CheckpointPublisher,
    imports: Mapping[str, Any],
    controller_state_writer: ControllerStateWriter | None = None,
    timeout_seconds: float = CANCEL_CHECKPOINT_TIMEOUT_SECONDS,
) -> bool:
    """Write and publish ``checkpoint-<global_step>`` after a host cancellation.

    Returns ``True`` only when a new checkpoint was committed on disk and handed
    to tracking. Every other outcome is recorded as a ``cancel_checkpoint`` event
    with a reason, and the caller falls back to its existing recovery path.
    """

    if timeout_seconds <= 0:
        raise ValueError("cancellation checkpoint timeout must be positive")
    request = boundary.gate.requested
    if request is None or not isinstance(error, SystemExit | KeyboardInterrupt):
        return False
    started = time.monotonic()
    step = int(trainer.state.global_step)
    output_dir = Path(str(trainer.args.output_dir)).resolve()
    latest = imports["get_last_checkpoint"](str(output_dir)) if output_dir.is_dir() else None
    latest_step = _checkpoint_step(latest)
    cancelled_update = step + 1
    attributes: dict[str, JsonValue] = {
        "technique": publisher.technique,
        "global_step": step,
        "cancelled_update": cancelled_update,
        "previous_checkpoint_step": latest_step,
        "signal": request.signal_number,
        "deferred_by": request.deferred_by,
        "forced": boundary.gate.forced,
    }

    def outcome(name: str, reason: str | None = None, **extra: JsonValue) -> bool:
        record: dict[str, JsonValue] = {
            **attributes,
            "outcome": name,
            "reason": reason,
            "seconds": round(time.monotonic() - started, 3),
            **extra,
        }
        if name == "saved":
            _LOGGER.warning("cancel checkpoint saved at update %s (%.1fs)", step, record["seconds"])
        else:
            _LOGGER.warning("cancel checkpoint %s at update %s: %s", name, step, reason)
        try:
            context.event("cancel_checkpoint", record)
        except Exception as event_error:  # evidence must not replace the cancellation
            error.add_note(f"failed to record the cancellation checkpoint outcome: {event_error!r}")
        return name == "saved"

    snapshot = boundary.snapshot
    if boundary.torn:
        return outcome("skipped", "cancelled_inside_optimizer_update")
    save_strategy = getattr(trainer.args, "save_strategy", "steps")
    if str(getattr(save_strategy, "value", save_strategy)) == "no":
        return outcome("skipped", "checkpoints_disabled")
    if snapshot is None:
        return outcome("skipped", "no_update_completed")
    if snapshot.global_step != step:
        return outcome("skipped", "boundary_step_mismatch", boundary_step=snapshot.global_step)
    if latest_step is not None and latest_step >= step:
        return outcome("skipped", "already_checkpointed")

    target = output_dir / f"checkpoint-{step}"
    staging = output_dir / CANCEL_CHECKPOINT_STAGING
    if target.exists():
        return outcome("failed", "checkpoint_directory_exists")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    snapshot.restore(trainer)
    failures: list[BaseException] = []

    def save() -> None:
        try:
            _select_training_device(trainer)
            trainer._save_checkpoint(getattr(trainer, "model_wrapped", None) or trainer.model, None)
        except BaseException as failure:
            failures.append(failure)

    # Only this save writes into staging; a periodic save would also rotate
    # older checkpoints, which must not happen before this one is committed.
    trainer._get_output_dir = lambda trial=None: str(staging)
    worker = threading.Thread(target=save, name="posttrain-cancel-checkpoint", daemon=True)
    worker.start()
    worker.join(timeout_seconds)
    if worker.is_alive():
        # The daemon thread may still be writing into staging; it is never renamed.
        return outcome("failed", "timed_out", timeout_seconds=timeout_seconds)
    vars(trainer).pop("_get_output_dir", None)
    if failures:
        error.add_note(f"cancellation checkpoint save failed: {failures[0]!r}")
        return outcome("failed", "save_failed", error_type=type(failures[0]).__name__)
    staged = staging / target.name
    incomplete = _incomplete_reason(staged, step)
    if incomplete is not None:
        return outcome("failed", "incomplete_checkpoint", detail=incomplete)
    os.replace(staged, target)
    shutil.rmtree(staging, ignore_errors=True)

    try:
        publisher.publish(
            target,
            step=step,
            interrupted=True,
            checkpoint_state_writer=_boundary_state_writer(controller_state_writer, snapshot),
        )
    except Exception as failure:
        error.add_note(f"cancellation checkpoint {target.name} was saved but not published: {failure!r}")
        return outcome("failed", "publication_failed", error_type=type(failure).__name__)
    # A run's logical metric steps never decrease, and the update the cancel
    # interrupted (step + 1) may already have logged rollout metrics at its
    # step, so the metric is recorded at that update; its value is the saved
    # update. Recording it at ``step`` was rejected by the tracker whenever the
    # cancel landed after a rollout batch of the next update had finished.
    try:
        context.metric("train/cancel_checkpoint_step", step, step=cancelled_update)
    except Exception as metric_error:
        error.add_note(f"failed to record the cancellation checkpoint metric: {metric_error!r}")
        return outcome("saved", metric_error=repr(metric_error))
    return outcome("saved")


def _boundary_state_writer(
    writer: ControllerStateWriter | None,
    snapshot: UpdateBoundarySnapshot,
) -> Callable[[Path], None] | None:
    state = snapshot.controller_state
    if writer is None or state is None:
        return None
    bound_writer = writer
    bound_state = state
    return lambda checkpoint: bound_writer(checkpoint, bound_state)


def _write_missing_controller_state(writer: Callable[[Path], None] | None) -> Callable[[Path], None] | None:
    """Never overwrite the controller snapshot a periodic save already wrote.

    The live controller has usually advanced past that checkpoint's step, so
    rewriting it would make same-run recovery restore a later curriculum.
    """

    if writer is None:
        return None
    bound_writer = writer

    def write(checkpoint: Path) -> None:
        if not (checkpoint / CURRICULUM_SNAPSHOT_NAME).is_file():
            bound_writer(checkpoint)

    return write


def _present_fields(owner: Any, names: tuple[str, ...]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name in names:
        value = getattr(owner, name, _MISSING)
        if value is not _MISSING:
            values[name] = value
    return values


def _capture_rng() -> dict[str, Any]:
    """Copy the generator states transformers saves with a checkpoint."""

    states: dict[str, Any] = {"python": random.getstate()}
    numpy = sys.modules.get("numpy")
    if numpy is not None:
        states["numpy"] = numpy.random.get_state()
    torch = sys.modules.get("torch")
    if torch is not None:
        states["cpu"] = torch.random.get_rng_state()
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            states["cuda"] = torch.cuda.random.get_rng_state_all()
    return states


def _restore_rng(states: Mapping[str, Any]) -> None:
    if "python" in states:
        random.setstate(states["python"])
    numpy = sys.modules.get("numpy")
    if numpy is not None and "numpy" in states:
        numpy.random.set_state(states["numpy"])
    torch = sys.modules.get("torch")
    if torch is not None:
        if "cpu" in states:
            torch.random.set_rng_state(states["cpu"])
        if "cuda" in states:
            torch.cuda.random.set_rng_state_all(states["cuda"])


def _select_training_device(trainer: Any) -> None:
    """Make the worker thread read the training device's CUDA generator."""

    torch = sys.modules.get("torch")
    device = getattr(trainer.args, "device", None)
    if torch is None or getattr(device, "type", None) != "cuda" or not torch.cuda.is_available():
        return
    torch.cuda.set_device(device)


def _checkpoint_step(path: str | None) -> int | None:
    if path is None:
        return None
    suffix = Path(path).name.removeprefix("checkpoint-")
    return int(suffix) if suffix.isdigit() else None


def _incomplete_reason(checkpoint: Path, step: int) -> str | None:
    """Apply the same completeness rule as same-run interruption recovery."""

    if not checkpoint.is_dir():
        return "checkpoint directory was not written"
    missing = [
        name for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt") if not (checkpoint / name).is_file()
    ]
    if not any(checkpoint.glob("rng_state*.pth")):
        missing.append("rng_state*.pth")
    if missing:
        return "missing " + ", ".join(missing)
    try:
        saved_step = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))["global_step"]
    except (OSError, ValueError, KeyError, TypeError):
        return "trainer_state.json is unreadable"
    if saved_step != step:
        return f"trainer_state.json records global_step {saved_step!r}"
    return None


__all__ = [
    "CANCEL_CHECKPOINT_STAGING",
    "CANCEL_CHECKPOINT_TIMEOUT_SECONDS",
    "UpdateBoundary",
    "UpdateBoundarySnapshot",
    "retain_checkpoint_after_interruption",
    "save_cancellation_checkpoint",
    "update_boundary_callback_type",
    "update_boundary_trainer_type",
]
