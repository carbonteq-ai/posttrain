"""Task-neutral adaptive curriculum state and queued file persistence."""

from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import random
import threading
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from .profiles import AdaptiveCurriculum

_STATE_VERSION = 1


class CurriculumStateBackend(Protocol):
    """Persistence boundary for ordered controller records and snapshots."""

    def append(self, record: Mapping[str, object]) -> None: ...

    def flush(self) -> None: ...

    def snapshot(self, path: Path, state: Mapping[str, object]) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class CurriculumDecision:
    index: int
    step: int
    task_ids: tuple[str, ...]
    class_probabilities: Mapping[str, float]
    task_probabilities: Mapping[str, float]
    selected_classes: Mapping[str, int]
    selected_tasks: Mapping[str, int]
    selection_kind: str = "initial_batch"
    round_index: int | None = None

    def as_record(self) -> dict[str, object]:
        return {
            "type": "decision",
            "version": _STATE_VERSION,
            "decision_index": self.index,
            "step": self.step,
            "task_ids": list(self.task_ids),
            "class_probabilities": dict(self.class_probabilities),
            "task_probabilities": dict(self.task_probabilities),
            "selected_classes": dict(self.selected_classes),
            "selected_tasks": dict(self.selected_tasks),
            "selection_kind": self.selection_kind,
            "round_index": self.round_index,
        }


@dataclass(frozen=True, slots=True)
class CurriculumObservation:
    step: int
    observed_groups: int
    invalid_groups: int
    task_signals: Mapping[str, float | None]
    class_signals: Mapping[str, float | None]

    def as_record(self) -> dict[str, object]:
        return {
            "type": "observation",
            "version": _STATE_VERSION,
            "step": self.step,
            "observed_groups": self.observed_groups,
            "invalid_groups": self.invalid_groups,
            "task_signals": dict(self.task_signals),
            "class_signals": dict(self.class_signals),
        }


@dataclass(frozen=True, slots=True)
class _Write:
    text: str


@dataclass(frozen=True, slots=True)
class _Flush:
    done: threading.Event


@dataclass(frozen=True, slots=True)
class _Snapshot:
    path: Path
    text: str
    done: threading.Event


class _Stop:
    pass


class QueuedJsonlCurriculumStateBackend:
    """Append state records in one ordered writer thread.

    Enqueueing is normally non-blocking. A bounded queue provides backpressure
    instead of dropping controller state when storage cannot keep up.
    """

    def __init__(self, journal_path: Path, *, queue_size: int = 256) -> None:
        if queue_size < 1:
            raise ValueError("curriculum state queue size must be positive")
        self.journal_path = journal_path.resolve()
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        self._queue: queue.Queue[_Write | _Flush | _Snapshot | _Stop] = queue.Queue(maxsize=queue_size)
        self._error: BaseException | None = None
        self._closed = False
        self._thread = threading.Thread(target=self._write_loop, name="curriculum-state-writer", daemon=True)
        self._thread.start()

    def append(self, record: Mapping[str, object]) -> None:
        self._ensure_open()
        text = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        self._queue.put(_Write(text))
        self._raise_writer_error()

    def flush(self) -> None:
        self._ensure_open()
        done = threading.Event()
        self._queue.put(_Flush(done))
        done.wait()
        self._raise_writer_error()

    def snapshot(self, path: Path, state: Mapping[str, object]) -> None:
        self._ensure_open()
        text = json.dumps(state, indent=2, sort_keys=True, allow_nan=False) + "\n"
        done = threading.Event()
        self._queue.put(_Snapshot(path.resolve(), text, done))
        done.wait()
        self._raise_writer_error()

    def close(self) -> None:
        if self._closed:
            self._raise_writer_error()
            return
        flush_error: BaseException | None = None
        try:
            self.flush()
        except BaseException as error:
            flush_error = error
        self._closed = True
        self._queue.put(_Stop())
        self._thread.join()
        if flush_error is not None:
            raise flush_error
        self._raise_writer_error()

    @staticmethod
    def read_snapshot(path: Path) -> Mapping[str, object]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("adaptive curriculum snapshot must be a JSON object")
        return cast(dict[str, object], value)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("adaptive curriculum state backend is closed")
        self._raise_writer_error()

    def _raise_writer_error(self) -> None:
        if self._error is not None:
            raise RuntimeError("adaptive curriculum state writer failed") from self._error

    def _write_loop(self) -> None:
        try:
            with self.journal_path.open("a", encoding="utf-8") as stream:
                while True:
                    item = self._queue.get()
                    try:
                        if isinstance(item, _Stop):
                            return
                        if self._error is not None:
                            continue
                        if isinstance(item, _Write):
                            stream.write(item.text)
                        elif isinstance(item, _Flush):
                            stream.flush()
                            os.fsync(stream.fileno())
                        else:
                            stream.flush()
                            os.fsync(stream.fileno())
                            item.path.parent.mkdir(parents=True, exist_ok=True)
                            temporary = item.path.with_name(f".{item.path.name}.{os.getpid()}.tmp")
                            with temporary.open("w", encoding="utf-8") as snapshot_stream:
                                snapshot_stream.write(item.text)
                                snapshot_stream.flush()
                                os.fsync(snapshot_stream.fileno())
                            os.replace(temporary, item.path)
                    except BaseException as error:
                        self._error = error
                    finally:
                        if isinstance(item, _Flush | _Snapshot):
                            item.done.set()
                        self._queue.task_done()
        except BaseException as error:
            self._error = error
            while True:
                item = self._queue.get()
                if isinstance(item, _Flush | _Snapshot):
                    item.done.set()
                self._queue.task_done()
                if isinstance(item, _Stop):
                    return


class AdaptiveCurriculumController:
    """Choose classes and tasks using recent observed reward variance."""

    def __init__(
        self,
        task_classes: Mapping[str, str],
        settings: AdaptiveCurriculum,
        backend: CurriculumStateBackend,
        *,
        restored_state: Mapping[str, object] | None = None,
    ) -> None:
        if not task_classes:
            raise ValueError("adaptive curriculum requires at least one task")
        normalized = {str(task_id): str(class_id) for task_id, class_id in task_classes.items()}
        if any(not task_id or not class_id for task_id, class_id in normalized.items()):
            raise ValueError("adaptive curriculum task and class identities must be non-empty")
        self.settings = settings
        self.backend = backend
        self.task_classes = dict(sorted(normalized.items()))
        grouped: defaultdict[str, list[str]] = defaultdict(list)
        for task_id, class_id in self.task_classes.items():
            grouped[class_id].append(task_id)
        self.tasks_by_class = {class_id: tuple(sorted(task_ids)) for class_id, task_ids in sorted(grouped.items())}
        self._history = {
            task_id: deque(maxlen=settings.history_groups) for task_id in self.task_classes
        }
        self._decision_index = 0
        self._inventory_digest = _inventory_digest(self.task_classes)
        if restored_state is not None:
            self._restore(restored_state)
        self.backend.append(
            {
                "type": "controller_started",
                "version": _STATE_VERSION,
                "inventory_digest": self._inventory_digest,
                "task_count": len(self.task_classes),
                "class_count": len(self.tasks_by_class),
                "settings": self._settings_record(),
                "restored": restored_state is not None,
                "decision_index": self._decision_index,
            }
        )

    @property
    def decision_index(self) -> int:
        return self._decision_index

    def select(
        self,
        group_count: int,
        *,
        step: int,
        selection_kind: str = "initial_batch",
        round_index: int | None = None,
    ) -> CurriculumDecision:
        if group_count < 1:
            raise ValueError("adaptive curriculum group count must be positive")
        class_signals = self.class_signals()
        class_ids = tuple(self.tasks_by_class)
        class_probabilities = _mixed_probabilities(class_ids, class_signals, self.settings.exploration)
        rng = random.Random(_decision_seed(self.settings.seed, self._decision_index))
        class_counts = _apportion(class_probabilities, group_count, rng)

        selected: list[str] = []
        task_probabilities: dict[str, float] = {}
        for class_id in class_ids:
            task_ids = self.tasks_by_class[class_id]
            task_signals = {task_id: self.task_signal(task_id) for task_id in task_ids}
            within_class = _mixed_probabilities(task_ids, task_signals, self.settings.exploration)
            task_probabilities.update(
                {task_id: class_probabilities[class_id] * probability for task_id, probability in within_class.items()}
            )
            counts = _apportion(within_class, class_counts[class_id], rng)
            for task_id, count in counts.items():
                selected.extend([task_id] * count)
        rng.shuffle(selected)
        selected_classes = _counts(self.task_classes[task_id] for task_id in selected)
        selected_tasks = _counts(selected)
        decision = CurriculumDecision(
            index=self._decision_index,
            step=step,
            task_ids=tuple(selected),
            class_probabilities=class_probabilities,
            task_probabilities=task_probabilities,
            selected_classes=selected_classes,
            selected_tasks=selected_tasks,
            selection_kind=selection_kind,
            round_index=round_index,
        )
        self._decision_index += 1
        self.backend.append(decision.as_record())
        return decision

    def observe(
        self,
        groups: Sequence[tuple[str, Sequence[float]]],
        *,
        step: int,
    ) -> CurriculumObservation:
        invalid = 0
        observed_task_ids: set[str] = set()
        for task_id, rewards in groups:
            if task_id not in self.task_classes:
                raise ValueError(f"adaptive curriculum observed unknown task {task_id!r}")
            finite = [float(value) for value in rewards if math.isfinite(float(value))]
            if len(finite) < 2:
                invalid += 1
                continue
            mean = sum(finite) / len(finite)
            variance = sum((value - mean) ** 2 for value in finite) / len(finite)
            self._history[task_id].append(variance)
            observed_task_ids.add(task_id)
        task_signals = {task_id: self.task_signal(task_id) for task_id in sorted(observed_task_ids)}
        observation = CurriculumObservation(
            step=step,
            observed_groups=len(groups) - invalid,
            invalid_groups=invalid,
            task_signals=task_signals,
            class_signals=self.class_signals(),
        )
        self.backend.append(observation.as_record())
        return observation

    def task_signal(self, task_id: str) -> float | None:
        history = self._history[task_id]
        return sum(history) / len(history) if history else None

    def class_signals(self) -> dict[str, float | None]:
        result: dict[str, float | None] = {}
        for class_id, task_ids in self.tasks_by_class.items():
            values = [signal for task_id in task_ids if (signal := self.task_signal(task_id)) is not None]
            result[class_id] = sum(values) / len(values) if values else None
        return result

    def state(self) -> dict[str, object]:
        return {
            "version": _STATE_VERSION,
            "inventory_digest": self._inventory_digest,
            "settings": self._settings_record(),
            "decision_index": self._decision_index,
            "history": {task_id: list(values) for task_id, values in self._history.items()},
        }

    def snapshot(self, path: Path) -> None:
        self.backend.snapshot(path, self.state())

    def flush(self) -> None:
        self.backend.flush()

    def close(self) -> None:
        self.backend.close()

    def _settings_record(self) -> dict[str, object]:
        return {
            "class_field": self.settings.class_field,
            "exploration": self.settings.exploration,
            "history_groups": self.settings.history_groups,
            "seed": self.settings.seed,
        }

    def _restore(self, state: Mapping[str, object]) -> None:
        if state.get("version") != _STATE_VERSION:
            raise ValueError("adaptive curriculum snapshot version does not match")
        if state.get("inventory_digest") != self._inventory_digest:
            raise ValueError("adaptive curriculum snapshot inventory does not match")
        if state.get("settings") != self._settings_record():
            raise ValueError("adaptive curriculum snapshot settings do not match")
        decision_index = state.get("decision_index")
        history = state.get("history")
        if not isinstance(decision_index, int) or decision_index < 0 or not isinstance(history, dict):
            raise ValueError("adaptive curriculum snapshot state is malformed")
        for task_id, values in history.items():
            if task_id not in self._history or not isinstance(values, list):
                raise ValueError("adaptive curriculum snapshot history is malformed")
            numeric = [float(value) for value in values]
            if any(not math.isfinite(value) or value < 0 for value in numeric):
                raise ValueError("adaptive curriculum snapshot contains an invalid signal")
            self._history[task_id].extend(numeric[-self.settings.history_groups :])
        self._decision_index = decision_index


def _mixed_probabilities(
    identities: Sequence[str],
    signals: Mapping[str, float | None],
    exploration: float,
) -> dict[str, float]:
    if not identities:
        return {}
    base = 1.0 / len(identities)
    priorities = {identity: max(float(signals.get(identity) or 0.0), 0.0) for identity in identities}
    total = sum(priorities.values())
    if total <= 0:
        return {identity: base for identity in identities}
    return {
        identity: exploration * base + (1 - exploration) * priorities[identity] / total
        for identity in identities
    }


def _apportion(probabilities: Mapping[str, float], total: int, rng: random.Random) -> dict[str, int]:
    if total < 0:
        raise ValueError("allocation total cannot be negative")
    counts = {identity: math.floor(total * probability) for identity, probability in probabilities.items()}
    remaining = total - sum(counts.values())
    remainders = defaultdict(list)
    for identity, probability in probabilities.items():
        remainders[total * probability - counts[identity]].append(identity)
    ordered: list[str] = []
    for remainder in sorted(remainders, reverse=True):
        ties = remainders[remainder]
        rng.shuffle(ties)
        ordered.extend(ties)
    for identity in ordered[:remaining]:
        counts[identity] += 1
    return counts


def _counts(values: Iterable[str]) -> dict[str, int]:
    result: defaultdict[str, int] = defaultdict(int)
    for value in values:
        result[str(value)] += 1
    return dict(sorted(result.items()))


def _decision_seed(seed: int, decision_index: int) -> int:
    digest = hashlib.sha256(f"{seed}:{decision_index}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _inventory_digest(task_classes: Mapping[str, str]) -> str:
    payload = json.dumps(task_classes, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "AdaptiveCurriculumController",
    "CurriculumDecision",
    "CurriculumObservation",
    "CurriculumStateBackend",
    "QueuedJsonlCurriculumStateBackend",
]
