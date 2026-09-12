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

_STATE_VERSION = 3
_CLASS_PRIOR_STRENGTH = 2.0


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
    task_priorities: Mapping[str, float]
    selected_classes: Mapping[str, int]
    selected_tasks: Mapping[str, int]
    selection_kind: str = "initial_batch"
    round_index: int | None = None
    discovery_reserved: int = 0
    discovery_fulfilled: int = 0
    new_tasks_selected: int = 0
    discovery_shortfall: int = 0
    duplicate_fallbacks: int = 0
    selection_reasons: tuple[str, ...] = ()
    selection_components: tuple[str, ...] = ()
    selection_probabilities: tuple[float, ...] = ()

    def as_record(self) -> dict[str, object]:
        return {
            "type": "decision",
            "version": _STATE_VERSION,
            "decision_index": self.index,
            "step": self.step,
            "task_ids": list(self.task_ids),
            "class_probabilities": dict(self.class_probabilities),
            "task_probabilities": dict(self.task_probabilities),
            "task_priorities": dict(self.task_priorities),
            "selected_classes": dict(self.selected_classes),
            "selected_tasks": dict(self.selected_tasks),
            "selection_kind": self.selection_kind,
            "round_index": self.round_index,
            "discovery_reserved": self.discovery_reserved,
            "discovery_fulfilled": self.discovery_fulfilled,
            "new_tasks_selected": self.new_tasks_selected,
            "discovery_shortfall": self.discovery_shortfall,
            "duplicate_fallbacks": self.duplicate_fallbacks,
            "selection_reasons": list(self.selection_reasons),
            "selection_components": list(self.selection_components),
            "selection_probabilities": list(self.selection_probabilities),
        }


@dataclass(frozen=True, slots=True)
class CurriculumObservation:
    step: int
    observed_groups: int
    invalid_groups: int
    task_signals: Mapping[str, float | None]
    class_signals: Mapping[str, float | None]
    task_rewards: Mapping[str, float]
    task_variances: Mapping[str, float]

    def as_record(self) -> dict[str, object]:
        return {
            "type": "observation",
            "version": _STATE_VERSION,
            "step": self.step,
            "observed_groups": self.observed_groups,
            "invalid_groups": self.invalid_groups,
            "task_signals": dict(self.task_signals),
            "class_signals": dict(self.class_signals),
            "task_rewards": dict(self.task_rewards),
            "task_variances": dict(self.task_variances),
        }


@dataclass(frozen=True, slots=True)
class _Evidence:
    step: int
    reward_sum: float
    count: int
    mean: float
    variance: float

    def as_record(self) -> dict[str, object]:
        return {
            "step": self.step,
            "reward_sum": self.reward_sum,
            "count": self.count,
            "mean": self.mean,
            "variance": self.variance,
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
    """Choose classes and tasks from observed reward and coverage evidence."""

    def __init__(
        self,
        task_classes: Mapping[str, str],
        settings: AdaptiveCurriculum,
        backend: CurriculumStateBackend,
        *,
        group_size: int = 4,
        restored_state: Mapping[str, object] | None = None,
    ) -> None:
        if not task_classes:
            raise ValueError("adaptive curriculum requires at least one task")
        normalized = {str(task_id): str(class_id) for task_id, class_id in task_classes.items()}
        if any(not task_id or not class_id for task_id, class_id in normalized.items()):
            raise ValueError("adaptive curriculum task and class identities must be non-empty")
        if group_size < 2:
            raise ValueError("adaptive curriculum group size must be at least two")
        self.settings = settings
        self.backend = backend
        self.group_size = group_size
        self.task_classes = dict(sorted(normalized.items()))
        grouped: defaultdict[str, list[str]] = defaultdict(list)
        for task_id, class_id in self.task_classes.items():
            grouped[class_id].append(task_id)
        self.tasks_by_class = {class_id: tuple(sorted(task_ids)) for class_id, task_ids in sorted(grouped.items())}
        self._history = {
            task_id: deque(maxlen=settings.history_groups) for task_id in self.task_classes
        }
        self._first_evidence: dict[str, _Evidence] = {}
        self._seen: set[str] = set()
        self._last_selected = {task_id: -1 for task_id in self.task_classes}
        self._active_step: int | None = None
        self._step_selected: set[str] = set()
        self._candidate_count = 0
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
        if self._active_step is None or step > self._active_step:
            self._active_step = step
            self._step_selected.clear()
        elif step < self._active_step:
            raise ValueError("adaptive curriculum step cannot move backwards")

        rng = random.Random(_decision_seed(self.settings.seed, self._decision_index))
        discovery_reserved = _cumulative_quota(
            self._candidate_count,
            group_count,
            self.settings.task_discovery,
        )
        selected: list[str] = []
        reasons: list[str] = []
        components: list[str] = []
        selection_probabilities: list[float] = []
        task_probabilities: dict[str, float] = {}
        task_priorities: dict[str, float] = {}
        class_probability_sums = {class_id: 0.0 for class_id in self.tasks_by_class}
        duplicate_fallbacks = 0
        reserved_fulfilled = 0
        new_selected = 0
        for offset in range(group_count):
            wants_discovery = offset < discovery_reserved
            task_id, probability, class_probabilities, reason, component, repeated = self._select_one(
                wants_discovery=wants_discovery,
                rng=rng,
            )
            was_seen = task_id in self._seen
            selected.append(task_id)
            reasons.append(reason)
            components.append(component)
            selection_probabilities.append(probability)
            task_probabilities[task_id] = probability
            task_priorities[task_id] = self.task_priority(task_id)
            for class_id, value in class_probabilities.items():
                class_probability_sums[class_id] += value
            if repeated:
                duplicate_fallbacks += 1
            if not was_seen:
                new_selected += 1
                self._seen.add(task_id)
                if wants_discovery:
                    reserved_fulfilled += 1
            self._step_selected.add(task_id)
            self._last_selected[task_id] = self._candidate_count
            self._candidate_count += 1

        class_probabilities = {
            class_id: value / group_count for class_id, value in class_probability_sums.items()
        }
        selected_classes = _counts(self.task_classes[task_id] for task_id in selected)
        selected_tasks = _counts(selected)
        decision = CurriculumDecision(
            index=self._decision_index,
            step=step,
            task_ids=tuple(selected),
            class_probabilities=class_probabilities,
            task_probabilities=task_probabilities,
            task_priorities=task_priorities,
            selected_classes=selected_classes,
            selected_tasks=selected_tasks,
            selection_kind=selection_kind,
            round_index=round_index,
            discovery_reserved=discovery_reserved,
            discovery_fulfilled=reserved_fulfilled,
            new_tasks_selected=new_selected,
            discovery_shortfall=discovery_reserved - reserved_fulfilled,
            duplicate_fallbacks=duplicate_fallbacks,
            selection_reasons=tuple(reasons),
            selection_components=tuple(components),
            selection_probabilities=tuple(selection_probabilities),
        )
        self._decision_index += 1
        self.backend.append(decision.as_record())
        return decision

    def _select_one(
        self,
        *,
        wants_discovery: bool,
        rng: random.Random,
    ) -> tuple[str, float, dict[str, float], str, str, bool]:
        unseen = [task_id for task_id in self.task_classes if task_id not in self._seen]
        familiar = [task_id for task_id in self.task_classes if task_id in self._seen]
        unseen_available = [task_id for task_id in unseen if task_id not in self._step_selected]
        familiar_available = [task_id for task_id in familiar if task_id not in self._step_selected]
        if wants_discovery and unseen_available:
            pool = unseen_available
            reason = "reserved_discovery"
        elif familiar_available:
            pool = familiar_available
            reason = "adaptive_practice"
        elif unseen_available:
            pool = unseen_available
            reason = "additional_discovery"
        else:
            pool = familiar or unseen or list(self.task_classes)
            reason = "duplicate_fallback"

        eligible = [task_id for task_id in pool if task_id not in self._step_selected]
        repeated = False
        if not eligible:
            eligible = pool
            repeated = True
            reason = "duplicate_fallback"

        tasks_by_class = {
            class_id: [task_id for task_id in eligible if self.task_classes[task_id] == class_id]
            for class_id in self.tasks_by_class
        }
        tasks_by_class = {class_id: task_ids for class_id, task_ids in tasks_by_class.items() if task_ids}
        class_ids = tuple(tasks_by_class)
        base = {class_id: 1.0 / len(class_ids) for class_id in class_ids}
        class_scores = {
            class_id: (
                self._class_discovery_priority(class_id)
                if reason in {"reserved_discovery", "additional_discovery"}
                else sum(self.task_priority(task_id) for task_id in task_ids) / len(task_ids)
            )
            for class_id, task_ids in tasks_by_class.items()
        }
        adaptive = _normalized_scores(class_ids, class_scores)
        use_coverage = rng.random() < self.settings.class_exploration or adaptive is None
        if use_coverage:
            route_class_probabilities = base
            component = "coverage"
        else:
            assert adaptive is not None
            route_class_probabilities = adaptive
            component = "adaptive"
        if adaptive is None:
            class_probabilities = base
        else:
            class_probabilities = {
                class_id: self.settings.class_exploration * base[class_id]
                + (1 - self.settings.class_exploration) * adaptive[class_id]
                for class_id in class_ids
            }
        class_id = _draw(route_class_probabilities, rng)
        class_tasks = tasks_by_class[class_id]

        if reason in {"reserved_discovery", "additional_discovery"}:
            route_within = {task_id: 1.0 / len(class_tasks) for task_id in class_tasks}
            task_id = _draw(route_within, rng)
            probability = class_probabilities[class_id] * route_within[task_id]
        else:
            oldest = min(self._last_selected[task_id] for task_id in class_tasks)
            choices = [task_id for task_id in class_tasks if self._last_selected[task_id] == oldest]
            reassessment = {task_id: 1.0 / len(choices) for task_id in choices}
            adaptive_within = _normalized_scores(
                class_tasks,
                {task_id: self.task_priority(task_id) for task_id in class_tasks},
            ) or {task_id: 1.0 / len(class_tasks) for task_id in class_tasks}
            route_within = reassessment if use_coverage else adaptive_within
            task_id = _draw(route_within, rng)
            if adaptive is None:
                probability = base[class_id] * reassessment.get(task_id, 0.0)
            else:
                probability = (
                    self.settings.class_exploration
                    * base[class_id]
                    * reassessment.get(task_id, 0.0)
                    + (1 - self.settings.class_exploration)
                    * adaptive[class_id]
                    * adaptive_within.get(task_id, 0.0)
                )
        if use_coverage and reason == "adaptive_practice":
            reason = "reassessment"
        return task_id, probability, class_probabilities, reason, component, repeated

    def observe(
        self,
        groups: Sequence[tuple[str, Sequence[float]]],
        *,
        step: int,
    ) -> CurriculumObservation:
        invalid = 0
        observed_task_ids: set[str] = set()
        task_rewards: dict[str, float] = {}
        task_variances: dict[str, float] = {}
        for task_id, rewards in groups:
            if task_id not in self.task_classes:
                raise ValueError(f"adaptive curriculum observed unknown task {task_id!r}")
            finite = [float(value) for value in rewards if math.isfinite(float(value))]
            if len(finite) < 2 or any(value < 0 or value > 1 for value in finite):
                invalid += 1
                continue
            mean = sum(finite) / len(finite)
            variance = sum((value - mean) ** 2 for value in finite) / len(finite)
            evidence = _Evidence(
                step=step,
                reward_sum=sum(finite),
                count=len(finite),
                mean=mean,
                variance=variance,
            )
            self._history[task_id].append(evidence)
            self._first_evidence.setdefault(task_id, evidence)
            observed_task_ids.add(task_id)
            task_rewards[task_id] = mean
            task_variances[task_id] = variance
        task_signals = {task_id: self.task_signal(task_id) for task_id in sorted(observed_task_ids)}
        observation = CurriculumObservation(
            step=step,
            observed_groups=len(groups) - invalid,
            invalid_groups=invalid,
            task_signals=task_signals,
            class_signals=self.class_signals(),
            task_rewards=task_rewards,
            task_variances=task_variances,
        )
        self.backend.append(observation.as_record())
        return observation

    def task_signal(self, task_id: str) -> float | None:
        history = self._history[task_id]
        return sum(item.variance for item in history) / len(history) if history else None

    def task_reward(self, task_id: str) -> float | None:
        history = self._history[task_id]
        return sum(item.mean for item in history) / len(history) if history else None

    def task_priority(self, task_id: str) -> float:
        alpha, beta = self._class_prior(self.task_classes[task_id], omit=task_id)
        for evidence in self._history[task_id]:
            useful = float(evidence.variance > 0)
            alpha += useful
            beta += 1 - useful
        return alpha / (alpha + beta)

    def _class_discovery_priority(self, class_id: str) -> float:
        alpha, beta = self._class_prior(class_id)
        return alpha / (alpha + beta)

    def _class_prior(self, class_id: str, *, omit: str | None = None) -> tuple[float, float]:
        useful_groups = 0
        observed_groups = 0
        for task_id in self.tasks_by_class[class_id]:
            if task_id == omit or task_id not in self._first_evidence:
                continue
            evidence = self._first_evidence[task_id]
            observed_groups += 1
            useful_groups += int(evidence.variance > 0)
        class_rate = (1 + useful_groups) / (2 + observed_groups)
        return (
            class_rate * _CLASS_PRIOR_STRENGTH,
            (1 - class_rate) * _CLASS_PRIOR_STRENGTH,
        )

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
            "group_size": self.group_size,
            "candidate_count": self._candidate_count,
            "seen": sorted(self._seen),
            "active_step": self._active_step,
            "step_selected": sorted(self._step_selected),
            "last_selected": dict(self._last_selected),
            "history": {
                task_id: [item.as_record() for item in values]
                for task_id, values in self._history.items()
            },
            "first_evidence": {
                task_id: evidence.as_record() for task_id, evidence in self._first_evidence.items()
            },
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
            "class_exploration": self.settings.class_exploration,
            "task_discovery": self.settings.task_discovery,
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
        if state.get("group_size") != self.group_size:
            raise ValueError("adaptive curriculum snapshot group size does not match")
        decision_index = state.get("decision_index")
        candidate_count = state.get("candidate_count")
        history = state.get("history")
        seen = state.get("seen")
        active_step = state.get("active_step")
        step_selected = state.get("step_selected")
        last_selected = state.get("last_selected")
        first_evidence = state.get("first_evidence")
        if (
            not isinstance(decision_index, int)
            or decision_index < 0
            or not isinstance(candidate_count, int)
            or candidate_count < 0
            or not isinstance(history, dict)
            or not isinstance(seen, list)
            or (active_step is not None and (not isinstance(active_step, int) or active_step < 1))
            or not isinstance(step_selected, list)
            or not isinstance(last_selected, dict)
            or not isinstance(first_evidence, dict)
        ):
            raise ValueError("adaptive curriculum snapshot state is malformed")
        if any(task_id not in self.task_classes for task_id in [*seen, *step_selected]):
            raise ValueError("adaptive curriculum snapshot contains an unknown task")
        for task_id, values in history.items():
            if task_id not in self._history or not isinstance(values, list):
                raise ValueError("adaptive curriculum snapshot history is malformed")
            self._history[task_id].extend(
                _decode_evidence(value) for value in values[-self.settings.history_groups :]
            )
        for task_id, value in first_evidence.items():
            if task_id not in self.task_classes:
                raise ValueError("adaptive curriculum snapshot first evidence is malformed")
            self._first_evidence[task_id] = _decode_evidence(value)
        for task_id, value in last_selected.items():
            if task_id not in self._last_selected or not isinstance(value, int):
                raise ValueError("adaptive curriculum snapshot selection history is malformed")
            self._last_selected[task_id] = value
        self._decision_index = decision_index
        self._candidate_count = candidate_count
        self._seen = {str(task_id) for task_id in seen}
        self._active_step = active_step
        self._step_selected = {str(task_id) for task_id in step_selected}


def _normalized_scores(
    identities: Sequence[str],
    scores: Mapping[str, float],
) -> dict[str, float] | None:
    positive = {identity: max(float(scores.get(identity, 0.0)), 0.0) for identity in identities}
    total = sum(positive.values())
    if total <= 0:
        return None
    return {identity: positive[identity] / total for identity in identities}


def _draw(probabilities: Mapping[str, float], rng: random.Random) -> str:
    threshold = rng.random()
    last = tuple(probabilities)[-1]
    for identity, probability in probabilities.items():
        threshold -= probability
        if threshold <= 0:
            return identity
    return last


def _cumulative_quota(previous: int, requested: int, fraction: float) -> int:
    return math.floor((previous + requested) * fraction) - math.floor(previous * fraction)


def _decode_evidence(value: object) -> _Evidence:
    if not isinstance(value, dict):
        raise ValueError("adaptive curriculum snapshot evidence is malformed")
    try:
        step = value["step"]
        count = value["count"]
        reward_sum = float(value["reward_sum"])
        mean = float(value["mean"])
        variance = float(value["variance"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("adaptive curriculum snapshot evidence is malformed") from error
    if (
        not isinstance(step, int)
        or step < 1
        or not isinstance(count, int)
        or count < 2
        or any(not math.isfinite(number) for number in (reward_sum, mean, variance))
        or not 0 <= reward_sum <= count
        or not 0 <= mean <= 1
        or variance < 0
    ):
        raise ValueError("adaptive curriculum snapshot evidence is malformed")
    return _Evidence(step=step, reward_sum=reward_sum, count=count, mean=mean, variance=variance)


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
