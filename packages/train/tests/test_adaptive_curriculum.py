from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest
from posttrain.common import CatalogRef
from posttrain.train import AdaptiveCurriculum
from posttrain.train.adaptive_curriculum import (
    AdaptiveCurriculumController,
    QueuedJsonlCurriculumStateBackend,
)
from posttrain.train.backends.trl.policy_curriculum import (
    AdaptiveCurriculumRuntime,
    adaptive_curriculum_trainer_type,
)
from posttrain.train.catalog_schema import decode_training_selection
from posttrain.train.profiles import GRPOSettings


class RecordingBackend:
    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []
        self.snapshots: dict[Path, dict[str, object]] = {}
        self.closed = False

    def append(self, record: Mapping[str, object]) -> None:
        self.records.append(dict(record))

    def flush(self) -> None:
        pass

    def snapshot(self, path: Path, state: Mapping[str, object]) -> None:
        self.snapshots[path] = dict(state)

    def close(self) -> None:
        self.closed = True


def _controller(
    backend: RecordingBackend,
    *,
    history_groups: int = 2,
    restored_state: Mapping[str, object] | None = None,
) -> AdaptiveCurriculumController:
    return AdaptiveCurriculumController(
        {"a1": "a", "a2": "a", "b1": "b", "b2": "b"},
        AdaptiveCurriculum("domain", exploration=0.2, history_groups=history_groups, seed=7),
        backend,
        restored_state=restored_state,
    )


def test_controller_starts_equal_then_prioritizes_observed_variance() -> None:
    backend = RecordingBackend()
    controller = _controller(backend)

    initial = controller.select(10, step=1)
    assert initial.class_probabilities == {"a": 0.5, "b": 0.5}
    assert initial.selected_classes == {"a": 5, "b": 5}

    observation = controller.observe(
        [("a1", [0.0, 1.0, 0.0, 1.0]), ("b1", [0.0, 0.0, 0.0, 0.0])],
        step=1,
    )
    assert observation.task_signals == {"a1": 0.25, "b1": 0.0}
    assert observation.class_signals == {"a": 0.25, "b": 0.0}

    adapted = controller.select(10, step=2)
    assert adapted.class_probabilities == pytest.approx({"a": 0.9, "b": 0.1})
    assert adapted.selected_classes == {"a": 9, "b": 1}
    assert adapted.selected_tasks["a1"] == 8
    assert adapted.selected_tasks["a2"] == 1
    assert adapted.selected_tasks.get("b1", 0) + adapted.selected_tasks.get("b2", 0) == 1


def test_all_zero_signal_returns_to_equal_base_allocation() -> None:
    controller = _controller(RecordingBackend())
    controller.observe(
        [
            ("a1", [0.0, 0.0]),
            ("a2", [1.0, 1.0]),
            ("b1", [0.0, 0.0]),
            ("b2", [1.0, 1.0]),
        ],
        step=1,
    )

    decision = controller.select(8, step=2)

    assert decision.class_probabilities == {"a": 0.5, "b": 0.5}
    assert decision.task_probabilities == {"a1": 0.25, "a2": 0.25, "b1": 0.25, "b2": 0.25}
    assert decision.selected_tasks == {"a1": 2, "a2": 2, "b1": 2, "b2": 2}


def test_history_advances_only_when_a_task_is_observed() -> None:
    controller = _controller(RecordingBackend(), history_groups=2)
    controller.observe([("a1", [0.0, 1.0])], step=1)
    controller.observe([("a1", [0.0, 0.0])], step=4)
    assert controller.task_signal("a1") == pytest.approx(0.125)
    assert controller.task_signal("a2") is None

    controller.observe([("a1", [0.0, 1.0])], step=9)
    assert controller.task_signal("a1") == pytest.approx(0.125)
    assert controller.task_signal("a2") is None


def test_invalid_group_does_not_become_zero_signal() -> None:
    controller = _controller(RecordingBackend())

    observation = controller.observe([("a1", [float("nan")])], step=1)

    assert observation.observed_groups == 0
    assert observation.invalid_groups == 1
    assert observation.task_signals == {}
    assert controller.task_signal("a1") is None


def test_snapshot_restore_reproduces_the_next_decision() -> None:
    first_backend = RecordingBackend()
    first = _controller(first_backend)
    first.select(7, step=1)
    first.observe([("a1", [0.0, 1.0]), ("b1", [0.0, 0.0])], step=1)
    state = first.state()

    expected = first.select(11, step=2)
    restored = _controller(RecordingBackend(), restored_state=state)
    actual = restored.select(11, step=2)

    assert actual == expected


def test_queued_file_backend_preserves_order_and_writes_atomic_snapshot(tmp_path: Path) -> None:
    journal = tmp_path / "controller" / "state.jsonl"
    snapshot = tmp_path / "checkpoint-10" / "adaptive-curriculum-state.json"
    backend = QueuedJsonlCurriculumStateBackend(journal, queue_size=2)

    backend.append({"sequence": 1})
    backend.append({"sequence": 2})
    backend.snapshot(snapshot, {"version": 1, "decision_index": 2})
    backend.append({"sequence": 3})
    backend.close()

    records = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    assert [record["sequence"] for record in records] == [1, 2, 3]
    assert QueuedJsonlCurriculumStateBackend.read_snapshot(snapshot) == {
        "version": 1,
        "decision_index": 2,
    }
    assert not tuple(snapshot.parent.glob("*.tmp"))


def test_adaptive_curriculum_settings_validate_small_public_surface() -> None:
    with pytest.raises(ValueError, match="class field"):
        AdaptiveCurriculum("")
    with pytest.raises(ValueError, match="exploration"):
        AdaptiveCurriculum("domain", exploration=0.0)
    with pytest.raises(ValueError, match="history groups"):
        AdaptiveCurriculum("domain", history_groups=0)


def test_catalog_decodes_adaptive_curriculum_independently_of_algorithm() -> None:
    settings = decode_training_selection(
        CatalogRef("training", "tests/adaptive-grpo"),
        {
            "selection_type": "grpo-settings",
            "id": "tests/adaptive-grpo",
            "loop": {"max_steps": 1, "per_device_batch_size": 2},
            "num_prompts_per_step": 1,
            "num_generations": 2,
            "adaptive_curriculum": {
                "class_field": "domain",
                "exploration": 0.25,
                "history_groups": 3,
                "seed": 19,
            },
        },
        {},
    )

    assert isinstance(settings, GRPOSettings)
    assert settings.algorithm == "grpo"
    assert settings.adaptive_curriculum == AdaptiveCurriculum(
        "domain",
        exploration=0.25,
        history_groups=3,
        seed=19,
    )


class EventContext:
    def __init__(self) -> None:
        self.events: list[tuple[str, Mapping[str, object]]] = []

    def event(self, name: str, attributes: Mapping[str, object]) -> None:
        self.events.append((name, attributes))


def _runtime(tmp_path: Path, context: EventContext) -> AdaptiveCurriculumRuntime:
    rows = [
        {"example_id": "a1", "domain": "a", "prompt": "a1"},
        {"example_id": "a2", "domain": "a", "prompt": "a2"},
        {"example_id": "b1", "domain": "b", "prompt": "b1"},
        {"example_id": "b2", "domain": "b", "prompt": "b2"},
    ]
    return AdaptiveCurriculumRuntime(
        context,  # type: ignore[arg-type]
        rows,
        AdaptiveCurriculum("domain", exploration=0.2, history_groups=2, seed=7),
        num_generations=2,
        state_dir=tmp_path / "state",
        resume_checkpoint=None,
    )


def test_trainer_composition_selects_before_generation_and_observes_raw_rewards(tmp_path: Path) -> None:
    context = EventContext()
    runtime = _runtime(tmp_path, context)

    class Parent:
        def __init__(self) -> None:
            self.model = SimpleNamespace(training=True)
            self.accelerator = SimpleNamespace(num_processes=1)
            self.args = SimpleNamespace(steps_per_generation=1)
            self.num_iterations = 1
            self._step = 0
            self._buffered_inputs = None
            self.state = SimpleNamespace(global_step=0)
            self.reward_weights = [1.0]

        def _prepare_inputs(self, generation_batch: list[dict[str, object]]) -> dict[str, object]:
            self.prepared = generation_batch
            return {"prepared": generation_batch}

        def _calculate_rewards(
            self,
            inputs: list[dict[str, object]],
            prompts: list[object],
            completions: list[object],
            completion_ids_list: list[list[int]],
        ) -> list[list[float]]:
            del prompts, completions, completion_ids_list
            rewards: list[list[float]] = []
            for index, row in enumerate(inputs):
                rewards.append([float(index % 2)] if row["domain"] == "a" else [0.0])
            return rewards

    try:
        trainer = adaptive_curriculum_trainer_type(Parent, runtime)()
        scheduled = [{"example_id": "ignored"}] * 8
        trainer._prepare_inputs(scheduled)
        selected = trainer.prepared
        assert len(selected) == 8
        assert all(
            selected[index]["example_id"] == selected[index + 1]["example_id"]
            for index in range(0, len(selected), 2)
        )

        trainer._calculate_rewards(selected, [], [], [])
        evidence_events = [event for event in context.events if event[0] == "adaptive_curriculum_evidence_observed"]
        assert evidence_events[-1][1]["observed_groups"] == 4
        assert runtime.controller.class_signals() == {"a": 0.25, "b": 0.0}
    finally:
        runtime.close()


def test_runtime_rejects_missing_class_metadata(tmp_path: Path) -> None:
    context = EventContext()
    with pytest.raises(ValueError, match="class field 'domain'"):
        AdaptiveCurriculumRuntime(
            context,  # type: ignore[arg-type]
            [{"example_id": "task-1", "prompt": "hello"}],
            AdaptiveCurriculum("domain"),
            num_generations=2,
            state_dir=tmp_path / "state",
            resume_checkpoint=None,
        )

