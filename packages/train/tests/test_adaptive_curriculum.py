from __future__ import annotations

import json
from collections import defaultdict
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
    task_classes: Mapping[str, str] | None = None,
    restored_state: Mapping[str, object] | None = None,
) -> AdaptiveCurriculumController:
    return AdaptiveCurriculumController(
        task_classes or {**{f"a{i}": "a" for i in range(1, 7)}, **{f"b{i}": "b" for i in range(1, 7)}},
        AdaptiveCurriculum(
            "domain",
            class_exploration=0.2,
            task_discovery=0.2,
            history_groups=history_groups,
            seed=7,
        ),
        backend,
        restored_state=restored_state,
    )


def test_controller_starts_equal_without_reading_hidden_task_difficulty() -> None:
    backend = RecordingBackend()
    controller = _controller(backend)

    initial = controller.select(8, step=1)

    assert initial.class_probabilities == {"a": 0.5, "b": 0.5}
    assert len(initial.task_ids) == len(set(initial.task_ids)) == 8
    assert initial.discovery_reserved == 1
    assert initial.discovery_fulfilled == 1
    assert initial.new_tasks_selected == 8
    assert set(initial.selection_reasons) == {"reserved_discovery", "additional_discovery"}


def test_low_and_high_reward_have_less_predicted_contrast_than_mixed_reward() -> None:
    controller = _controller(
        RecordingBackend(),
        task_classes={"a1": "low", "a2": "high", "b1": "mixed"},
    )
    controller.observe(
        [
            ("a1", [0.0, 0.0, 0.0, 0.0]),
            ("a2", [1.0, 1.0, 1.0, 1.0]),
            ("b1", [0.0, 1.0, 0.0, 1.0]),
        ],
        step=1,
    )

    assert controller.task_reward("a1") == 0.0
    assert controller.task_reward("a2") == 1.0
    assert controller.task_reward("b1") == 0.5
    assert controller.task_signal("a1") == 0.0
    assert controller.task_signal("a2") == 0.0
    assert controller.task_signal("b1") == 0.25
    assert controller.task_priority("b1") > controller.task_priority("a1")
    assert controller.task_priority("b1") > controller.task_priority("a2")


def test_representative_class_evidence_guides_unseen_tasks_without_marking_them_solved() -> None:
    controller = _controller(RecordingBackend())
    controller.observe(
        [
            *((f"a{i}", [1.0, 1.0, 1.0, 1.0]) for i in range(1, 5)),
            *((f"b{i}", [0.0, 1.0, 0.0, 1.0]) for i in range(1, 5)),
        ],
        step=1,
    )

    assert controller.task_reward("a5") is None
    assert controller.task_reward("b5") is None
    assert controller.task_priority("b5") > controller.task_priority("a5")


def test_current_step_memory_avoids_duplicates_then_degrades_without_failing() -> None:
    controller = _controller(
        RecordingBackend(),
        task_classes={"a1": "a", "a2": "a", "b1": "b", "b2": "b"},
    )

    first = controller.select(2, step=1, selection_kind="active_sampling_refill", round_index=1)
    second = controller.select(2, step=1, selection_kind="active_sampling_refill", round_index=2)
    exhausted = controller.select(2, step=1, selection_kind="active_sampling_refill", round_index=3)

    assert len(set(first.task_ids + second.task_ids)) == 4
    assert first.duplicate_fallbacks == second.duplicate_fallbacks == 0
    assert exhausted.duplicate_fallbacks == 2
    assert set(exhausted.selection_reasons) == {"duplicate_fallback"}
    next_step = controller.select(2, step=2)
    assert next_step.duplicate_fallbacks == 0


def test_duplicate_observations_are_recorded_without_failing() -> None:
    controller = _controller(RecordingBackend(), task_classes={"a1": "a"})

    observation = controller.observe(
        [("a1", [0.0, 1.0]), ("a1", [1.0, 1.0])],
        step=1,
    )

    assert observation.observed_groups == 2
    assert observation.invalid_groups == 0
    assert controller.task_reward("a1") == pytest.approx(0.75)


def test_cumulative_discovery_reserve_survives_small_requests() -> None:
    controller = _controller(RecordingBackend())
    controller.select(5, step=1)

    decisions = [controller.select(1, step=2) for _ in range(5)]

    assert [decision.discovery_reserved for decision in decisions] == [0, 0, 0, 0, 1]
    assert decisions[-1].discovery_shortfall == 0
    assert "reserved_discovery" in decisions[-1].selection_reasons


def test_fresh_mixed_evidence_changes_later_selection_probabilities() -> None:
    controller = _controller(RecordingBackend())
    controller.select(8, step=1)

    observation = controller.observe(
        [("a1", [0.0, 1.0, 0.0, 1.0]), ("b1", [0.0, 0.0, 0.0, 0.0])],
        step=1,
    )
    assert observation.task_signals == {"a1": 0.25, "b1": 0.0}
    assert observation.class_signals == {"a": 0.25, "b": 0.0}

    adapted = controller.select(4, step=2)
    assert adapted.task_probabilities
    assert all(count == 1 for count in adapted.selected_tasks.values())


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

    out_of_range = controller.observe([("a1", [-1.0, 2.0])], step=1)
    assert out_of_range.invalid_groups == 1
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


def test_snapshot_restore_preserves_current_step_exclusions() -> None:
    first = _controller(RecordingBackend())
    selected = first.select(5, step=1)

    restored = _controller(RecordingBackend(), restored_state=first.state())
    refill = restored.select(5, step=1, selection_kind="active_sampling_refill", round_index=2)

    assert set(selected.task_ids).isdisjoint(refill.task_ids)
    assert refill.duplicate_fallbacks == 0


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
    with pytest.raises(ValueError, match="class exploration"):
        AdaptiveCurriculum("domain", class_exploration=0.0)
    with pytest.raises(ValueError, match="task discovery"):
        AdaptiveCurriculum("domain", task_discovery=-0.1)
    with pytest.raises(ValueError, match="history groups"):
        AdaptiveCurriculum("domain", history_groups=0)


def test_legacy_exploration_decodes_to_both_new_controls() -> None:
    legacy = AdaptiveCurriculum("domain", exploration=0.3)

    assert legacy.class_exploration == 0.3
    assert legacy.task_discovery == 0.3
    with pytest.raises(ValueError, match="cannot be combined"):
        AdaptiveCurriculum("domain", class_exploration=0.4, exploration=0.3)


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
                "class_exploration": 0.25,
                "task_discovery": 0.3,
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
        class_exploration=0.25,
        task_discovery=0.3,
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
        {"example_id": "a3", "domain": "a", "prompt": "a3"},
        {"example_id": "a4", "domain": "a", "prompt": "a4"},
        {"example_id": "b1", "domain": "b", "prompt": "b1"},
        {"example_id": "b2", "domain": "b", "prompt": "b2"},
        {"example_id": "b3", "domain": "b", "prompt": "b3"},
        {"example_id": "b4", "domain": "b", "prompt": "b4"},
    ]
    return AdaptiveCurriculumRuntime(
        context,  # type: ignore[arg-type]
        rows,
        AdaptiveCurriculum(
            "domain",
            class_exploration=0.2,
            task_discovery=0.2,
            history_groups=2,
            seed=7,
        ),
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
        decisions = [event for event in context.events if event[0] == "adaptive_curriculum_allocation_selected"]
        assert decisions[-1][1]["selection_kind"] == "initial_batch"
        assert decisions[-1][1]["round_index"] is None
    finally:
        runtime.close()


def test_olmo_active_sampling_selects_each_refill_from_fresh_evidence(tmp_path: Path) -> None:
    import torch

    context = EventContext()
    runtime = _runtime(tmp_path, context)

    class Accelerator:
        num_processes = 1
        device = torch.device("cpu")

        @staticmethod
        def gather(value: object) -> object:
            if isinstance(value, torch.Tensor) and value.ndim == 0:
                return value.reshape(1)
            return value

    class Parent:
        def __init__(self) -> None:
            self.model = SimpleNamespace(training=True)
            self.accelerator = Accelerator()
            self.active_sampling = True
            self.active_sampling_max_batches = 3
            self.active_sampling_reward_std_epsilon = 0.0
            self.num_generations = 2
            self.state = SimpleNamespace(global_step=0)
            self.reward_weights = [1.0]
            self._metrics = {"train": defaultdict(list)}
            self.generated_task_ids: list[tuple[str, ...]] = []
            self.reward_calls = 0

        def _calculate_rewards(
            self,
            inputs: list[dict[str, object]],
            prompts: list[object],
            completions: list[object],
            completion_ids_list: list[list[int]],
        ) -> list[list[float]]:
            del prompts, completions, completion_ids_list
            self.reward_calls += 1
            rewards: list[list[float]] = []
            for offset in range(0, len(inputs), self.num_generations):
                # Retain one group in the first round, then retain the refill.
                varied = self.reward_calls > 1 or offset == 0
                rewards.extend([[0.0], [1.0]] if varied else [[0.0], [0.0]])
            return rewards

        def _generate_and_score_completions(
            self,
            inputs: list[dict[str, object]],
        ) -> dict[str, object]:
            self.generated_task_ids.append(
                tuple(str(inputs[offset]["example_id"]) for offset in range(0, len(inputs), self.num_generations))
            )
            rewards = self._calculate_rewards(inputs, [], [], [])
            group_std: list[float] = []
            for offset in range(0, len(rewards), self.num_generations):
                values = torch.tensor([row[0] for row in rewards[offset : offset + self.num_generations]])
                group_std.extend([float(values.std(unbiased=False))] * self.num_generations)
            return {
                "completion_ids": torch.arange(len(inputs)).reshape(-1, 1),
                "completion_mask": torch.ones((len(inputs), 1), dtype=torch.bool),
                "group_reward_std": torch.tensor(group_std),
                "example_id": [row["example_id"] for row in inputs],
                "domain": [row["domain"] for row in inputs],
            }

        @staticmethod
        def _select_dynamic_sampling_rows(batch: dict[str, object], keep: object) -> dict[str, object]:
            assert isinstance(keep, torch.Tensor)
            indices = keep.nonzero(as_tuple=False).flatten().tolist()
            selected: dict[str, object] = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    selected[key] = value if value.ndim == 0 else value[keep]
                elif isinstance(value, list):
                    selected[key] = [value[index] for index in indices]
                else:
                    selected[key] = value
            return selected

        @staticmethod
        def _concatenate_dynamic_sampling_batches(
            batches: list[dict[str, object]],
        ) -> dict[str, object]:
            result: dict[str, object] = {}
            for key in batches[0]:
                values = [batch[key] for batch in batches]
                if all(isinstance(value, torch.Tensor) for value in values):
                    result[key] = torch.cat(values)  # type: ignore[arg-type]
                elif all(isinstance(value, list) for value in values):
                    result[key] = [item for value in values for item in value]  # type: ignore[union-attr]
                else:
                    result[key] = values[0]
            return result

    try:
        trainer = adaptive_curriculum_trainer_type(Parent, runtime)()
        # The scheduled rows define capacity only. Task identities must be chosen lazily per refill.
        candidate_capacity = [{"example_id": "ignored"}] * 12
        retained = trainer._prepare_active_sampling_inputs(candidate_capacity)

        generated_task_ids = [task_id for refill in trainer.generated_task_ids for task_id in refill]
        assert len(generated_task_ids) == len(set(generated_task_ids)) == 3
        assert len(retained["completion_ids"]) == 4
        assert trainer._metrics["train"]["active_sampling/generation_rounds"] == [2]
        assert trainer._metrics["train"]["active_sampling/generated_rows"] == [6]
        decisions = [event[1] for event in context.events if event[0] == "adaptive_curriculum_allocation_selected"]
        assert [decision["selection_kind"] for decision in decisions] == [
            "active_sampling_refill",
            "active_sampling_refill",
        ]
        assert [decision["round_index"] for decision in decisions] == [1, 2]
        event_names = [name for name, _ in context.events]
        first_decision = event_names.index("adaptive_curriculum_allocation_selected")
        observation = event_names.index("adaptive_curriculum_evidence_observed", first_decision + 1)
        second_decision = event_names.index("adaptive_curriculum_allocation_selected", observation + 1)
        assert first_decision < observation < second_decision
        assert decisions[0]["duplicate_fallbacks"] == 0
        assert decisions[1]["duplicate_fallbacks"] == 0
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
