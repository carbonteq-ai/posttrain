from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from posttrain.common import CatalogRef
from posttrain.train import AdaptiveCurriculum
from posttrain.train.adaptive_curriculum import (
    AdaptiveCurriculumController,
    CurriculumCapacityError,
    QueuedJsonlCurriculumStateBackend,
)
from posttrain.train.backends.trl.policy_curriculum import (
    AdaptiveCurriculumRuntime,
    _native_algorithm_reward_rows,
    _native_group_reward_std,
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


def test_native_algorithm_rewards_reach_trl_as_scalar_rows() -> None:
    import torch

    class Accelerator:
        device = torch.device("cpu")

        @staticmethod
        def gather(value: object) -> object:
            return value

    trainer = SimpleNamespace(accelerator=Accelerator(), reward_funcs=[object()])

    rewards = _native_algorithm_reward_rows(
        trainer,
        [
            {"algorithm_reward": 0.0},
            {"algorithm_reward": 1.0},
            {"algorithm_reward": 1.0},
            {"algorithm_reward": 0.0},
        ],
    )

    assert isinstance(rewards, torch.Tensor)
    torch.testing.assert_close(rewards, torch.tensor([[0.0], [1.0], [1.0], [0.0]]))
    assert rewards.std(unbiased=False).item() == pytest.approx(0.5)


def test_native_algorithm_rewards_reject_partial_population() -> None:
    import torch

    trainer = SimpleNamespace(
        accelerator=SimpleNamespace(device=torch.device("cpu")),
        reward_funcs=[object()],
    )

    with pytest.raises(RuntimeError, match="every completion"):
        _native_algorithm_reward_rows(trainer, [{"algorithm_reward": 0.0}, {}])


@pytest.mark.parametrize("value", [True, {"partial_credit": {"score": 1.0, "weight": 1.0}}])
def test_native_algorithm_rewards_reject_unprojected_environment_values(value: object) -> None:
    import torch

    trainer = SimpleNamespace(
        accelerator=SimpleNamespace(device=torch.device("cpu")),
        reward_funcs=[object()],
    )

    with pytest.raises(RuntimeError, match="must be scalar numbers"):
        _native_algorithm_reward_rows(trainer, [{"algorithm_reward": value}])


def test_generic_reward_path_remains_available_for_other_environments() -> None:
    import torch

    class Accelerator:
        device = torch.device("cpu")

        @staticmethod
        def gather(value: object) -> object:
            return value

    trainer = SimpleNamespace(accelerator=Accelerator(), reward_funcs=[object(), object()])

    assert _native_algorithm_reward_rows(trainer, [{"prompt": "environment-owned"}]) is None


def test_native_group_reward_std_repeats_each_prompt_group_statistic() -> None:
    import torch

    class Accelerator:
        device = torch.device("cpu")

        @staticmethod
        def gather(value: object) -> object:
            return value

    trainer = SimpleNamespace(
        accelerator=Accelerator(),
        reward_funcs=[object()],
        num_generations=4,
        mask_truncated_completions=False,
    )

    group_std = _native_group_reward_std(
        trainer,
        [{"algorithm_reward": value} for value in (0.0, 1.0, 1.0, 0.0, 0.6, 0.6, 0.6, 0.6)],
    )

    assert isinstance(group_std, torch.Tensor)
    torch.testing.assert_close(group_std, torch.tensor([0.57735026] * 4 + [0.0] * 4))


def test_native_group_reward_std_excludes_masked_truncations() -> None:
    import torch

    class Accelerator:
        device = torch.device("cpu")
        process_index = 0

        @staticmethod
        def gather(value: object) -> object:
            return value

    trainer = SimpleNamespace(
        accelerator=Accelerator(),
        reward_funcs=[object()],
        num_generations=4,
        mask_truncated_completions=True,
    )
    rows = [
        {"algorithm_reward": 0.0, "is_truncated": True},
        {"algorithm_reward": 1.0, "is_truncated": False},
        {"algorithm_reward": 1.0, "is_truncated": False},
        {"algorithm_reward": 1.0, "is_truncated": False},
    ]

    group_std = _native_group_reward_std(trainer, rows)

    assert isinstance(group_std, torch.Tensor)
    torch.testing.assert_close(group_std, torch.zeros(4))


def test_adaptive_trainer_uses_native_rewards_without_generic_rescoring() -> None:
    import torch

    class Accelerator:
        device = torch.device("cpu")

        @staticmethod
        def gather(value: object) -> object:
            return value

    class Runtime:
        observed: tuple[list[dict[str, object]], list[list[float]], list[float], int] | None = None

        def observe_rewards(
            self,
            inputs: list[dict[str, object]],
            rewards: list[list[float]],
            weights: list[float],
            *,
            step: int,
        ) -> None:
            self.observed = (inputs, rewards, weights, step)

    class Parent:
        def __init__(self) -> None:
            self.model = SimpleNamespace(training=True)
            self.accelerator = Accelerator()
            self.reward_funcs = [object()]
            self.reward_weights = torch.tensor([1.0])
            self.num_generations = 2
            self.mask_truncated_completions = False
            self.state = SimpleNamespace(global_step=2)

        def _calculate_rewards(self, *_args: object, **_kwargs: object) -> object:
            raise AssertionError("native rewards must not be sent through the generic reward callback")

    runtime = Runtime()
    trainer = adaptive_curriculum_trainer_type(Parent, cast(Any, runtime))()
    inputs = [
        {"example_id": "task", "algorithm_reward": 0.0},
        {"example_id": "task", "algorithm_reward": 1.0},
    ]

    rewards = trainer._calculate_rewards(inputs, [], [], [])

    torch.testing.assert_close(rewards, torch.tensor([[0.0], [1.0]]))
    assert runtime.observed == (inputs, [[0.0], [1.0]], [1.0], 3)


def _controller(
    backend: RecordingBackend,
    *,
    history_groups: int = 2,
    task_classes: Mapping[str, str] | None = None,
    restored_state: Mapping[str, object] | None = None,
    warm_start_state: Mapping[str, object] | None = None,
    class_exploration: float = 0.2,
    task_discovery: float = 0.2,
    policy: str = "quota",
    exploration_share: float = 0.2,
    uncertainty_weight: float = 4.0,
) -> AdaptiveCurriculumController:
    return AdaptiveCurriculumController(
        task_classes or {**{f"a{i}": "a" for i in range(1, 7)}, **{f"b{i}": "b" for i in range(1, 7)}},
        AdaptiveCurriculum(
            "domain",
            class_exploration=class_exploration,
            task_discovery=task_discovery,
            history_groups=history_groups,
            seed=7,
            policy=cast(Any, policy),
            exploration_share=exploration_share,
            uncertainty_weight=uncertainty_weight,
        ),
        backend,
        restored_state=restored_state,
        warm_start_state=warm_start_state,
    )


def test_controller_starts_equal_without_reading_hidden_task_difficulty() -> None:
    backend = RecordingBackend()
    controller = _controller(backend)

    initial = controller.select(8, step=1)

    assert initial.class_discovery_priorities["a"] == initial.class_discovery_priorities["b"]
    assert sum(initial.class_probabilities.values()) == pytest.approx(1.0)
    assert set(initial.task_priorities) == set(initial.task_ids)
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


def test_constant_fractional_reward_is_not_mistaken_for_future_variance() -> None:
    controller = _controller(
        RecordingBackend(),
        task_classes={"constant": "shared", "mixed": "shared"},
    )
    observation = controller.observe(
        [
            ("constant", [0.2, 0.2, 0.2, 0.2]),
            ("mixed", [0.0, 0.4, 0.0, 0.4]),
        ],
        step=1,
    )

    assert observation.task_rewards == {"constant": 0.2, "mixed": 0.2}
    assert observation.task_variances["constant"] == pytest.approx(0.0)
    assert observation.task_variances["mixed"] == pytest.approx(0.04)
    assert controller.task_priority("mixed") > controller.task_priority("constant")


def test_class_evidence_is_a_bounded_prior_not_a_replacement_for_task_history() -> None:
    task_classes = {"zero": "shared", "positive": "shared"}
    task_classes.update({f"class-positive-{index}": "shared" for index in range(20)})
    controller = _controller(RecordingBackend(), task_classes=task_classes)
    controller.observe(
        [
            ("zero", [0.2, 0.2, 0.2, 0.2]),
            ("positive", [0.0, 0.4, 0.0, 0.4]),
            *((f"class-positive-{index}", [0.0, 1.0, 0.0, 1.0]) for index in range(20)),
        ],
        step=1,
    )

    assert controller.task_priority("positive") > controller.task_priority("zero")


def test_class_discovery_uses_sample_uncertainty_without_counting_repeat_groups_as_tasks() -> None:
    task_classes = {
        **{f"few-{index}": "few" for index in range(5)},
        **{f"many-{index}": "many" for index in range(100)},
    }
    controller = _controller(RecordingBackend(), task_classes=task_classes)
    useful = [0.0, 1.0, 0.0, 1.0]
    constant = [1.0, 1.0, 1.0, 1.0]
    controller.observe(
        [
            *((f"few-{index}", useful if index < 2 else constant) for index in range(5)),
            *((f"many-{index}", useful if index < 2 else constant) for index in range(100)),
        ],
        step=1,
    )

    decision = controller.select(1, step=2)

    assert decision.class_discovery_priorities["few"] > decision.class_discovery_priorities["many"]


def test_recent_task_outcomes_have_more_weight_than_older_outcomes() -> None:
    older_useful = _controller(RecordingBackend(), task_classes={"task": "shared"}, history_groups=4)
    newer_useful = _controller(RecordingBackend(), task_classes={"task": "shared"}, history_groups=4)
    useful = [0.0, 1.0, 0.0, 1.0]
    constant = [1.0, 1.0, 1.0, 1.0]
    older_useful.observe([("task", useful)], step=1)
    older_useful.observe([("task", constant)], step=2)
    newer_useful.observe([("task", constant)], step=1)
    newer_useful.observe([("task", useful)], step=2)

    assert newer_useful.task_priority("task") > older_useful.task_priority("task")


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


def test_current_step_memory_fails_before_repeating_a_task() -> None:
    backend = RecordingBackend()
    controller = _controller(
        backend,
        task_classes={"a1": "a", "a2": "a", "b1": "b", "b2": "b"},
    )

    first = controller.select(2, step=1, selection_kind="active_sampling_refill", round_index=1)
    second = controller.select(2, step=1, selection_kind="active_sampling_refill", round_index=2)
    state_before = controller.state()
    records_before = len(backend.records)
    with pytest.raises(CurriculumCapacityError, match="requested=2 available=0"):
        controller.select(2, step=1, selection_kind="active_sampling_refill", round_index=3)

    assert len(set(first.task_ids + second.task_ids)) == 4
    assert first.duplicate_fallbacks == second.duplicate_fallbacks == 0
    assert controller.state() == state_before
    assert len(backend.records) == records_before
    next_step = controller.select(2, step=2)
    assert next_step.duplicate_fallbacks == 0


def test_yield_first_starts_class_balanced_and_uses_full_pool() -> None:
    controller = _controller(
        RecordingBackend(),
        task_classes={"a1": "a", "a2": "a", "a3": "a", "b1": "b"},
        policy="yield_first",
        exploration_share=1.0,
    )
    first = controller.select(1, step=1)
    second = controller.select(1, step=1, selection_kind="active_sampling_refill", round_index=2)

    assert first.policy == second.policy == "yield_first"
    assert first.class_probabilities == pytest.approx({"a": 0.5, "b": 0.5})
    assert first.selection_reasons == ("uncertainty_exploration",)
    assert set(first.task_ids).isdisjoint(second.task_ids)
    assert first.discovery_reserved == second.discovery_reserved == 0


def test_yield_first_can_fill_worst_case_vortex_refills_without_repeating_tasks() -> None:
    inventory = {f"task-{index}": f"class-{index % 7}" for index in range(160)}
    controller = _controller(RecordingBackend(), task_classes=inventory, policy="yield_first")

    selected = [
        task_id
        for round_index in range(1, 11)
        for task_id in controller.select(
            10,
            step=1,
            selection_kind="active_sampling_refill",
            round_index=round_index,
        ).task_ids
    ]

    assert len(selected) == len(set(selected)) == 100
    assert len(inventory.keys() - set(selected)) == 60


def test_yield_first_uncertainty_ages_and_resume_preserves_step_exclusions() -> None:
    inventory = {"a1": "a", "a2": "a", "a3": "a", "b1": "b"}
    controller = _controller(
        RecordingBackend(), task_classes=inventory, policy="yield_first", exploration_share=1.0
    )
    for _ in range(12):
        controller.observe([("a1", [0.0, 1.0, 0.0, 1.0])], step=1)
    current = controller._yield_uncertainties(1)
    assert current["a2"] > current["a1"]
    aged = controller._yield_uncertainties(81)
    assert aged["a1"] > current["a1"]

    first = controller.select(2, step=81)
    state = controller.state()
    expected = controller.select(2, step=81, selection_kind="active_sampling_refill", round_index=2)
    restored = _controller(
        RecordingBackend(), task_classes=inventory, policy="yield_first",
        exploration_share=1.0, restored_state=state,
    )
    actual = restored.select(2, step=81, selection_kind="active_sampling_refill", round_index=2)
    assert actual == expected
    assert set(first.task_ids).isdisjoint(actual.task_ids)
    with pytest.raises(CurriculumCapacityError, match="available=0"):
        restored.select(1, step=81, selection_kind="active_sampling_refill", round_index=3)


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


def test_cumulative_class_coverage_survives_small_requests() -> None:
    controller = _controller(
        RecordingBackend(),
        class_exploration=0.2,
        task_discovery=0.0,
    )

    decisions = [controller.select(1, step=index + 1) for index in range(5)]

    assert [decision.class_coverage_reserved for decision in decisions] == [0, 0, 0, 0, 1]
    assert decisions[-1].class_coverage_fulfilled == 1
    assert decisions[-1].selection_components == ("coverage",)


def test_discovery_fraction_is_a_floor_when_familiar_tasks_predict_no_contrast() -> None:
    controller = _controller(
        RecordingBackend(),
        task_classes={f"task-{index}": "shared" for index in range(40)},
    )
    initial = controller.select(10, step=1)
    controller.observe(
        [(task_id, [1.0, 1.0, 1.0, 1.0]) for task_id in initial.task_ids],
        step=1,
    )

    adapted = controller.select(10, step=2)

    assert adapted.discovery_reserved == 2
    assert adapted.new_tasks_selected > adapted.discovery_reserved
    assert "additional_discovery" in adapted.selection_reasons


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
        self.metric_points: list[tuple[str, float, int | None, Mapping[str, object]]] = []

    def event(self, name: str, attributes: Mapping[str, object]) -> None:
        self.events.append((name, attributes))

    def metric(
        self,
        name: str,
        value: float,
        *,
        step: int | None = None,
        attributes: Mapping[str, object] | None = None,
    ) -> None:
        self.metric_points.append((name, float(value), step, dict(attributes or {})))

    def metrics(
        self,
        values: Mapping[str, float],
        *,
        step: int | None = None,
        attributes: Mapping[str, object] | None = None,
    ) -> None:
        for name, value in values.items():
            self.metric(name, value, step=step, attributes=attributes)


def _runtime(
    tmp_path: Path,
    context: EventContext,
    *,
    policy: str = "quota",
    state_name: str = "state",
    warm_start_state_dir: Path | None = None,
) -> AdaptiveCurriculumRuntime:
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
            policy=cast(Any, policy),
        ),
        num_generations=2,
        state_dir=tmp_path / state_name,
        resume_checkpoint=None,
        warm_start_state_dir=warm_start_state_dir,
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
            self.reward_funcs = [object()]
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
            selected[index]["example_id"] == selected[index + 1]["example_id"] for index in range(0, len(selected), 2)
        )

        trainer._calculate_rewards(selected, [], [], [])
        evidence_events = [event for event in context.events if event[0] == "adaptive_curriculum_evidence_observed"]
        assert evidence_events[-1][1]["observed_groups"] == 4
        assert runtime.controller.class_signals() == {"a": 0.25, "b": 0.0}
        decisions = [event for event in context.events if event[0] == "adaptive_curriculum_allocation_selected"]
        assert decisions[-1][1]["selection_kind"] == "initial_batch"
        assert decisions[-1][1]["round_index"] is None
        metric_values = {name: value for name, value, _, _ in context.metric_points}
        assert metric_values["train/rl/curriculum/candidate_groups"] == 4
        assert metric_values["train/rl/curriculum/unique_tasks"] == 4
        assert metric_values["train/rl/curriculum/new_tasks"] == 4
        class_points = [
            (value, step, attributes["class_id"])
            for name, value, step, attributes in context.metric_points
            if name == "train/rl/curriculum/class_candidate_groups"
        ]
        assert sum(value for value, _, _ in class_points) == 4
        assert len({step for _, step, _ in class_points}) == 1
        assert {class_id for _, _, class_id in class_points} <= {"a", "b"}
    finally:
        runtime.close()


@pytest.mark.parametrize("policy", ["quota", "yield_first"])
def test_olmo_active_sampling_selects_each_refill_from_fresh_evidence(tmp_path: Path, policy: str) -> None:
    import torch

    context = EventContext()
    runtime = _runtime(tmp_path, context, policy=policy)

    class Accelerator:
        num_processes = 1
        device = torch.device("cpu")

        @staticmethod
        def gather(value: object) -> object:
            value_tensor = cast(Any, value)
            if isinstance(value, torch.Tensor) and value_tensor.ndim == 0:
                return value_tensor.reshape(1)
            return value

    class Parent:
        def __init__(self) -> None:
            self.model = SimpleNamespace(training=True)
            self.accelerator = Accelerator()
            self.active_sampling = True
            self.active_sampling_max_batches = 3
            self.active_sampling_reward_std_epsilon = 0.0
            self.num_generations = 2
            self.mask_truncated_completions = False
            self.state = SimpleNamespace(global_step=0)
            self.reward_funcs = [object()]
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
            self.reward_calls += 1
            for offset in range(0, len(inputs), self.num_generations):
                # Retain one group in the first round, then retain the refill.
                varied = self.reward_calls > 1 or offset == 0
                values = (0.0, 1.0) if varied else (0.0, 0.0)
                for row, reward in zip(inputs[offset : offset + self.num_generations], values, strict=True):
                    row["algorithm_reward"] = reward
            self._calculate_rewards(inputs, [], [], [])
            return {
                "completion_ids": torch.arange(len(inputs)).reshape(-1, 1),
                "completion_mask": torch.ones((len(inputs), 1), dtype=torch.bool),
                # Reproduce the live failure: generic TRL statistics lost the
                # native variance even though the rollout rewards were mixed.
                "group_reward_std": torch.zeros(len(inputs)),
                # Group-centred advantages: a mixed group is (-0.5, 0.5), a tied group (0, 0).
                "advantages": torch.tensor(
                    [
                        cast(float, row["algorithm_reward"]) - 0.5 if self.reward_calls > 1 or index < 2 else 0.0
                        for index, row in enumerate(inputs)
                    ]
                ),
                "example_id": [row["example_id"] for row in inputs],
                "domain": [row["domain"] for row in inputs],
            }

        @staticmethod
        def _select_dynamic_sampling_rows(batch: dict[str, object], keep: object) -> dict[str, object]:
            assert isinstance(keep, torch.Tensor)
            keep_tensor = cast(Any, keep)
            indices = keep_tensor.nonzero(as_tuple=False).flatten().tolist()
            selected: dict[str, object] = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    value_tensor = cast(Any, value)
                    selected[key] = value_tensor if value_tensor.ndim == 0 else value_tensor[keep_tensor]
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
        # Groups, not rows; and the optimized batch holds only mixed-reward groups.
        assert trainer._metrics["train"]["active_sampling/generated_groups"] == [3]
        assert trainer._metrics["train"]["active_sampling/retained_groups"] == [2]
        assert trainer._metrics["train"]["trained/frac_reward_zero_std"] == [0.0]
        assert trainer._metrics["train"]["trained/advantages_zero_fraction"] == [0.0]
        assert trainer._metrics["train"]["active_sampling/native_reward_std_max_delta"] == pytest.approx(
            [2**-0.5, 2**-0.5]
        )
        decisions = [event[1] for event in context.events if event[0] == "adaptive_curriculum_allocation_selected"]
        assert [decision["selection_kind"] for decision in decisions] == [
            "active_sampling_refill",
            "active_sampling_refill",
        ]
        assert [decision["round_index"] for decision in decisions] == [1, 2]
        assert [decision["policy"] for decision in decisions] == [policy, policy]
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


def _yield_first_source(inventory: Mapping[str, str]) -> AdaptiveCurriculumController:
    """A yield-first controller with five optimizer steps of mixed evidence."""

    controller = _controller(RecordingBackend(), task_classes=inventory, policy="yield_first", exploration_share=0.5)
    for step in range(1, 6):
        decision = controller.select(2, step=step)
        controller.observe(
            [(task_id, [0.0, 1.0, 1.0, 0.0] if index == 0 else [1.0, 1.0, 1.0, 1.0]) for index, task_id in enumerate(decision.task_ids)],
            step=step,
        )
    return controller


def test_yield_first_warm_start_keeps_task_evidence_and_resets_run_counters() -> None:
    inventory = {f"a{i}": "a" for i in range(1, 5)} | {f"b{i}": "b" for i in range(1, 5)}
    source = _yield_first_source(inventory)
    backend = RecordingBackend()
    warm = _controller(
        backend, task_classes=inventory, policy="yield_first", exploration_share=0.5, warm_start_state=source.state()
    )

    assert warm._seen == source._seen and warm._seen
    assert warm._history == source._history
    assert warm._first_evidence == source._first_evidence
    assert (warm.decision_index, warm._candidate_count, warm._active_step, warm._step_selected) == (0, 0, None, set())
    assert backend.records[0]["warm_started"] is True and backend.records[0]["restored"] is False
    # A resumed controller refuses to go back to step 1; a warm-started one starts there.
    with pytest.raises(ValueError, match="cannot move backwards"):
        _controller(
            RecordingBackend(), task_classes=inventory, policy="yield_first", exploration_share=0.5, restored_state=source.state()
        ).select(1, step=1)
    assert warm.select(2, step=1).index == 0


def test_yield_first_warm_start_continues_evidence_ageing_across_runs() -> None:
    inventory = {f"a{i}": "a" for i in range(1, 5)} | {f"b{i}": "b" for i in range(1, 5)}
    source = _yield_first_source(inventory)
    state = source.state()
    warm = _controller(RecordingBackend(), task_classes=inventory, policy="yield_first", exploration_share=0.5, warm_start_state=state)
    resumed = _controller(RecordingBackend(), task_classes=inventory, policy="yield_first", exploration_share=0.5, restored_state=state)

    # Step k of the new run carries exactly the decay of k further source steps (source ended at step 5).
    for k in (1, 7, 40):
        assert warm._yield_uncertainties(k) == pytest.approx(resumed._yield_uncertainties(5 + k))


def test_warm_start_keeps_never_selected_tasks_least_recently_used() -> None:
    inventory = {f"a{i}": "a" for i in range(1, 5)} | {f"b{i}": "b" for i in range(1, 5)}
    source = _yield_first_source(inventory)
    warm = _controller(
        RecordingBackend(), task_classes=inventory, policy="yield_first", exploration_share=0.5, warm_start_state=source.state()
    )

    never = [task_id for task_id, value in source._last_selected.items() if value < 0]
    selected = [task_id for task_id, value in source._last_selected.items() if value >= 0]
    assert never and selected
    assert all(warm._last_selected[task_id] < 0 for task_id in inventory)
    assert max(warm._last_selected[task_id] for task_id in never) < min(warm._last_selected[task_id] for task_id in selected)
    assert sorted(selected, key=warm._last_selected.__getitem__) == sorted(selected, key=source._last_selected.__getitem__)


def test_warm_start_requires_a_matching_curriculum_and_excludes_resume() -> None:
    inventory = {f"a{i}": "a" for i in range(1, 5)} | {f"b{i}": "b" for i in range(1, 5)}
    state = _yield_first_source(inventory).state()
    with pytest.raises(ValueError, match="inventory does not match"):
        _controller(RecordingBackend(), task_classes={**inventory, "c1": "c"}, policy="yield_first", exploration_share=0.5, warm_start_state=state)
    with pytest.raises(ValueError, match="settings do not match"):
        _controller(RecordingBackend(), task_classes=inventory, policy="yield_first", exploration_share=0.3, warm_start_state=state)
    with pytest.raises(ValueError, match="both resume"):
        _controller(
            RecordingBackend(), task_classes=inventory, policy="yield_first", exploration_share=0.5,
            restored_state=state, warm_start_state=state,
        )


def test_runtime_warm_starts_from_a_published_curriculum_state_directory(tmp_path: Path) -> None:
    source = _runtime(tmp_path, EventContext(), policy="yield_first", state_name="source")
    decision = source.controller.select(2, step=1)
    source.controller.observe([(task_id, [0.0, 1.0]) for task_id in decision.task_ids], step=1)
    published = source.save_final_state().parent
    source.close()

    context = EventContext()
    warm = _runtime(tmp_path, context, policy="yield_first", state_name="warm", warm_start_state_dir=published)
    started = dict(next(attributes for name, attributes in context.events if name == "adaptive_curriculum_started"))
    assert started["warm_started"] is True and started["restored"] is False
    assert warm.controller._seen == set(decision.task_ids)
    warm.close()

    with pytest.raises(RuntimeError, match="warm-start snapshot is missing"):
        _runtime(tmp_path, EventContext(), policy="yield_first", state_name="missing", warm_start_state_dir=tmp_path / "empty")

