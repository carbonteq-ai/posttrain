"""Contracts for the simple yield-first soft-routing replay."""
import sys
from collections import defaultdict
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from controller_policy_experiment import NullBackend  # noqa: E402
from posttrain.train import AdaptiveCurriculum  # noqa: E402
from posttrain.train.adaptive_curriculum import AdaptiveCurriculumController  # noqa: E402
from vortex_next import Sampler, Settings, World  # noqa: E402
from vortex_next import run as run_previous
from vortex_simple import run_simple, score_tasks  # noqa: E402


def test_soft_routing_starts_class_balanced_and_never_gates_familiar_tasks():
    inventory = {'a': 'small', 'b': 'large', 'c': 'large'}
    controller = AdaptiveCurriculumController(
        inventory, AdaptiveCurriculum('class', seed=0), NullBackend(), group_size=4
    )
    sampler = Sampler(inventory, Settings(stability_penalty=0, uncertainty_weight=0.2), binary=True)
    scores = score_tasks(controller, sampler, 1)
    for explore in (False, True):
        assert sampler.probabilities(list(inventory), scores, explore) == pytest.approx({
            'a': 0.5, 'b': 0.25, 'c': 0.25,
        })
    controller.observe([('a', [0, 0, 1, 1])], step=1)
    sampler.observe('a', [0, 0, 1, 1], 1)
    probabilities = sampler.probabilities(list(inventory), score_tasks(controller, sampler, 2), True)
    assert all(probability > 0 for probability in probabilities.values())


def test_simple_replay_is_deterministic_and_preserves_admission_rule():
    groups = [
        {'task_id': str(index), 'class': str(index % 4), 'rewards': [0, 0, 1, 1],
         'input_tokens': 100, 'output_tokens': 20}
        for index in range(40)
    ]
    settings = Settings(stability_penalty=0, uncertainty_weight=0.2)
    first = run_simple(groups, 'soft_yield_explore', 3, settings=settings)
    second = run_simple(groups, 'soft_yield_explore', 3, settings=settings)
    assert first == second
    assert first['completed_steps'] == 100
    assert first['candidates'] == first['retained'] == 800
    assert first['explore_selected'] + first['variance_selected'] == 800


def test_simple_policy_rejects_unknown_variant():
    with pytest.raises(ValueError, match='unsupported simple policy'):
        run_simple([], 'unknown', 1)


@pytest.mark.parametrize('policy', ['current_controller', 'binary_soft20'])
def test_baseline_selection_matches_previous_replay(policy):
    groups = [
        {'task_id': str(index), 'class': str(index % 4),
         'rewards': [0, 0, 1, 1] if index % 3 else [0, 0, 0, 0],
         'input_tokens': 100, 'output_tokens': 20}
        for index in range(40)
    ]
    settings = Settings(stability_penalty=0, uncertainty_weight=1)
    previous = run_previous(groups, policy, 3, settings=settings)
    simple = run_simple(groups, policy, 3, settings=settings)
    for metric in ('completed_steps', 'candidates', 'retained', 'covered_tasks',
                   'candidates_per_retained', 'zero_variance_fraction', 'new_fraction'):
        assert simple[metric] == previous[metric]


@pytest.mark.parametrize('policy', [
    'current_controller', 'binary_soft20', 'soft_yield', 'soft_yield_explore', 'production_yield_first',
])
def test_no_task_repeats_within_step_across_refill_rounds(policy, monkeypatch):
    groups = [
        {'task_id': str(index), 'class': str(index % 4), 'rewards': [0, 0, 0, 0],
         'input_tokens': 100, 'output_tokens': 20}
        for index in range(40)
    ]
    seen = defaultdict(set)
    original_group = World.group

    def assert_distinct(self, task, step):
        assert task not in seen[step], f'duplicate task {task} in step {step}'
        seen[step].add(task)
        return original_group(self, task, step)

    monkeypatch.setattr(World, 'group', assert_distinct)
    run_simple(groups, policy, 3, settings=Settings(stability_penalty=0, uncertainty_weight=0.2))
    assert max(map(len, seen.values())) == 32
    assert len(set().union(*seen.values())) < sum(map(len, seen.values()))  # later-step reuse is valid


def test_soft_selector_reports_shortfall_when_distinct_pool_cannot_fill_refill():
    groups = [
        {'task_id': str(index), 'class': 'x', 'rewards': [0, 0, 0, 0],
         'input_tokens': 100, 'output_tokens': 20}
        for index in range(10)
    ]
    result = run_simple(groups, 'soft_yield_explore', 3,
                        settings=Settings(stability_penalty=0, uncertainty_weight=0.2))
    assert result['completed_steps'] == 0
    assert result['candidates'] == 800  # one distinct eight-task round per step, then shortfall
    assert all(item['rounds'] == 1 for item in result['per_step'])
