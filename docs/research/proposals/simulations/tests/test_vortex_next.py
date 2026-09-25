"""Behavioral contracts for the research VORTEX next sampler."""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from vortex_next import Sampler, Settings, run, signal  # noqa: E402


def test_equal_classes_and_tasks_on_cold_start():
    sampler = Sampler({'a': 'small', 'b': 'large', 'c': 'large'})
    scores = sampler.scores(1)
    for explore in (False, True):
        p = sampler.probabilities(list(sampler.inventory), scores, explore)
        assert p == pytest.approx({'a': 0.5, 'b': 0.25, 'c': 0.25})


def test_exploration_converges_to_normal_when_uncertainty_vanishes():
    sampler = Sampler({'a': 'x', 'b': 'x'})
    scores = {'a': (0.7, 0), 'b': (0.3, 0)}
    assert sampler.probabilities(['a', 'b'], scores, True) == sampler.probabilities(['a', 'b'], scores, False)


def test_mean_reward_does_not_penalize_low_reward_variation():
    assert signal([0, 0, 0.2, 0.2]) == pytest.approx(signal([0.8, 0.8, 1, 1]))
    assert signal([1] * 4) == signal([0] * 4) == 0
    assert signal([0, 0, 0.001, 0.001]) < signal([0, 0, 0.2, 0.2])
    with pytest.raises(ValueError):
        signal([0, 1, 1, float('nan')])


def test_one_spike_cannot_outweigh_repeated_consistent_variation():
    sampler = Sampler({'steady': 'x', 'spike': 'x'}, Settings(half_life=1e9))
    for step in range(1, 31):
        sampler.observe('steady', [0.4, 0.4, 0.6, 0.6], step)
        sampler.observe('spike', [0, 0, 1, 1] if step == 30 else [0] * 4, step)
    scores = sampler.scores(31)
    assert scores['steady'][0] > scores['spike'][0]


def test_equal_average_signal_prefers_stable_history():
    sampler = Sampler({'steady': 'x', 'erratic': 'x'}, Settings(half_life=1e9))
    high_signal = signal([0, 0, 1, 1])
    target_signal = high_signal / 2
    variance = 0.1**2 * target_signal / (1 - target_signal)
    gap = 2 * math.sqrt(variance)
    for step in range(1, 41):
        sampler.observe('steady', [0, 0, gap, gap], step)
        sampler.observe('erratic', [0, 0, 1, 1] if step % 2 else [0] * 4, step)
    scores = sampler.scores(41)
    assert scores['steady'][0] > scores['erratic'][0]


def test_aging_restores_uncertainty_for_rechecking():
    sampler = Sampler({'mastered': 'x'})
    for _ in range(60):
        sampler.observe('mastered', [1] * 4, 1)
    recent = sampler.scores(1)['mastered']
    stale = sampler.scores(101)['mastered']
    assert stale[1] > recent[1]


def test_replay_accounting_and_seed_reproducibility():
    groups = [{'task_id': str(i), 'class': str(i % 4), 'rewards': [0, 0, 1, 1],
               'input_tokens': 100, 'output_tokens': 20} for i in range(40)]
    a = run(groups, 'vortex_next', 3)
    b = run(groups, 'vortex_next', 3)
    assert a == b
    assert a['completed_steps'] == 100
    assert a['candidates'] == a['retained'] == 800
    assert a['explore_selected'] + a['variance_selected'] == 800
