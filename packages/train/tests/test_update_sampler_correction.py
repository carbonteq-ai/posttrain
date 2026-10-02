"""Independent correction references, separate from native execution admission."""
import math
from collections.abc import MutableMapping
from typing import cast

import pytest
from posttrain.train.profiles import CAPOSettings, GDPOSettings, GRPOSettings, SAMPOSettings, TrainingLoop
from posttrain.train.update_records import ActionRef, InvalidPolicyUpdate
from posttrain.train.update_sampler_correction import recipe_sampler_correction_weights, sampler_correction_weights

from .test_update_plan import five_turns


def scores():
    snapshot, _ = five_turns()
    actions = [record.action for record in snapshot.actions]
    old = dict.fromkeys(actions, 0.0)
    sampled = {a: -math.log(w) for a, w in zip(actions, (2., .5, 1., 3., .25), strict=True)}
    return snapshot, old, sampled


@pytest.mark.parametrize(('mode', 'expected'), [
    ('token_truncate', [2., .5, 1., 2., .5]),
    ('token_mask', [2., .5, 1., 0., 0.]),
    # A's three ratios multiply to1; B's two multiply to.75.
    ('sequence_truncate', [1., 1., 1., .75, .75]),
    ('sequence_mask', [1., 1., 1., .75, .75]),
])
def test_complete_episode_and_token_reference(mode, expected):
    snapshot, old, sampled = scores()
    actual = sampler_correction_weights(snapshot, old, sampled, mode=mode, lower=.5, upper=2.)
    assert list(actual.values()) == pytest.approx(expected)
    # Turn selection or packing consumes these fixed weights, not a new reduction.
    assert actual[snapshot.actions[3].action] == pytest.approx(expected[3])
    with pytest.raises(TypeError):
        cast(MutableMapping[ActionRef, float], actual)[snapshot.actions[0].action] = 0


@pytest.mark.parametrize('mode', ['token_truncate', 'sequence_truncate', 'token_mask', 'sequence_mask'])
def test_large_log_ratio_uses_selected_bounds_without_overflow(mode):
    snapshot, old, sampled = scores()
    sampled = dict.fromkeys(sampled, -1001.)
    actual = sampler_correction_weights(snapshot, old, sampled, mode=mode, lower=None, upper=2.)
    assert set(actual.values()) == ({2.} if mode.endswith('truncate') else {0.})


def test_missing_extra_or_nonfinite_scores_reject_before_weights():
    snapshot, old, sampled = scores()
    action = snapshot.actions[0].action
    for invalid in ({a:v for a,v in sampled.items() if a != action}, {**sampled, action: math.nan},
                    {**sampled, action: True}, {**sampled, action: "0"}):
        with pytest.raises(InvalidPolicyUpdate):
            sampler_correction_weights(snapshot, old, invalid, mode='token_truncate', lower=None, upper=2.)
    with pytest.raises(InvalidPolicyUpdate, match='uncapped'):
        sampler_correction_weights(snapshot, old, dict.fromkeys(sampled, -1001.),
                                   mode='token_truncate', lower=None, upper=None)


@pytest.mark.parametrize('bounds', [(0., 2.), (1., 1.), (2., 1.), (None, math.inf), (False, 2.)])
def test_invalid_bounds_reject(bounds):
    snapshot, old, sampled = scores()
    with pytest.raises(InvalidPolicyUpdate, match='bounds'):
        sampler_correction_weights(snapshot, old, sampled, mode='token_truncate', lower=bounds[0], upper=bounds[1])


@pytest.mark.parametrize('kind,expected', [
    ('grpo', [1., 1., 1., .75, .75]),
    ('sampo', [2., .5, 1., 2., .25]),
    ('gdpo', [2., .5, 1., 3., .25]),
    ('capo', [2., .5, 1., 3., .25]),
])
def test_selected_recipe_defaults_are_shared_by_backends(kind, expected):
    args = {'id': 'correction-test', 'loop': TrainingLoop(max_steps=1, per_device_batch_size=2)}
    if kind == 'gdpo':
        selected = GDPOSettings(**args, component_names=('outcome',), component_weights=(1.,))
    elif kind == 'capo':
        selected = CAPOSettings(**args)
    elif kind == 'sampo':
        selected = SAMPOSettings(**args)
    else:
        selected = GRPOSettings(**args)
    snapshot, old, sampled = scores()
    actual = recipe_sampler_correction_weights(selected, snapshot, old, sampled)
    assert list(actual.values()) == pytest.approx(expected)
