"""Independent correction references, separate from native execution admission."""

import math

import numpy as np
import pytest
from posttrain.train.profiles import CAPOSettings, GDPOSettings, GRPOSettings, SAMPOSettings, TrainingLoop
from posttrain.train.update_records import ConditioningView, InvalidPolicyUpdate, PolicyVersions, PopulationSnapshot
from posttrain.train.update_sampler_correction import recipe_sampler_correction_weights, sampler_correction_weights


def five_turns() -> PopulationSnapshot:
    """Two episodes: A has three one-token turns, B has two."""
    return PopulationSnapshot(
        "five-turns",
        "native:five-turns",
        "native-digest",
        tuple(
            ConditioningView(
                turn,
                "native:five-turns",
                "tokens",
                "attention",
                "positions",
                "template@1",
                turn,
                10,
                episode,
                "branch",
                (0,),
            )
            for episode, turn in (("A", "A1"), ("A", "A2"), ("A", "A3"), ("B", "B1"), ("B", "B2"))
        ),
        (),
        (),
        PolicyVersions("sampler@1", "old@1", "current@1", None),
        "selector@1",
    )


def scores():
    snapshot = five_turns()
    old = np.zeros(snapshot.size)
    sampled = -np.log(np.array([2.0, 0.5, 1.0, 3.0, 0.25]))
    return snapshot, old, sampled


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("token_truncate", [2.0, 0.5, 1.0, 2.0, 0.5]),
        ("token_mask", [2.0, 0.5, 1.0, 0.0, 0.0]),
        # A's three ratios multiply to1; B's two multiply to.75.
        ("sequence_truncate", [1.0, 1.0, 1.0, 0.75, 0.75]),
        ("sequence_mask", [1.0, 1.0, 1.0, 0.75, 0.75]),
    ],
)
def test_complete_episode_and_token_reference(mode, expected):
    snapshot, old, sampled = scores()
    actual = sampler_correction_weights(snapshot, old, sampled, mode=mode, lower=0.5, upper=2.0)
    assert actual.tolist() == pytest.approx(expected)
    # Turn selection or packing consumes these fixed weights, not a new reduction.
    assert actual[snapshot.view_positions(snapshot.view_index("B1"))].tolist() == pytest.approx([expected[3]])
    with pytest.raises(ValueError, match="read-only"):
        actual[0] = 0


@pytest.mark.parametrize("mode", ["token_truncate", "sequence_truncate", "token_mask", "sequence_mask"])
def test_large_log_ratio_uses_selected_bounds_without_overflow(mode):
    snapshot, old, sampled = scores()
    sampled = np.full_like(sampled, -1001.0)
    actual = sampler_correction_weights(snapshot, old, sampled, mode=mode, lower=None, upper=2.0)
    assert set(actual.tolist()) == ({2.0} if mode.endswith("truncate") else {0.0})


def test_missing_extra_or_nonfinite_scores_reject_before_weights():
    snapshot, old, sampled = scores()
    for invalid in (
        sampled[:-1],
        np.append(sampled, 0.0),
        np.where(np.arange(snapshot.size) == 0, math.nan, sampled),
        np.where(np.arange(snapshot.size) == 0, math.inf, sampled),
    ):
        with pytest.raises(InvalidPolicyUpdate):
            sampler_correction_weights(snapshot, old, invalid, mode="token_truncate", lower=None, upper=2.0)
    with pytest.raises(InvalidPolicyUpdate, match="uncapped"):
        sampler_correction_weights(
            snapshot, old, np.full_like(sampled, -1001.0), mode="token_truncate", lower=None, upper=None
        )


@pytest.mark.parametrize("bounds", [(0.0, 2.0), (1.0, 1.0), (2.0, 1.0), (None, math.inf), (False, 2.0)])
def test_invalid_bounds_reject(bounds):
    snapshot, old, sampled = scores()
    with pytest.raises(InvalidPolicyUpdate, match="bounds"):
        sampler_correction_weights(snapshot, old, sampled, mode="token_truncate", lower=bounds[0], upper=bounds[1])


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("grpo", [1.0, 1.0, 1.0, 0.75, 0.75]),
        ("sampo", [2.0, 0.5, 1.0, 2.0, 0.25]),
        ("gdpo", [2.0, 0.5, 1.0, 3.0, 0.25]),
        ("capo", [2.0, 0.5, 1.0, 3.0, 0.25]),
    ],
)
def test_selected_recipe_defaults_are_shared_by_backends(kind, expected):
    args = {"id": "correction-test", "loop": TrainingLoop(max_steps=1, per_device_batch_size=2)}
    if kind == "gdpo":
        selected = GDPOSettings(**args, component_names=("outcome",), component_weights=(1.0,))
    elif kind == "capo":
        selected = CAPOSettings(**args)
    elif kind == "sampo":
        selected = SAMPOSettings(**args)
    else:
        selected = GRPOSettings(**args)
    snapshot, old, sampled = scores()
    actual = recipe_sampler_correction_weights(selected, snapshot, old, sampled)
    assert actual.tolist() == pytest.approx(expected)
