"""Scalar native credit uses complete groups before optimizer scheduling."""

import math
from dataclasses import replace

import pytest
from posttrain.train.profiles import ActiveGroupSampling, GRPOSettings
from posttrain.train.update_records import InvalidPolicyUpdate

from .test_update_evidence import native_rollout
from .test_update_resolution import native_resolution, settings


def selected(algorithm, scaling):
    base = settings()
    return GRPOSettings(id="scalar-native", algorithm=algorithm, loop=base.loop,
                        num_prompts_per_step=2, advantage_scaling=scaling, policy_updates=base.policy_updates)


@pytest.mark.parametrize("algorithm", ["grpo", "dapo"])
@pytest.mark.parametrize("scaling", ["group", "batch", "none"])
def test_scalar_credit_matches_independent_group_reference_before_minibatching(algorithm, scaling):
    # Interleaved groups have very different reward spreads; turns within each
    # episode receive the same scalar credit, unlike SAMPO's hierarchical credit.
    rollouts = (native_rollout("a", 0, 1), native_rollout("b", 0, 5),
                native_rollout("a", 1, 0), native_rollout("b", 1, 1))
    result = native_resolution(rollouts, selected(algorithm, scaling))
    batch_std = math.sqrt(14.75 / 3)
    divisors = (1, 1) if scaling == "none" else (
        (batch_std + 1e-4, batch_std + 1e-4) if scaling == "batch" else
        (math.sqrt(0.5) + 1e-4, math.sqrt(8) + 1e-4)
    )
    expected = {"episode-a-0": 0.5 / divisors[0], "episode-a-1": -0.5 / divisors[0],
                "episode-b-0": 2 / divisors[1], "episode-b-1": -2 / divisors[1]}
    for value in result.credit.values:
        assert value.advantage == pytest.approx(expected[value.action.episode_id])
    assert len(result.updates) == 4
    assert len({update.objective.credit_digest for update in result.updates}) == 1
    assert result.credit.estimator_id.startswith(f"{algorithm}-scalar-credit@1:")


def test_scalar_credit_shapes_raw_rewards_without_mutating_native_evidence():
    rollouts = (native_rollout("a", 0, 1), replace(native_rollout("a", 1, 1), is_truncated=True))
    result = native_resolution(rollouts, replace(selected("grpo", "none"), truncation_penalty=0.5))
    assert [value.advantage for value in result.credit.values] == pytest.approx([0.25] * 4 + [-0.25] * 4)
    assert rollouts[0].reward == rollouts[1].reward == 1


def test_scalar_equal_rewards_produce_finite_zero_credit_and_reject_other_recipes():
    rollouts = (native_rollout("a", 0, 1), native_rollout("a", 1, 1))
    result = native_resolution(rollouts, selected("grpo", "group"))
    assert all(value.advantage == 0 for value in result.credit.values)
    with pytest.raises(InvalidPolicyUpdate, match="supported GRPO/DAPO"):
        native_resolution(rollouts, replace(selected("grpo", "none"), algorithm="olmo3",
            clip_epsilon_low=0.2, clip_epsilon_high=0.272,
            importance_sampling_mode="token_truncate", importance_sampling_clip_min=None,
            importance_sampling_clip_max=2.0, active_sampling=ActiveGroupSampling()))


@pytest.mark.parametrize("scaling", ["group", "batch", "none"])
def test_scalar_credit_agrees_with_selected_native_verl_estimator(scaling):
    torch = pytest.importorskip("torch")
    pytest.importorskip("verl")
    import numpy as np
    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage

    rollouts = tuple(native_rollout(group, index, reward)
                     for group, index, reward in (("a", 0, 1), ("a", 1, 0), ("b", 0, 5), ("b", 1, 1)))
    result = native_resolution(rollouts, selected("grpo", scaling))
    mask = torch.tensor([row.env_mask for row in rollouts], dtype=torch.float32)
    rewards = torch.zeros_like(mask)
    rewards[:, -1] = torch.tensor([1., 0., 5., 1.])
    actual, _ = compute_grpo_outcome_advantage(rewards, mask, np.array(["a", "a", "b", "b"]),
        epsilon=1e-4, norm_adv_by_std_in_grpo=scaling != "none", std_scope="group" if scaling == "none" else scaling,
        trl_statistics=True)
    expected = torch.tensor([value.advantage for value in result.credit.values])
    torch.testing.assert_close(actual[mask.bool()], expected, rtol=1e-6, atol=1e-6)
