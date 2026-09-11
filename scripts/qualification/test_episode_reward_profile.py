import pytest
from episode_reward_profile import EPISODE_COMPONENT_WEIGHTS, episode_component_weights


def test_priorities_are_name_aligned_and_completion_led():
    names = tuple(reversed(tuple(EPISODE_COMPONENT_WEIGHTS)[1:]))
    weights = episode_component_weights(names)
    assert weights[0] == 0.55
    assert sum(weights) == pytest.approx(1.0)
    assert dict(zip(("partial_credit", *names), weights, strict=True)) == EPISODE_COMPONENT_WEIGHTS
    assert weights[0] > max(weights[1:])


@pytest.mark.parametrize("names", [(), ("unknown",), tuple(EPISODE_COMPONENT_WEIGHTS)])
def test_mismatched_rubrics_fail_before_training(names):
    with pytest.raises(ValueError, match="rubric names differ"):
        episode_component_weights(names)
