"""Native streaming reward fields shared with veRL V1 group filtering."""

from __future__ import annotations


def training_response_mask(
    env_mask: tuple[bool, ...],
    *,
    is_truncated: bool,
    mask_truncated_completions: bool,
    requires_complete_group: bool,
) -> list[int]:
    """Resolve the trainable mask or reject a trajectory that needs replacement."""

    if is_truncated and mask_truncated_completions:
        if requires_complete_group:
            raise RuntimeError(
                "SAMPO requires replacement for truncated trajectories whose policy tokens are masked"
            )
        return [0] * len(env_mask)
    return [int(value) for value in env_mask]


def streaming_reward_extra_info(
    *,
    task_reward: float,
    algorithm_reward: float,
) -> dict[str, float]:
    """Expose the pre-batch metric used by bounded dynamic group sampling."""

    return {
        "seq_reward": algorithm_reward,
        "task_reward": task_reward,
    }


__all__ = ["streaming_reward_extra_info", "training_response_mask"]
