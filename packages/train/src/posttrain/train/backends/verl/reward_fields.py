"""Native streaming reward fields shared with veRL V1 group filtering."""

from __future__ import annotations

import math

from ...online_rl import EnvironmentRollout
from ...profiles import shape_rollout_reward
from ...reward_evidence import InvalidRewardEvidence


def structured_reward_metadata(
    rollout: EnvironmentRollout, *, component_names: tuple[str, ...], require_process: bool
) -> dict[str, object]:
    """Transport validated raw evidence, with process credit in sampled-token order.

    The receiving trainer scatters against its own response mask after repadding;
    it must never interpret original trace offsets as padded tensor positions.
    """
    evidence = rollout.reward_evidence
    if evidence is None or rollout.is_truncated:
        raise InvalidRewardEvidence("structured rewards require complete nontruncated evidence")
    values = evidence.require_components(component_names)
    result: dict[str, object] = {
        "prompt_group_id": evidence.prompt_group_id,
        "rollout_id": evidence.rollout_id,
        "trace_id": evidence.trace_id,
        "projection_id": evidence.projection_id,
        "components": dict(zip(component_names, values, strict=True)),
    }
    if require_process:
        if evidence.process is None:
            raise InvalidRewardEvidence("CAPO requires retained process evidence")
        errors = evidence.process.error_mask(rollout.env_mask)
        result["process_error_mask"] = [
            value for value, sampled in zip(errors, rollout.env_mask, strict=True) if sampled
        ]
    return result


def shaped_rollout_reward(
    rollout: EnvironmentRollout,
    *,
    max_completion_tokens: int,
    overlong_buffer_tokens: int | None,
    overlong_penalty_factor: float | None,
    truncation_penalty: float | None,
) -> float:
    """The reward veRL groups, filters and trains on: the TRL path's shaping rule."""

    if overlong_buffer_tokens is not None and overlong_penalty_factor is None:
        raise ValueError("veRL DAPO overlong shaping requires a penalty factor")
    return shape_rollout_reward(
        rollout.reward,
        len(rollout.completion_ids),
        is_truncated=rollout.is_truncated,
        max_completion_tokens=max_completion_tokens,
        overlong_buffer_tokens=overlong_buffer_tokens,
        overlong_penalty_factor=overlong_penalty_factor if overlong_penalty_factor is not None else 1.0,
        truncation_penalty=truncation_penalty,
    )


def streaming_reward_extra_info(
    *,
    task_reward: float,
    algorithm_reward: float,
    excluded: bool = False,
) -> dict[str, float]:
    """Pre-batch metrics read by veRL's group filters, active sampling and prompt selector.

    ``group_reward`` is the value group-spread checks use: NaN for a trajectory TRL
    excludes from group statistics (a masked truncated completion), as TRL's NaN
    reward does. ``seq_reward`` stays finite and is what the curriculum observes.
    """

    return {
        "seq_reward": algorithm_reward,
        "task_reward": task_reward,
        "group_reward": math.nan if excluded else algorithm_reward,
    }


__all__ = ["shaped_rollout_reward", "streaming_reward_extra_info"]
