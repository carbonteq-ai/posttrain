"""Native streaming reward fields shared with veRL V1 group filtering."""

from __future__ import annotations

from ...online_rl import EnvironmentRollout
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
            raise RuntimeError("SAMPO requires replacement for truncated trajectories whose policy tokens are masked")
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
