"""The fork-only native names Posttrain records per veRL revision exist in that veRL.

``worker._FORK_NATIVE_NAME_REVISIONS`` is what the adapter trusts when it
accepts a revision for GDPO, CAPO, OLMo 3, SAMPO and the TRL-equivalent
GRPO settings. This test checks the entries
for the installed veRL version against veRL's own registries, so the table
cannot claim a name a release does not register. It needs an isolated
environment with the CarbonTeq veRL fork (see Concrete Steps in
``docs/plan/verl-vortex-port.md``) and skips elsewhere.
"""

from __future__ import annotations

from importlib.metadata import version

import pytest

torch = pytest.importorskip("torch")
core_algos = pytest.importorskip("verl.trainer.ppo.core_algos", reason="requires the CarbonTeq veRL fork")
replay_buffer = pytest.importorskip("verl.trainer.ppo.v1.replay_buffer")

from posttrain.train.backends.verl.worker import _FORK_NATIVE_NAME_REVISIONS  # noqa: E402

KL_NAMES = frozenset({"k3_unclipped"})


def _registers(name: str) -> bool:
    """Whether the installed veRL provides the native name Posttrain requests."""

    import inspect
    from dataclasses import fields

    from verl.trainer.config import algorithm

    def config_fields(config: type) -> set[str]:
        return {field.name for field in fields(config)}

    if name == "active_sampling":
        return hasattr(replay_buffer, "ActiveSamplingReplayBuffer") and hasattr(algorithm, "ActiveSamplingConfig")
    if name == "prompt_selector":
        from verl.trainer.ppo.v1 import prompt_selector

        return callable(prompt_selector.load_prompt_selector)
    if name == "trl_sampler_correction":
        return {"rollout_is_clip_min", "rollout_is_log_ratio_bound"} <= config_fields(algorithm.RolloutCorrectionConfig)
    if name == "grpo_scaling":
        parameters = inspect.signature(core_algos.compute_grpo_outcome_advantage).parameters
        return {"grpo_std_epsilon", "grpo_std_scope"} <= config_fields(algorithm.AlgoConfig) and {
            "std_scope",
            "trl_statistics",
        } <= set(parameters)
    if name == "row_exclusion":
        return "exclude_flagged_rows" in config_fields(algorithm.AlgoConfig)
    if name == "admission_retries":
        return "failed_group_attempts" in inspect.signature(replay_buffer.ReplayBuffer).parameters
    if name == "linear_lr":
        from verl.utils import torch_functional

        return callable(torch_functional.get_linear_schedule_with_warmup)
    if name == "candidate_batches":
        return "candidate_batches" in config_fields(algorithm.FilterGroupsConfig) and hasattr(
            replay_buffer, "CandidateBatchReplayBuffer"
        )
    if name in KL_NAMES:
        logprob = torch.tensor([-0.5, -1.0])
        reference = torch.tensor([-0.4, -2.0])
        return bool(torch.isfinite(core_algos.kl_penalty(logprob, reference, name)).all())
    return callable(core_algos.get_policy_loss_fn(name))


def test_installed_verl_registers_every_name_recorded_for_its_version() -> None:
    installed = version("verl")
    entries = [names for release, names in _FORK_NATIVE_NAME_REVISIONS.values() if release == installed]
    if not entries:
        pytest.skip(f"no fork-native-name record for installed veRL {installed}")
    missing = sorted(name for name in frozenset().union(*entries) if not _registers(name))
    assert not missing, f"veRL {installed} lacks {missing}"
