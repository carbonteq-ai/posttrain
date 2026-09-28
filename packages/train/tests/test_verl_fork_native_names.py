"""The fork-only native names Posttrain records per veRL revision exist in that veRL.

``worker._FORK_NATIVE_NAME_REVISIONS`` is what the adapter trusts when it
accepts a revision for GDPO, CAPO and OLMo 3. This test checks the entries
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


def test_installed_verl_registers_every_name_recorded_for_its_version() -> None:
    installed = version("verl")
    entries = [names for release, names in _FORK_NATIVE_NAME_REVISIONS.values() if release == installed]
    if not entries:
        pytest.skip(f"no fork-native-name record for installed veRL {installed}")
    logprob = torch.tensor([-0.5, -1.0])
    reference = torch.tensor([-0.4, -2.0])
    for name in frozenset().union(*entries):
        if name == "active_sampling":
            assert hasattr(replay_buffer, "ActiveSamplingReplayBuffer")
        elif name in KL_NAMES:
            assert torch.isfinite(core_algos.kl_penalty(logprob, reference, name)).all()
        else:
            assert callable(core_algos.get_policy_loss_fn(name))
