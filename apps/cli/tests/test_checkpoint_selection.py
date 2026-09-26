from types import SimpleNamespace

import pytest
from posttrain_cli.commands.work_package import _select_checkpoint_output


def _link(name, digest, metadata, kind="training-checkpoint"):
    return SimpleNamespace(
        direction="output", kind=kind, name=name, artifact=SimpleNamespace(digest=digest, provider_metadata=metadata)
    )


def _select(links, step):
    return _select_checkpoint_output(
        tuple(links), source_run_id="source", kinds=frozenset({"training-checkpoint"}), step=step
    )


def test_final_checkpoint_registered_twice_at_the_same_step_selects_the_periodic_view():
    periodic = _link("checkpoint-00000020-recovery", "sha256:a", {"checkpoint_step": 20, "global_step": 20})
    final = _link("recovery-checkpoint", "sha256:a", {"global_step": 20})
    earlier = _link("checkpoint-00000010-recovery", "sha256:b", {"checkpoint_step": 10, "global_step": 10})
    assert _select([final, periodic, earlier], 20) is periodic
    assert _select([final, periodic, earlier], 10) is earlier


def test_different_checkpoints_at_one_step_are_still_ambiguous():
    first = _link("checkpoint-00000020-recovery", "sha256:a", {"checkpoint_step": 20})
    second = _link("recovery-checkpoint", "sha256:c", {"global_step": 20})
    with pytest.raises(Exception, match="2 matching checkpoint model outputs at step 20"):
        _select([first, second], 20)


def test_missing_digests_never_count_as_identical():
    first = _link("checkpoint-00000020-recovery", None, {"checkpoint_step": 20})
    second = _link("recovery-checkpoint", None, {"global_step": 20})
    with pytest.raises(Exception, match="expected 1"):
        _select([first, second], 20)


def test_a_cancelled_runs_republication_yields_to_the_committed_periodic_view():
    periodic = _link("checkpoint-00000040-recovery", "sha256:a", {"checkpoint_step": 40, "interrupted": False})
    republished = _link("checkpoint-00000040-recovery", "sha256:d", {"checkpoint_step": 40, "interrupted": True})
    assert _select([republished, periodic], 40) is periodic
    other = _link("checkpoint-00000040-recovery", "sha256:e", {"checkpoint_step": 40, "interrupted": True})
    with pytest.raises(Exception, match="expected 1"):
        _select([republished, other], 40)
