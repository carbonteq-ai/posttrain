"""Complete driver seals reuse the existing resolved actor recovery contract."""

from dataclasses import replace

import pytest
from posttrain.train.backends.verl.policy_checkpoint import (
    inspect_driver_checkpoint,
    latest_driver_checkpoint,
    prune_staged_checkpoint_sources,
    publish_driver_checkpoint,
    seal_driver_checkpoint,
    stage_driver_checkpoint,
)
from posttrain.train.update_records import InvalidPolicyUpdate
from posttrain.train.update_recovery import FILENAME, commit_update_recovery

from .test_update_recovery import boundary


def actor_checkpoint(path):
    actor = path / "actor"
    actor.mkdir(parents=True)
    commit_update_recovery(actor, boundary(actor))


def test_job_seal_requires_data_and_ignores_interrupted_newer_save(tmp_path):
    checkpoint = tmp_path / "global_step_1"
    actor_checkpoint(checkpoint)
    with pytest.raises(InvalidPolicyUpdate, match="missing or unreadable"):
        seal_driver_checkpoint(checkpoint)
    assert latest_driver_checkpoint(tmp_path) is None
    (checkpoint / "data.pt").write_bytes(b"native dataloader cursor")
    (checkpoint / "selector.json").write_bytes(b"selector state")
    state = seal_driver_checkpoint(checkpoint)
    assert inspect_driver_checkpoint(checkpoint) == state
    assert seal_driver_checkpoint(checkpoint) == state
    actor_checkpoint(tmp_path / "global_step_2")
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("2")
    assert latest_driver_checkpoint(tmp_path) == checkpoint


@pytest.mark.parametrize("relative", ["data.pt", "selector.json", f"actor/{FILENAME}", "actor/scores.bin"])
def test_committed_driver_corruption_is_rejected_without_fallback(tmp_path, relative):
    checkpoint = tmp_path / "global_step_1"
    actor_checkpoint(checkpoint)
    (checkpoint / "data.pt").write_bytes(b"cursor")
    (checkpoint / "selector.json").write_bytes(b"selector")
    seal_driver_checkpoint(checkpoint)
    (checkpoint / relative).write_bytes(b"changed after commit")
    with pytest.raises(InvalidPolicyUpdate):
        inspect_driver_checkpoint(checkpoint)
    with pytest.raises(InvalidPolicyUpdate):
        latest_driver_checkpoint(tmp_path)


def test_mislabeled_driver_step_cannot_be_committed(tmp_path):
    checkpoint = tmp_path / "global_step_2"
    actor_checkpoint(checkpoint)
    (checkpoint / "data.pt").write_bytes(b"cursor")
    with pytest.raises(InvalidPolicyUpdate, match="differs from its native actor"):
        seal_driver_checkpoint(checkpoint)
    assert not (checkpoint / FILENAME).exists()


def test_transported_context_validates_the_actual_framework_identity(tmp_path):
    from posttrain.common import RunContext
    from posttrain.train.backends.verl.contracts import VerlRunContext
    from pydantic import ValidationError

    context = RunContext("project", "work", "run", "train.sampo", "job@1", tmp_path)
    transported = VerlRunContext.model_validate({**context.identity_attributes, "workspace": context.workspace})
    assert RunContext(**transported.model_dump()).identity_attributes == context.identity_attributes
    with pytest.raises(ValidationError):
        VerlRunContext.model_validate({**transported.model_dump(), "workspace": "relative"})
    with pytest.raises(ValidationError):
        VerlRunContext.model_validate({**transported.model_dump(), "run_id": ""})


def sealed_checkpoint(root, step):
    checkpoint = root / f"global_step_{step}"
    actor = checkpoint / "actor"
    actor.mkdir(parents=True)
    state = boundary(actor)
    state = replace(
        state,
        identity=replace(state.identity, applied_update_offset=step - 1, attempt_offset=step - 1),
        native_applied_updates=step,
    )
    commit_update_recovery(actor, state)
    (checkpoint / "data.pt").write_bytes(f"native cursor {step}".encode())
    seal_driver_checkpoint(checkpoint)
    return checkpoint


def test_rotated_sources_leave_verified_publication_copies(tmp_path):
    from types import SimpleNamespace
    from typing import cast

    from posttrain.common import Observer, RunContext

    sources, staged = tmp_path / "checkpoints", tmp_path / "publications"
    artifacts = []
    context = RunContext(
        "project",
        "work",
        "run",
        "train.grpo",
        "job@1",
        tmp_path,
        observer=cast(Observer, SimpleNamespace(artifact=artifacts.append)),
    )
    for step in range(1, 4):
        checkpoint = sealed_checkpoint(sources, step)
        retained = publish_driver_checkpoint(context, checkpoint, staged)
        assert stage_driver_checkpoint(checkpoint, staged) == retained
        prune_staged_checkpoint_sources(sources, staged, 1)
    assert sorted(path.name for path in sources.iterdir()) == ["global_step_3"]
    assert len(artifacts) == 3
    assert all(item.metadata["checkpoint_snapshot_id"].startswith("run/step-") for item in artifacts)
    assert all(item.role == "recovery" and item.kind == "training-checkpoint" for item in artifacts)
    assert [
        inspect_driver_checkpoint(staged / f"global_step_{step}").native_applied_updates for step in range(1, 4)
    ] == [1, 2, 3]


def test_missing_or_corrupt_publication_stage_prevents_all_pruning(tmp_path):
    sources, staged = tmp_path / "checkpoints", tmp_path / "publications"
    checkpoints = [sealed_checkpoint(sources, step) for step in range(1, 4)]
    stage_driver_checkpoint(checkpoints[0], staged)
    with pytest.raises(InvalidPolicyUpdate):
        prune_staged_checkpoint_sources(sources, staged, 1)
    assert all(path.exists() for path in checkpoints)
    stage_driver_checkpoint(checkpoints[1], staged)
    (staged / "global_step_1/data.pt").write_bytes(b"corrupt staged data")
    with pytest.raises(InvalidPolicyUpdate):
        prune_staged_checkpoint_sources(sources, staged, 1)
    assert all(path.exists() for path in checkpoints)


def test_failed_host_publication_keeps_native_sources(tmp_path):
    from types import SimpleNamespace
    from typing import cast

    from posttrain.common import Observer, RunContext

    sources, staged = tmp_path / "checkpoints", tmp_path / "publications"
    checkpoint = sealed_checkpoint(sources, 1)

    def reject(artifact):
        raise RuntimeError("host unavailable")

    context = RunContext(
        "project",
        "work",
        "run",
        "train.grpo",
        "job@1",
        tmp_path,
        observer=cast(Observer, SimpleNamespace(artifact=reject)),
    )
    with pytest.raises(RuntimeError, match="host unavailable"):
        publish_driver_checkpoint(context, checkpoint, staged)
    assert checkpoint.exists()
    assert inspect_driver_checkpoint(staged / checkpoint.name).native_applied_updates == 1
