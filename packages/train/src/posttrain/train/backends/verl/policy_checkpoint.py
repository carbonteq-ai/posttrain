"""Complete synchronous veRL job seals over native actor and driver state."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

from posttrain.common import LocalArtifactRef, ProducedArtifact, RunContext

from ...update_records import InvalidPolicyUpdate
from ...update_recovery import (
    FILENAME,
    UpdateRecoveryState,
    checkpoint_component,
    commit_update_recovery,
    inspect_update_recovery,
)


def checkpoint_step(checkpoint: Path) -> int:
    name = checkpoint.name
    if not name.startswith("global_step_"):
        raise InvalidPolicyUpdate("native job checkpoint lacks an applied step directory")
    suffix = name.removeprefix("global_step_")
    if not suffix.isascii() or not suffix.isdecimal() or str(int(suffix)) != suffix:
        raise InvalidPolicyUpdate("native job checkpoint step is not canonical")
    return int(suffix)


def seal_driver_checkpoint(checkpoint: Path) -> UpdateRecoveryState:
    """Publish readiness only after native actor and dataloader writes finish."""
    actor = inspect_update_recovery(checkpoint / "actor")
    if actor.native_applied_updates != checkpoint_step(checkpoint):
        raise InvalidPolicyUpdate("driver checkpoint step differs from its native actor")
    components = tuple(
        checkpoint_component(checkpoint, f"actor/{item.relative_path}", role=item.role) for item in actor.components
    )
    components += (
        checkpoint_component(checkpoint, f"actor/{FILENAME}", role="actor-recovery-seal"),
        checkpoint_component(checkpoint, "data.pt", role="native-driver-state"),
    )
    # Native prompt selectors may write additional state. Bind every additional
    # driver file, rather than guessing a selector-specific serialization name.
    for path in sorted(checkpoint.rglob("*")):
        relative = path.relative_to(checkpoint)
        if relative.parts[0] == "actor" or relative.as_posix() in (FILENAME, "data.pt"):
            continue
        if path.is_symlink():
            raise InvalidPolicyUpdate("native driver checkpoint contains a symlink")
        if path.is_file():
            components += (checkpoint_component(checkpoint, relative.as_posix(), role="native-driver-state"),)
    state = replace(actor, components=components)
    commit_update_recovery(checkpoint, state)
    return state


def inspect_driver_checkpoint(checkpoint: Path) -> UpdateRecoveryState:
    state = inspect_update_recovery(checkpoint)
    actor = inspect_update_recovery(checkpoint / "actor")
    if (
        state.native_applied_updates != checkpoint_step(checkpoint)
        or replace(state, components=actor.components) != actor
        or not any(item.relative_path == "data.pt" and item.role == "native-driver-state" for item in state.components)
        or not any(
            item.relative_path == f"actor/{FILENAME}" and item.role == "actor-recovery-seal"
            for item in state.components
        )
    ):
        raise InvalidPolicyUpdate("driver seal does not bind its complete native update boundary")
    return state


def latest_driver_checkpoint(directory: Path) -> Path | None:
    """Ignore interrupted unsealed saves; reject corruption of a committed save.

    The stock latest-step text file can be written before the job seal. It is
    therefore advisory here, never the authority for resolved recovery.
    """
    candidates = []
    for checkpoint in directory.glob("global_step_*"):
        if checkpoint.is_symlink():
            raise InvalidPolicyUpdate("native checkpoint discovery encountered a symlink")
        if checkpoint.is_dir() and (checkpoint / FILENAME).exists():
            inspect_driver_checkpoint(checkpoint)
            candidates.append(checkpoint)
    return max(candidates, key=checkpoint_step) if candidates else None


def stage_driver_checkpoint(checkpoint: Path, destination_root: Path) -> Path:
    """Keep an immutable publication copy before pruning mutable native sources.

    Observation delivery can be asynchronous. Its artifact source must outlive
    native checkpoint rotation; the host retains ownership of staged copies.
    """
    expected = inspect_driver_checkpoint(checkpoint)
    if destination_root.resolve().is_relative_to(checkpoint.resolve()):
        raise InvalidPolicyUpdate("checkpoint publication stage cannot be inside its source")
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / checkpoint.name
    if destination.is_symlink() or destination_root.is_symlink():
        raise InvalidPolicyUpdate("checkpoint publication staging cannot use symlinks")
    if destination.exists():
        if inspect_driver_checkpoint(destination) != expected:
            raise InvalidPolicyUpdate("checkpoint publication stage differs from committed source")
        return destination
    if any(path.is_symlink() for path in checkpoint.rglob("*")):
        raise InvalidPolicyUpdate("checkpoint publication source contains a symlink")
    temporary = Path(tempfile.mkdtemp(prefix=".resolved-checkpoint-", dir=destination_root))
    try:
        shutil.copytree(checkpoint, temporary / checkpoint.name)
        staged = temporary / checkpoint.name
        if inspect_driver_checkpoint(staged) != expected or inspect_driver_checkpoint(checkpoint) != expected:
            raise InvalidPolicyUpdate("native checkpoint changed while staging publication")
        # Reuse the seal writer's component durability checks on the copied tree.
        for path in staged.rglob("*"):
            if path.is_file():
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
        for directory in sorted(
            (path for path in staged.rglob("*") if path.is_dir()), key=lambda path: len(path.parts), reverse=True
        ) + [staged]:
            descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        os.rename(staged, destination)
        descriptor = os.open(destination_root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        shutil.rmtree(temporary)
    return destination


def publish_driver_checkpoint(context: RunContext, checkpoint: Path, destination_root: Path) -> Path:
    staged = stage_driver_checkpoint(checkpoint, destination_root)
    state = inspect_driver_checkpoint(staged)
    digest = hashlib.sha256()
    for path in sorted(item for item in staged.rglob("*") if item.is_file()):
        digest.update(path.relative_to(staged).as_posix().encode())
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    step = state.native_applied_updates
    context.artifact(
        ProducedArtifact(
            f"training/checkpoint-{step:08d}/recovery",
            "training-checkpoint",
            LocalArtifactRef(staged, digest.hexdigest()),
            role="recovery",
            metadata={
                "checkpoint_view": "recovery",
                "checkpoint_step": step,
                "global_step": step,
                "checkpoint_snapshot_id": f"{context.run_id}/step-{step:08d}",
                "trainer_checkpoint_schema": "posttrain.resolved-update.v2",
                "runtime_identity": state.identity.runtime_identity,
            },
        )
    )
    return staged


def prune_staged_checkpoint_sources(checkpoint_root: Path, publication_root: Path, limit: int) -> None:
    """Remove only complete superseded sources with independently verified copies."""
    if type(limit) is not int or limit < 1:
        raise InvalidPolicyUpdate("native checkpoint retention requires a positive limit")
    if checkpoint_root.is_symlink() or publication_root.is_symlink():
        raise InvalidPolicyUpdate("native checkpoint retention roots cannot be symlinks")
    if checkpoint_root.resolve().is_relative_to(
        publication_root.resolve()
    ) or publication_root.resolve().is_relative_to(checkpoint_root.resolve()):
        raise InvalidPolicyUpdate("native retention and publication roots must be separate")
    candidates = []
    for checkpoint in checkpoint_root.glob("global_step_*"):
        if checkpoint.is_symlink():
            raise InvalidPolicyUpdate("native checkpoint retention encountered a symlink")
        if checkpoint.is_dir() and (checkpoint / FILENAME).exists():
            inspect_driver_checkpoint(checkpoint)
            candidates.append(checkpoint)
    removals = sorted(candidates, key=checkpoint_step)[:-limit]
    # Validate all copies before any deletion; failed staging cannot partially
    # rotate a set whose old evidence is still awaiting host publication.
    for checkpoint in removals:
        if inspect_driver_checkpoint(publication_root / checkpoint.name) != inspect_driver_checkpoint(checkpoint):
            raise InvalidPolicyUpdate("native retention lacks an identical complete publication stage")
    for checkpoint in removals:
        shutil.rmtree(checkpoint)
