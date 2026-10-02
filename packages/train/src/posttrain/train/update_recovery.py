"""Content-bound resolved-update state at complete native checkpoint boundaries.

Native adapters save model/optimizer/scheduler/RNG/scaler state first, then seal
the directory with this manifest. Readers ignore unsealed directories. This
module does not save native state or attest its semantics; backend qualification
must establish that the named native boundary restores the applied step.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from .checkpoints import CheckpointComponent
from .update_records import InvalidPolicyUpdate, record_digest, require_identity

FILENAME = "posttrain-resolved-update.json"


@dataclass(frozen=True, slots=True)
class UpdateRecoveryIdentity:
    population_digest: str
    credit_digest: str
    objective_digest: str
    update_digests: tuple[str, ...]
    execution_digest: str
    sampler_correction_digest: str
    runtime_identity: str
    score_contract: str
    score_temperature: float
    world_size: int
    applied_update_offset: int = 0
    attempt_offset: int = 0

    def __post_init__(self) -> None:
        import math

        require_identity(self.population_digest, self.credit_digest, self.objective_digest,
                         self.execution_digest, self.sampler_correction_digest,
                         self.runtime_identity, self.score_contract, *self.update_digests)
        if not self.update_digests or len(set(self.update_digests)) != len(self.update_digests):
            raise InvalidPolicyUpdate("recovery requires distinct ordered resolved occurrences")
        if type(self.world_size) is not int or self.world_size < 1:
            raise InvalidPolicyUpdate("recovery requires a positive qualified world size")
        if any(type(value) is not int or value < 0 for value in (self.applied_update_offset, self.attempt_offset)):
            raise InvalidPolicyUpdate("recovery population offsets must be nonnegative integers")
        if self.attempt_offset < self.applied_update_offset:
            raise InvalidPolicyUpdate("recovery prior attempts cannot be fewer than prior applied updates")
        if isinstance(self.score_temperature, bool) or not math.isfinite(self.score_temperature) or self.score_temperature <= 0:
            raise InvalidPolicyUpdate("recovery requires a finite positive score temperature")

    @property
    def digest(self) -> str:
        return record_digest(self)


@dataclass(frozen=True, slots=True)
class UpdateRecoveryState:
    identity: UpdateRecoveryIdentity
    next_update: int
    applied_updates: int
    attempts: int
    native_applied_updates: int
    parameter_version: str
    components: tuple[CheckpointComponent, ...]

    def __post_init__(self) -> None:
        require_identity(self.parameter_version)
        counters = (self.next_update, self.applied_updates, self.attempts, self.native_applied_updates)
        if any(type(value) is not int or value < 0 for value in counters):
            raise InvalidPolicyUpdate("recovery counters must be nonnegative integers")
        # Current native adapters qualify only applied occurrences, not omission.
        if self.next_update != self.applied_updates or self.native_applied_updates != (
            self.identity.applied_update_offset + self.applied_updates
        ):
            raise InvalidPolicyUpdate("recovery native and resolved applied boundaries disagree")
        if self.next_update > len(self.identity.update_digests) or self.attempts < self.applied_updates:
            raise InvalidPolicyUpdate("recovery cursor or attempt count exceeds declared schedule")
        if not self.components or len({item.relative_path for item in self.components}) != len(self.components):
            raise InvalidPolicyUpdate("recovery requires unique retained checkpoint components")
        roles = {item.role for item in self.components}
        if not {"native-checkpoint", "frozen-policy-scores", "resolved-population"} <= roles:
            raise InvalidPolicyUpdate("recovery must bind native state, frozen scores and resolved population")
        if any(item.relative_path == FILENAME for item in self.components):
            raise InvalidPolicyUpdate("recovery seal cannot be its own checkpoint component")


def _component_path(checkpoint: Path, relative: str) -> Path:
    path = checkpoint / relative
    if path.is_symlink() or not path.resolve().is_relative_to(checkpoint.resolve()):
        raise InvalidPolicyUpdate("recovery component escapes the retained checkpoint")
    return path


def checkpoint_component(checkpoint: Path, relative: str, *, role: str) -> CheckpointComponent:
    # Validate before filesystem access, including absolute/parent traversal.
    CheckpointComponent(role, relative, 0, "0" * 64)
    path = _component_path(checkpoint, relative)
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
                size += len(block)
    except OSError as error:
        raise InvalidPolicyUpdate("recovery component is missing or unreadable") from error
    return CheckpointComponent(role, relative, size, digest.hexdigest())


def _verify_components(checkpoint: Path, state: UpdateRecoveryState) -> None:
    for component in state.components:
        if checkpoint_component(checkpoint, component.relative_path, role=component.role) != component:
            raise InvalidPolicyUpdate("recovery component bytes changed after checkpoint commit")


def _read_update_recovery(checkpoint: Path) -> UpdateRecoveryState:
    try:
        payload = json.loads((checkpoint / FILENAME).read_text())
        if set(payload) != {"schema", "state"} or payload["schema"] not in {
            "posttrain.resolved-update.v1", "posttrain.resolved-update.v2",
        }:
            raise InvalidPolicyUpdate("unsupported resolved-update recovery seal")
        values = dict(payload["state"])
        retained_identity = dict(values.pop("identity"))
        offsets = ("applied_update_offset", "attempt_offset")
        if payload["schema"] == "posttrain.resolved-update.v2" and not all(key in retained_identity for key in offsets):
            raise InvalidPolicyUpdate("recovery v2 requires explicit population offsets")
        if payload["schema"] == "posttrain.resolved-update.v1" and any(retained_identity.get(key, 0) != 0 for key in offsets):
            raise InvalidPolicyUpdate("legacy recovery cannot declare nonzero population offsets")
        retained_identity["update_digests"] = tuple(retained_identity["update_digests"])
        values["identity"] = UpdateRecoveryIdentity(**retained_identity)
        values["components"] = tuple(CheckpointComponent(**item) for item in values["components"])
        state = UpdateRecoveryState(**values)
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise InvalidPolicyUpdate("checkpoint lacks a valid complete resolved-update seal") from error
    return state


def inspect_update_recovery(checkpoint: Path) -> UpdateRecoveryState:
    """Verify retained bytes for host inspection, without admitting a selection.

    The host must independently check runtime, scoring and resolved recipe
    compatibility before using the retained identity to restore native state.
    Integrity alone does not qualify the saved algorithm or execution path.
    """
    state = _read_update_recovery(checkpoint)
    _verify_components(checkpoint, state)
    return state


def load_update_recovery(checkpoint: Path, identity: UpdateRecoveryIdentity) -> UpdateRecoveryState:
    state = _read_update_recovery(checkpoint)
    if state.identity != identity:
        raise InvalidPolicyUpdate("recovery population, algorithm, runtime or schedule identity changed")
    _verify_components(checkpoint, state)
    return state


def commit_update_recovery(checkpoint: Path, state: UpdateRecoveryState) -> None:
    """Seal a native checkpoint after verifying all retained component bytes.

    The checkpoint writer owns exclusive access to this directory. A replay of
    the identical commit is idempotent; a different state cannot replace a seal.
    fsync plus atomic rename keeps incomplete temporary metadata off the resume
    path. Publication must occur only after this method returns successfully.
    """
    target = checkpoint / FILENAME
    if target.exists():
        if load_update_recovery(checkpoint, state.identity) != state:
            raise InvalidPolicyUpdate("cannot replace an already committed update boundary")
        return
    _verify_components(checkpoint, state)
    payload = json.dumps({"schema": "posttrain.resolved-update.v2", "state": asdict(state)},
                         sort_keys=True, allow_nan=False)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=checkpoint, prefix=".resolved-update-", delete=False) as stream:
            temporary = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # Native checkpoint files must be durable before publishing the seal.
        for component in state.components:
            with _component_path(checkpoint, component.relative_path).open("rb") as stream:
                os.fsync(stream.fileno())
        os.replace(temporary, target)
        temporary = None
        directory = os.open(checkpoint, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
