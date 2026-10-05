"""Population-frozen score transport beside a complete native checkpoint."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from ..checkpoints import CheckpointComponent
from ..update_records import InvalidPolicyUpdate, record_digest
from ..update_recovery import (
    FILENAME,
    UpdateRecoveryIdentity,
    UpdateRecoveryState,
    checkpoint_component,
    commit_update_recovery,
    load_update_recovery,
)
from ..update_transport import RetainedResolvedPopulation, decode_population_payload
from ..update_transport import population_payload as _population_payload
from .policy_update_scoring import FrozenPopulationScores

SCORES_FILENAME = "posttrain-frozen-policy-scores.pt"
POPULATION_FILENAME = "posttrain-update-population.json"
NATIVE_EVIDENCE_FILENAME = "posttrain-native-population.bin"
CORRECTION_FILENAME = "posttrain-sampler-correction.json"


def _correction_payload(correction: np.ndarray | None) -> Any:
    return None if correction is None else np.asarray(correction, dtype=np.float64).tolist()


def _correction_digest(correction: np.ndarray | None) -> str:
    return hashlib.sha256(
        json.dumps(_correction_payload(correction), sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def load_sampler_correction(checkpoint: Path, identity: UpdateRecoveryIdentity) -> np.ndarray | None:
    """Read detached correction only after every checkpoint component verifies.

    The caller still admits runtime/recipe identities independently. Uncorrected
    checkpoints keep their existing identity and need no extra component.
    """
    state = load_update_recovery(checkpoint, identity)
    components = [item for item in state.components if item.role == "sampler-correction"]
    if identity.sampler_correction_digest == _correction_digest(None):
        if components:
            raise InvalidPolicyUpdate("uncorrected checkpoint unexpectedly retains sampler correction")
        return None
    if len(components) != 1 or components[0].relative_path != CORRECTION_FILENAME:
        raise InvalidPolicyUpdate("corrected checkpoint lacks sealed sampler correction")
    try:
        payload = json.loads((checkpoint / CORRECTION_FILENAME).read_text())
        if set(payload) != {"schema", "weights"} or payload["schema"] != "posttrain.sampler-correction.v2":
            raise ValueError("unsupported correction schema")
        weights = payload["weights"]
        if not isinstance(weights, list) or any(
            type(value) not in (float, int) or not math.isfinite(value) or value < 0 for value in weights
        ):
            raise ValueError("invalid correction weights")
        correction = np.asarray(weights, dtype=np.float64)
        retained = decode_population_payload(json.loads((checkpoint / POPULATION_FILENAME).read_text()))
        if correction.shape != (retained.resolved.snapshot.size,) or (
            _correction_digest(correction) != identity.sampler_correction_digest
        ):
            raise ValueError("correction differs from complete frozen population")
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise InvalidPolicyUpdate("retained sampler correction differs from the sealed population identity") from error
    correction.setflags(write=False)
    return correction


def load_retained_population(
    checkpoint: Path,
    identity: UpdateRecoveryIdentity,
    *,
    sampler_correction: np.ndarray | None,
) -> RetainedResolvedPopulation:
    """Reconstruct verified sidecar records before loading native parameters.

    The caller supplies the selected expected recovery identity and retained
    correction values. Neither is inferred from unverified population JSON.
    Frozen old/reference tensors and native state are loaded by the backend's
    existing recovery path after this input composition succeeds.
    """
    load_update_recovery(checkpoint, identity)
    try:
        payload = json.loads((checkpoint / POPULATION_FILENAME).read_text())
    except (OSError, ValueError) as error:
        raise InvalidPolicyUpdate("verified checkpoint lacks retained population records") from error
    retained = decode_population_payload(payload)
    resolved = retained.resolved
    candidate = SimpleNamespace(
        updates=resolved.updates,
        credit=resolved.credit,
        spec=resolved.spec,
        execution=resolved.execution,
        capabilities=resolved.capabilities,
        sampler_correction=sampler_correction,
        score_contract=identity.score_contract,
        score_temperature=identity.score_temperature,
        max_overflow_retries=retained.max_overflow_retries,
        applied_update_offset=retained.applied_update_offset,
        attempt_offset=retained.attempt_offset,
    )
    if (
        population_recovery_identity(
            candidate, runtime_identity=identity.runtime_identity, world_size=identity.world_size
        )
        != identity
    ):
        raise InvalidPolicyUpdate("retained population reconstruction differs from the selected recovery identity")
    return retained


def population_recovery_identity(population: Any, *, runtime_identity: str, world_size: int) -> UpdateRecoveryIdentity:
    correction = population.sampler_correction
    correction_digest = _correction_digest(correction)
    capabilities = asdict(population.capabilities)
    if capabilities["context_layout"] == "ragged":
        # Preserve the executable identity of legacy v1/v2 ragged checkpoints.
        capabilities.pop("context_layout")
    execution_payload = {
        "budget": asdict(population.execution),
        "capabilities": capabilities,
        "max_overflow_retries": getattr(population, "max_overflow_retries", 0),
    }
    execution_digest = hashlib.sha256(
        json.dumps(execution_payload, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    return UpdateRecoveryIdentity(
        population.updates[0].population.digest,
        population.credit.digest,
        record_digest(population.spec),
        tuple(update.digest for update in population.updates),
        execution_digest,
        correction_digest,
        runtime_identity,
        population.score_contract,
        population.score_temperature,
        world_size,
        getattr(population, "applied_update_offset", 0),
        getattr(population, "attempt_offset", 0),
    )


def save_population_recovery(
    population: Any,
    checkpoint: Path,
    *,
    runtime_identity: str,
    world_size: int,
    native_applied_updates: int,
    native_components: tuple[str, ...],
) -> UpdateRecoveryState:
    """Bind already saved native components and freeze scores before publication."""
    if getattr(population, "_pending", None) is not None:
        raise InvalidPolicyUpdate("cannot checkpoint an incomplete resolved optimizer update")
    identity = population_recovery_identity(population, runtime_identity=runtime_identity, world_size=world_size)
    snapshot = population.updates[0].population
    if population.old is None:
        raise InvalidPolicyUpdate("cannot checkpoint a population before freezing its old scores")
    if (checkpoint / FILENAME).exists():
        state = load_update_recovery(checkpoint, identity)
        if (state.next_update, state.applied_updates, state.attempts, state.native_applied_updates) != (
            population.next_update,
            population.applied_updates,
            population.attempts,
            native_applied_updates,
        ):
            raise InvalidPolicyUpdate("committed checkpoint differs from the current resolved boundary")
        return state
    values = {}
    for role, frozen, version in (
        ("old", population.old, snapshot.versions.old_score),
        ("reference", population.reference, snapshot.versions.reference),
    ):
        if frozen is None:
            continue
        if version is None:
            raise InvalidPolicyUpdate("frozen reference has no declared policy version")
        frozen.validate(
            snapshot,
            policy_version=version,
            score_contract=population.score_contract,
            score_temperature=population.score_temperature,
        )
        values[role] = frozen.values.detach().cpu().clone()
    if population.spec.beta and "reference" not in values:
        raise InvalidPolicyUpdate("KL recovery requires retained population-frozen reference scores")
    # Check counters and native component presence before any sidecar mutation.
    components = tuple(
        checkpoint_component(checkpoint, relative, role="native-checkpoint") for relative in native_components
    )
    state = (
        UpdateRecoveryState(
            identity,
            population.next_update,
            population.applied_updates,
            population.attempts,
            native_applied_updates,
            f"{snapshot.versions.current}/applied-{population.applied_updates}",
            components
            + (
                CheckpointComponent("frozen-policy-scores", SCORES_FILENAME, 0, "0" * 64),
                CheckpointComponent("resolved-population", POPULATION_FILENAME, 0, "0" * 64),
            ),
        )
        if components
        else None
    )
    if state is None:
        raise InvalidPolicyUpdate("resolved recovery requires saved native checkpoint components")
    from .policy_update_inputs import NativePopulationInputs

    correction_component = ()
    if population.sampler_correction is not None:
        correction = np.asarray(population.sampler_correction, dtype=np.float64)
        if correction.shape != (snapshot.size,) or not np.isfinite(correction).all() or (correction < 0).any():
            raise InvalidPolicyUpdate("checkpoint correction requires complete finite detached action weights")
        payload = {"schema": "posttrain.sampler-correction.v2", "weights": _correction_payload(correction)}
        with (checkpoint / CORRECTION_FILENAME).open("x") as stream:
            json.dump(payload, stream, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        correction_component = (checkpoint_component(checkpoint, CORRECTION_FILENAME, role="sampler-correction"),)
    evidence_component = ()
    reader = getattr(population, "read_input", None)
    if isinstance(reader, NativePopulationInputs):
        evidence = reader.retained_evidence
        if reader.evidence.digest != snapshot.native_evidence_digest:
            raise InvalidPolicyUpdate("checkpoint native evidence differs from its frozen population")
        with (checkpoint / NATIVE_EVIDENCE_FILENAME).open("xb") as stream:
            stream.write(evidence)
            stream.flush()
            os.fsync(stream.fileno())
        evidence_component = (
            checkpoint_component(checkpoint, NATIVE_EVIDENCE_FILENAME, role="native-rollout-evidence"),
        )
    with (checkpoint / SCORES_FILENAME).open("xb") as stream:
        torch.save(values, stream)
        stream.flush()
        os.fsync(stream.fileno())
    with (checkpoint / POPULATION_FILENAME).open("x") as stream:
        json.dump(_population_payload(population), stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    scores = checkpoint_component(checkpoint, SCORES_FILENAME, role="frozen-policy-scores")
    contract = checkpoint_component(checkpoint, POPULATION_FILENAME, role="resolved-population")
    state = UpdateRecoveryState(
        identity,
        state.next_update,
        state.applied_updates,
        state.attempts,
        state.native_applied_updates,
        state.parameter_version,
        components + (scores, contract) + evidence_component + correction_component,
    )
    commit_update_recovery(checkpoint, state)
    return state


def restore_population_recovery(
    population: Any,
    checkpoint: Path,
    *,
    runtime_identity: str,
    world_size: int,
    native_applied_updates: int,
    device: Any,
) -> UpdateRecoveryState:
    """Restore scores/cursor after the adapter verifies native state restoration."""
    identity = population_recovery_identity(population, runtime_identity=runtime_identity, world_size=world_size)
    state = load_update_recovery(checkpoint, identity)
    if state.native_applied_updates != native_applied_updates:
        raise InvalidPolicyUpdate("restored native optimizer boundary differs from resolved recovery")
    snapshot = population.updates[0].population
    if state.parameter_version != f"{snapshot.versions.current}/applied-{state.applied_updates}":
        raise InvalidPolicyUpdate("recovery parameter version differs from its applied cursor")
    try:
        retained = json.loads((checkpoint / POPULATION_FILENAME).read_text())
    except (OSError, ValueError) as error:
        raise InvalidPolicyUpdate("recovery lacks retained population and prepared credit") from error
    expected = json.loads(json.dumps(_population_payload(population), allow_nan=False))
    decoded = decode_population_payload(retained)
    resolved = decoded.resolved
    canonical = _population_payload(
        SimpleNamespace(
            updates=resolved.updates,
            credit=resolved.credit,
            spec=resolved.spec,
            execution=resolved.execution,
            capabilities=resolved.capabilities,
            max_overflow_retries=decoded.max_overflow_retries,
            applied_update_offset=decoded.applied_update_offset,
            attempt_offset=decoded.attempt_offset,
        )
    )
    if json.loads(json.dumps(canonical, allow_nan=False)) != expected:
        raise InvalidPolicyUpdate("recovery retained population/credit differs from selected resolved inputs")
    try:
        payload = torch.load(checkpoint / SCORES_FILENAME, weights_only=True, map_location=device)
    except (OSError, ValueError, RuntimeError) as error:
        raise InvalidPolicyUpdate("retained frozen policy scores cannot be loaded") from error
    expected_roles = {"old"} | ({"reference"} if population.spec.beta else set())
    if not isinstance(payload, dict) or not expected_roles <= set(payload) or not set(payload) <= {"old", "reference"}:
        raise InvalidPolicyUpdate("recovery frozen score roles differ from the objective")
    frozen = {}
    for role, tensor in payload.items():
        version = snapshot.versions.old_score if role == "old" else snapshot.versions.reference
        if (
            version is None
            or not isinstance(tensor, torch.Tensor)
            or not tensor.is_floating_point()
            or tensor.shape != (snapshot.size,)
        ):
            raise InvalidPolicyUpdate("recovery scores do not align with the admitted action population")
        scores = FrozenPopulationScores(
            snapshot.digest,
            version,
            population.score_contract,
            population.score_temperature,
            tensor.detach().clone(),
        )
        scores.validate(
            snapshot,
            policy_version=version,
            score_contract=population.score_contract,
            score_temperature=population.score_temperature,
        )
        frozen[role] = scores
    # Mutate live cursor only after every identity/file/score check succeeded.
    population.old = frozen["old"]
    population.reference = frozen.get("reference")
    population.next_update, population.applied_updates, population.attempts = (
        state.next_update,
        state.applied_updates,
        state.attempts,
    )
    population._pending = None
    return state
