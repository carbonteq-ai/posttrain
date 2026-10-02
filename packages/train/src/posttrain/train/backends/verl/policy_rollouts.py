"""Native episode receipts carried through veRL's agent-loop transport."""

from __future__ import annotations

import hashlib
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from posttrain.common import LocalArtifactRef, ProducedArtifact
from pydantic import BaseModel, ConfigDict, field_serializer

from ...online_rl import EnvironmentRollout
from ...profiles import CAPOSettings, GDPOSettings, GRPOSettings, SAMPOSettings
from ...update_plan import ExecutionCapabilities
from ...update_records import InvalidPolicyUpdate
from ..policy_update_admission import AdmittedNativePopulation


class NativeEpisodeReceipt(BaseModel):
    """Collection evidence, not admission or prepared training credit.

    The native artifact remains replay authority. JSON retains typed rollout
    metadata and original conditioning coordinates across Ray/object-array
    transport, without relying on Python pickle or flattened trainer rows.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    schema_version: Literal[1] = 1
    artifact: ProducedArtifact
    rollout: EnvironmentRollout

    @field_serializer("artifact")
    def serialize_artifact(self, artifact: ProducedArtifact) -> dict[str, Any]:
        if not isinstance(artifact.reference, LocalArtifactRef):
            raise InvalidPolicyUpdate("veRL receipt serialization requires a retained local artifact")
        return {"name": artifact.name, "kind": artifact.kind, "required": artifact.required,
                "role": artifact.role, "metadata": dict(artifact.metadata),
                "reference": {"path": str(artifact.reference.path), "digest": artifact.reference.digest}}

    def verify_artifact(self) -> bytes:
        artifact = self.artifact
        reference = artifact.reference
        if (not isinstance(reference, LocalArtifactRef) or artifact.kind != "evaluation-traces"
                or not artifact.required or artifact.metadata.get("replay_authority") is not True
                or artifact.metadata.get("format") not in ("verifiers-native-episodes", "verifiers-native-traces")
                or artifact.metadata.get("trace_ids") != [self.rollout.trace.external_id]):
            raise InvalidPolicyUpdate("veRL episode receipt requires its retained native replay artifact")
        evidence = reference.path.read_bytes()
        if hashlib.sha256(evidence).hexdigest() != reference.digest:
            raise InvalidPolicyUpdate("veRL episode receipt native artifact differs from its digest")
        return evidence


def retain_episode_receipt(bridge: Any, rollout: EnvironmentRollout, *, sampler_step: int) -> str:
    """Freeze one completed episode before conversion to native trainer rows."""
    info = rollout.trace.payload.get("info")
    if (getattr(bridge, "policy_update_context_contract", None) != "causal-text@1"
            or not callable(getattr(bridge, "retain_population", None))
            or not rollout.conditioning_records or not isinstance(info, Mapping)
            or any(not isinstance(info.get(key), str) or not info[key] for key in (
                "posttrain_prompt_group_id", "posttrain_episode_id", "posttrain_rollout_id"))
            or rollout.behavior_policy is None or rollout.behavior_policy.start != sampler_step
            or rollout.behavior_policy.end != sampler_step):
        raise InvalidPolicyUpdate("veRL resolved collection requires original contexts, identities and synchronous policy")
    receipt = NativeEpisodeReceipt(artifact=bridge.retain_population((rollout,)), rollout=rollout)
    receipt.verify_artifact()
    def mapping_value(value: Any) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise TypeError(f"unsupported native receipt value: {type(value).__name__}")
        return dict(value)

    return receipt.model_dump_json(fallback=mapping_value)


def admit_episode_receipts(
    encoded: tuple[str, ...], settings: GRPOSettings | SAMPOSettings | GDPOSettings | CAPOSettings,
    capabilities: ExecutionCapabilities, *, destination: Path, **admission: Any,
) -> tuple[AdmittedNativePopulation, ProducedArtifact]:
    """Authenticate native receipts, then resolve complete groups through the common boundary.

    Exact repeated envelopes are retained once (an episode may contain sibling
    traces). Selected rollout duplicates still reject. The native decoder and
    shared admission verify schema, contexts, group completeness and credit.
    Hosts must publish the returned artifact before optimizer execution.
    """
    from ...integrations.verifiers_population_artifact import retain_native_population

    if not encoded:
        raise InvalidPolicyUpdate("veRL population requires native episode receipts")
    receipts = tuple(NativeEpisodeReceipt.model_validate_json(value) for value in encoded)
    trace_ids = tuple(receipt.rollout.trace.external_id for receipt in receipts)
    if len(set(trace_ids)) != len(trace_ids):
        raise InvalidPolicyUpdate("veRL population duplicates selected native trajectories")
    formats: set[str] = set()
    evidence = []
    for receipt in receipts:
        evidence.append(receipt.verify_artifact())
        format = receipt.artifact.metadata.get("format")
        if not isinstance(format, str):
            raise InvalidPolicyUpdate("veRL population requires a named native replay format")
        formats.add(format)
    if len(formats) != 1:
        raise InvalidPolicyUpdate("veRL population mixes native replay formats")
    lines = dict.fromkeys(line for raw in evidence for line in raw.splitlines(keepends=True))
    if any(not line.endswith(b"\n") for line in lines):
        raise InvalidPolicyUpdate("veRL population requires complete native JSONL records")
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".native-assembly-", dir=destination) as temporary:
        source = Path(temporary) / "source.jsonl"
        source.write_bytes(b"".join(lines))
        artifact = retain_native_population(source, destination, trace_ids,
            episodes=next(iter(formats)) == "verifiers-native-episodes")
    admitted = AdmittedNativePopulation.from_retained_artifact(
        artifact, tuple(receipt.rollout for receipt in receipts), settings, capabilities, **admission,
    )
    return admitted, artifact
