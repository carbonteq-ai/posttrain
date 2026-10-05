"""Compose durable native evidence with resolved collection and recovery.

The host owns artifact persistence and native decoding. This boundary reads the
retained artifact back before returning a population to either native executor.
It does not collect task groups, publish artifacts or construct native models.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from posttrain.common import LocalArtifactRef, ProducedArtifact

from ..online_rl import EnvironmentRollout
from ..profiles import CAPOSettings, GDPOSettings, GRPOSettings, SAMPOSettings
from ..update_plan import ExecutionCapabilities
from ..update_records import ActionRef, InvalidPolicyUpdate, PolicyVersions, SemanticSpan
from ..update_recovery import UpdateRecoveryIdentity
from ..update_resolution import ResolvedPolicyPopulation, resolve_policy_population, resolve_rollout_population
from .policy_update_inputs import NativePopulationInputs


@dataclass(frozen=True, slots=True)
class AdmittedNativePopulation:
    """A resolved population whose original conditioning inputs are available."""

    resolved: ResolvedPolicyPopulation
    read_input: NativePopulationInputs
    applied_update_offset: int = 0
    attempt_offset: int = 0
    max_overflow_retries: int = 0

    def __post_init__(self) -> None:
        self._validate_counters(self.applied_update_offset, self.attempt_offset, self.max_overflow_retries)
        if (
            hashlib.sha256(self.read_input.retained_evidence).hexdigest()
            != self.resolved.snapshot.native_evidence_digest
        ):
            raise InvalidPolicyUpdate("admitted input reader lost its original native evidence bytes")
        for view in self.resolved.snapshot.conditioning:
            self.read_input(view)

    @classmethod
    def from_retained_artifact(
        cls,
        artifact: ProducedArtifact,
        rollouts: tuple[EnvironmentRollout, ...],
        settings: GRPOSettings | SAMPOSettings | GDPOSettings | CAPOSettings,
        capabilities: ExecutionCapabilities,
        *,
        population_id: str,
        template_revision: str,
        versions: PolicyVersions,
        sampler_step: int,
        selector_digest: str,
        spans: tuple[SemanticSpan, ...] = (),
        process_credit: Any = None,
        applied_update_offset: int = 0,
        attempt_offset: int = 0,
        max_overflow_retries: int = 0,
    ) -> AdmittedNativePopulation:
        """Consume the bridge's durable local native artifact before promotion.

        Use its logical artifact name in frozen references, never a machine
        path. Recovery hosts resolve that name to the retained/promoted bytes.
        This method does not claim that observer submission completed promotion.
        """
        from ..integrations.verifiers_population_artifact import decode_native_population

        cls._validate_counters(applied_update_offset, attempt_offset, max_overflow_retries)
        reference = artifact.reference
        format = artifact.metadata.get("format")
        if (
            not isinstance(reference, LocalArtifactRef)
            or artifact.kind != "evaluation-traces"
            or not artifact.required
            or artifact.metadata.get("replay_authority") is not True
            or format not in ("verifiers-native-episodes", "verifiers-native-traces")
        ):
            raise InvalidPolicyUpdate("native admission requires a retained native replay artifact")
        evidence = reference.path.read_bytes()
        if hashlib.sha256(evidence).hexdigest() != reference.digest:
            raise InvalidPolicyUpdate("retained native artifact differs from its declared digest")
        return cls.from_rollouts(
            rollouts,
            settings,
            capabilities,
            population_id=population_id,
            native_evidence_ref=artifact.name,
            read_evidence=lambda _: evidence,
            decode=lambda raw: decode_native_population(raw, format=format, content="conditioning"),
            template_revision=template_revision,
            versions=versions,
            sampler_step=sampler_step,
            selector_digest=selector_digest,
            spans=spans,
            process_credit=process_credit,
            applied_update_offset=applied_update_offset,
            attempt_offset=attempt_offset,
            max_overflow_retries=max_overflow_retries,
        )

    @classmethod
    def from_rollouts(
        cls,
        rollouts: tuple[EnvironmentRollout, ...],
        settings: GRPOSettings | SAMPOSettings | GDPOSettings | CAPOSettings,
        capabilities: ExecutionCapabilities,
        *,
        population_id: str,
        native_evidence_ref: str,
        read_evidence: Callable[[str], bytes],
        decode: Callable[[bytes], Mapping[str, Any]],
        template_revision: str,
        versions: PolicyVersions,
        sampler_step: int,
        selector_digest: str,
        spans: tuple[SemanticSpan, ...] = (),
        process_credit: Any = None,
        applied_update_offset: int = 0,
        attempt_offset: int = 0,
        max_overflow_retries: int = 0,
    ) -> AdmittedNativePopulation:
        """Resolve complete groups against bytes already retained by the host.

        Artifact resolution may use local or remote storage. A missing artifact
        is an admission failure; a caller cannot replace it with flattened rows.
        Evidence readers/decoders propagate their errors without backend mutation.
        """
        cls._validate_counters(applied_update_offset, attempt_offset, max_overflow_retries)
        if type(sampler_step) is not int or sampler_step != applied_update_offset:
            raise InvalidPolicyUpdate("fresh native admission requires sampling at the current applied boundary")
        evidence = cls._read_evidence(read_evidence, native_evidence_ref)
        updates = settings.policy_updates
        if not spans and updates is not None and updates.objective_variant == "semantic-spans":
            # Derive environment-owned reasoning/answer spans from the retained
            # original traces, so selection addresses exactly admitted actions.
            from ..update_spans import reasoning_answer_spans

            spans = reasoning_answer_spans(rollouts, decode(evidence))
        resolved = resolve_rollout_population(
            rollouts,
            settings,
            capabilities,
            population_id=population_id,
            native_evidence_ref=native_evidence_ref,
            native_evidence_digest=hashlib.sha256(evidence).hexdigest(),
            template_revision=template_revision,
            versions=versions,
            sampler_step=sampler_step,
            selector_digest=selector_digest,
            spans=spans,
        )
        reader = NativePopulationInputs.from_evidence(resolved.snapshot, evidence, decode)
        selected_estimator = updates.credit_estimator if updates is not None else None
        if (selected_estimator is None) != (process_credit is None):
            raise InvalidPolicyUpdate("process credit requires both a selected estimator and an injected provider")
        if process_credit is not None:
            # The composition-owned provider scores retained spans and returns
            # validated detached credit; the algorithm's own credit is replaced.
            if process_credit.estimator_id != selected_estimator:
                raise InvalidPolicyUpdate("injected process credit differs from the selected estimator")
            credit = process_credit.prepare(resolved.snapshot, reader)
            if credit.estimator_id != selected_estimator or credit.population_digest != resolved.snapshot.digest:
                raise InvalidPolicyUpdate(
                    "process credit provider returned credit for a different estimator or population"
                )
            resolved = resolve_policy_population(resolved.snapshot, credit, settings, capabilities)
        views = {view.id: view for view in resolved.snapshot.conditioning}
        for rollout in rollouts:
            for record, indices in zip(
                rollout.conditioning_records, rollout.conditioning_completion_indices, strict=True
            ):
                inputs = reader(views[f"{record.trace_id}/node-{record.node_index}"])
                targets = {index: inputs.token_ids[position] for index, position in inputs.action_positions}
                if any(
                    rollout.completion_ids[completion] != targets[native]
                    for native, completion in zip(record.sampled_token_indices, indices, strict=True)
                ):
                    raise InvalidPolicyUpdate("native rollout sampled tokens differ from retained original inputs")
        return cls(resolved, reader, applied_update_offset, attempt_offset, max_overflow_retries)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Path,
        identity: UpdateRecoveryIdentity,
        *,
        sampler_correction: Mapping[ActionRef, float] | None,
        read_evidence: Callable[[str], bytes] | None = None,
        decode: Callable[[bytes], Mapping[str, Any]],
    ) -> AdmittedNativePopulation:
        """Verify the native seal, restore frozen credit, then admit original inputs.

        With no artifact resolver, require the original evidence bytes inside
        the sealed checkpoint. Legacy checkpoints need an explicit resolver.
        This performs no credit estimation or native state loading. The backend
        still restores its sealed old scores, model, optimizer, RNG and scaler.
        """
        from ..update_recovery import load_update_recovery
        from .policy_update_recovery import NATIVE_EVIDENCE_FILENAME, load_retained_population

        retained = load_retained_population(checkpoint, identity, sampler_correction=sampler_correction)
        resolved = retained.resolved
        if read_evidence is None:
            state = load_update_recovery(checkpoint, identity)
            if not any(
                component.role == "native-rollout-evidence" and component.relative_path == NATIVE_EVIDENCE_FILENAME
                for component in state.components
            ):
                raise InvalidPolicyUpdate(
                    "checkpoint lacks sealed native evidence; supply a retained artifact resolver"
                )
            evidence = (checkpoint / NATIVE_EVIDENCE_FILENAME).read_bytes()
        else:
            evidence = cls._read_evidence(read_evidence, resolved.snapshot.native_evidence_ref)
        reader = NativePopulationInputs.from_evidence(resolved.snapshot, evidence, decode)
        return cls(
            resolved, reader, retained.applied_update_offset, retained.attempt_offset, retained.max_overflow_retries
        )

    @staticmethod
    def _read_evidence(read_evidence: Callable[[str], bytes], reference: str) -> bytes:
        evidence = read_evidence(reference)
        if not isinstance(evidence, bytes):
            raise InvalidPolicyUpdate("native artifact reader must return original retained bytes")
        return evidence

    @staticmethod
    def _validate_counters(applied: int, attempts: int, retries: int) -> None:
        if any(type(value) is not int or value < 0 for value in (applied, attempts, retries)) or attempts < applied:
            raise InvalidPolicyUpdate("native admission requires coherent applied/attempt offsets and retry budget")
