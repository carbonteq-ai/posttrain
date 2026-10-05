"""Actor-owned resolved population state behind native veRL worker RPCs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from posttrain.common import NullObserver, Observer, RunContext

from ...profiles import CAPOSettings, GDPOSettings, GRPOSettings, SAMPOSettings
from ...update_objectives import resolve_objective_term
from ...update_plan import ExecutionCapabilities
from ...update_records import InvalidPolicyUpdate
from ...update_recovery import inspect_update_recovery
from ...update_sampler_correction import sampler_correction_recipe
from ...update_telemetry import update_metrics
from ..policy_update_admission import AdmittedNativePopulation
from .contracts import VerlLaunchManifest
from .policy_job import ResolvedVeRLCollectionHost
from .policy_updates import ResolvedVeRLPopulation


@dataclass
class ResolvedVeRLActorSession:
    """One optimizer owner; driver receipts are supplied only for fresh collection.

    Native worker/driver transport remains responsible for RPC dispatch, policy
    synchronization and rollout storage. This session never invokes a driver
    callback from inside an actor RPC, avoiding cyclic driver/actor calls.
    """

    context: RunContext
    settings: GRPOSettings | SAMPOSettings | GDPOSettings | CAPOSettings
    capabilities: ExecutionCapabilities
    engine: Any
    restore_population: Callable[[Path], ResolvedVeRLPopulation]
    runtime_identity: str
    template_revision: str
    score_temperature: float
    score_contract: str
    max_overflow_retries: int = 0
    reference_scores: Callable[[ResolvedVeRLPopulation], Any] | None = None
    host: ResolvedVeRLCollectionHost = field(init=False)
    _receipts: tuple[str, ...] | None = field(default=None, init=False)
    _collection_boundary: tuple[int, int] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.host = ResolvedVeRLCollectionHost(
            self.context,
            self.settings,
            self.capabilities,
            self.engine,
            self._take_receipts,
            self.restore_population,
            self.runtime_identity,
            self.template_revision,
            self.score_temperature,
            self.score_contract,
            self.max_overflow_retries,
            self.reference_scores,
        )

    def state(self) -> dict[str, int | bool]:
        current = self.host.run.current
        applied = self.engine.lr_scheduler.last_epoch
        if type(applied) is not int or applied < 0 or (current is None and applied != 0):
            raise InvalidPolicyUpdate("native actor counter lacks its resolved population state")
        if current is not None and (current.global_applied_updates != applied or current._pending is not None):
            raise InvalidPolicyUpdate("resolved actor is not at a complete native update boundary")
        return {
            "applied": applied,
            "attempts": current.global_attempts if current is not None else 0,
            "needs_population": current is None or current.next_update == len(current.updates),
        }

    def _take_receipts(self, applied: int, attempts: int) -> Any:
        import numpy as np
        from verl.protocol import DataProto  # pyright: ignore[reportMissingImports]

        if self._receipts is None or self._collection_boundary != (applied, attempts):
            raise InvalidPolicyUpdate("resolved actor collection lacks receipts at its exact policy boundary")
        receipts = self._receipts
        self._receipts = self._collection_boundary = None
        return DataProto(non_tensor_batch={"posttrain_native_episode_receipt": np.asarray(receipts, dtype=object)})

    def update(self, receipts: tuple[str, ...] | None, *, expected_applied: int) -> Mapping[str, float]:
        before = self.state()
        if type(expected_applied) is not int or before["applied"] != expected_applied or self._receipts is not None:
            raise InvalidPolicyUpdate("driver update differs from the native actor boundary")
        if before["needs_population"]:
            if (
                not isinstance(receipts, tuple)
                or len(receipts) != (self.settings.num_prompts_per_step * self.settings.num_generations)
                or any(not isinstance(value, str) or not value for value in receipts)
            ):
                raise InvalidPolicyUpdate("resolved actor requires one complete native receipt population")
            self._receipts = receipts
            self._collection_boundary = (expected_applied, int(before["attempts"]))
        elif receipts is not None:
            raise InvalidPolicyUpdate("retained actor population cannot consume a fresh driver batch")
        output = self.host.run_update()
        after = self.state()
        if after["applied"] != expected_applied + 1:
            raise InvalidPolicyUpdate("native actor did not commit exactly one resolved optimizer update")
        population = self.host.run.active()
        adjoints = population.last_adjoints
        term = resolve_objective_term(
            population.updates[population.next_update - 1], population.spec, population.credit
        )
        evaluation = adjoints.evaluation
        metrics = {
            "train/rl/loss": float(output["loss"]),
            "train/rl/policy_loss": float(output["policy_loss"]),
            "train/rl/kl_loss": float(output["kl_loss"]),
            "train/rl/applied_optimizer_updates": float(after["applied"]),
            "train/rl/optimizer_attempts": float(after["attempts"]),
            "train/rl/selected_policy_actions": float(len(term.policy_weights)),
            "train/rl/selected_kl_actions": float(len(term.kl_weights)),
            "train/grad_norm": float(output["metrics"]["grad_norm"]),
        }
        # The same selected-action credit, clipping and correction evidence as
        # the resolved TRL job. Entropy is not measured on the score-adjoint path.
        metrics.update(
            update_metrics(
                term,
                population.credit,
                clipped_ratios={
                    action: float(evaluation.ratios[action].detach()) for action in evaluation.clipped_actions
                },
                sampler_correction=population.sampler_correction,
                correction_recipe=sampler_correction_recipe(self.settings),
                old_scores=(
                    None
                    if population.old is None
                    else {action: float(value) for action, value in population.old.values.items()}
                ),
                sampled_scores=population.sampled_scores,
            )
        )
        for name, value in self.engine.last_loss_scale_metrics.items():
            metrics[f"train/{name}"] = float(value)
        return metrics

    def save_checkpoint(self, checkpoint: Path, *, expected_applied: int) -> Any:
        if self.state()["applied"] != expected_applied or self._receipts is not None:
            raise InvalidPolicyUpdate("driver checkpoint differs from the committed actor boundary")
        return self.host.save_checkpoint(checkpoint)

    def load_checkpoint(self, checkpoint: Path) -> Any:
        if self.host.run.current is not None or self._receipts is not None:
            raise InvalidPolicyUpdate("resolved actor recovery cannot replace an active population")
        return self.host.load_checkpoint(checkpoint)


def actor_session_from_manifest(
    manifest: VerlLaunchManifest,
    engine: Any,
    observer: Observer,
    *,
    capabilities: ExecutionCapabilities,
    runtime_identity: str,
    template_revision: str,
    score_temperature: float,
    score_contract: str,
    max_overflow_retries: int = 0,
    reference_scores: Callable[[ResolvedVeRLPopulation], Any] | None = None,
) -> ResolvedVeRLActorSession:
    """Bind the real host identity and restore sealed credit without recomputing it.

    Native composition supplies qualified runtime/score identities and an
    observer transport; it must not substitute a NullObserver for a launched
    job. Model construction and reference-policy ownership remain native.
    """
    selected = manifest.payload.resolved_settings
    if selected is None or manifest.run_context is None:
        raise InvalidPolicyUpdate("resolved native actor requires typed settings and the actual host context")
    if isinstance(observer, NullObserver):
        raise InvalidPolicyUpdate("resolved native actor requires a publishing observer transport")
    context = RunContext(**manifest.run_context.model_dump(), observer=observer)

    def decode(evidence: bytes) -> Mapping[str, Any]:
        import json

        from ...integrations.verifiers_population_artifact import decode_native_population

        if not evidence or not evidence.endswith(b"\n"):
            raise InvalidPolicyUpdate("resolved actor recovery requires complete native JSONL")
        first = json.loads(evidence.splitlines()[0])
        if not isinstance(first, dict):
            raise InvalidPolicyUpdate("resolved actor recovery requires native record objects")
        return decode_native_population(
            evidence, format=("verifiers-native-episodes" if "traces" in first else "verifiers-native-traces")
        )

    def restore(checkpoint: Path) -> ResolvedVeRLPopulation:
        from ..policy_update_recovery import load_sampler_correction

        state = inspect_update_recovery(checkpoint)
        if (
            state.identity.runtime_identity != runtime_identity
            or state.identity.world_size != engine.get_data_parallel_size()
        ):
            raise InvalidPolicyUpdate("resolved actor recovery differs from the actual native runtime")
        correction = load_sampler_correction(checkpoint, state.identity)
        admitted = AdmittedNativePopulation.from_checkpoint(
            checkpoint, state.identity, sampler_correction=correction, decode=decode
        )
        return ResolvedVeRLPopulation.from_admitted(
            admitted, score_temperature=score_temperature, score_contract=score_contract, sampler_correction=correction
        )

    return ResolvedVeRLActorSession(
        context,
        selected.settings,
        capabilities,
        engine,
        restore,
        runtime_identity,
        template_revision,
        score_temperature,
        score_contract,
        max_overflow_retries,
        reference_scores,
    )
