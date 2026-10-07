"""Bind native veRL collection batches to the resolved engine lifecycle."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from posttrain.common import RunContext

from ...profiles import CAPOSettings, GDPOSettings, GRPOSettings, SAMPOSettings
from ...update_plan import ExecutionCapabilities
from ...update_records import InvalidPolicyUpdate, PolicyVersions
from ...update_resolution import resolve_policy_population
from ...update_sampler_correction import recipe_sampler_correction_weights
from ..policy_update_admission import AdmittedNativePopulation
from .policy_rollouts import admit_episode_receipts
from .policy_updates import ResolvedVeRLPopulation, ResolvedVeRLRun


def episode_receipts_from_batch(batch: Any, *, expected_rows: int) -> tuple[str, ...]:
    """Read veRL's native non-tensor column without consulting flattened tokens."""
    columns = getattr(batch, "non_tensor_batch", None)
    if not isinstance(columns, Mapping) or "posttrain_native_episode_receipt" not in columns:
        raise InvalidPolicyUpdate("veRL native batch lacks episode receipts")
    values = columns["posttrain_native_episode_receipt"]
    if getattr(values, "ndim", None) != 1 or len(values) != expected_rows:
        raise InvalidPolicyUpdate("veRL native receipt column differs from the complete population size")
    receipts = tuple(values.tolist())
    if any(not isinstance(value, str) or not value for value in receipts):
        raise InvalidPolicyUpdate("veRL native receipt column contains missing or invalid episodes")
    return receipts


@dataclass
class ResolvedVeRLCollectionHost:
    """Keep one native engine while collecting only at population boundaries.

    The worker supplies native collection and authenticated population restoration.
    Engine/runtime initialization and rollout-policy synchronization remain worker
    responsibilities. This binding does not qualify those responsibilities.
    """

    context: RunContext
    settings: GRPOSettings | SAMPOSettings | GDPOSettings | CAPOSettings
    capabilities: ExecutionCapabilities
    engine: Any
    collect_batch: Callable[[int, int], Any]
    restore_population: Callable[[Path], ResolvedVeRLPopulation]
    runtime_identity: str
    template_revision: str
    score_temperature: float
    score_contract: str
    max_overflow_retries: int = 0
    reference_scores: Callable[[ResolvedVeRLPopulation], Any] | None = None
    run: ResolvedVeRLRun = field(init=False)

    def __post_init__(self) -> None:
        if self.settings.policy_updates is None:
            raise InvalidPolicyUpdate("veRL resolved host requires selected policy_updates")
        AdmittedNativePopulation._validate_counters(0, 0, self.max_overflow_retries)  # noqa: SLF001
        if self.settings.beta and self.reference_scores is None:
            raise InvalidPolicyUpdate("veRL resolved host requires selected frozen reference scoring")
        self.run = ResolvedVeRLRun(self.collect, self.restore)

    def restore(self, checkpoint: Path) -> ResolvedVeRLPopulation:
        population = self.restore_population(checkpoint)
        if population.sampler_correction is None:
            raise InvalidPolicyUpdate("selected ordinary recipe requires retained sampler correction")
        snapshot = population.updates[0].population
        selected = resolve_policy_population(snapshot, population.credit, self.settings, self.capabilities)
        version = f"{self.runtime_identity}/actor-{population.applied_update_offset}"
        versions = PolicyVersions(
            version, version, version, f"{self.runtime_identity}/reference" if self.settings.beta else None
        )
        if (
            # Updates are identified by digest (their population snapshots compare by identity).
            tuple(update.digest for update in population.updates) != tuple(update.digest for update in selected.updates)
            or population.spec != selected.spec
            or population.execution != selected.execution
            or population.capabilities != selected.capabilities
            or population.max_overflow_retries != self.max_overflow_retries
            or snapshot.versions != versions
            or population.score_contract != self.score_contract
            or population.score_temperature != self.score_temperature
            or any(view.template_revision != self.template_revision for view in snapshot.conditioning)
        ):
            raise InvalidPolicyUpdate("veRL restored population differs from selected job contracts")
        return population

    def collect(self, applied: int, attempts: int) -> ResolvedVeRLPopulation:
        if self.engine.lr_scheduler.last_epoch != applied:
            raise InvalidPolicyUpdate("veRL collection differs from the native applied boundary")
        batch = self.collect_batch(applied, attempts)
        receipts = episode_receipts_from_batch(
            batch, expected_rows=self.settings.num_prompts_per_step * self.settings.num_generations
        )
        version = f"{self.runtime_identity}/actor-{applied}"
        admitted, artifact = admit_episode_receipts(
            receipts,
            self.settings,
            self.capabilities,
            destination=self.context.workspace / "native-populations",
            population_id=f"{self.context.run_id}/population-at-{applied}",
            template_revision=self.template_revision,
            versions=PolicyVersions(
                version, version, version, f"{self.runtime_identity}/reference" if self.settings.beta else None
            ),
            sampler_step=applied,
            selector_digest=f"{self.runtime_identity}/native-selection-{applied}",
            applied_update_offset=applied,
            attempt_offset=attempts,
            max_overflow_retries=self.max_overflow_retries,
        )
        population = ResolvedVeRLPopulation.from_admitted(
            admitted,
            score_temperature=self.score_temperature,
            score_contract=self.score_contract,
            sampler_correction=None,
        )
        sampled = admitted.read_input.sampling_log_scores(admitted.resolved.snapshot)
        population.sampled_scores = sampled
        population.prepare_sampler_correction = lambda old: recipe_sampler_correction_weights(
            self.settings, admitted.resolved.snapshot, old, sampled
        )
        if self.reference_scores is not None and population.spec.beta:
            population.reference = self.reference_scores(population)
        self.context.artifact(artifact)
        return population

    def run_update(self) -> Any:
        return self.run.run_update(self.engine)

    def save_checkpoint(self, checkpoint: Path) -> Any:
        return self.run.save_native_checkpoint(self.engine, checkpoint, runtime_identity=self.runtime_identity)

    def load_checkpoint(self, checkpoint: Path) -> Any:
        return self.run.load_native_checkpoint(self.engine, checkpoint, runtime_identity=self.runtime_identity)
