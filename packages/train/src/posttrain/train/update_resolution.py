"""Resolve one admitted population before either native backend mutates policy.

Collection supplies complete evidence and prepared credit. This boundary binds
the selected objective and schedule and checks every execution pack up front;
it does not collect trajectories or replace backend capability qualification.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .online_rl import EnvironmentRollout
from .profiles import CAPOSettings, GDPOSettings, GRPOSettings, SAMPOSettings
from .update_credit import (
    PreparedCredit,
    SampoCreditEstimator,
    ScalarGroupCreditEstimator,
    StructuredCreditEstimator,
    prepare_credit,
)
from .update_evidence import population_from_rollouts
from .update_objectives import ObjectiveSpec, objective_population
from .update_plan import (
    ExecutionCapabilities,
    ExecutionPack,
    PolicyExecutionBudget,
    ResolvedUpdate,
    plan_packs,
    resolve_updates,
)
from .update_records import InvalidPolicyUpdate, PolicyVersions, PopulationSnapshot, SemanticSpan, payload_digest


@dataclass(frozen=True, slots=True, eq=False)
class ResolvedPolicyPopulation:
    """An admitted population with its credit, objective, occurrences and packs.

    Built by ``resolve_policy_population`` (or restored from a checkpoint that
    recomputes the same structures); the checks here bind the parts together
    by identity and do not resolve them again.
    """

    snapshot: PopulationSnapshot
    credit: PreparedCredit
    spec: ObjectiveSpec
    updates: tuple[ResolvedUpdate, ...]
    packs: tuple[tuple[ExecutionPack, ...], ...]
    execution: PolicyExecutionBudget
    capabilities: ExecutionCapabilities

    def __post_init__(self) -> None:
        if not self.updates or len(self.packs) != len(self.updates):
            raise InvalidPolicyUpdate("resolved population requires every occurrence and execution plan")
        self.credit.validate(self.snapshot)
        for update, packs in zip(self.updates, self.packs, strict=True):
            if update.population.digest != self.snapshot.digest:
                raise InvalidPolicyUpdate("resolved occurrence belongs to different native evidence")
            if update.objective.credit_digest != self.credit.digest or update.objective.contract_digest != (
                self.spec.digest
            ):
                raise InvalidPolicyUpdate("resolved occurrence objective differs from its credit or specification")
            # An occurrence that selects nothing (zero-weight episodes) scores no turns and has no packs.
            if bool(packs) != bool(update.views) or any(
                pack.update_digest != update.digest or pack.index != index for index, pack in enumerate(packs)
            ):
                raise InvalidPolicyUpdate("resolved execution plans changed after population validation")
            if sorted(view for pack in packs for view in pack.views) != sorted(update.views):
                raise InvalidPolicyUpdate("resolved execution plans must cover each occurrence's turns exactly once")

    @property
    def digest(self) -> str:
        """Identity of the whole resolved contract: evidence, credit, objective, occurrences, packs and budgets."""
        return payload_digest(
            {
                "snapshot": self.snapshot.digest,
                "credit": self.credit.digest,
                "spec": self.spec.digest,
                "updates": [update.digest for update in self.updates],
                "packs": [[asdict(pack) for pack in packs] for packs in self.packs],
                "execution": asdict(self.execution),
                "capabilities": asdict(self.capabilities),
            }
        )


def resolve_policy_population(
    snapshot: PopulationSnapshot,
    credit: PreparedCredit,
    settings: GRPOSettings | SAMPOSettings | GDPOSettings | CAPOSettings,
    capabilities: ExecutionCapabilities,
) -> ResolvedPolicyPopulation:
    """Translate supported typed selections without changing algorithm meaning.

    Credit must already be prepared on the complete population by its admitted
    estimator. Fail the entire population if any occurrence exceeds capacity;
    callers must not discover an invalid later pack after earlier updates apply.
    """
    selection = settings.policy_updates
    if selection is None:
        raise InvalidPolicyUpdate("resolved population requires explicit policy_updates")
    selection.validate_legacy_loop(
        max_steps=settings.loop.max_steps,
        per_device_batch_size=settings.loop.per_device_batch_size,
        gradient_accumulation_steps=settings.loop.gradient_accumulation_steps,
    )
    if settings.mask_truncated_completions:
        raise InvalidPolicyUpdate("resolved completion-truncation masking requires a qualified objective variant")
    if isinstance(settings, SAMPOSettings):
        identity = {"algorithm": "sampo@1", "semantic-spans": "sampo-spans@1", "turn-rows": "sampo-turns@1"}[
            selection.objective_variant
        ]
        clip_high = settings.clip_epsilon_high
    else:
        if selection.objective_variant != "algorithm":
            raise InvalidPolicyUpdate(
                "selected objective variant (semantic spans or turn rows) requires an explicitly supported algorithm"
            )
        if isinstance(settings, GRPOSettings):
            if settings.algorithm not in {"grpo", "dapo"}:
                raise InvalidPolicyUpdate("resolved population does not support this GRPO recipe")
            identity = f"{settings.algorithm}@1"
            clip_high = settings.resolved_clip_epsilon_high
        else:
            identity = "gdpo@1" if isinstance(settings, GDPOSettings) else "capo@1"
            clip_high = settings.clip_epsilon_high
    spec = ObjectiveSpec(
        identity,
        clip_low=settings.clip_epsilon_low,
        clip_high=clip_high,
        beta=settings.beta,
        policy_selection=selection.policy_selection,
        kl_selection=selection.kl_selection,
        denominator=selection.denominator,
        empty_policy=selection.empty_policy,
    )
    objective = objective_population(snapshot, spec, credit)
    updates = resolve_updates(snapshot, selection.schedule, objective)
    packs = tuple(plan_packs(update, selection.execution, capabilities) for update in updates)
    return ResolvedPolicyPopulation(snapshot, credit, spec, updates, packs, selection.execution, capabilities)


def resolve_rollout_population(
    rollouts: tuple[EnvironmentRollout, ...],
    settings: GRPOSettings | SAMPOSettings | GDPOSettings | CAPOSettings,
    capabilities: ExecutionCapabilities,
    *,
    population_id: str,
    native_evidence_ref: str,
    native_evidence_digest: str,
    template_revision: str,
    versions: PolicyVersions,
    sampler_step: int,
    selector_digest: str,
    spans: tuple[SemanticSpan, ...] = (),
) -> ResolvedPolicyPopulation:
    """Prepare existing native credit once on a complete admitted population.

    The collector must persist native bytes and admit complete groups first.
    It supplies original unshaped rewards. SAMPO shaping affects estimator input
    only: native traces and action/context identities remain replay authority.
    GRPO/DAPO delegate to the explicit scalar-group adapter; they cannot fall
    through to a hierarchical or structured estimator.
    """
    if not isinstance(settings, GRPOSettings | SAMPOSettings | GDPOSettings | CAPOSettings):
        raise InvalidPolicyUpdate("native rollout resolution requires a supported credit adapter")
    snapshot, rows = population_from_rollouts(
        rollouts,
        population_id=population_id,
        native_evidence_ref=native_evidence_ref,
        native_evidence_digest=native_evidence_digest,
        template_revision=template_revision,
        versions=versions,
        sampler_step=sampler_step,
        num_generations=settings.num_generations,
        selector_digest=selector_digest,
        spans=spans,
        anchor_fallback=isinstance(settings, SAMPOSettings) and settings.anchor_fallback == "environment-state",
    )
    if isinstance(settings, GRPOSettings):
        required = tuple(relation.id for relation in snapshot.relations if relation.kind == "prompt-group")
        credit = prepare_credit(snapshot, ScalarGroupCreditEstimator(settings, rows, required))
    elif isinstance(settings, SAMPOSettings):
        required = tuple(relation.id for relation in snapshot.relations)
        credit = prepare_credit(snapshot, SampoCreditEstimator(settings, rows, required))
    else:
        if any(rollout.is_truncated for rollout in rows.rollouts):
            raise InvalidPolicyUpdate("structured native credit requires nontruncated admitted rollouts")
        required = tuple(relation.id for relation in snapshot.relations if relation.kind == "prompt-group")
        credit = prepare_credit(snapshot, StructuredCreditEstimator(settings, rows, required))
    return resolve_policy_population(snapshot, credit, settings, capabilities)
