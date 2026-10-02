"""Step-addressed process credit: injected scorers and explicit external estimators.

A scorer is injected by the host composition and returns ``SpanAssessment``
evidence for retained span IDs: quality scores with their scorer revision,
observed input and observation scope. Assessments are never advantages. An
explicitly identified external estimator turns a complete set of assessments
into detached per-span advantages; ``prepare_credit`` then validates them like
any other estimator's output. Training reads the retained assessments and the
estimator identity, not a replacement mask.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from .reward_evidence import ObservationScope, SpanAssessment
from .update_credit import ActionCredit, PreparedCredit
from .update_records import ActionRef, InvalidPolicyUpdate, PopulationSnapshot, SemanticSpan, require_identity


class SpanScorer(Protocol):
    """Composition-injected scorer; train never imports a scorer backend."""

    @property
    def revision(self) -> str: ...

    @property
    def observation_scope(self) -> ObservationScope: ...

    def assess(self, snapshot: PopulationSnapshot, spans: tuple[SemanticSpan, ...],
               read_input: Callable[[Any], Any]) -> tuple[SpanAssessment, ...]: ...


def assess_population_spans(
    snapshot: PopulationSnapshot, scorer: SpanScorer, read_input: Callable[[Any], Any], *, roles: tuple[str, ...],
) -> tuple[SpanAssessment, ...]:
    """Score every retained span of the selected roles exactly once, by span ID."""
    require_identity(scorer.revision, *roles)
    spans = tuple(span for span in snapshot.spans if span.role in roles)
    if not spans:
        raise InvalidPolicyUpdate("process scoring requires retained spans of the selected roles")
    assessments = scorer.assess(snapshot, spans, read_input)
    if not isinstance(assessments, tuple) or not all(isinstance(value, SpanAssessment) for value in assessments):
        raise InvalidPolicyUpdate("span scorer must return retained span assessments")
    by_span = {value.span_id: value for value in assessments}
    if len(by_span) != len(assessments) or set(by_span) != {span.id for span in spans}:
        raise InvalidPolicyUpdate("span scorer must assess every selected retained span exactly once")
    for value in assessments:
        if (value.scorer_revision != scorer.revision or value.observation_scope != scorer.observation_scope
                or value.evidence_ref != snapshot.native_evidence_ref):
            raise InvalidPolicyUpdate("span assessment differs from the injected scorer or retained evidence")
    return assessments


@dataclass(frozen=True, slots=True)
class ExternalSpanCreditEstimator:
    """Independently specified estimator from span assessments to detached advantages.

    ``estimate`` receives the complete assessments of one population and returns
    one finite advantage per assessed span. Actions outside assessed spans get
    zero credit. Overlapping assessed spans are rejected: combining step credit
    on shared actions needs its own declared estimator.
    """

    estimator_id: str
    revision: str
    assessments: tuple[SpanAssessment, ...]
    estimate: Callable[[tuple[SpanAssessment, ...], PopulationSnapshot], Mapping[str, float]]
    evidence_digests: tuple[str, ...]
    required_relations: tuple[str, ...] = ()

    @property
    def id(self) -> str:
        return f"{self.estimator_id}@{self.revision}"

    def prepare(self, snapshot: PopulationSnapshot) -> PreparedCredit:
        require_identity(self.estimator_id, self.revision)
        if not self.assessments:
            raise InvalidPolicyUpdate("external span credit requires retained assessments")
        scope: ObservationScope = self.assessments[0].observation_scope
        if any(value.observation_scope != scope for value in self.assessments):
            raise InvalidPolicyUpdate("external span credit requires one declared observation scope")
        spans = {span.id: span for span in snapshot.spans}
        values = self.estimate(self.assessments, snapshot)
        if not isinstance(values, Mapping) or set(values) != {value.span_id for value in self.assessments}:
            raise InvalidPolicyUpdate("external estimator must return one advantage per assessed span")
        credit: dict[ActionRef, float] = {record.action: 0.0 for record in snapshot.actions}
        credited: set[ActionRef] = set()
        for span_id, advantage in values.items():
            if isinstance(advantage, bool) or not isinstance(advantage, int | float) or not math.isfinite(advantage):
                raise InvalidPolicyUpdate("external estimator returned a non-finite or non-numeric advantage")
            actions = set(spans[span_id].actions())
            if actions & credited:
                raise InvalidPolicyUpdate("overlapping assessed spans need a declared credit-combination estimator")
            credited |= actions
            for action in actions:
                credit[action] = float(advantage)
        return PreparedCredit(snapshot.digest, self.id,
                              tuple(ActionCredit(action, value) for action, value in sorted(credit.items())),
                              self.required_relations, (("process", 1.0),), f"{self.estimator_id}-detached@1",
                              scope, self.evidence_digests)


class ProcessCreditProvider(Protocol):
    """Composition-injected source of detached process credit for one population."""

    @property
    def estimator_id(self) -> str: ...

    def prepare(self, snapshot: PopulationSnapshot, read_input: Callable[[Any], Any]) -> PreparedCredit: ...


@dataclass(frozen=True, slots=True)
class ScoredSpanCreditProvider:
    """Score retained spans with an injected scorer, then apply an explicit estimator.

    ``estimator_id`` is the full identity a settings selection names (for
    example ``group-centered-likelihood@1``); assessments stay evidence and only
    the estimator's validated output becomes detached credit.
    """

    scorer: SpanScorer
    roles: tuple[str, ...]
    estimator_name: str
    estimator_revision: str
    estimate: Callable[[tuple[SpanAssessment, ...], PopulationSnapshot], Mapping[str, float]]
    last_assessments: list[tuple[SpanAssessment, ...]] = field(default_factory=list)

    @property
    def estimator_id(self) -> str:
        return f"{self.estimator_name}@{self.estimator_revision}"

    def prepare(self, snapshot: PopulationSnapshot, read_input: Callable[[Any], Any]) -> PreparedCredit:
        from .update_credit import prepare_credit

        assessments = assess_population_spans(snapshot, self.scorer, read_input, roles=self.roles)
        self.last_assessments.append(assessments)
        return prepare_credit(snapshot, ExternalSpanCreditEstimator(
            self.estimator_name, self.estimator_revision, assessments, self.estimate,
            (snapshot.native_evidence_digest,)))
