"""A real local span scorer: mean scorer-model log-probability of original tokens.

The scorer reads each span's exact original conditioning input through the
population reader, so it observes only the span's prefix (and the span itself),
never later turns. It returns quality evidence, not advantages; an explicitly
identified estimator must turn its assessments into detached credit.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from posttrain.train.reward_evidence import ObservationScope, RewardValue, SpanAssessment


class LikelihoodSpanScorer:
    """Score retained spans by the injected model's mean token log-probability."""

    observation_scope: ObservationScope = "prefix"

    def __init__(self, model: Any, *, model_id: str, model_revision: str, device: Any, temperature: float = 1.0):
        if not model_id.strip() or not model_revision.strip() or not temperature > 0:
            raise ValueError("likelihood scorer requires a pinned model identity and positive temperature")
        self.model, self.device, self.temperature = model, device, temperature
        self.model_id, self.model_revision = model_id, model_revision

    @property
    def revision(self) -> str:
        return f"likelihood-span-scorer@1/{self.model_id}@{self.model_revision}/t{self.temperature!r}"

    def assess(
        self, snapshot: Any, spans: tuple[Any, ...], read_input: Callable[[Any], Any]
    ) -> tuple[SpanAssessment, ...]:
        import torch

        views = {view.id: view for view in snapshot.conditioning}
        # Per sampled turn keep only log-probabilities at original action
        # positions; the full vocabulary distribution is freed immediately.
        cache: dict[str, tuple[Any, dict[int, float]]] = {}
        results = []
        self.model.eval()
        for span in spans:
            turns = {interval.start.turn_id for interval in span.action_intervals}
            if len(turns) != 1:
                raise ValueError("likelihood scoring requires one original sampled turn per span")
            view_id = next(iter(turns))
            if view_id not in cache:
                inputs = read_input(views[view_id])
                ids = torch.tensor([inputs.token_ids], device=self.device)
                positions = dict(inputs.action_positions)
                with torch.no_grad():
                    logits = self.model(input_ids=ids).logits[0].float() / self.temperature
                    rows = torch.tensor([position - 1 for position in positions.values()], device=logits.device)
                    targets = torch.tensor(
                        [inputs.token_ids[position] for position in positions.values()], device=logits.device
                    )
                    selected = logits[rows].gather(1, targets.unsqueeze(1)).squeeze(1) - logits[rows].logsumexp(-1)
                del logits
                cache[view_id] = (
                    inputs,
                    {index: float(value) for index, value in zip(positions, selected.tolist(), strict=True)},
                )
            inputs, logprobs = cache[view_id]
            values = [
                logprobs[index]
                for interval in span.action_intervals
                for index in range(interval.start.token_index, interval.end)
            ]
            results.append(
                SpanAssessment(
                    snapshot.native_evidence_ref,
                    span.id,
                    (RewardValue("mean_token_logprob", "valid", sum(values) / len(values)),),
                    self.revision,
                    f"{view_id}@{inputs.record.input_digest}",
                    "prefix",
                    span.role,
                    f"{self.model_id}@{self.model_revision}",
                )
            )
        return tuple(results)


def group_centered_likelihood_estimate(assessments: tuple[SpanAssessment, ...], snapshot: Any) -> dict[str, float]:
    """``group-centered-likelihood@1``: span mean log-prob minus its prompt group's mean.

    Groups are the snapshot's complete prompt-group relations; a span belongs to
    the group containing its original actions. Spans are compared only within a
    group, so credit is relative quality among samples of one prompt.
    """
    spans = {span.id: span for span in snapshot.spans}
    # Relations name whole turns; a span's group is that of the turns it covers.
    member_group = {
        turn: relation.id
        for relation in snapshot.relations
        if relation.kind == "prompt-group"
        for turn in relation.members
    }
    grouped: dict[str, list[tuple[str, float]]] = {}
    for assessment in assessments:
        groups = {member_group.get(interval.start.turn_id) for interval in spans[assessment.span_id].action_intervals}
        if len(groups) != 1 or None in groups:
            raise ValueError("process span must belong to exactly one complete prompt group")
        value = assessment.components[0].value
        if value is None:
            raise ValueError("group-centered likelihood requires valid span scores")
        grouped.setdefault(next(iter(groups)) or "", []).append((assessment.span_id, float(value)))
    estimate: dict[str, float] = {}
    for members in grouped.values():
        mean = sum(value for _, value in members) / len(members)
        estimate.update({span_id: value - mean for span_id, value in members})
    return estimate


def likelihood_process_credit(
    model: Any, *, model_id: str, model_revision: str, device: Any, temperature: float = 1.0
) -> Any:
    """Composition factory for ``credit_estimator: group-centered-likelihood@1``."""
    from posttrain.train.update_process_credit import ScoredSpanCreditProvider

    scorer = LikelihoodSpanScorer(
        model, model_id=model_id, model_revision=model_revision, device=device, temperature=temperature
    )
    return ScoredSpanCreditProvider(
        scorer, ("reasoning",), "group-centered-likelihood", "1", group_centered_likelihood_estimate
    )
