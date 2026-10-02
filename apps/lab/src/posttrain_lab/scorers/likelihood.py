"""A real local span scorer: mean scorer-model log-probability of original tokens.

The scorer reads each span's exact original conditioning input through the
population reader, so it observes only the span's prefix (and the span itself),
never later turns. It returns quality evidence, not advantages; an explicitly
identified estimator must turn its assessments into detached credit.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from posttrain.train.reward_evidence import RewardValue, SpanAssessment


class LikelihoodSpanScorer:
    """Score retained spans by the injected model's mean token log-probability."""

    observation_scope = "prefix"

    def __init__(self, model: Any, *, model_id: str, model_revision: str, device: Any, temperature: float = 1.0):
        if not model_id.strip() or not model_revision.strip() or not temperature > 0:
            raise ValueError("likelihood scorer requires a pinned model identity and positive temperature")
        self.model, self.device, self.temperature = model, device, temperature
        self.model_id, self.model_revision = model_id, model_revision

    @property
    def revision(self) -> str:
        return f"likelihood-span-scorer@1/{self.model_id}@{self.model_revision}/t{self.temperature!r}"

    def assess(self, snapshot: Any, spans: tuple[Any, ...],
               read_input: Callable[[Any], Any]) -> tuple[SpanAssessment, ...]:
        import torch

        views = {view.id: view for view in snapshot.conditioning}
        cache: dict[str, tuple[Any, Any, dict[int, int]]] = {}
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
                with torch.no_grad():
                    logits = self.model(input_ids=ids).logits[0].float() / self.temperature
                cache[view_id] = (inputs, logits.log_softmax(-1), dict(inputs.action_positions))
            inputs, logprobs, positions = cache[view_id]
            values = [float(logprobs[positions[action.token_index] - 1, inputs.token_ids[positions[action.token_index]]])
                      for interval in span.action_intervals for action in interval.actions()]
            results.append(SpanAssessment(
                snapshot.native_evidence_ref, span.id,
                (RewardValue("mean_token_logprob", "valid", sum(values) / len(values)),),
                self.revision, f"{view_id}@{inputs.record.input_digest}", "prefix", span.role,
                f"{self.model_id}@{self.model_revision}"))
        return tuple(results)
