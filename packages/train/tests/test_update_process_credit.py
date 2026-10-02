"""Process credit transport over the two-episode reasoning fixture (fake scorer only).

The fake scorer demonstrates transport of span-addressed quality evidence; the
independently specified estimator below supplies known detached advantages. The
live gate must replace the fake with a real injected local scorer.
"""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from posttrain.common import TraceObservation
from posttrain.environment.verifiers_conditioning import native_conditioning_records
from posttrain.train.backends.policy_update_admission import AdmittedNativePopulation
from posttrain.train.online_rl import AgenticTurn, BehaviorPolicySpan, EnvironmentRollout
from posttrain.train.reward_evidence import RewardValue, SpanAssessment
from posttrain.train.update_credit import prepare_credit
from posttrain.train.update_plan import PolicyExecutionBudget, PolicyUpdateSchedule, PolicyUpdateSettings
from posttrain.train.update_process_credit import ExternalSpanCreditEstimator, assess_population_spans
from posttrain.train.update_records import ActionSelection, InvalidPolicyUpdate, PolicyVersions
from posttrain.train.update_resolution import resolve_policy_population

from .test_update_resolution import capabilities, settings

FIXTURE = Path(__file__).parent / "fixtures" / "policy_updates" / "two-episode-reasoning.json"


def _decode(raw: bytes):
    return {record["id"]: SimpleNamespace(
        id=record["id"], nodes=[SimpleNamespace(**node) for node in record["nodes"]],
        calls=[SimpleNamespace(node=call["node"], usage=SimpleNamespace(**call["usage"])) for call in record["calls"]])
        for record in json.loads(raw)}


def _rollouts():
    traces = _decode(FIXTURE.read_bytes())
    return tuple(EnvironmentRollout(
        "task", (1, 2, 3), tuple(traces[name].nodes[1].token_ids[1:]), (-1.,) * 4, (True,) * 4, reward, False,
        TraceObservation("verifiers", name, {"info": {
            "posttrain_episode_id": f"episode-{name}", "posttrain_prompt_group_id": "group",
            "posttrain_rollout_id": f"rollout-{name}"}}),
        turns=(AgenticTurn(0, 4, "shared"),), behavior_policy=BehaviorPolicySpan(3, 3),
        conditioning_records=native_conditioning_records(traces[name], sampled_node_indices=(1,),
                                                        context_contract="causal-text@1"),
        selected_branch_id="1", conditioning_completion_indices=((0, 1, 2, 3),),
    ) for name, reward in (("trace-a", 1.), ("trace-b", 0.)))


def _population():
    evidence = FIXTURE.read_bytes()
    rollouts = _rollouts()
    selected = replace(settings(), policy_updates=PolicyUpdateSettings(
        PolicyUpdateSchedule("episode", 1), PolicyExecutionBudget(2, 100, 1000), objective_variant="semantic-spans",
        policy_selection=ActionSelection("roles", roles=("reasoning",)),
        kl_selection=ActionSelection("roles", roles=("reasoning", "answer"))))
    admitted = AdmittedNativePopulation.from_rollouts(
        rollouts, selected, capabilities(), population_id="population@3", native_evidence_ref="artifact:fixture",
        read_evidence=lambda reference: evidence, decode=_decode, template_revision="template@1",
        versions=PolicyVersions("sampler@3", "old@3", "current@3", None), sampler_step=3,
        selector_digest="fixture@1", applied_update_offset=3, attempt_offset=3)
    return admitted, selected


class FakeStepScorer:
    """Scores a reasoning span by its first original token; transport only."""

    revision = "fake-step-scorer@1"
    observation_scope = "prefix"

    def assess(self, snapshot, spans, read_input):
        views = {view.id: view for view in snapshot.conditioning}
        values = []
        for span in spans:
            first = span.action_intervals[0].start
            inputs = read_input(views[first.turn_id])
            token = dict(inputs.action_positions)[first.token_index]
            values.append(SpanAssessment(snapshot.native_evidence_ref, span.id,
                                         (RewardValue("step_quality", "valid", float(inputs.token_ids[token])),),
                                         self.revision, f"{first.turn_id}@prefix", "prefix", "reasoning-step",
                                         "fake-snapshot@1"))
        return tuple(values)


def _known_estimator(assessments, snapshot):
    # Independently specified: centered step quality, scaled to unit spread.
    quality = {value.span_id: value.components[0].value for value in assessments}
    mean = sum(quality.values()) / len(quality)
    return {span: (score - mean) / 2.0 for span, score in quality.items()}


def test_scored_reasoning_spans_become_validated_detached_credit():
    admitted, selected = _population()
    snapshot = admitted.resolved.snapshot
    assessments = assess_population_spans(snapshot, cast(Any, FakeStepScorer()), admitted.read_input, roles=("reasoning",))
    assert {value.span_id: value.components[0].value for value in assessments} == {
        "trace-a/node-1/reasoning": 10.0, "trace-b/node-1/reasoning": 12.0}
    credit = prepare_credit(snapshot, ExternalSpanCreditEstimator(
        "known-centered-step", "1", assessments, _known_estimator, (snapshot.native_evidence_digest,)))
    by_action = {(value.action.turn_id, value.action.token_index): value.advantage for value in credit.values}
    # Reasoning spans: trace-a tokens 1-2 (quality 10), trace-b tokens 1-3 (quality 12);
    # answer tokens carry no process credit.
    assert by_action == {("trace-a/node-1", 1): -0.5, ("trace-a/node-1", 2): -0.5,
                         ("trace-a/node-1", 3): 0.0, ("trace-a/node-1", 4): 0.0,
                         ("trace-b/node-1", 1): 0.5, ("trace-b/node-1", 2): 0.5, ("trace-b/node-1", 3): 0.5,
                         ("trace-b/node-1", 4): 0.0}
    assert credit.observation_scope == "prefix" and credit.estimator_id == "known-centered-step@1"
    resolved = resolve_policy_population(snapshot, credit, selected, capabilities())
    assert {update.objective.definition_id for update in resolved.updates} == {"sampo-spans@1"}


def test_quality_scores_and_incomplete_or_foreign_assessments_are_rejected():
    admitted, _ = _population()
    snapshot = admitted.resolved.snapshot
    assessments = assess_population_spans(snapshot, cast(Any, FakeStepScorer()), admitted.read_input, roles=("reasoning",))

    class Masquerade:
        id = "raw-quality"
        required_relations = ()

        def prepare(self, snapshot):
            return assessments  # scorer evidence, not prepared advantages

    with pytest.raises(InvalidPolicyUpdate, match="not scorer evidence"):
        prepare_credit(snapshot, cast(Any, Masquerade()))

    class PartialScorer(FakeStepScorer):
        def assess(self, snapshot, spans, read_input):
            return super().assess(snapshot, spans[:1], read_input)

    with pytest.raises(InvalidPolicyUpdate, match="exactly once"):
        assess_population_spans(snapshot, cast(Any, PartialScorer()), admitted.read_input, roles=("reasoning",))

    class ForeignScorer(FakeStepScorer):
        revision = "other@1"

        def assess(self, snapshot, spans, read_input):
            return tuple(replace(value, scorer_revision="fake-step-scorer@1")
                         for value in super().assess(snapshot, spans, read_input))

    with pytest.raises(InvalidPolicyUpdate, match="injected scorer"):
        assess_population_spans(snapshot, cast(Any, ForeignScorer()), admitted.read_input, roles=("reasoning",))
    with pytest.raises(InvalidPolicyUpdate, match="one advantage per assessed span"):
        prepare_credit(snapshot, ExternalSpanCreditEstimator(
            "bad", "1", assessments, lambda values, population: {}, (snapshot.native_evidence_digest,)))


def test_selected_estimator_replaces_algorithm_credit_through_injected_provider():
    from posttrain.train.update_process_credit import ScoredSpanCreditProvider

    provider = ScoredSpanCreditProvider(cast(Any, FakeStepScorer()), ("reasoning",), "known-centered-step", "1",
                                        _known_estimator)
    algorithm, _ = _population()
    evidence = FIXTURE.read_bytes()
    base = algorithm.resolved.updates[0].population
    rollouts = _rollouts()
    selected = replace(settings(), policy_updates=PolicyUpdateSettings(
        PolicyUpdateSchedule("episode", 1), PolicyExecutionBudget(2, 100, 1000), objective_variant="semantic-spans",
        policy_selection=ActionSelection("roles", roles=("reasoning",)), credit_estimator="known-centered-step@1"))
    admitted = AdmittedNativePopulation.from_rollouts(
        rollouts, selected, capabilities(), population_id="population@3", native_evidence_ref="artifact:fixture",
        read_evidence=lambda reference: evidence, decode=_decode, template_revision="template@1",
        versions=PolicyVersions("sampler@3", "old@3", "current@3", None), sampler_step=3,
        selector_digest="fixture@1", applied_update_offset=3, attempt_offset=3, process_credit=provider)
    assert admitted.resolved.credit.estimator_id == "known-centered-step@1"
    assert admitted.resolved.snapshot.digest == base.digest
    assert len(provider.last_assessments) == 1
    by_action = {(value.action.turn_id, value.action.token_index): value.advantage for value in admitted.resolved.credit.values}
    assert by_action[("trace-a/node-1", 1)] == -0.5 and by_action[("trace-b/node-1", 4)] == 0.0
    for wrong, estimator in ((provider, None), (None, "known-centered-step@1"),
                             (provider, "other-estimator@1")):
        assert selected.policy_updates is not None
        changed = replace(selected, policy_updates=replace(selected.policy_updates, credit_estimator=estimator))
        with pytest.raises(InvalidPolicyUpdate, match="process credit"):
            AdmittedNativePopulation.from_rollouts(
                rollouts, changed, capabilities(), population_id="population@3", native_evidence_ref="artifact:fixture",
                read_evidence=lambda reference: evidence, decode=_decode, template_revision="template@1",
                versions=PolicyVersions("sampler@3", "old@3", "current@3", None), sampler_step=3,
                selector_digest="fixture@1", applied_update_offset=3, attempt_offset=3, process_credit=wrong)


@pytest.mark.parametrize(("estimator", "variant", "backend", "admitted"), [
    ("group-centered-likelihood@1", "semantic-spans", "trl@1", True),
    ("known-centered-step@1", "semantic-spans", "trl@1", False),
    ("group-centered-likelihood@1", "algorithm", "trl@1", False),
    ("group-centered-likelihood@1", "semantic-spans", "verl@1", False),
])
def test_public_guard_admits_only_gpu_qualified_process_credit(estimator, variant, backend, admitted):
    from posttrain.train.requests import _resolved_selection_problem

    updates = replace(PolicyUpdateSettings(PolicyUpdateSchedule("episode", 1), PolicyExecutionBudget(2, 100, 1000)),
                      objective_variant=variant, credit_estimator=estimator)
    problem = _resolved_selection_problem("SAMPO", replace(settings(), policy_updates=updates),
                                          cast(Any, SimpleNamespace(backend=backend, target=SimpleNamespace(placement={}))),
                                          cast(Any, SimpleNamespace(backend="transformers@1")))
    assert (problem is None) == admitted, problem
