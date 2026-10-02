"""Real local likelihood scorer over original reasoning spans (tiny random model)."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from posttrain.common import TraceObservation  # noqa: E402
from posttrain.environment.verifiers_conditioning import native_conditioning_records  # noqa: E402
from posttrain.train.backends.policy_update_admission import AdmittedNativePopulation  # noqa: E402
from posttrain.train.online_rl import AgenticTurn, BehaviorPolicySpan, EnvironmentRollout  # noqa: E402
from posttrain.train.profiles import SAMPOSettings, TrainingLoop  # noqa: E402
from posttrain.train.update_credit import prepare_credit  # noqa: E402
from posttrain.train.update_plan import (  # noqa: E402
    ExecutionCapabilities,
    PolicyExecutionBudget,
    PolicyUpdateSchedule,
    PolicyUpdateSettings,
)
from posttrain.train.update_process_credit import ExternalSpanCreditEstimator, assess_population_spans  # noqa: E402
from posttrain.train.update_records import ActionSelection, PolicyVersions  # noqa: E402
from posttrain.train.update_resolution import resolve_policy_population  # noqa: E402
from posttrain_lab.scorers import LikelihoodSpanScorer  # noqa: E402

FIXTURE = (Path(__file__).resolve().parents[3] / "packages/train/tests/fixtures/policy_updates"
           / "two-episode-reasoning.json")


def _decode(raw):
    return {record["id"]: SimpleNamespace(
        id=record["id"], nodes=[SimpleNamespace(**node) for node in record["nodes"]],
        calls=[SimpleNamespace(node=call["node"], usage=SimpleNamespace(**call["usage"])) for call in record["calls"]])
        for record in json.loads(raw)}


def _rollouts():
    traces = _decode(FIXTURE.read_bytes())
    return tuple(EnvironmentRollout(
        "task", (1, 2, 3), tuple(traces[name].nodes[1].token_ids[1:]), (-1.,) * 4, (True,) * 4, reward, False,
        TraceObservation("verifiers", name, {"info": {"posttrain_episode_id": f"episode-{name}",
            "posttrain_prompt_group_id": "group", "posttrain_rollout_id": f"rollout-{name}"}}),
        turns=(AgenticTurn(0, 4, "shared"),), behavior_policy=BehaviorPolicySpan(3, 3),
        conditioning_records=native_conditioning_records(traces[name], sampled_node_indices=(1,),
                                                        context_contract="causal-text@1"),
        selected_branch_id="1", conditioning_completion_indices=((0, 1, 2, 3),),
    ) for name, reward in (("trace-a", 1.), ("trace-b", 0.)))


def _admitted():
    evidence = FIXTURE.read_bytes()
    rollouts = _rollouts()
    settings = SAMPOSettings(id="scorer", loop=TrainingLoop(max_steps=1, per_device_batch_size=1),
        policy_updates=PolicyUpdateSettings(PolicyUpdateSchedule("episode", 1), PolicyExecutionBudget(2, 100, 1000),
            objective_variant="semantic-spans", policy_selection=ActionSelection("roles", roles=("reasoning",))))
    capabilities = ExecutionCapabilities(("sampo-spans@1",), ("sampled-logp", "old-logp", "reference-logp"), 100, True)
    admitted = AdmittedNativePopulation.from_rollouts(
        rollouts, settings, capabilities, population_id="population@3", native_evidence_ref="artifact:fixture",
        read_evidence=lambda reference: evidence, decode=_decode, template_revision="template@1",
        versions=PolicyVersions("sampler@3", "old@3", "current@3", None), sampler_step=3,
        selector_digest="fixture@1", applied_update_offset=3, attempt_offset=3)
    return admitted, settings, capabilities


def test_likelihood_scorer_scores_original_reasoning_spans_with_retained_provenance():
    torch.manual_seed(0)
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(vocab_size=32, n_positions=16, n_embd=16,
                                                                 n_layer=1, n_head=2))
    scorer = LikelihoodSpanScorer(model, model_id="tiny-gpt2", model_revision="seed0", device="cpu")
    admitted, settings, capabilities = _admitted()
    snapshot = admitted.resolved.snapshot
    assessments = assess_population_spans(snapshot, cast(Any, scorer), admitted.read_input, roles=("reasoning",))
    views = {view.id: view for view in snapshot.conditioning}
    for assessment in assessments:
        span = next(span for span in snapshot.spans if span.id == assessment.span_id)
        inputs = admitted.read_input(views[span.action_intervals[0].start.turn_id])
        with torch.no_grad():
            logprobs = model(input_ids=torch.tensor([inputs.token_ids])).logits[0].log_softmax(-1)
        positions = dict(inputs.action_positions)
        expected = [float(logprobs[positions[action.token_index] - 1, inputs.token_ids[positions[action.token_index]]])
                    for action in span.actions()]
        assert assessment.components[0].value == pytest.approx(sum(expected) / len(expected), abs=1e-6)
        assert assessment.scorer_revision == scorer.revision and assessment.observation_scope == "prefix"
        assert assessment.scorer_snapshot == "tiny-gpt2@seed0" and assessment.semantic_kind == "reasoning"
    quality = {value.span_id: float(cast(float, value.components[0].value)) for value in assessments}
    mean = sum(quality.values()) / len(quality)
    credit = prepare_credit(snapshot, ExternalSpanCreditEstimator(
        "centered-likelihood", "1", assessments, lambda values, population: {key: value - mean for key, value in quality.items()},
        (snapshot.native_evidence_digest,)))
    resolved = resolve_policy_population(snapshot, credit, settings, capabilities)
    assert {update.objective.definition_id for update in resolved.updates} == {"sampo-spans@1"}
    assert replace(credit, estimator_id=credit.estimator_id).observation_scope == "prefix"


def test_likelihood_process_credit_replaces_algorithm_credit_at_admission():
    from posttrain_lab.scorers import likelihood_process_credit

    torch.manual_seed(0)
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(vocab_size=32, n_positions=16, n_embd=16,
                                                                 n_layer=1, n_head=2))
    provider = likelihood_process_credit(model, model_id="tiny-gpt2", model_revision="seed0", device="cpu")
    evidence = FIXTURE.read_bytes()
    reference, settings, capabilities = _admitted()
    assert settings.policy_updates is not None
    selected = replace(settings, policy_updates=replace(settings.policy_updates,
                                                         credit_estimator="group-centered-likelihood@1"))
    rollouts = reference.resolved.updates[0].population  # population identity is unchanged by credit choice
    admitted = AdmittedNativePopulation.from_rollouts(
        _rollouts(), selected, capabilities, population_id="population@3", native_evidence_ref="artifact:fixture",
        read_evidence=lambda ref: evidence, decode=_decode, template_revision="template@1",
        versions=PolicyVersions("sampler@3", "old@3", "current@3", None), sampler_step=3,
        selector_digest="fixture@1", applied_update_offset=3, attempt_offset=3, process_credit=provider)
    assert admitted.resolved.snapshot.digest == rollouts.digest
    assert admitted.resolved.credit.estimator_id == "group-centered-likelihood@1"
    [assessments] = provider.last_assessments
    quality = {value.span_id: float(cast(float, value.components[0].value)) for value in assessments}
    mean = sum(quality.values()) / len(quality)  # the fixture's two episodes share one prompt group
    spans = {span.id: span for span in admitted.resolved.snapshot.spans}
    expected = {action: quality[span_id] - mean for span_id in quality for action in spans[span_id].actions()}
    for value in admitted.resolved.credit.values:
        assert value.advantage == pytest.approx(expected.get(value.action, 0.0), abs=1e-12)
