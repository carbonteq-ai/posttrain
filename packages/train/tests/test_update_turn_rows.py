"""Turn ratios and equal-turn reduction are independent of episode packing."""

import json
import math
from dataclasses import replace
from types import SimpleNamespace

import pytest
from posttrain.train.update_credit import ActionCredit, PreparedCredit
from posttrain.train.update_objectives import ObjectiveSpec, objective_population, resolve_objective_term
from posttrain.train.update_plan import PolicyUpdateSchedule, resolve_updates
from posttrain.train.update_records import ActionRecord, InvalidPolicyUpdate
from posttrain.train.update_resolution import resolve_policy_population

from .test_update_plan import five_turns
from .test_update_resolution import capabilities, settings


def fixture(identity="sampo-turns@1"):
    snapshot, _ = five_turns()
    # Two episodes, five turns, eight actions; turn lengths 3,1,1,2,1.
    records = []
    for record, length in zip(snapshot.actions, (3, 1, 1, 2, 1), strict=True):
        records.extend(ActionRecord(replace(record.action, token_index=index), record.conditioning_id, record.native_ref)
                       for index in range(length))
    snapshot = replace(snapshot, actions=tuple(records))
    advantages = {"A1": 1., "A2": -1., "A3": .5, "B1": .3, "B2": -.3}
    credit = PreparedCredit(snapshot.digest, "fixture@1",
        tuple(ActionCredit(record.action, advantages[record.action.turn_id]) for record in records),
        (), (), "none@1", "full-trajectory", ("evidence",))
    spec = ObjectiveSpec(identity)
    population = objective_population(snapshot, spec, credit)
    update = resolve_updates(snapshot, PolicyUpdateSchedule("episode", 2), population)[0]
    return snapshot, credit, spec, update, resolve_objective_term(update, spec, credit)


def test_unequal_lengths_use_equal_turn_weight_and_turn_only_dependencies():
    snapshot, credit, spec, update, term = fixture()
    weights = {value.action: value.weight for value in term.policy_weights}
    for support in term.ratio_support:
        assert len({action.turn_id for action in support}) == 1
        assert sum(weights[action] for action in support) == pytest.approx(1/5)
    assert sorted(len(support) for support in term.ratio_support) == [1, 1, 1, 2, 3]
    assert sorted(length for _, length in term.policy_denominators) == [1, 1, 1, 2, 3]
    partial = resolve_updates(snapshot, PolicyUpdateSchedule("turn", 1), update.objective)[0]
    selected = resolve_objective_term(partial, spec, credit)
    assert len(partial.dependencies) == len(selected.ratio_support[0]) == 3
    assert len({action.turn_id for action in partial.dependencies}) == 1
    assert [weight.weight for weight in selected.policy_weights] == [1/3]*3


def test_named_selection_is_sampo_only_and_preserves_legacy_objective():
    snapshot, credit, _, _, _ = fixture()
    base = settings()
    assert base.policy_updates is not None
    selected = replace(base, policy_updates=replace(base.policy_updates, objective_variant="turn-rows"))
    supported = replace(capabilities(), definition_ids=capabilities().definition_ids + ("sampo-turns@1",))
    result = resolve_policy_population(snapshot, credit, selected, supported)
    assert result.spec.definition_id == "sampo-turns@1" and result.credit is credit
    assert resolve_policy_population(snapshot, credit, base, supported).spec.definition_id == "sampo@1"
    from posttrain.train.profiles import GRPOSettings

    with pytest.raises(InvalidPolicyUpdate, match="selected objective variant"):
        resolve_policy_population(snapshot, credit, GRPOSettings(id="unsupported", loop=base.loop,
                                                                policy_updates=selected.policy_updates), supported)
    from posttrain.train.update_transport import decode_population_payload, population_payload

    retained = decode_population_payload(json.loads(json.dumps(population_payload(SimpleNamespace(
        updates=result.updates, credit=result.credit, spec=result.spec,
        execution=result.execution, capabilities=result.capabilities,
        max_overflow_retries=2, applied_update_offset=3, attempt_offset=4,
    )))))
    assert retained.resolved == result
    assert retained.applied_update_offset == 3 and retained.attempt_offset == 4


@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16", "float16"])
def test_loss_and_gradients_match_independent_author_turn_row_formula(dtype_name):
    torch = pytest.importorskip("torch")
    from posttrain.train.backends.policy_update_math import ScoreBundle, evaluate
    from posttrain.train.backends.policy_update_replay import prepare_score_adjoints

    snapshot, credit, _, _, term = fixture()
    deltas = {"A1": .3, "A2": -1., "A3": .1, "B1": .1, "B2": -.1}
    current = {record.action: torch.tensor(deltas[record.action.turn_id], dtype=getattr(torch, dtype_name), requires_grad=True)
               for record in snapshot.actions}
    old = {action: torch.zeros((), dtype=torch.float32) for action in current}
    score = ScoreBundle(current, old, term.parameter_version)
    actual = evaluate(term, credit, score)
    advantages = {value.action: value.advantage for value in credit.values}
    # Direct author-style row mean, local token derivative, clipping, token mean,
    # then row mean. No production objective/reduction helper supplies the reference.
    row_losses = []
    for turn in ("A1", "A2", "A3", "B1", "B2"):
        actions = [record.action for record in snapshot.actions if record.action.turn_id == turn]
        logits = torch.stack([current[action].float() for action in actions])
        ratio = (logits.mean().detach() + logits - logits.detach()).exp()
        advantage = torch.tensor([advantages[action] for action in actions])
        row_losses.append(torch.maximum(-ratio*advantage, -ratio.clamp(.8, 1.2)*advantage).mean())
    expected = torch.stack(row_losses).mean()
    torch.testing.assert_close(actual.loss, expected)
    actual_grad = torch.autograd.grad(actual.loss, tuple(current.values()), retain_graph=True)
    expected_grad = torch.autograd.grad(expected, tuple(current.values()), retain_graph=True)
    torch.testing.assert_close(torch.stack(actual_grad), torch.stack(expected_grad))
    assert {action.turn_id for action in actual.clipped_actions} == {"A1", "A2"}
    assert math.isclose(float(actual.ratios[snapshot.actions[0].action].detach()),
                        math.exp(float(current[snapshot.actions[0].action].detach())), rel_tol=1e-6)
    # Native veRL's bounded replay uses these same complete-objective adjoints.
    prepared = prepare_score_adjoints(term, credit, score)
    for action, gradient in zip(current, actual_grad, strict=True):
        torch.testing.assert_close(prepared.adjoints[action], gradient.float(), atol=.001, rtol=.01)
    _, episode_credit, _, _, episode = fixture("sampo@1")
    episode_result = evaluate(episode, episode_credit, ScoreBundle(current, old, episode.parameter_version))
    assert episode_result.clipped_actions == ()
