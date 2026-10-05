"""Turn ratios and equal-turn reduction are independent of episode packing."""

import json
import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from posttrain.train.update_credit import PreparedCredit
from posttrain.train.update_objectives import ObjectiveSpec, objective_population, resolve_objective_term
from posttrain.train.update_plan import PolicyUpdateSchedule, resolve_updates
from posttrain.train.update_records import InvalidPolicyUpdate
from posttrain.train.update_resolution import resolve_policy_population

from .test_update_plan import five_turns
from .test_update_resolution import capabilities, settings


def fixture(identity="sampo-turns@1"):
    snapshot, _ = five_turns()
    # Two episodes, five turns, eight actions; turn lengths 3,1,1,2,1.
    snapshot = replace(
        snapshot,
        conditioning=tuple(
            replace(view, sampled=tuple(range(length)))
            for view, length in zip(snapshot.conditioning, (3, 1, 1, 2, 1), strict=True)
        ),
    )
    advantages = {"A1": 1.0, "A2": -1.0, "A3": 0.5, "B1": 0.3, "B2": -0.3}
    credit = PreparedCredit(
        snapshot.digest,
        "fixture@1",
        np.array([advantages[action.turn_id] for action in snapshot.actions()]),
        (),
        (),
        "none@1",
        "full-trajectory",
        ("evidence",),
    )
    spec = ObjectiveSpec(identity)
    population = objective_population(snapshot, spec, credit)
    update = resolve_updates(snapshot, PolicyUpdateSchedule("episode", 2), population)[0]
    return snapshot, credit, spec, update, resolve_objective_term(update, spec, credit)


def test_unequal_lengths_use_equal_turn_weight_and_turn_only_dependencies():
    snapshot, credit, spec, update, term = fixture()
    assert snapshot.size == 8
    supports = [np.flatnonzero(term.ratio_segment == segment) for segment in range(term.segment_count)]
    for support in supports:
        assert len({int(snapshot.view_of[position]) for position in support}) == 1
        assert term.policy_weight[support].sum() == pytest.approx(1 / 5)
    assert sorted(len(support) for support in supports) == [1, 1, 1, 2, 3]
    assert sorted(length for _, length in term.policy_denominators) == [1, 1, 1, 2, 3]
    partial = resolve_updates(snapshot, PolicyUpdateSchedule("turn", 1), update.objective)[0]
    selected = resolve_objective_term(partial, spec, credit)
    assert partial.views == (0,)
    assert partial.dependency_count == int((selected.ratio_segment == 0).sum()) == 3
    assert selected.policy_weight[selected.policy_positions].tolist() == [1 / 3] * 3


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
        resolve_policy_population(
            snapshot,
            credit,
            GRPOSettings(id="unsupported", loop=base.loop, policy_updates=selected.policy_updates),
            supported,
        )
    from posttrain.train.update_transport import decode_population_payload, population_payload

    retained = decode_population_payload(
        json.loads(
            json.dumps(
                population_payload(
                    SimpleNamespace(
                        updates=result.updates,
                        credit=result.credit,
                        spec=result.spec,
                        execution=result.execution,
                        capabilities=result.capabilities,
                        max_overflow_retries=2,
                        applied_update_offset=3,
                        attempt_offset=4,
                    )
                )
            )
        )
    )
    restored = retained.resolved
    assert restored.snapshot.digest == result.snapshot.digest
    assert restored.credit.digest == result.credit.digest
    assert restored.spec == result.spec
    assert [update.digest for update in restored.updates] == [update.digest for update in result.updates]
    assert restored.packs == result.packs
    assert retained.applied_update_offset == 3 and retained.attempt_offset == 4


@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16", "float16"])
def test_loss_and_gradients_match_independent_author_turn_row_formula(dtype_name):
    torch = pytest.importorskip("torch")
    from posttrain.train.backends.policy_update_math import ScoreBundle, evaluate
    from posttrain.train.backends.policy_update_replay import prepare_score_adjoints

    snapshot, credit, _, _, term = fixture()
    actions = snapshot.actions()
    deltas = {"A1": 0.3, "A2": -1.0, "A3": 0.1, "B1": 0.1, "B2": -0.1}
    current = torch.tensor(
        [deltas[action.turn_id] for action in actions], dtype=getattr(torch, dtype_name), requires_grad=True
    )
    old = torch.zeros(snapshot.size, dtype=torch.float32)
    scored = np.ones(snapshot.size, dtype=bool)
    score = ScoreBundle(current, old, term.parameter_version, scored)
    actual = evaluate(term, credit, score)
    # Direct author-style row mean, local token derivative, clipping, token mean,
    # then row mean. No production objective/reduction helper supplies the reference.
    row_losses = []
    for turn in ("A1", "A2", "A3", "B1", "B2"):
        positions = [position for position, action in enumerate(actions) if action.turn_id == turn]
        logits = current[positions].float()
        ratio = (logits.mean().detach() + logits - logits.detach()).exp()
        advantage = torch.tensor(credit.advantages[positions], dtype=torch.float32)
        row_losses.append(torch.maximum(-ratio * advantage, -ratio.clamp(0.8, 1.2) * advantage).mean())
    expected = torch.stack(row_losses).mean()
    torch.testing.assert_close(actual.loss, expected)
    (actual_grad,) = torch.autograd.grad(actual.loss, (current,), retain_graph=True)
    (expected_grad,) = torch.autograd.grad(expected, (current,), retain_graph=True)
    torch.testing.assert_close(actual_grad, expected_grad)
    assert {actions[int(position)].turn_id for position in np.flatnonzero(actual.clipped)} == {"A1", "A2"}
    assert math.isclose(
        float(actual.ratios[0].detach()),
        math.exp(float(current[0].detach())),
        rel_tol=1e-6,
    )
    # Native veRL's bounded replay uses these same complete-objective adjoints.
    prepared = prepare_score_adjoints(term, credit, score)
    torch.testing.assert_close(prepared.adjoints, actual_grad.float(), atol=0.001, rtol=0.01)
    _, episode_credit, _, _, episode = fixture("sampo@1")
    episode_result = evaluate(episode, episode_credit, ScoreBundle(current, old, episode.parameter_version, scored))
    assert not episode_result.clipped.any()
