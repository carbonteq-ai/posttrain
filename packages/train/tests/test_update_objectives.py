"""Independent loss/derivative references; no native-support claim from logits."""

import math
from dataclasses import replace

import pytest
from posttrain.train.update_credit import ActionCredit, PreparedCredit
from posttrain.train.update_objectives import (
    ActionSelection,
    ObjectiveSpec,
    distributed_weight,
    objective_population,
    resolve_objective_term,
)
from posttrain.train.update_plan import PolicyExecutionBudget, PolicyUpdateSchedule, plan_packs, resolve_updates
from posttrain.train.update_records import ActionInterval, InvalidPolicyUpdate, SemanticSpan

from .test_update_plan import capabilities, five_turns


def resolved(identity="sampo@1", *, spec=None, advantages=None):
    snapshot, _ = five_turns()
    values = advantages or (1.0, -1.0, 0.0, 0.5, -0.5)
    credit = PreparedCredit(
        snapshot.digest,
        "fixture-estimator@1",
        tuple(ActionCredit(record.action, value) for record, value in zip(snapshot.actions, values, strict=True)),
        (),
        (("outcome", 1.0),),
        "fixture@1",
        "full-trajectory",
        ("fixture-evidence",),
    )
    spec = spec or ObjectiveSpec(identity)
    population = objective_population(snapshot, spec, credit)
    update = resolve_updates(snapshot, PolicyUpdateSchedule("episode", 2), population)[0]
    return snapshot, credit, update, resolve_objective_term(update, spec, credit)


def test_unequal_episode_lengths_resolve_global_weights() -> None:
    _, _, _, term = resolved()
    assert [item.weight for item in term.policy_weights] == pytest.approx([1 / 6] * 3 + [1 / 4] * 2)
    assert term.policy_denominators == (("A", 3), ("B", 2))
    _, _, _, token = resolved("dapo@1")
    assert [item.weight for item in token.policy_weights] == pytest.approx([1 / 5] * 5)


def test_policy_and_kl_support_and_empty_episode_weighting_are_independent() -> None:
    snapshot, _ = five_turns()
    a, b = snapshot.actions[0].action, snapshot.actions[3].action
    snapshot = replace(
        snapshot,
        spans=(
            SemanticSpan("think", "reasoning", "extract@1", (ActionInterval(a, 1),)),
            SemanticSpan("regularize", "assistant", "extract@1", (ActionInterval(b, 1),)),
        ),
    )
    credit = PreparedCredit(
        snapshot.digest,
        "fixture@1",
        tuple(ActionCredit(record.action, 1.0) for record in snapshot.actions),
        (),
        (),
        "none@1",
        "prefix",
        ("evidence",),
    )
    spec = ObjectiveSpec(
        "sampo-spans@1",
        beta=0.1,
        policy_selection=ActionSelection("spans", ("think",)),
        kl_selection=ActionSelection("spans", ("regularize",)),
        denominator="original-eligible",
        empty_policy="zero",
    )
    update = resolve_updates(
        snapshot, PolicyUpdateSchedule("episode", 2), objective_population(snapshot, spec, credit)
    )[0]
    term = resolve_objective_term(update, spec, credit)
    assert [(item.action, item.weight) for item in term.policy_weights] == [(a, 1 / 6)]
    assert [(item.action, item.weight) for item in term.kl_weights] == [(b, 1 / 4)]
    assert term.zero_policy_episodes == ("B",)
    assert term.zero_kl_episodes == ("A",)
    assert len(term.ratio_support[0]) == 3
    with pytest.raises(InvalidPolicyUpdate, match="empty objective selection"):
        objective_population(snapshot, replace(spec, empty_policy="reject"), credit)


def test_closed_definitions_reject_unqualified_combinations() -> None:
    with pytest.raises(InvalidPolicyUpdate, match="unsupported versioned"):
        ObjectiveSpec("ppo-without-critic@1")
    with pytest.raises(InvalidPolicyUpdate, match="legacy objective"):
        ObjectiveSpec("sampo@1", policy_selection=ActionSelection("spans", ("thinking",)))
    with pytest.raises(InvalidPolicyUpdate, match="absolute positive"):
        ObjectiveSpec("cispo-upper@1")


def test_changed_objective_or_credit_cannot_reuse_an_update() -> None:
    _, credit, update, term = resolved()
    with pytest.raises(InvalidPolicyUpdate, match="differs from resolved"):
        resolve_objective_term(update, replace(term.spec, clip_high=0.3), credit)
    with pytest.raises(InvalidPolicyUpdate, match="credit identity changed"):
        resolve_objective_term(update, term.spec, replace(credit, evidence_digests=("other-evidence",)))
    refreshed = resolve_objective_term(update, term.spec, credit, parameter_version="current@2")
    assert refreshed.parameter_version == "current@2"
    assert refreshed.digest != term.digest
    assert credit.digest == refreshed.credit_digest


def test_partial_episode_uses_original_ratio_but_update_selection_denominator() -> None:
    snapshot, credit, _, term = resolved()
    objective = objective_population(snapshot, term.spec, credit)
    update = resolve_updates(snapshot, PolicyUpdateSchedule("turn", 1), objective)[0]
    partial = resolve_objective_term(update, term.spec, credit)
    assert len(partial.policy_weights) == 1
    assert partial.policy_weights[0].weight == 1.0
    assert len(partial.ratio_support[0]) == 3


def test_empty_policy_zero_and_omit_have_distinct_episode_denominators() -> None:
    snapshot, _ = five_turns()
    first = snapshot.actions[0].action
    snapshot = replace(
        snapshot, spans=(SemanticSpan("one-step", "reasoning", "extract@1", (ActionInterval(first, 1),)),)
    )
    credit = PreparedCredit(
        snapshot.digest,
        "fixture@1",
        tuple(ActionCredit(record.action, 1.0) for record in snapshot.actions),
        (),
        (),
        "none@1",
        "prefix",
        ("evidence",),
    )
    weights = []
    for empty in ("zero", "omit"):
        spec = ObjectiveSpec(
            "sampo-spans@1", policy_selection=ActionSelection("spans", ("one-step",)), empty_policy=empty
        )
        update = resolve_updates(
            snapshot, PolicyUpdateSchedule("episode", 2), objective_population(snapshot, spec, credit)
        )[0]
        weights.append(resolve_objective_term(update, spec, credit).policy_weights[0].weight)
    assert weights == [0.5, 1.0]


def test_distributed_averaging_preserves_unequal_global_contributions() -> None:
    _, _, _, term = resolved()
    losses = [2.0, 5.0, -1.0, 8.0, 1.0]
    reference = sum(value * item.weight for value, item in zip(losses, term.policy_weights, strict=True))
    # Unequal 1/4 partitions; a second layout leaves one rank entirely empty.
    for partitions in (([0], [1, 2, 3, 4]), ([], [0, 1, 2, 3, 4])):
        ranks = [
            sum(
                losses[index] * distributed_weight(term.policy_weights[index].weight, averaging_ranks=2)
                for index in partition
            )
            for partition in partitions
        ]
        assert sum(ranks) / 2 == pytest.approx(reference)


def test_local_and_coupled_sequence_gradients_differ_despite_equal_values() -> None:
    torch = pytest.importorskip("torch")
    from posttrain.train.backends.policy_update_math import ScoreBundle, evaluate

    gradients = []
    losses = []
    for identity in ("sampo@1", "sequence-coupled@1"):
        snapshot, credit, _, term = resolved(identity)
        logps = torch.full((5,), 0.1, requires_grad=True)
        actions = [record.action for record in snapshot.actions]
        evaluation = evaluate(
            term,
            credit,
            ScoreBundle(
                dict(zip(actions, logps.unbind(), strict=True)),
                {action: torch.tensor(0.0) for action in actions},
                "current@1",
            ),
        )
        evaluation.loss.backward()
        gradients.append(logps.grad)
        losses.append(evaluation.loss.item())
    assert losses[0] == pytest.approx(losses[1])
    assert gradients[0].tolist() == pytest.approx(
        [-math.exp(0.1) / 6, math.exp(0.1) / 6, 0, -math.exp(0.1) / 8, math.exp(0.1) / 8]
    )
    assert gradients[1].tolist() == pytest.approx([0.0] * 5, abs=1e-7)


def test_clipping_positive_and_negative_branches_have_independent_analytic_gradients() -> None:
    torch = pytest.importorskip("torch")
    from posttrain.train.backends.policy_update_math import ScoreBundle, evaluate

    snapshot, credit, _, term = resolved("grpo@1", advantages=(1.0, -1.0, 1.0, -1.0, 1.0))
    ratios = (1.5, 0.5, 0.5, 1.5, 1.0)
    logps = torch.tensor([math.log(value) for value in ratios], requires_grad=True)
    actions = [record.action for record in snapshot.actions]
    evaluation = evaluate(
        term,
        credit,
        ScoreBundle(
            dict(zip(actions, logps.unbind(), strict=True)),
            {action: torch.tensor(0.0) for action in actions},
            "current@1",
        ),
    )
    evaluation.loss.backward()
    # High positive and low negative ratios clip; opposite signs keep gradients.
    assert evaluation.clipped_actions == tuple(actions[:2])
    assert logps.grad.tolist() == pytest.approx([0, 0, -0.5 / 6, 1.5 / 4, -1 / 4])


def test_cispo_weight_is_detached_even_at_a_nonunit_uncapped_ratio() -> None:
    torch = pytest.importorskip("torch")
    from posttrain.train.backends.policy_update_math import ScoreBundle, evaluate

    spec = ObjectiveSpec("cispo-upper@1", cispo_max_weight=1.2)
    snapshot, credit, _, term = resolved(spec=spec)
    ratios = (1.1, 1.5, 0.5, 2.0, 0.75)
    logps = torch.tensor([math.log(value) for value in ratios], requires_grad=True)
    actions = [record.action for record in snapshot.actions]
    result = evaluate(
        term,
        credit,
        ScoreBundle(
            dict(zip(actions, logps.unbind(), strict=True)),
            {action: torch.tensor(0.0) for action in actions},
            "current@1",
        ),
    )
    result.loss.backward()
    assert logps.grad.tolist() == pytest.approx([-1.1 / 5, 1.2 / 5, 0, -0.6 / 5, 0.375 / 5])


def test_independent_kl_selection_matches_analytic_k3_derivative() -> None:
    torch = pytest.importorskip("torch")
    from posttrain.train.backends.policy_update_math import ScoreBundle, evaluate

    spec = ObjectiveSpec("grpo@1", beta=0.3)
    snapshot, credit, _, term = resolved(spec=spec, advantages=(0.0,) * 5)
    # Near-zero input exercises FP32 cancellation-safe sampled k3 arithmetic.
    differences = (1e-5, -1e-5, 0.1, -0.5, 1.0)
    current = torch.zeros(5, requires_grad=True)
    actions = [record.action for record in snapshot.actions]
    result = evaluate(
        term,
        credit,
        ScoreBundle(
            dict(zip(actions, current.unbind(), strict=True)),
            {action: torch.tensor(0.0) for action in actions},
            "current@1",
            {action: torch.tensor(value) for action, value in zip(actions, differences, strict=True)},
        ),
    )
    result.loss.backward()
    expected_values = [
        0.3 * item.weight * (math.expm1(value) - value)
        for item, value in zip(term.kl_weights, differences, strict=True)
    ]
    expected_gradients = [
        -0.3 * item.weight * math.expm1(value) for item, value in zip(term.kl_weights, differences, strict=True)
    ]
    assert result.kl_loss.item() == pytest.approx(sum(expected_values), abs=1e-8)
    assert current.grad.tolist() == pytest.approx(expected_gradients, rel=2e-6, abs=1e-10)


def test_unused_scores_can_be_nonfinite_but_selected_ratio_dependencies_cannot() -> None:
    torch = pytest.importorskip("torch")
    from posttrain.train.backends.policy_update_math import ScoreBundle, evaluate
    from posttrain.train.update_records import ActionRef

    snapshot, credit, _, term = resolved()
    actions = [record.action for record in snapshot.actions]
    current = {action: torch.tensor(0.0, requires_grad=True) for action in actions}
    # A context/tool position is not an eligible ActionRecord and is never read.
    current[ActionRef("A", "branch", "tool-result", 99)] = torch.tensor(float("nan"))
    old = {action: torch.tensor(0.0) for action in actions}
    assert torch.isfinite(evaluate(term, credit, ScoreBundle(current, old, "current@1")).loss)
    with pytest.raises(InvalidPolicyUpdate, match="different parameter version"):
        evaluate(term, credit, ScoreBundle(current, old, "current@2"))
    current[actions[2]] = torch.tensor(float("nan"))
    with pytest.raises(InvalidPolicyUpdate, match="non-finite scalar"):
        evaluate(term, credit, ScoreBundle(current, old, "current@1"))


@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_execution_packs_match_monolithic_objective_and_gradient(dtype) -> None:
    torch = pytest.importorskip("torch")
    from posttrain.train.backends.policy_update_math import ScoreBundle, evaluate

    snapshot, credit, update, term = resolved("grpo@1")
    actions = [record.action for record in snapshot.actions]
    native_capabilities = replace(capabilities(), definition_ids=("grpo@1",), statistics=("sampled-logp", "old-logp"))
    initial = torch.tensor(
        [[0.3, -0.1], [-0.2, 0.4], [0.0, 0.1], [0.1, -0.1], [-0.1, 0.2]], dtype=getattr(torch, dtype)
    )
    advantages = torch.tensor([1.0, -1.0, 0.0, 0.5, -0.5])
    old = torch.tensor([-0.6] * 5)
    oracle = initial.clone().requires_grad_()
    current = oracle.float().log_softmax(-1)[:, 0]
    ratio = (current - old).exp()
    # Direct independent whole-population expression, with explicit 3/2 lengths.
    per_action = torch.maximum(-ratio * advantages, -ratio.clamp(0.8, 1.2) * advantages)
    reference = (per_action[:3].mean() + per_action[3:].mean()) / 2
    reference.backward()
    for record_budget in (1, 2, 5):
        parameters = initial.clone().requires_grad_()
        scores = {}
        for pack in plan_packs(update, PolicyExecutionBudget(record_budget, 100, 100), native_capabilities):
            indices = [actions.index(action) for action in pack.actions]
            local = parameters[indices].float().log_softmax(-1)[:, 0]
            scores.update(zip(pack.actions, local.unbind(), strict=True))
        result = evaluate(term, credit, ScoreBundle(scores, dict(zip(actions, old.unbind(), strict=True)), "current@1"))
        result.loss.backward()
        torch.testing.assert_close(result.loss, reference.detach(), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(parameters.grad, oracle.grad, atol=1e-4 if dtype == "float16" else 1e-3, rtol=1e-2)
