"""Global objective dependencies survive exclusive context ownership."""

from dataclasses import replace

import pytest
from posttrain.train.update_distribution import resolve_score_ownership, resolve_score_rounds
from posttrain.train.update_records import InvalidPolicyUpdate

from .test_update_objectives import resolved


@pytest.mark.parametrize("owners", [(("A1",), ("A2", "A3", "B1", "B2")), ((), ("A1", "A2", "A3", "B1", "B2"))])
def test_unequal_and_empty_partitions_preserve_global_dependencies(owners):
    _, _, update, term = resolved("sequence-coupled@1")
    plan = resolve_score_ownership(update, term, owners)
    assert plan.term_digest == term.digest and plan.update_digest == update.digest
    assert set(plan.actions) == set(update.dependencies)
    assert len(plan.actions) == 5
    assert [len(part.actions) for part in plan.partitions] == [len(part) for part in owners]
    # Ratios remain episode-wide even though A's contexts cross the rank boundary.
    assert sorted(len(support) for support in term.ratio_support) == [2, 3]
    assert term.policy_denominators == (("A", 3), ("B", 2))
    assert plan == resolve_score_ownership(update, term, owners)


@pytest.mark.parametrize(
    "owners",
    [(), (("A1",),), (("A1",), ("A1", "A2", "A3", "B1", "B2")), (("A1",), ("A2", "A3", "B1", "B2", "foreign"))],
)
def test_missing_duplicate_or_foreign_context_ownership_rejected(owners):
    _, _, update, term = resolved()
    with pytest.raises(InvalidPolicyUpdate, match="exactly once"):
        resolve_score_ownership(update, term, owners)


def test_changed_term_cannot_reuse_ownership():
    _, _, update, term = resolved()
    with pytest.raises(InvalidPolicyUpdate, match="different resolved term"):
        resolve_score_ownership(update, replace(term, update_digest="other"), (("A1", "A2", "A3", "B1", "B2"),))


@pytest.mark.parametrize("owners", [(("A1",), ("A2", "A3", "B1", "B2")), ((), ("A1", "A2", "A3", "B1", "B2"))])
def test_matched_rounds_preserve_owned_actions_and_pad_with_admitted_contexts(owners):
    _, _, update, term = resolved("sequence-coupled@1")
    plan = resolve_score_ownership(update, term, owners)
    rounds = resolve_score_rounds(update, term, plan)
    assert len(rounds) == max(len(part) for part in owners)
    assert all(len(row) == 2 for row in rounds)
    for rank in range(2):
        real = tuple(action for row in rounds if row[rank].contributes for action in row[rank].actions)
        assert real == plan.partitions[rank].actions
        assert sum(not row[rank].contributes for row in rounds) == len(rounds) - len(owners[rank])
    assert all(set(work.actions) <= set(update.dependencies) for row in rounds for work in row)
    assert all(
        work.context_id in {view.id for view in update.population.conditioning} for row in rounds for work in row
    )


def test_forged_context_action_mapping_cannot_be_used_for_native_rounds():
    _, _, update, term = resolved()
    plan = resolve_score_ownership(update, term, (("A1",), ("A2", "A3", "B1", "B2")))
    parts = (
        replace(plan.partitions[0], context_ids=("A2",)),
        replace(plan.partitions[1], context_ids=("A1", "A3", "B1", "B2")),
    )
    with pytest.raises(InvalidPolicyUpdate, match="differs"):
        resolve_score_rounds(update, term, replace(plan, partitions=parts))


def test_owned_carriers_route_global_coupled_adjoints_and_reject_drift():
    torch = pytest.importorskip("torch")
    from posttrain.train.backends.policy_update_distribution import distributed_score_carrier
    from posttrain.train.backends.policy_update_math import ScoreBundle
    from posttrain.train.backends.policy_update_replay import prepare_score_adjoints

    snapshot, credit, update, term = resolved("sequence-coupled@1")
    scores = {record.action: torch.tensor(0.1, requires_grad=True) for record in snapshot.actions}
    prepared = prepare_score_adjoints(
        term, credit, ScoreBundle(scores, {action: torch.tensor(0.0) for action in scores}, term.parameter_version)
    )
    plan = resolve_score_ownership(update, term, (("A1",), ("A2", "A3", "B1", "B2")))
    owned = {action: scores[action] for action in plan.partitions[1].actions}
    loss = distributed_score_carrier(plan, prepared, owned, rank=1, parameter_version=term.parameter_version)
    gradients = torch.autograd.grad(loss, tuple(owned.values()))
    for action, gradient in zip(owned, gradients, strict=True):
        torch.testing.assert_close(gradient, prepared.adjoints[action] * 2)
    with pytest.raises(InvalidPolicyUpdate, match="owned"):
        distributed_score_carrier(plan, prepared, scores, rank=1, parameter_version=term.parameter_version)
    with pytest.raises(InvalidPolicyUpdate, match="replay drift"):
        distributed_score_carrier(
            plan,
            prepared,
            {action: value + 0.01 for action, value in owned.items()},
            rank=1,
            parameter_version=term.parameter_version,
        )
    with pytest.raises(InvalidPolicyUpdate, match="identity"):
        distributed_score_carrier(plan, prepared, owned, rank=True, parameter_version=term.parameter_version)


def test_empty_rank_requires_an_actual_zero_gradient_native_anchor():
    torch = pytest.importorskip("torch")
    from posttrain.train.backends.policy_update_distribution import distributed_score_carrier
    from posttrain.train.backends.policy_update_math import ScoreBundle
    from posttrain.train.backends.policy_update_replay import prepare_score_adjoints

    snapshot, credit, update, term = resolved()
    scores = {record.action: torch.tensor(0.1) for record in snapshot.actions}
    prepared = prepare_score_adjoints(
        term, credit, ScoreBundle(scores, {action: torch.tensor(0.0) for action in scores}, term.parameter_version)
    )
    plan = resolve_score_ownership(update, term, ((), ("A1", "A2", "A3", "B1", "B2")))
    parameter = torch.tensor(3.0, requires_grad=True)
    carrier = distributed_score_carrier(
        plan, prepared, {}, rank=0, parameter_version=term.parameter_version, empty_anchor=parameter.square()
    )
    carrier.backward()
    assert parameter.grad.item() == 0
    for anchor in (None, torch.tensor(0.0), parameter * float("nan")):
        with pytest.raises(InvalidPolicyUpdate, match="anchor"):
            distributed_score_carrier(
                plan, prepared, {}, rank=0, parameter_version=term.parameter_version, empty_anchor=anchor
            )
