"""Global objective dependencies survive exclusive context ownership."""

from dataclasses import replace

import numpy as np
import pytest
from posttrain.train.update_credit import PreparedCredit
from posttrain.train.update_distribution import (
    RankScorePartition,
    owned_positions,
    resolve_score_ownership,
    resolve_score_rounds,
)
from posttrain.train.update_objectives import ObjectiveSpec, objective_population, resolve_objective_term
from posttrain.train.update_plan import PolicyUpdateSchedule, resolve_updates
from posttrain.train.update_records import ConditioningView, InvalidPolicyUpdate, PolicyVersions, PopulationSnapshot

TURNS = (("A", "A1"), ("A", "A2"), ("A", "A3"), ("B", "B1"), ("B", "B2"))


def resolved(identity="sampo@1"):
    """Episodes A (three one-token turns) and B (two), resolved as one update."""
    snapshot = PopulationSnapshot(
        "five-turns",
        "native:five-turns",
        "native-digest",
        tuple(
            ConditioningView(
                turn,
                "native:five-turns",
                "tokens",
                "attention",
                "positions",
                "template@1",
                turn,
                10,
                episode,
                "branch",
                (0,),
            )
            for episode, turn in TURNS
        ),
        (),
        (),
        PolicyVersions("sampler@1", "old@1", "current@1", None),
        "selector@1",
    )
    credit = PreparedCredit(
        snapshot.digest,
        "fixture-estimator@1",
        np.array((1.0, -1.0, 0.0, 0.5, -0.5)),
        (),
        (("outcome", 1.0),),
        "fixture@1",
        "full-trajectory",
        ("fixture-evidence",),
    )
    spec = ObjectiveSpec(identity)
    update = resolve_updates(
        snapshot, PolicyUpdateSchedule("episode", 2), objective_population(snapshot, spec, credit)
    )[0]
    return snapshot, credit, update, resolve_objective_term(update, spec, credit)


def views(snapshot, *names):
    return tuple(snapshot.view_index(name) for name in names)


def owners(snapshot, *parts):
    return tuple(views(snapshot, *part) for part in parts)


SPLIT = (("A1",), ("A2", "A3", "B1", "B2"))
EMPTY_FIRST = ((), ("A1", "A2", "A3", "B1", "B2"))


@pytest.mark.parametrize("names", [SPLIT, EMPTY_FIRST])
def test_unequal_and_empty_partitions_preserve_global_dependencies(names):
    snapshot, _, update, term = resolved("sequence-coupled@1")
    plan = resolve_score_ownership(update, term, owners(snapshot, *names))
    assert plan.term_digest == term.digest and plan.update_digest == update.digest
    assert set(plan.views) == set(update.views)
    assert len(plan.views) == 5
    assert [len(part.views) for part in plan.partitions] == [len(part) for part in names]
    # Ratios remain episode-wide even though A's contexts cross the rank boundary.
    segments = term.ratio_segment[term.ratio_segment >= 0]
    assert sorted(np.bincount(segments).tolist()) == [2, 3]
    assert term.policy_denominators == (("A", 3), ("B", 2))
    assert plan == resolve_score_ownership(update, term, owners(snapshot, *names))


@pytest.mark.parametrize(
    "names",
    [(), (("A1",),), (("A1",), ("A1", "A2", "A3", "B1", "B2")), (("A1",), ("A2", "A3", "B1", "B2", "foreign"))],
)
def test_missing_duplicate_or_foreign_context_ownership_rejected(names):
    snapshot, _, update, term = resolved()
    # A foreign turn is any index outside the occurrence's admitted turns.
    assigned = tuple(
        tuple(len(snapshot.conditioning) if name == "foreign" else snapshot.view_index(name) for name in part)
        for part in names
    )
    with pytest.raises(InvalidPolicyUpdate, match="exactly once"):
        resolve_score_ownership(update, term, assigned)


def test_changed_term_cannot_reuse_ownership():
    snapshot, _, update, term = resolved()
    with pytest.raises(InvalidPolicyUpdate, match="different resolved term"):
        resolve_score_ownership(update, replace(term, update_digest="other"), owners(snapshot, EMPTY_FIRST[1]))


@pytest.mark.parametrize("names", [SPLIT, EMPTY_FIRST])
def test_matched_rounds_preserve_owned_actions_and_pad_with_admitted_contexts(names):
    snapshot, _, update, term = resolved("sequence-coupled@1")
    plan = resolve_score_ownership(update, term, owners(snapshot, *names))
    rounds = resolve_score_rounds(update, term, plan)
    assert len(rounds) == max(len(part) for part in names)
    assert all(len(row) == 2 for row in rounds)
    for rank in range(2):
        real = tuple(row[rank].view for row in rounds if row[rank].contributes)
        assert real == plan.partitions[rank].views
        assert sum(not row[rank].contributes for row in rounds) == len(rounds) - len(names[rank])
    assert all(work.view in update.views for row in rounds for work in row)
    assert all(0 <= work.view < len(update.population.conditioning) for row in rounds for work in row)


def test_owned_positions_cover_each_rank_turn_exactly():
    snapshot, _, update, term = resolved("sequence-coupled@1")
    plan = resolve_score_ownership(update, term, owners(snapshot, *SPLIT))
    first, second = (owned_positions(update, plan, rank) for rank in range(2))
    assert first.tolist() == [snapshot.view_positions(snapshot.view_index("A1")).start]
    assert sorted(first.tolist() + second.tolist()) == list(range(snapshot.size))


def test_forged_or_unbound_ownership_cannot_be_used_for_native_rounds():
    # Rewritten by design: partitions now name whole turns only, so a separate
    # context-to-action mapping cannot be forged. Rounds still re-derive the
    # ownership and reject a plan bound to another term or a non-canonical one.
    snapshot, _, update, term = resolved()
    plan = resolve_score_ownership(update, term, owners(snapshot, *SPLIT))
    with pytest.raises(InvalidPolicyUpdate, match="differs"):
        resolve_score_rounds(update, term, replace(plan, term_digest="forged"))
    reordered = (plan.partitions[0], RankScorePartition(1, tuple(reversed(plan.partitions[1].views))))
    with pytest.raises(InvalidPolicyUpdate, match="differs"):
        resolve_score_rounds(update, term, replace(plan, partitions=reordered))


def test_owned_carriers_route_global_coupled_adjoints_and_reject_drift():
    torch = pytest.importorskip("torch")
    from posttrain.train.backends.policy_update_distribution import distributed_score_carrier
    from posttrain.train.backends.policy_update_math import ScoreBundle
    from posttrain.train.backends.policy_update_replay import prepare_score_adjoints
    from posttrain.train.backends.policy_update_scoring import PositionScores

    snapshot, credit, update, term = resolved("sequence-coupled@1")
    scored = np.ones(snapshot.size, dtype=bool)
    prepared = prepare_score_adjoints(
        term,
        credit,
        ScoreBundle(torch.full((snapshot.size,), 0.1), torch.zeros(snapshot.size), term.parameter_version, scored),
    )
    plan = resolve_score_ownership(update, term, owners(snapshot, *SPLIT))
    positions = owned_positions(update, plan, 1)
    values = torch.full((positions.size,), 0.1, requires_grad=True)
    owned = PositionScores(positions, values)
    loss = distributed_score_carrier(plan, snapshot, prepared, owned, rank=1, parameter_version=term.parameter_version)
    (gradient,) = torch.autograd.grad(loss, (values,))
    torch.testing.assert_close(gradient, prepared.adjoints[torch.as_tensor(positions)] * 2)
    everything = PositionScores(np.arange(snapshot.size), torch.full((snapshot.size,), 0.1))
    with pytest.raises(InvalidPolicyUpdate, match="owned"):
        distributed_score_carrier(
            plan, snapshot, prepared, everything, rank=1, parameter_version=term.parameter_version
        )
    with pytest.raises(InvalidPolicyUpdate, match="replay drift"):
        distributed_score_carrier(
            plan,
            snapshot,
            prepared,
            PositionScores(positions, values + 0.01),
            rank=1,
            parameter_version=term.parameter_version,
        )
    with pytest.raises(InvalidPolicyUpdate, match="identity"):
        distributed_score_carrier(plan, snapshot, prepared, owned, rank=True, parameter_version=term.parameter_version)


def test_empty_rank_requires_an_actual_zero_gradient_native_anchor():
    torch = pytest.importorskip("torch")
    from posttrain.train.backends.policy_update_distribution import distributed_score_carrier
    from posttrain.train.backends.policy_update_math import ScoreBundle
    from posttrain.train.backends.policy_update_replay import prepare_score_adjoints

    snapshot, credit, update, term = resolved()
    scored = np.ones(snapshot.size, dtype=bool)
    prepared = prepare_score_adjoints(
        term,
        credit,
        ScoreBundle(torch.full((snapshot.size,), 0.1), torch.zeros(snapshot.size), term.parameter_version, scored),
    )
    plan = resolve_score_ownership(update, term, owners(snapshot, *EMPTY_FIRST))
    parameter = torch.tensor(3.0, requires_grad=True)
    carrier = distributed_score_carrier(
        plan,
        snapshot,
        prepared,
        None,
        rank=0,
        parameter_version=term.parameter_version,
        empty_anchor=parameter.square(),
    )
    carrier.backward()
    assert parameter.grad.item() == 0
    for anchor in (None, torch.tensor(0.0), parameter * float("nan")):
        with pytest.raises(InvalidPolicyUpdate, match="anchor"):
            distributed_score_carrier(
                plan, snapshot, prepared, None, rank=0, parameter_version=term.parameter_version, empty_anchor=anchor
            )
