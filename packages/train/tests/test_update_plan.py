from dataclasses import replace

import numpy as np
import pytest
from posttrain.train.update_plan import (
    ContributionRef,
    ExecutionCapabilities,
    ObjectivePopulation,
    PolicyExecutionBudget,
    PolicyUpdateSchedule,
    ResolvedUpdate,
    execution_context_tokens,
    plan_packs,
    population_context_width,
    resolve_updates,
    turn_order,
)
from posttrain.train.update_records import (
    ConditioningView,
    InvalidPolicyUpdate,
    PolicyVersions,
    PopulationSnapshot,
)

TURNS = (("A", "A1"), ("A", "A2"), ("A", "A3"), ("B", "B1"), ("B", "B2"))


def five_turns() -> tuple[PopulationSnapshot, ObjectivePopulation]:
    """Two episodes (A: three turns, B: two turns), one sampled token per turn."""
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
    everything = np.ones(snapshot.size, dtype=bool)
    objective = ObjectivePopulation(
        "fixture-local@1",
        "credit-digest",
        tuple(ContributionRef(turn, index, (index,)) for index, (_, turn) in enumerate(TURNS)),
        everything,
        everything,
        ("sampled-logp",),
        4,
    )
    return snapshot, objective


def capabilities(*, coupled: bool = False) -> ExecutionCapabilities:
    return ExecutionCapabilities(("fixture-local@1",), ("sampled-logp",), 100, coupled)


def turn_ids(snapshot: PopulationSnapshot, views: tuple[int, ...]) -> tuple[str, ...]:
    return tuple(snapshot.conditioning[view].id for view in views)


def test_two_episodes_five_turns_three_packs_one_update() -> None:
    snapshot, objective = five_turns()
    updates = resolve_updates(snapshot, PolicyUpdateSchedule("episode", 2), objective)
    assert len(updates) == 1
    packs = plan_packs(updates[0], PolicyExecutionBudget(2, 100, 100), capabilities())
    assert [turn_ids(snapshot, pack.views) for pack in packs] == [("A1", "A2"), ("A3", "B1"), ("B2",)]
    assert {pack.update_digest for pack in packs} == {updates[0].digest}
    repacked = plan_packs(updates[0], PolicyExecutionBudget(5, 100, 100), capabilities())
    assert len(repacked) == 1
    assert repacked[0].update_digest == packs[0].update_digest


def test_frozen_population_reuse_creates_distinct_occurrences() -> None:
    snapshot, objective = five_turns()
    schedule = PolicyUpdateSchedule("episode", 1, epochs=2, order="shuffle", seed=12)
    updates = resolve_updates(snapshot, schedule, objective)
    assert [update.digest for update in updates] == [
        update.digest for update in resolve_updates(snapshot, schedule, objective)
    ]
    assert len(updates) == 4
    occurrences = [occurrence for update in updates for occurrence in update.occurrence_ids]
    assert len(set(occurrences)) == len(occurrences) == 10
    assert {update.population.versions.old_score for update in updates} == {"old@1"}
    assert {update.objective.credit_digest for update in updates} == {"credit-digest"}


def test_partial_episode_retains_unselected_ratio_dependencies() -> None:
    snapshot, objective = five_turns()
    selected = replace(objective.contributions[0], dependency_views=(0, 1, 2))
    objective = replace(objective, contributions=(selected,))
    update = resolve_updates(snapshot, PolicyUpdateSchedule("turn", 1), objective)[0]
    contribution = update.objective.contributions[update.contributions[0]]
    assert int(objective.policy[snapshot.view_positions(contribution.view)].sum()) == 1
    assert update.views == (0, 1, 2)
    assert update.dependency_count == 3
    with pytest.raises(InvalidPolicyUpdate, match="dependency crosses packs"):
        plan_packs(update, PolicyExecutionBudget(1, 100, 100), capabilities())
    packs = plan_packs(update, PolicyExecutionBudget(1, 100, 100), capabilities(coupled=True))
    assert len(packs) == 3


def test_contribution_dependency_closure_must_include_its_turn() -> None:
    with pytest.raises(InvalidPolicyUpdate, match="dependency closure"):
        ContributionRef("A1", 0, (1, 2))
    with pytest.raises(InvalidPolicyUpdate, match="cannot duplicate"):
        ContributionRef("A1", 0, (0, 0))
    snapshot, objective = five_turns()
    with pytest.raises(InvalidPolicyUpdate, match="identities must be unique"):
        replace(objective, contributions=objective.contributions + objective.contributions[:1])
    with pytest.raises(InvalidPolicyUpdate, match="different population"):
        ResolvedUpdate(
            replace(snapshot, conditioning=snapshot.conditioning[:4]),
            objective,
            "schedule",
            0,
            0,
            (0,),
            ("occurrence",),
            (0,),
            (),
        )


def test_final_minibatch_policies_are_explicit() -> None:
    snapshot, objective = five_turns()
    included = resolve_updates(snapshot, PolicyUpdateSchedule("turn", 2), objective)
    assert [len(update.contributions) for update in included] == [2, 2, 1]
    dropped = resolve_updates(snapshot, PolicyUpdateSchedule("turn", 2, final_policy="drop"), objective)
    assert len(dropped) == 2
    assert dropped[0].discarded_contributions == ("B2",)
    with pytest.raises(InvalidPolicyUpdate, match="incomplete final"):
        resolve_updates(snapshot, PolicyUpdateSchedule("turn", 2, final_policy="error"), objective)
    with pytest.raises(InvalidPolicyUpdate, match="drops every"):
        resolve_updates(snapshot, PolicyUpdateSchedule("turn", 10, final_policy="drop"), objective)


def test_token_budget_cannot_truncate_an_atomic_contribution() -> None:
    snapshot, objective = five_turns()
    # Turn A1 samples three tokens: one atomic contribution costing three selected tokens.
    snapshot = replace(
        snapshot, conditioning=(replace(snapshot.conditioning[0], sampled=(0, 1, 2)), *snapshot.conditioning[1:])
    )
    everything = np.ones(snapshot.size, dtype=bool)
    objective = replace(objective, policy=everything, kl=everything)
    updates = resolve_updates(snapshot, PolicyUpdateSchedule("selected-token", 2), objective)
    assert [len(update.contributions) for update in updates] == [1, 2, 2]
    assert updates[0].views == (0,)
    assert updates[0].dependency_count == 3


def test_capacity_and_statistics_fail_before_execution() -> None:
    snapshot, objective = five_turns()
    update = resolve_updates(snapshot, PolicyUpdateSchedule("episode", 2), objective)[0]
    for budget, capability, message in (
        (PolicyExecutionBudget(2, 100, 1), capabilities(), "statistics exceed"),
        (PolicyExecutionBudget(2, 5, 100), capabilities(), "conditioning exceeds"),
        (PolicyExecutionBudget(2, 100, 100), replace(capabilities(), max_context_tokens=5), "context capacity"),
        (PolicyExecutionBudget(2, 100, 100), replace(capabilities(), statistics=()), "lacks required statistics"),
        (PolicyExecutionBudget(2, 100, 100), replace(capabilities(), definition_ids=()), "qualified objective"),
    ):
        with pytest.raises(InvalidPolicyUpdate, match=message):
            plan_packs(update, budget, capability)


def test_statistic_capacity_counts_every_position_of_dependency_turns() -> None:
    snapshot, objective = five_turns()
    # A2 samples four tokens; A1's ratio depends on A1-A3, so six positions are retained.
    snapshot = replace(
        snapshot,
        conditioning=(
            snapshot.conditioning[0],
            replace(snapshot.conditioning[1], sampled=(0, 1, 2, 3)),
            *snapshot.conditioning[2:],
        ),
    )
    everything = np.ones(snapshot.size, dtype=bool)
    objective = replace(
        objective,
        contributions=(replace(objective.contributions[0], dependency_views=(0, 1, 2)),),
        policy=everything,
        kl=everything,
    )
    update = resolve_updates(snapshot, PolicyUpdateSchedule("turn", 1), objective)[0]
    assert update.dependency_count == 6
    assert len(plan_packs(update, PolicyExecutionBudget(3, 100, 24), capabilities())) == 1
    with pytest.raises(InvalidPolicyUpdate, match="statistics exceed"):
        plan_packs(update, PolicyExecutionBudget(3, 100, 23), capabilities())


@pytest.mark.parametrize(
    "layout,sizes,costs",
    [
        ("ragged", [2, 2, 1], [12, 18, 5]),
        ("dense-pack", [2, 1, 2], [18, 11, 14]),
        ("dense-population", [1, 1, 1, 1, 1], [11, 11, 11, 11, 11]),
    ],
)
def test_physical_padding_cost_changes_only_execution_boundaries(layout, sizes, costs):
    snapshot, objective = five_turns()
    snapshot = replace(
        snapshot,
        conditioning=tuple(
            replace(view, context_tokens=length)
            for view, length in zip(snapshot.conditioning, (3, 9, 11, 7, 5), strict=True)
        ),
    )
    update = resolve_updates(snapshot, PolicyUpdateSchedule("episode", 2), objective)[0]
    original_digest = update.digest
    layout_capabilities = replace(capabilities(), context_layout=layout)
    packs = plan_packs(update, PolicyExecutionBudget(2, 18, 100), layout_capabilities)
    assert [len(pack.views) for pack in packs] == sizes
    assert [pack.context_tokens for pack in packs] == costs
    assert [execution_context_tokens(snapshot, pack.views, layout_capabilities) for pack in packs] == costs
    assert all(pack.context_tokens <= 18 for pack in packs)
    assert tuple(view for pack in packs for view in pack.views) == update.views
    assert update.views == turn_order(snapshot, set(update.views))
    assert update.digest == original_digest and {pack.update_digest for pack in packs} == {original_digest}


def test_population_padding_includes_unselected_sampled_contexts_and_rejects_oversized_width():
    snapshot, objective = five_turns()
    snapshot = replace(
        snapshot,
        conditioning=tuple(
            replace(view, context_tokens=length)
            for view, length in zip(snapshot.conditioning, (3, 9, 11, 7, 5), strict=True)
        ),
    )
    selected = replace(objective, contributions=objective.contributions[:1])
    update = resolve_updates(snapshot, PolicyUpdateSchedule("turn", 1), selected)[0]
    dense = replace(capabilities(), context_layout="dense-population")
    assert plan_packs(update, PolicyExecutionBudget(1, 11, 100), dense)[0].context_tokens == 11
    with pytest.raises(InvalidPolicyUpdate, match="conditioning exceeds"):
        plan_packs(update, PolicyExecutionBudget(1, 10, 100), dense)
    with pytest.raises(InvalidPolicyUpdate, match="padding width exceeds"):
        plan_packs(update, PolicyExecutionBudget(1, 11, 100), replace(dense, max_context_tokens=10))
    # Every conditioning view samples at least one token, so an added unselected
    # turn widens the population padding even though no contribution selects it.
    unused = replace(snapshot.conditioning[0], id="unselected", context_tokens=99)
    assert population_context_width(replace(snapshot, conditioning=(*snapshot.conditioning, unused))) == 99
    with pytest.raises(InvalidPolicyUpdate, match="sampled token indices"):
        replace(unused, sampled=())
