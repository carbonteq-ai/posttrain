"""One forward per covering context: planning and scoring equivalence."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from posttrain.environment.verifiers_conditioning import materialize_native_conditioning, native_conditioning_records
from posttrain.train.update_objectives import ObjectiveSpec
from posttrain.train.update_plan import (
    ContributionRef,
    ExecutionCapabilities,
    ObjectivePopulation,
    PolicyExecutionBudget,
    PolicyUpdateSchedule,
    plan_packs,
    prefix_covers,
    resolve_updates,
)
from posttrain.train.update_records import (
    ConditioningView,
    InvalidPolicyUpdate,
    PolicyVersions,
    PopulationSnapshot,
)

torch = pytest.importorskip("torch")

from posttrain.train.backends.policy_update_scoring import (  # noqa: E402
    PositionScores,
    dense_scores,
    freeze_population_scores,
    score_covers,
    score_views,
)

from .test_update_scoring import CausalModel  # noqa: E402


def node(parent, role, tokens, sampled=False):
    return SimpleNamespace(
        parent=parent,
        sampled=sampled,
        message={"role": role},
        token_ids=list(tokens),
        mask=[sampled and index > 0 for index in range(len(tokens))],
    )


def trace():
    """A system prompt, three chained assistant turns, and one branch off the first tool result."""
    nodes = [
        node(None, "system", [1, 2]),
        node(0, "assistant", [3, 4, 5], sampled=True),
        node(1, "tool", [6]),
        node(2, "assistant", [0, 2, 4, 6], sampled=True),
        node(3, "tool", [1, 3]),
        node(4, "assistant", [5, 2], sampled=True),
        node(2, "assistant", [6, 1, 4], sampled=True),
    ]
    return SimpleNamespace(id="trace", nodes=nodes, calls=[SimpleNamespace(node=index) for index in (1, 3, 5, 6)])


def population():
    native = trace()
    records = native_conditioning_records(native, sampled_node_indices=(1, 3, 5, 6), context_contract="causal-text@1")
    views = tuple(
        ConditioningView(
            f"trace/node-{record.node_index}",
            "native",
            json.dumps(
                {"trace_id": "trace", "prefix_nodes": record.prefix_node_indices, "node_index": record.node_index},
                sort_keys=True,
                separators=(",", ":"),
            ),
            "causal-text@1/attention",
            "causal-text@1/positions",
            "template@1",
            record.input_digest,
            record.context_tokens,
            "episode",
            "branch",
            record.sampled_token_indices,
        )
        for record in records
    )
    snapshot = PopulationSnapshot(
        "population", "native", "digest", views, (), (), PolicyVersions("s@1", "o@1", "c@1", None), "all@1"
    )
    inputs = {
        view.id: materialize_native_conditioning(native, record) for view, record in zip(views, records, strict=True)
    }
    return snapshot, lambda view: inputs[view.id]


def test_chained_turns_share_the_longest_context_and_branches_keep_their_own():
    snapshot, _ = population()
    covers = prefix_covers(snapshot, (0, 1, 2, 3))
    # Turns 0, 1, 2 form one chain; turn 3 branches off turn 0's tool result.
    assert covers == ((2, (0, 1, 2)), (3, (3,)))


def test_turns_without_native_coordinates_cover_only_themselves():
    snapshot, _ = population()
    plain = PopulationSnapshot(
        snapshot.id,
        snapshot.native_evidence_ref,
        snapshot.native_evidence_digest,
        tuple(
            ConditioningView(
                view.id,
                view.native_ref,
                "opaque",
                view.attention_ref,
                view.positions_ref,
                view.template_revision,
                view.digest,
                view.context_tokens,
                view.episode_id,
                view.branch_id,
                view.sampled,
            )
            for view in snapshot.conditioning
        ),
        (),
        (),
        snapshot.versions,
        snapshot.selector_digest,
    )
    assert prefix_covers(plain, (0, 1, 2, 3)) == ((0, (0,)), (1, (1,)), (2, (2,)), (3, (3,)))


def resolved_update(snapshot):
    """One occurrence that depends on every turn of the episode (as an episode-geometric ratio does)."""
    objective = ObjectivePopulation(
        "sampo@1",
        "credit",
        tuple(ContributionRef(f"c{view}", view, (0, 1, 2, 3)) for view in range(4)),
        np.ones(snapshot.size, dtype=bool),
        np.zeros(snapshot.size, dtype=bool),
        ("sampled-logp", "old-logp"),
        16,
        ObjectiveSpec("sampo@1").digest,
    )
    return resolve_updates(snapshot, PolicyUpdateSchedule("episode", 1), objective)[0]


def capabilities(*, prefix_sharing):
    return ExecutionCapabilities(("sampo@1",), ("sampled-logp", "old-logp"), 100, True, prefix_sharing=prefix_sharing)


def test_packs_charge_each_forwarded_context_once_and_never_split_a_cover():
    snapshot, _ = population()
    update = resolved_update(snapshot)
    budget = PolicyExecutionBudget(records=1, context_tokens=1000, statistic_bytes=10**9)
    shared = plan_packs(update, budget, capabilities(prefix_sharing=True))
    assert [pack.covers for pack in shared] == [((2, (0, 1, 2)),), ((3, (3,)),)]
    lengths = [view.context_tokens for view in snapshot.conditioning]
    assert [pack.context_tokens for pack in shared] == [lengths[2], lengths[3]]
    separate = plan_packs(update, budget, capabilities(prefix_sharing=False))
    assert [pack.covers for pack in separate] == [((view, (view,)),) for view in (0, 1, 2, 3)]
    assert sum(pack.context_tokens for pack in separate) == sum(lengths)
    with pytest.raises(InvalidPolicyUpdate, match="hard pack capacity"):
        plan_packs(
            update,
            PolicyExecutionBudget(records=4, context_tokens=lengths[2] - 1, statistic_bytes=10**9),
            capabilities(prefix_sharing=True),
        )


def test_cover_scores_equal_per_turn_scores_and_gradients():
    torch.manual_seed(3)
    model = CausalModel()
    snapshot, read_input = population()
    common = dict(read_input=read_input, device=torch.device("cpu"), score_temperature=0.8)
    entropies_each, entropies_shared = np.full(snapshot.size, np.nan), np.full(snapshot.size, np.nan)
    each = score_views(model, snapshot, (0, 1, 2, 3), entropies=entropies_each, **common)
    shared = score_covers(model, snapshot, prefix_covers(snapshot, (0, 1, 2, 3)), entropies=entropies_shared, **common)
    dense_each = dense_scores(snapshot.size, (each,), device=torch.device("cpu"))
    dense_shared = dense_scores(snapshot.size, (shared,), device=torch.device("cpu"))
    torch.testing.assert_close(dense_shared, dense_each, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(entropies_shared, entropies_each, rtol=1e-6)
    gradients = []
    for scores in (dense_each, dense_shared):
        model.zero_grad()
        (scores * torch.linspace(-1, 1, snapshot.size)).sum().backward()
        gradients.append([parameter.grad.clone() for parameter in model.parameters()])
    for left, right in zip(*gradients, strict=True):
        torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)
    shared_frozen = freeze_population_scores(
        model, snapshot, policy_version="o@1", score_contract="causal@1", prefix_sharing=True, **common
    )
    each_frozen = freeze_population_scores(model, snapshot, policy_version="o@1", score_contract="causal@1", **common)
    torch.testing.assert_close(shared_frozen.values, each_frozen.values, rtol=1e-6, atol=1e-6)


def test_a_cover_must_contain_its_turns_contexts():
    snapshot, read_input = population()
    with pytest.raises(InvalidPolicyUpdate, match="not a prefix"):
        score_covers(
            CausalModel(),
            snapshot,
            ((3, (1, 3)),),
            read_input=read_input,
            device=torch.device("cpu"),
            score_temperature=1.0,
        )
    assert isinstance(
        score_covers(
            CausalModel(),
            snapshot,
            ((2, (0, 2)),),
            read_input=read_input,
            device=torch.device("cpu"),
            score_temperature=1.0,
        ),
        PositionScores,
    )
