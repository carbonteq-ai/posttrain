"""Score-and-replay preserves complete objectives across native backward packs."""

import copy

import numpy as np
import pytest
from posttrain.train.update_objectives import resolve_objective_term
from posttrain.train.update_records import InvalidPolicyUpdate

torch = pytest.importorskip("torch")

from posttrain.train.backends.policy_update_math import ScoreBundle, evaluate  # noqa: E402
from posttrain.train.backends.policy_update_replay import prepare_score_adjoints  # noqa: E402
from posttrain.train.backends.policy_update_scoring import PositionScores, dense_scores, score_views  # noqa: E402

from .test_update_execution import resolved  # noqa: E402
from .test_update_scoring import CausalModel  # noqa: E402


@pytest.mark.parametrize("identity", ["grpo@1", "sampo@1", "sequence-coupled@1"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_replay_model_gradients_match_graph_retention_with_clipping(identity, dtype):
    torch.manual_seed(29)
    snapshot, source, credit, spec, update, _ = resolved(identity)
    full = CausalModel().to(dtype)
    replay = copy.deepcopy(full)
    cpu = torch.device("cpu")
    kwargs = dict(read_input=lambda view: source, device=cpu, score_temperature=0.7)
    scored = snapshot.views_mask(update.views)
    assert scored.all()
    current = dense_scores(snapshot.size, (score_views(full, snapshot, update.views, **kwargs),), device=cpu)
    # Deliberately exercise clipping and mixed signs, independent of LR/recipe:
    # the first turn's old scores sit below current, the second turn's above.
    old = current.detach() - torch.tensor([0.4, 0.4, -0.4, -0.4])
    term = resolve_objective_term(update, spec, credit)
    direct = evaluate(term, credit, ScoreBundle(current, old, term.parameter_version, scored))
    direct.loss.backward()
    with torch.no_grad():
        frozen = dense_scores(snapshot.size, (score_views(replay, snapshot, update.views, **kwargs),), device=cpu)
        adjoints = prepare_score_adjoints(term, credit, ScoreBundle(frozen, old, term.parameter_version, scored))
    assert adjoints.evaluation.clipped.any()
    torch.testing.assert_close(adjoints.evaluation.loss, direct.loss.detach())
    # One backward pack per turn.
    assert len(update.views) == 2
    for views in ((update.views[0],), (update.views[1],)):
        scores = score_views(replay, snapshot, views, **kwargs)
        adjoints.carrier(scores, parameter_version=term.parameter_version).backward()
    tolerance = 0.005 if dtype == torch.bfloat16 else 0.0007 if dtype == torch.float16 else 1e-6
    for left, right in zip(full.parameters(), replay.parameters(), strict=True):
        torch.testing.assert_close(left.grad, right.grad, rtol=tolerance, atol=tolerance)
    assert not adjoints.adjoints.requires_grad


def test_replay_rejects_changed_parameters_or_scores_before_backward():
    snapshot, source, credit, spec, update, _ = resolved("sampo@1")
    model = CausalModel()
    cpu = torch.device("cpu")
    scores = score_views(
        model,
        snapshot,
        update.views,
        read_input=lambda view: source,
        device=cpu,
        score_temperature=1,
    )
    dense = dense_scores(snapshot.size, (scores,), device=cpu)
    scored = np.ones(snapshot.size, dtype=bool)
    term = resolve_objective_term(update, spec, credit)
    adjoints = prepare_score_adjoints(term, credit, ScoreBundle(dense, dense, term.parameter_version, scored))
    with pytest.raises(InvalidPolicyUpdate, match="parameter version changed"):
        adjoints.carrier(scores, parameter_version="later")
    with pytest.raises(InvalidPolicyUpdate, match="drift exceeds"):
        adjoints.carrier(
            PositionScores(scores.positions, scores.values + 0.01), parameter_version=term.parameter_version
        )
