"""Score-and-replay preserves complete objectives across native backward packs."""

import copy

import pytest
from posttrain.train.update_objectives import resolve_objective_term
from posttrain.train.update_records import InvalidPolicyUpdate

torch = pytest.importorskip("torch")

from posttrain.train.backends.policy_update_math import ScoreBundle, evaluate  # noqa: E402
from posttrain.train.backends.policy_update_replay import prepare_score_adjoints  # noqa: E402
from posttrain.train.backends.policy_update_scoring import score_actions  # noqa: E402

from .test_update_execution import resolved  # noqa: E402
from .test_update_scoring import CausalModel  # noqa: E402


@pytest.mark.parametrize("identity", ["grpo@1", "sampo@1", "sequence-coupled@1"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_replay_model_gradients_match_graph_retention_with_clipping(identity, dtype):
    torch.manual_seed(29)
    snapshot, source, credit, spec, update, _ = resolved(identity)
    full = CausalModel().to(dtype)
    replay = copy.deepcopy(full)
    kwargs = dict(read_input=lambda view: source, device=torch.device("cpu"), score_temperature=0.7)
    current = score_actions(full, snapshot, update.dependencies, **kwargs)
    # Deliberately exercise clipping and mixed signs, independent of LR/recipe.
    old = {
        action: value.detach() - (0.4 if index < 2 else -0.4) for index, (action, value) in enumerate(current.items())
    }
    term = resolve_objective_term(update, spec, credit)
    direct = evaluate(term, credit, ScoreBundle(current, old, term.parameter_version))
    direct.loss.backward()
    with torch.no_grad():
        frozen = score_actions(replay, snapshot, update.dependencies, **kwargs)
        adjoints = prepare_score_adjoints(term, credit, ScoreBundle(frozen, old, term.parameter_version))
    assert len(adjoints.evaluation.clipped_actions) > 0
    torch.testing.assert_close(adjoints.evaluation.loss, direct.loss.detach())
    for actions in (update.dependencies[:2], update.dependencies[2:]):
        scores = score_actions(replay, snapshot, actions, **kwargs)
        adjoints.carrier(scores, parameter_version=term.parameter_version).backward()
    tolerance = 0.005 if dtype == torch.bfloat16 else 0.0007 if dtype == torch.float16 else 1e-6
    for left, right in zip(full.parameters(), replay.parameters(), strict=True):
        torch.testing.assert_close(left.grad, right.grad, rtol=tolerance, atol=tolerance)
    assert all(not value.requires_grad for value in adjoints.adjoints.values())


def test_replay_rejects_changed_parameters_or_scores_before_backward():
    snapshot, source, credit, spec, update, _ = resolved("sampo@1")
    model = CausalModel()
    scores = score_actions(
        model,
        snapshot,
        update.dependencies,
        read_input=lambda view: source,
        device=torch.device("cpu"),
        score_temperature=1,
    )
    term = resolve_objective_term(update, spec, credit)
    adjoints = prepare_score_adjoints(term, credit, ScoreBundle(scores, scores, term.parameter_version))
    with pytest.raises(InvalidPolicyUpdate, match="parameter version changed"):
        adjoints.carrier(scores, parameter_version="later")
    with pytest.raises(InvalidPolicyUpdate, match="drift exceeds"):
        adjoints.carrier(
            {action: value + 0.01 for action, value in scores.items()}, parameter_version=term.parameter_version
        )
