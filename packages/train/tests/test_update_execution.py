"""Model-graph bridge checks; native trainer qualification is a separate gate."""

import copy
from dataclasses import replace

import numpy as np
import pytest
from posttrain.train.update_credit import PreparedCredit
from posttrain.train.update_objectives import ObjectiveSpec, objective_population, resolve_objective_term
from posttrain.train.update_plan import (
    ExecutionCapabilities,
    PolicyExecutionBudget,
    PolicyUpdateSchedule,
    plan_packs,
    resolve_updates,
)
from posttrain.train.update_records import InvalidPolicyUpdate, PopulationSnapshot

torch = pytest.importorskip("torch")

from posttrain.train.backends.policy_update_execution import compute_resolved_loss  # noqa: E402
from posttrain.train.backends.policy_update_scoring import freeze_population_scores  # noqa: E402

from .test_update_scoring import CausalModel, score_population  # noqa: E402


def resolved(identity):
    """Two one-turn episodes over the same original context: positions 0-1 and 2-3."""
    snapshot, source, _ = score_population()
    view = replace(snapshot.conditioning[0], id="context-b", episode_id="episode-b")
    snapshot = PopulationSnapshot(
        snapshot.id,
        snapshot.native_evidence_ref,
        snapshot.native_evidence_digest,
        snapshot.conditioning + (view,),
        snapshot.spans,
        snapshot.relations,
        snapshot.versions,
        snapshot.selector_digest,
    )
    credit = PreparedCredit(
        snapshot.digest,
        "external-fixture@1",
        np.array((1, -0.5, 0.25, 0.75)),
        (),
        (),
        "fixture@1",
        "prefix",
        ("evidence",),
    )
    spec = ObjectiveSpec(identity)
    objective = objective_population(snapshot, spec, credit)
    update = resolve_updates(snapshot, PolicyUpdateSchedule("episode", 2), objective)[0]
    capabilities = ExecutionCapabilities((identity,), objective.required_statistics, 100, True)
    return snapshot, source, credit, spec, update, capabilities


@pytest.mark.parametrize("packwise", [False, True])
@pytest.mark.parametrize("identity", ["grpo@1", "sampo@1"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_two_applied_transitions_match_monolithic_oracle_across_packs(identity, dtype, packwise):
    torch.manual_seed(11)
    snapshot, source, credit, spec, update, capabilities = resolved(identity)
    initial = CausalModel().to(dtype)
    reference = copy.deepcopy(initial)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    ids = torch.tensor([source.token_ids])

    def reference_logps(model):
        logits = model.head(model.embedding(ids).cumsum(1))[0, [2, 3]].float() / 0.7
        return logits[:, [4, 5]].diagonal() - torch.logsumexp(logits, dim=1)

    old_reference = reference_logps(reference).detach()

    def reference_loss(model):
        current = reference_logps(model)
        delta = current - old_reference
        if identity == "sampo@1":
            ratios = (delta.mean().detach() + current - current.detach()).exp()
        else:
            ratios = delta.exp()
        advantages = torch.tensor([[1.0, -0.5], [0.25, 0.75]])
        return torch.maximum(-ratios * advantages, -ratios.clamp(0.8, 1.2) * advantages).mean()

    expected_states, expected_gradients = [], []
    for _ in range(2):
        reference_optimizer.zero_grad()
        loss = reference_loss(reference)
        loss.backward()
        expected_gradients.append([parameter.grad.detach().clone() for parameter in reference.parameters()])
        reference_optimizer.step()
        expected_states.append(copy.deepcopy(reference.state_dict()))

    for count in (1, 2):
        model = copy.deepcopy(initial)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        # One turn per pack (two packs) versus both turns in one pack.
        packs = plan_packs(update, PolicyExecutionBudget(count, 100, 10000), capabilities)
        assert len(packs) == 3 - count
        kwargs = dict(read_input=lambda view: source, device=torch.device("cpu"), score_temperature=0.7)
        old = freeze_population_scores(model, snapshot, policy_version="old@1", score_contract="causal@1", **kwargs)
        for index in range(2):
            optimizer.zero_grad()
            term = resolve_objective_term(update, spec, credit, parameter_version=f"current@{index + 1}")
            actual = compute_resolved_loss(
                model,
                update,
                term,
                credit,
                packs,
                old=old,
                reference=None,
                score_contract="causal@1",
                sampler_correction=None,
                backward=(lambda loss: loss.backward()) if packwise else None,
                **kwargs,
            )
            # Pack-wise backward has already applied every pack's gradient and
            # returns a gradient-free leaf, so the trainer's backward adds nothing.
            assert actual.loss.requires_grad and actual.loss.is_leaf == packwise
            # Loss arithmetic is compared at identical parameters. Independent
            # half-parameter optimizer histories can differ by rounding after
            # separate graph accumulation; gradients and states are compared below.
            # The vectorized loss is a weighted sum (weights 1/4, exact in
            # binary) rather than a mean; the original tolerance still holds.
            torch.testing.assert_close(actual.loss, reference_loss(model).detach(), rtol=1e-6, atol=1e-6)
            actual.loss.backward()
            tolerance = 0.005 if dtype == torch.bfloat16 else 0.0007 if dtype == torch.float16 else 1e-6
            for parameter, gradient in zip(model.parameters(), expected_gradients[index], strict=True):
                torch.testing.assert_close(parameter.grad, gradient, rtol=tolerance, atol=tolerance)
            optimizer.step()
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, expected_states[index][name], rtol=tolerance, atol=tolerance)


def test_missing_dependency_pack_rejected_before_model_execution():
    snapshot, source, credit, spec, update, capabilities = resolved("sampo@1")
    model = CausalModel()
    kwargs = dict(read_input=lambda view: source, device=torch.device("cpu"), score_temperature=1)
    old = freeze_population_scores(model, snapshot, policy_version="old@1", score_contract="causal@1", **kwargs)
    packs = plan_packs(update, PolicyExecutionBudget(1, 100, 10000), capabilities)
    assert len(packs) == 2
    term = resolve_objective_term(update, spec, credit)
    with pytest.raises(InvalidPolicyUpdate, match="each resolved dependency"):
        compute_resolved_loss(
            model,
            update,
            term,
            credit,
            packs[:1],
            old=old,
            reference=None,
            score_contract="causal@1",
            sampler_correction=None,
            **kwargs,
        )


def test_packwise_backward_requires_every_ratio_segment_inside_one_pack():
    from posttrain.train.backends.policy_update_execution import _segments_within_packs

    _, _, credit, spec, update, capabilities = resolved("sampo@1")
    term = resolve_objective_term(update, spec, credit)
    split = plan_packs(update, PolicyExecutionBudget(1, 100, 10000), capabilities)
    joined = plan_packs(update, PolicyExecutionBudget(2, 100, 10000), capabilities)
    assert _segments_within_packs(update, term, split) and _segments_within_packs(update, term, joined)
    # One sequence ratio over both episodes cannot be split across two packs.
    spanning = replace(term, ratio_segment=np.where(term.ratio_segment >= 0, 0, -1), segment_count=1)
    assert not _segments_within_packs(update, spanning, split)
    assert _segments_within_packs(update, spanning, joined)
