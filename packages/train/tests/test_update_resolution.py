from dataclasses import replace

import pytest
from posttrain.train.profiles import CAPOSettings, GDPOSettings, GRPOSettings, SAMPOSettings, TrainingLoop
from posttrain.train.reward_advantages import compute_capo_advantages
from posttrain.train.reward_evidence import ProcessCredit, RewardEvidence, RewardValue
from posttrain.train.sampo_advantages import compute_sampo_advantages
from posttrain.train.update_plan import (
    ExecutionCapabilities,
    PolicyExecutionBudget,
    PolicyUpdateSchedule,
    PolicyUpdateSettings,
)
from posttrain.train.update_records import (
    ActionInterval,
    ActionSelection,
    InvalidPolicyUpdate,
    PolicyVersions,
    SemanticSpan,
)
from posttrain.train.update_resolution import resolve_policy_population, resolve_rollout_population

from .test_update_evidence import native_rollout
from .test_update_objectives import resolved


def settings():
    return SAMPOSettings(
        id="resolution-test",
        loop=TrainingLoop(max_steps=2, per_device_batch_size=1),
        policy_updates=PolicyUpdateSettings(
            PolicyUpdateSchedule("episode", 1),
            PolicyExecutionBudget(2, 100, 1000),
        ),
    )


def capabilities():
    return ExecutionCapabilities(
        ("sampo@1", "sampo-spans@1", "grpo@1", "dapo@1"), ("sampled-logp", "old-logp", "reference-logp"), 100, True
    )


def test_typed_selection_preserves_credit_and_episode_dependencies():
    snapshot, credit, _, _ = resolved()
    selected = settings()
    result = resolve_policy_population(snapshot, credit, selected, capabilities())
    assert result.credit is credit
    assert result.spec.clip_low == selected.clip_epsilon_low
    assert result.spec.clip_high == selected.clip_epsilon_high
    assert [update.views for update in result.updates] == [(0, 1, 2), (3, 4)]
    assert [update.dependency_count for update in result.updates] == [3, 2]
    assert [len(packs) for packs in result.packs] == [2, 1]
    again = resolve_policy_population(snapshot, credit, selected, capabilities())
    assert again.snapshot.digest == result.snapshot.digest and again.credit.digest == result.credit.digest
    assert [update.digest for update in again.updates] == [update.digest for update in result.updates]
    assert again.packs == result.packs


def test_every_occurrence_is_capacity_checked_before_return():
    snapshot, credit, _, _ = resolved()
    # Only episode B exceeds this backend's qualified context capacity.
    snapshot = replace(
        snapshot,
        conditioning=tuple(
            replace(view, context_tokens=101) if view.id == "B2" else view for view in snapshot.conditioning
        ),
    )
    credit = replace(credit, population_digest=snapshot.digest)
    with pytest.raises(InvalidPolicyUpdate, match="context capacity"):
        resolve_policy_population(snapshot, credit, settings(), capabilities())


def test_legacy_and_unsupported_selections_do_not_fall_through():
    snapshot, credit, _, _ = resolved()
    with pytest.raises(InvalidPolicyUpdate, match="explicit policy_updates"):
        resolve_policy_population(
            snapshot,
            credit,
            SAMPOSettings(
                id="legacy",
                loop=TrainingLoop(max_steps=2, per_device_batch_size=2),
            ),
            capabilities(),
        )
    with pytest.raises(InvalidPolicyUpdate, match="qualified objective"):
        resolve_policy_population(snapshot, credit, settings(), replace(capabilities(), definition_ids=()))
    selected = settings()
    assert selected.policy_updates is not None
    with pytest.raises(InvalidPolicyUpdate, match="semantic spans"):
        resolve_policy_population(
            snapshot,
            credit,
            GRPOSettings(
                id="unsupported-spans",
                loop=selected.loop,
                policy_updates=replace(selected.policy_updates, objective_variant="semantic-spans"),
            ),
            capabilities(),
        )


@pytest.mark.parametrize("algorithm,high", [("grpo", 0.2), ("dapo", 0.28)])
def test_grpo_recipe_coefficients_are_not_replaced_by_sampo_defaults(algorithm, high):
    snapshot, credit, _, _ = resolved()
    selected = GRPOSettings(
        id="recipe",
        algorithm=algorithm,
        loop=settings().loop,
        clip_epsilon_high=high,
        beta=0.15,
        policy_updates=settings().policy_updates,
    )
    result = resolve_policy_population(snapshot, credit, selected, capabilities())
    assert result.spec.definition_id == f"{algorithm}@1"
    assert result.spec.clip_high == high
    assert result.spec.beta == 0.15
    assert "reference-logp" in result.updates[0].objective.required_statistics


@pytest.mark.parametrize("algorithm", ["gdpo", "capo"])
def test_structured_credit_is_transported_without_renormalizing_minibatches(algorithm):
    snapshot, credit, _, _ = resolved()
    if algorithm == "gdpo":
        selected = GDPOSettings(
            id="structured",
            loop=settings().loop,
            beta=0.25,
            clip_epsilon_low=0.13,
            clip_epsilon_high=0.24,
            policy_updates=settings().policy_updates,
            component_names=("outcome",),
            component_weights=(1.0,),
        )
    else:
        selected = CAPOSettings(
            id="structured",
            loop=settings().loop,
            beta=0.25,
            clip_epsilon_low=0.13,
            clip_epsilon_high=0.24,
            policy_updates=settings().policy_updates,
        )
    result = resolve_policy_population(
        snapshot,
        credit,
        selected,
        replace(
            capabilities(),
            definition_ids=(f"{algorithm}@1",),
        ),
    )
    assert result.credit is credit
    assert result.spec.definition_id == f"{algorithm}@1"
    assert (result.spec.clip_low, result.spec.clip_high, result.spec.beta) == (0.13, 0.24, 0.25)


def _reasoning_selection(schedule: PolicyUpdateSchedule):
    snapshot, credit, _, _ = resolved()
    action = snapshot.action(0)
    snapshot = replace(
        snapshot, spans=(SemanticSpan("thinking", "reasoning", "fixture@1", (ActionInterval(action, 1),)),)
    )
    credit = replace(credit, population_digest=snapshot.digest)
    selected = settings()
    assert selected.policy_updates is not None
    selected = replace(
        selected,
        policy_updates=replace(
            selected.policy_updates,
            schedule=schedule,
            objective_variant="semantic-spans",
            policy_selection=ActionSelection("spans", ("thinking",)),
            empty_policy="zero",
        ),
    )
    return snapshot, resolve_policy_population(snapshot, credit, selected, capabilities())


def test_reasoning_selection_keeps_unselected_turn_ratio_dependencies_in_one_occurrence():
    # Both episodes in one occurrence: episode B selects nothing and keeps zero weight.
    snapshot, result = _reasoning_selection(PolicyUpdateSchedule("episode", 2))
    assert result.spec.definition_id == "sampo-spans@1"
    assert result.spec.policy_selection.resolve(snapshot).tolist() == [True, False, False, False, False]
    assert result.updates[0].views == (0, 1, 2)
    assert result.updates[0].dependency_count == 3


def test_reasoning_selection_keeps_unselected_episode_ratio_dependencies():
    snapshot, result = _reasoning_selection(PolicyUpdateSchedule("episode", 1))
    assert result.spec.definition_id == "sampo-spans@1"
    assert result.spec.policy_selection.resolve(snapshot).tolist() == [True, False, False, False, False]
    assert result.updates[0].views == (0, 1, 2)
    assert result.updates[0].dependency_count == 3
    assert result.updates[1].views == () and result.packs[1] == ()


def native_resolution(rollouts, selected):
    return resolve_rollout_population(
        rollouts,
        selected,
        capabilities(),
        population_id="admitted",
        native_evidence_ref="native:receipts",
        native_evidence_digest="native-receipt-digest",
        template_revision="native-template@1",
        versions=PolicyVersions("sampler@3", "old@3", "current@3", None),
        sampler_step=3,
        selector_digest="original-actions@1",
    )


def test_native_collection_credit_shapes_before_group_statistics_without_changing_evidence():
    rollouts = (native_rollout("a", 0, 1.0), replace(native_rollout("a", 1, 1.0), is_truncated=True))
    selected = replace(settings(), truncation_penalty=0.5)
    result = native_resolution(rollouts, selected)
    expected = compute_sampo_advantages(
        selected, [row.example_id for row in rollouts], [rollouts[0], replace(rollouts[1], reward=0.5)]
    )
    assert result.credit.advantages.tolist() == pytest.approx(
        [
            value
            for rollout, row in zip(rollouts, expected.token_advantages, strict=True)
            for value, eligible in zip(row, rollout.env_mask, strict=True)
            if eligible
        ]
    )
    assert (result.credit.advantages != 0).any()
    assert rollouts[1].reward == 1.0
    assert result.snapshot.native_evidence_digest == "native-receipt-digest"
    assert len(result.updates) == 2
    assert len({update.objective.credit_digest for update in result.updates}) == 1


def test_native_collection_normalizes_complete_groups_before_optimizer_slicing():
    rollouts = (native_rollout("a", 0, 1.0), native_rollout("a", 1, 0.0))
    split = native_resolution(rollouts, settings())
    selected = settings()
    assert selected.policy_updates is not None
    full = native_resolution(
        rollouts,
        replace(
            selected,
            policy_updates=replace(
                selected.policy_updates,
                schedule=PolicyUpdateSchedule("episode", 2),
            ),
        ),
    )
    assert split.credit.advantages.tolist() == full.credit.advantages.tolist()
    assert len(split.updates) == 2 and len(full.updates) == 1
    with pytest.raises(InvalidPolicyUpdate, match="complete prompt groups"):
        native_resolution(rollouts[:1], settings())


def test_unqualified_truncation_mask_does_not_silently_keep_all_actions():
    snapshot, credit, _, _ = resolved()
    with pytest.raises(InvalidPolicyUpdate, match="truncation masking"):
        resolve_policy_population(
            snapshot, credit, replace(settings(), mask_truncated_completions=True), capabilities()
        )


def test_resolved_population_cannot_replace_a_later_objective_or_pack():
    snapshot, credit, _, _ = resolved()
    prepared = resolve_policy_population(snapshot, credit, settings(), capabilities())
    changed = replace(
        prepared.updates[-1], objective=replace(prepared.updates[-1].objective, contract_digest="different-objective")
    )
    with pytest.raises(InvalidPolicyUpdate, match="objective differs"):
        replace(prepared, updates=(*prepared.updates[:-1], changed))
    with pytest.raises(InvalidPolicyUpdate, match="execution plans changed"):
        replace(prepared, packs=(prepared.packs[0], ()))
    with pytest.raises(InvalidPolicyUpdate, match="exactly once"):
        replace(prepared, packs=(prepared.packs[0][:1], prepared.packs[1]))
    with pytest.raises(InvalidPolicyUpdate, match="different native evidence"):
        replace(
            prepared,
            updates=(replace(prepared.updates[0], population=replace(snapshot, id="other")), *prepared.updates[1:]),
        )


def test_native_capo_process_credit_stays_on_original_positions_before_minibatching():
    evidence = tuple(
        RewardEvidence(
            "a",
            f"occurrence-a-{index}",
            f"trace-a-{index}",
            "7",
            "fixture@1",
            (RewardValue("outcome", "valid", float(index == 0)),),
            ProcessCredit("valid", "assistant-turns@1", f"critique:{index}", ((0, 1),)),
        )
        for index in range(2)
    )
    rollouts = tuple(
        replace(native_rollout("a", index, float(index == 0)), reward_evidence=evidence[index]) for index in range(2)
    )
    selected = CAPOSettings(id="native-capo", loop=settings().loop, policy_updates=settings().policy_updates)
    result = resolve_rollout_population(
        rollouts,
        selected,
        replace(capabilities(), definition_ids=("capo@1",)),
        population_id="admitted",
        native_evidence_ref="native:receipts",
        native_evidence_digest="native-digest",
        template_revision="native-template@1",
        versions=PolicyVersions("sampler@3", "old@3", "current@3", None),
        sampler_step=3,
        selector_digest="original-actions@1",
    )
    expected = compute_capo_advantages(evidence, [row.env_mask for row in rollouts], group_size=2)
    assert result.credit.advantages.tolist() == pytest.approx(
        [
            value
            for rollout, row in zip(rollouts, expected.token_advantages, strict=True)
            for value, eligible in zip(row, rollout.env_mask, strict=True)
            if eligible
        ]
    )
    assert len(result.updates) == 2
    assert result.credit.observation_scope == "full-trajectory"
    with pytest.raises(InvalidPolicyUpdate, match="nontruncated"):
        resolve_rollout_population(
            (replace(rollouts[0], is_truncated=True), rollouts[1]),
            selected,
            replace(capabilities(), definition_ids=("capo@1",)),
            population_id="admitted",
            native_evidence_ref="native:receipts",
            native_evidence_digest="native-digest",
            template_revision="native-template@1",
            versions=PolicyVersions("sampler@3", "old@3", "current@3", None),
            sampler_step=3,
            selector_digest="original-actions@1",
        )
