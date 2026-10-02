from dataclasses import replace

import pytest
from posttrain.train.profiles import CAPOSettings, GDPOSettings
from posttrain.train.reward_advantages import compute_capo_advantages, compute_gdpo_advantages
from posttrain.train.reward_evidence import ProcessCredit, RewardEvidence, RewardValue
from posttrain.train.update_credit import (
    ActionCredit,
    NativeCreditRows,
    PreparedCredit,
    SampoCreditEstimator,
    StructuredCreditEstimator,
    prepare_credit,
)
from posttrain.train.update_records import (
    ActionRecord,
    ActionRef,
    ConditioningView,
    InvalidPolicyUpdate,
    PolicyVersions,
    PopulationRelation,
    PopulationSnapshot,
)

from .test_sampo import _rollout, _settings
from .test_update_records import population


class ExternalEstimator:
    """Declared external estimator fixture, not a recommended PRM recipe."""

    id = "external-fixture@1"
    required_relations = ("prompt",)

    def prepare(self, snapshot: PopulationSnapshot) -> PreparedCredit:
        return PreparedCredit(
            snapshot.digest,
            self.id,
            tuple(
                ActionCredit(record.action, (-1.0, 0.0, 1.0)[index]) for index, record in enumerate(snapshot.actions)
            ),
            self.required_relations,
            (("step-return", 1.0),),
            "externally-prepared@1",
            "prefix",
            ("retained-step-assessment-digest",),
        )


def test_external_estimator_preserves_detached_credit_and_identity() -> None:
    snapshot = population()
    credit = prepare_credit(snapshot, ExternalEstimator())
    assert tuple(value.advantage for value in credit.values) == (-1.0, 0.0, 1.0)
    assert credit.digest == prepare_credit(snapshot, ExternalEstimator()).digest


def test_partial_population_rejected_before_estimator_is_called() -> None:
    snapshot = population()
    partial = replace(snapshot.relations[0], completeness="partial", members=snapshot.relations[0].members[:1])
    snapshot = replace(snapshot, relations=(partial, snapshot.relations[1]))
    with pytest.raises(InvalidPolicyUpdate, match="complete population"):
        prepare_credit(snapshot, ExternalEstimator())


def test_credit_must_cover_original_actions_and_frozen_evidence() -> None:
    snapshot = population()
    credit = prepare_credit(snapshot, ExternalEstimator())
    with pytest.raises(InvalidPolicyUpdate, match="cover eligible"):
        replace(credit, values=credit.values[:1]).validate(snapshot)
    with pytest.raises(InvalidPolicyUpdate, match="different frozen"):
        credit.validate(replace(snapshot, native_evidence_digest="different"))


@pytest.mark.parametrize("advantage", [float("nan"), float("inf"), True])
def test_credit_rejects_nonfinite_and_boolean_values(advantage: float) -> None:
    with pytest.raises(InvalidPolicyUpdate, match="finite numeric"):
        ActionCredit(population().actions[0].action, advantage)


def test_scorer_quality_cannot_be_relabelled_as_advantage() -> None:
    credit = prepare_credit(population(), ExternalEstimator())
    with pytest.raises(InvalidPolicyUpdate, match="quality scores"):
        replace(credit, meaning="quality")  # type: ignore[arg-type]


def native_rows() -> tuple[PopulationSnapshot, NativeCreditRows]:
    rollouts = tuple(
        replace(
            _rollout(float(index == 0), str(index)),
            reward_evidence=RewardEvidence(
                "prompt",
                f"episode-{index}",
                f"trace-{index}",
                "branch",
                "fixture@1",
                (RewardValue("outcome", "valid", float(index == 0)),),
                ProcessCredit("valid", "assistant-turns@1", f"critique:{index}", ((0, 1),)),
            ),
        )
        for index in range(2)
    )
    coordinates = tuple(
        tuple(
            ActionRef(f"episode-{index}", "branch", "first" if position < 2 else "second", position)
            if eligible
            else None
            for position, eligible in enumerate(rollout.env_mask)
        )
        for index, rollout in enumerate(rollouts)
    )
    actions = tuple(action for row in coordinates for action in row if action is not None)
    snapshot = PopulationSnapshot(
        "native-fixture",
        "native:fixture",
        "native-fixture-digest",
        tuple(ActionRecord(action, "context", "native:node") for action in actions),
        (ConditioningView("context", "native:fixture", "tokens", "attention", "positions", "template@1", "digest", 8),),
        (),
        (PopulationRelation("prompt", "prompt-group", actions, "complete", actions),),
        PolicyVersions("sampler@1", "old@1", "current@1", None),
        "selector@1",
    )
    return snapshot, NativeCreditRows(rollouts, coordinates, ("native-fixture-digest",))


def test_sampo_adapter_preserves_existing_sparse_terminal_credit() -> None:
    snapshot, rows = native_rows()
    credit = prepare_credit(snapshot, SampoCreditEstimator(_settings(), rows, ("prompt",)))
    assert [value.advantage for value in credit.values] == pytest.approx(
        [0.975, 0.975, 1.0, 1.0, -0.975, -0.975, -1.0, -1.0]
    )
    assert credit.normalization == "sampo-mean@1"


def test_sampo_native_adapter_applies_truncation_recipe_before_centering():
    snapshot, rows = native_rows()
    # Equal raw rewards must become unequal algorithm rewards when one native
    # rollout is truncated. Previously this adapter returned all-zero credit.
    rows = replace(
        rows, rollouts=(replace(rows.rollouts[0], reward=1.0), replace(rows.rollouts[1], reward=1.0, is_truncated=True))
    )
    selected = _settings(truncation_penalty=0.5)
    credit = prepare_credit(snapshot, SampoCreditEstimator(selected, rows, ("prompt",)))
    assert [value.advantage for value in credit.values] == pytest.approx(
        [0.4875, 0.4875, 0.5, 0.5, -0.4875, -0.4875, -0.5, -0.5]
    )
    assert rows.rollouts[1].reward == 1.0
    assert credit.estimator_id.startswith("sampo-credit@2:")


@pytest.mark.parametrize("algorithm", ["gdpo", "capo"])
def test_structured_adapter_delegates_without_changing_existing_estimator(algorithm) -> None:
    snapshot, rows = native_rows()
    loop = _settings().loop
    evidence: list[RewardEvidence] = []
    for rollout in rows.rollouts:
        assert rollout.reward_evidence is not None
        evidence.append(rollout.reward_evidence)
    masks = [rollout.env_mask for rollout in rows.rollouts]
    if algorithm == "gdpo":
        settings = GDPOSettings(
            id="fixture-gdpo",
            loop=loop,
            max_prompt_length=2,
            max_completion_length=6,
            component_names=("outcome",),
            component_weights=(1.0,),
        )
        expected = compute_gdpo_advantages(
            evidence, masks, component_names=("outcome",), component_weights=(1.0,), group_size=2
        )
    else:
        settings = CAPOSettings(id="fixture-capo", loop=loop, max_prompt_length=2, max_completion_length=6)
        expected = compute_capo_advantages(evidence, masks, group_size=2)
    credit = prepare_credit(snapshot, StructuredCreditEstimator(settings, rows, ("prompt",)))
    assert [value.advantage for value in credit.values] == pytest.approx(
        [
            value
            for row, mask in zip(expected.token_advantages, masks, strict=True)
            for value, eligible in zip(row, mask, strict=True)
            if eligible
        ]
    )


def test_native_projection_rejects_observation_credit_and_coordinate_loss() -> None:
    _, rows = native_rows()
    with pytest.raises(InvalidPolicyUpdate, match="sampled eligibility"):
        replace(rows, actions=(rows.actions[0][:-1], rows.actions[1]))
    with pytest.raises(InvalidPolicyUpdate, match="ineligible native"):
        rows.project(((1.0,) * 6,) * 2)
