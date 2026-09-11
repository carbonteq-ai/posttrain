"""Independent numerical expectations for complete structured-reward batches."""

import math
from dataclasses import replace
from decimal import Decimal, localcontext

import pytest
from posttrain.train.reward_advantages import compute_capo_advantages, compute_gdpo_advantages
from posttrain.train.reward_evidence import InvalidRewardEvidence, ProcessCredit, RewardEvidence, RewardValue


def _evidence(index, group, values, spans=()):
    return RewardEvidence(
        group,
        f"r{index}",
        f"t{index}",
        "branch",
        "fixture@1",
        tuple(RewardValue(name, "valid", value) for name, value in values.items()),
        ProcessCredit("valid", "align@1", f"derived/{index}", spans),
    )


def _decimal_normalize(values, epsilon):
    # Independent high-precision oracle using explicit sample variance.
    with localcontext() as context:
        context.prec = 60
        values = [Decimal(str(value)) for value in values]
        mean = sum(values, Decimal(0)) / len(values)
        sd = (sum(((value - mean) ** 2 for value in values), Decimal(0)) / (len(values) - 1)).sqrt()
        return [(value - mean) / (sd + Decimal(str(epsilon))) for value in values]


def test_gdpo_conflicting_components_unequal_lengths_and_interleaved_groups():
    evidence = [
        _evidence(i, group, {"a": a, "b": b})
        for i, (group, a, b) in enumerate(
            [
                ("g1", 1, 0),
                ("g2", 4, 1),
                ("g1", 0, 1),
                ("g2", 2, 0),
            ]
        )
    ]
    masks = [(True,), (True, False, True), (True, True, True), (True, True)]
    result = compute_gdpo_advantages(
        evidence, masks, component_names=("a", "b"), component_weights=(1, 2), group_size=2
    )
    expected = [Decimal(0)] * 4
    for indices in ((0, 2), (1, 3)):
        for component, weight in (("a", 1), ("b", 2)):
            column = [evidence[i].require_components((component,))[0] for i in indices]
            for i, value in zip(indices, _decimal_normalize(column, "0.0001"), strict=True):
                expected[i] += weight * value
    expected = _decimal_normalize(expected, "0.0001")
    for row, mask, value in zip(result.token_advantages, masks, expected, strict=True):
        assert row == pytest.approx([float(value) if sampled else 0 for sampled in mask], abs=1e-12)
    # Same scalar sum, distinct component-weighted signal in g1.
    assert result.token_advantages[0][0] < 0 < result.token_advantages[2][0]


def test_capo_group_population_excludes_tools_and_uses_all_response_tokens():
    evidence = [_evidence(0, "g", {"outcome": 1}, ((1, 2),)), _evidence(1, "g", {"outcome": 0}, ((0, 1),))]
    masks = [(True, True, False, True), (True,)]
    result = compute_capo_advantages(evidence, masks, group_size=2)
    assert result.raw_rewards == ((2, 1, 0, 2), (-1,))
    expected = [float(value) for value in _decimal_normalize([2, 1, 2, -1], "0.000001")]
    assert result.token_advantages[0] == pytest.approx([expected[0], expected[1], 0, expected[2]])
    assert result.token_advantages[1] == pytest.approx([expected[3]])


def test_gdpo_preserves_small_variation_on_large_offsets():
    evidence = [_evidence(i, "g", {"a": 1e12 + i}) for i in range(3)]
    result = compute_gdpo_advantages(
        evidence, [(True,)] * 3, component_names=("a",), component_weights=(1,), group_size=3
    )
    assert result.token_advantages[1] == (0.0,)
    assert result.token_advantages[0][0] == pytest.approx(-result.token_advantages[2][0], abs=1e-15)


@pytest.mark.parametrize("algorithm", ["gdpo", "capo"])
def test_constant_groups_remain_finite_zero(algorithm):
    evidence = [_evidence(i, "g", {"outcome": 1}) for i in range(2)]
    masks = [(True,), (True, False, True)]
    if algorithm == "gdpo":
        result = compute_gdpo_advantages(
            evidence, masks, component_names=("outcome",), component_weights=(1,), group_size=2
        )
    else:
        result = compute_capo_advantages(evidence, masks, group_size=2)
    assert all(value == 0 and math.isfinite(value) for row in result.token_advantages for value in row)


@pytest.mark.parametrize("defect", ["partial", "duplicate", "duplicate_trace", "missing"])
def test_group_admission_rejects_corrupt_population(defect):
    evidence = [_evidence(i, "g", {"outcome": i}) for i in range(2)]
    if defect == "partial":
        evidence[1] = replace(evidence[1], prompt_group_id="another")
    elif defect == "duplicate":
        evidence[1] = replace(evidence[1], rollout_id=evidence[0].rollout_id)
    elif defect == "duplicate_trace":
        evidence[1] = replace(evidence[1], trace_id=evidence[0].trace_id)
    else:
        evidence[1] = replace(evidence[1], components=(RewardValue("outcome", "failed"),))
    with pytest.raises(InvalidRewardEvidence):
        compute_capo_advantages(evidence, [(True,)] * 2, group_size=2)


def test_input_permutation_only_permutes_output():
    evidence = [_evidence(i, f"g{i // 2}", {"outcome": i % 2}) for i in range(4)]
    masks = [(True,) * (i + 1) for i in range(4)]
    original = compute_capo_advantages(evidence, masks, group_size=2)
    permutation = [3, 0, 2, 1]
    shuffled = compute_capo_advantages(
        [evidence[i] for i in permutation], [masks[i] for i in permutation], group_size=2
    )
    for row, index in zip(shuffled.token_advantages, permutation, strict=True):
        assert row == pytest.approx(original.token_advantages[index], abs=1e-12)
