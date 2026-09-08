"""Admission must distinguish missing evidence from valid zero rewards."""

from dataclasses import asdict

import pytest
from posttrain.train.reward_evidence import InvalidRewardEvidence, ProcessCredit, RewardEvidence, RewardValue
from pydantic import TypeAdapter


def test_reward_evidence_serialization_preserves_status_and_identity():
    value = RewardEvidence(
        "batch/occurrence",
        "response/0",
        "trace/0",
        "branch/0",
        "projection@1",
        (
            RewardValue("outcome", "valid", 0.0),
            RewardValue("quality", "abstained"),
        ),
        ProcessCredit("valid", "alignment@1", "trace-evidence/sha256:abc", ((0, 2),)),
    )
    adapter = TypeAdapter(RewardEvidence)
    restored = adapter.validate_json(adapter.dump_json(value))
    assert asdict(restored) == asdict(value)
    assert restored.require_components(("outcome",)) == (0.0,)
    with pytest.raises(InvalidRewardEvidence, match="unavailable"):
        restored.require_components(("quality",))


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), True])
def test_valid_status_requires_a_real_finite_score(value):
    with pytest.raises(InvalidRewardEvidence):
        RewardValue("quality", "valid", value)


def test_unavailable_score_cannot_be_silently_zero():
    with pytest.raises(InvalidRewardEvidence):
        RewardValue("quality", "failed", 0.0)


def test_error_spans_form_a_union_over_original_tokens():
    credit = ProcessCredit("valid", "align@1", "evidence/1", ((0, 2), (1, 3), (0, 2), (4, 5)))
    assert credit.error_mask((True, True, True, False, True)) == (True, True, True, False, True)


@pytest.mark.parametrize("spans", [((2, 4),), ((0, 7),)])
def test_error_spans_cannot_cross_tools_or_completion_boundary(spans):
    credit = ProcessCredit("valid", "align@1", "evidence/1", spans)
    with pytest.raises(InvalidRewardEvidence):
        credit.error_mask((True, True, True, False, True))


def test_failed_critique_is_not_an_empty_error_set():
    with pytest.raises(InvalidRewardEvidence, match="resolved"):
        ProcessCredit("failed", "align@1", "evidence/1").error_mask((True,))
