"""Critique votes resolve only against retained original token coordinates."""

import pytest
from posttrain.train.critique import CritiqueStep, CritiqueVote, resolve_critique
from posttrain.train.reward_evidence import InvalidRewardEvidence


def resolve(votes, **options):
    return resolve_critique(
        (CritiqueStep("step", ((0, 2), (3, 4))),),
        tuple(votes),
        (True, True, False, True),
        projection_id="steps@1",
        evidence_ref="panel/1",
        expected_votes=2,
        **options,
    )


@pytest.mark.parametrize(
    "rule,expected",
    [
        ("intersection", ()),
        ("strict_majority", ()),
        ("at_least_half", ((0, 2), (3, 4))),
    ],
)
def test_even_vote_tie_is_explicit(rule, expected):
    votes = [CritiqueVote("a", "valid", "judge/a", ("step", "step")), CritiqueVote("b", "valid", "judge/b")]
    assert resolve(votes, rule=rule).error_spans == expected


@pytest.mark.parametrize(
    "second",
    [
        CritiqueVote("a", "valid", "judge/b"),
        CritiqueVote("b", "valid", "judge/a"),
        CritiqueVote("b", "failed", "judge/b"),
        CritiqueVote("b", "valid", "judge/b", ("unknown",)),
    ],
)
def test_unresolved_or_duplicated_evidence_is_not_valid_zero(second):
    with pytest.raises(InvalidRewardEvidence):
        resolve([CritiqueVote("a", "valid", "judge/a"), second])


def test_all_error_votes_preserve_observation_holes():
    result = resolve(
        [CritiqueVote("a", "valid", "judge/a", ("step",)), CritiqueVote("b", "valid", "judge/b", ("step",))]
    )
    assert result.error_mask((True, True, False, True)) == (True, True, False, True)


def test_partial_panel_fails_closed():
    with pytest.raises(InvalidRewardEvidence, match="complete"):
        resolve([CritiqueVote("a", "valid", "judge/a")])
