"""The temporary general-episode fixture has enforceable, non-demonstration gates."""

from pathlib import Path

from automationbench_v1.episode_prompt import EpisodeVerdict
from calibrate_general_episode_prompt import assess_expectations, load_cases

FIXTURE = Path(__file__).parent / "fixtures/general_episode_judge_candidate_v1.json"


def test_fixture_materializes_five_role_separated_cases():
    _, cases = load_cases(FIXTURE)
    assert len(cases) == 5
    assert len({case["input_digest"] for case in cases}) == 5
    assert all([message["role"] for message in case["messages"]] == ["system", "user"] for case in cases)


def test_perfect_scores_fail_the_known_all_day_defect():
    _, cases = load_cases(FIXTURE)
    case = next(case for case in cases if case["id"] == "calendar-all-day-mismatch")
    verdict = EpisodeVerdict.model_validate(
        {
            "requirement_checks": [
                {
                    "requirement": "Create an event",
                    "outcome": "satisfied",
                    "explanation": "The tool ran.",
                    "evidence": ["message-0", "message-2"],
                    "relevant_dimensions": ["action_quality"],
                }
            ],
            "assessments": {
                name: {
                    "status": "valid",
                    "score": 1.0,
                    "reason": "The tool ran.",
                    "evidence": ["message-2"],
                }
                for name in (
                    "problem_understanding_planning",
                    "logical_correctness",
                    "verification_self_correction",
                    "progress_efficiency",
                    "action_quality",
                    "answer_quality",
                )
            },
        }
    )
    failures = assess_expectations(case, verdict)
    assert any("missing violated dimensions" in failure for failure in failures)
    assert any("action_quality" in failure for failure in failures)
