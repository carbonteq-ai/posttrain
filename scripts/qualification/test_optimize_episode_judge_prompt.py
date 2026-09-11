import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import optimize_episode_judge_prompt as optimizer
import pytest
from automationbench_v1.episode_prompt import (
    EPISODE_PROMPT_VERSION,
    EPISODE_RUBRICS,
    GENERAL_EPISODE_JUDGE_SYSTEM_PROMPT,
)
from optimize_episode_judge_prompt import (
    calibrate_vocabulary_profile,
    evaluate_report,
    materialize_candidates,
    select_candidate,
    self_author_prompt,
)


def _scores(value: float) -> dict[str, float]:
    return dict.fromkeys(EPISODE_RUBRICS, value)


def _labels() -> dict:
    return {
        "schema_version": 1,
        "cases": {
            "good": {"score_bounds": {"action_quality": {"min": 0.75}}},
            "unsupported": {
                "score_bounds": {
                    "answer_quality": {"max": 0.5},
                }
            },
        },
        "comparisons": [
            {
                "better": "good",
                "worse": "unsupported",
                "dimensions": ["answer_quality"],
                "minimum_margin": 0.25,
            }
        ],
        "gates": {
            "minimum_valid_rate": 1.0,
            "minimum_constraint_pass_rate": 1.0,
            "minimum_pairwise_accuracy": 1.0,
            "maximum_length_rate": 0.0,
            "maximum_false_perfect_count": 0,
        },
    }


def _report() -> dict:
    good = _scores(1.0)
    unsupported = _scores(0.75)
    unsupported["answer_quality"] = 0.5
    return {
        "label": "candidate",
        "cases": [
            {
                "case_id": "good",
                "structured_output_valid": True,
                "finish_reason": "stop",
                "completion_tokens": 400,
                "scores": good,
            },
            {
                "case_id": "unsupported",
                "structured_output_valid": True,
                "finish_reason": "stop",
                "completion_tokens": 600,
                "scores": unsupported,
            },
        ],
    }


def test_materialization_changes_only_the_system_prompt_and_identity() -> None:
    corpus = [
        {
            "case_id": "one",
            "input_digest": "old",
            "messages": [
                {"role": "system", "content": "old rubric"},
                {
                    "role": "user",
                    "content": json.dumps({"rubrics": EPISODE_RUBRICS, "trajectory": []}),
                },
            ],
        }
    ]
    candidates = {
        "schema_version": 1,
        "base_prompt_version": EPISODE_PROMPT_VERSION,
        "candidates": [
            {
                "id": "baseline",
                "system_prompt": GENERAL_EPISODE_JUDGE_SYSTEM_PROMPT,
                "rubrics": EPISODE_RUBRICS,
            },
            {
                "id": "self-authored",
                "system_prompt": "Use my preferred evaluator wording.",
                "rubrics": {name: f"My wording for {name}." for name in EPISODE_RUBRICS},
            },
        ],
    }

    result = materialize_candidates(corpus, candidates)

    assert set(result) == {"baseline", "self-authored"}
    assert json.loads(result["baseline"][0]["messages"][1]["content"])["trajectory"] == []
    self_authored_request = result["self-authored"][0]["messages"][1]
    assert self_authored_request["role"] == "user"
    assert json.loads(self_authored_request["content"])["rubrics"] != EPISODE_RUBRICS
    assert result["baseline"][0]["input_digest"] != result["self-authored"][0]["input_digest"]
    assert result["baseline"][0]["source_input_digest"] == "old"


def test_self_authoring_elicits_vocabulary_without_exposing_source_prose(monkeypatch) -> None:
    requests = []
    elicitation = {
        "task_explanation": "Judge the complete observed path against the user's requested outcome.",
        "decision_inputs": ["request", "attempt", "observation", "claim"],
        "key_distinctions": ["attempt versus result", "fact versus claim"],
        "dimension_language": {name: f"Natural meaning of {name}" for name in EPISODE_RUBRICS},
        "structured_response_guidance": "Check requirements, then independently rate dimensions.",
    }
    rewritten_rubrics = {
        name: f"Independently judge the observed episode for {name.replace('_', ' ')}." for name in EPISODE_RUBRICS
    }
    rewritten_prompt = (
        "Inspect the complete trajectory as evidence. Reconstruct the requested outcomes, compare "
        "attempts with observations and claims, cite message positions, and independently assess "
        "every fixed quality dimension. Use the supplied discrete anchors and return only the fixed "
        "structured record with concise explanations. "
    ) * 2
    responses = [
        {
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(elicitation)}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        },
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": json.dumps({"system_prompt": rewritten_prompt, "rubrics": rewritten_rubrics})
                    },
                }
            ],
            "usage": {"prompt_tokens": 30, "completion_tokens": 40},
        },
    ]

    def post_json(url, payload, timeout):
        requests.append(payload)
        return responses[len(requests) - 1]

    monkeypatch.setattr(optimizer, "_post_json", post_json)
    result = self_author_prompt(
        elicitation_system_prompt="Explain the problem in your language.",
        synthesis_system_prompt="Now write the generic prompt.",
        base_url="http://judge",
        model="judge/model",
        model_revision="a" * 40,
        runtime_identity="runtime",
        max_tokens=4096,
        timeout=60,
    )

    first_user_message = requests[0]["messages"][1]["content"]
    assert GENERAL_EPISODE_JUDGE_SYSTEM_PROMPT not in first_user_message
    assert "problem_understanding_planning" in first_user_message
    synthesis_input = json.loads(requests[1]["messages"][1]["content"])
    assert synthesis_input["fixed_contract"]["system_prompt"] == GENERAL_EPISODE_JUDGE_SYSTEM_PROMPT
    assert result["generation"]["method"] == "same-model-explain-then-rephrase"
    assert result["candidates"][1]["rubrics"] == rewritten_rubrics


def test_vocabulary_calibration_returns_guidance_without_rewriting_contract(monkeypatch) -> None:
    profile = {
        "task_explanation": (
            "Judge a complete observed agent episode against the original requested outcome and "
            "constraints, keeping the action attempt, environment observation, and final claim separate."
        ),
        "decision_inputs": ["request", "attempt", "observation", "claim"],
        "key_distinctions": [
            "attempt versus observation",
            "claim versus established fact",
            "task result versus independent dimensions",
            "missing evidence versus favorable assumption",
        ],
        "dimension_language": {name: f"Natural language for {name}." for name in EPISODE_RUBRICS},
        "structured_response_guidance": "List material requirements first and independently score every named dimension.",
    }
    requests = []

    def post_json(url, payload, timeout):
        requests.append(payload)
        return {
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(profile)}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        }

    monkeypatch.setattr(optimizer, "_post_json", post_json)
    artifact = calibrate_vocabulary_profile(
        elicitation_system_prompt="Explain the evaluator task in your own language.",
        base_url="http://judge",
        model="judge/model",
        model_revision="a" * 40,
        runtime_identity="runtime",
        max_tokens=2048,
        timeout=60,
    )

    assert len(requests) == 1
    assert artifact["kind"] == "episode-judge-vocabulary-profile"
    assert artifact["profile"] == profile
    assert "system_prompt" not in artifact
    assert "rubrics" not in artifact


def test_evaluation_requires_valid_nonperfect_defect_sensitive_scores() -> None:
    result = evaluate_report(_report(), _labels())

    assert result["passed"] is True
    assert result["metrics"]["valid_rate"] == 1.0
    assert result["metrics"]["constraint_pass_rate"] == 1.0
    assert result["metrics"]["pairwise_accuracy"] == 1.0
    assert result["metrics"]["false_perfect_count"] == 0


def test_false_perfect_candidate_is_rejected() -> None:
    report = copy.deepcopy(_report())
    report["cases"][1]["scores"] = _scores(1.0)

    result = evaluate_report(report, _labels())

    assert result["passed"] is False
    assert result["metrics"]["false_perfect_count"] == 1
    assert any("false_perfect_count" in failure for failure in result["failures"])


def test_missing_labeled_case_is_rejected() -> None:
    report = _report()
    report["cases"].pop()

    with pytest.raises(ValueError, match="missing labeled cases: unsupported"):
        evaluate_report(report, _labels())


def test_source_digest_mismatch_is_rejected() -> None:
    labels = _labels()
    labels["cases"]["good"]["source_input_digest"] = "expected"
    report = _report()
    report["cases"][0]["source_input_digest"] = "different"

    with pytest.raises(ValueError, match="source digest differs for case 'good'"):
        evaluate_report(report, labels)


def test_repetition_gate_is_enforced() -> None:
    labels = _labels()
    labels["gates"]["minimum_repetitions_per_case"] = 2

    result = evaluate_report(_report(), labels)

    assert result["passed"] is False
    assert result["metrics"]["minimum_repetitions_per_case"] == 1


def test_selection_uses_tokens_only_after_all_quality_gates_pass() -> None:
    assert (
        select_candidate(
            {
                "cheap-invalid": {
                    "passed": False,
                    "metrics": {"mean_completion_tokens": 10},
                },
                "valid-large": {
                    "passed": True,
                    "metrics": {"mean_completion_tokens": 600},
                },
                "valid-small": {
                    "passed": True,
                    "metrics": {"mean_completion_tokens": 400},
                },
            }
        )
        == "valid-small"
    )
