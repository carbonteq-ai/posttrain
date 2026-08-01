from __future__ import annotations

from dataclasses import replace

import pytest
from posttrain_lab.qualification.scenarios import (
    SCENARIOS,
    QualificationAcceptance,
    QualificationScenario,
    scenario_by_id,
)


def test_scenarios_round_trip_as_data_only_manifests() -> None:
    for scenario in SCENARIOS.values():
        assert QualificationScenario.from_manifest(scenario.to_manifest()) == scenario


def test_online_scenario_requires_environment_and_rollout_inference() -> None:
    scenario = scenario_by_id("automationbench-qwen35-08b-grpo-10")

    with pytest.raises(ValueError, match="requires one environment"):
        replace(scenario, environment_ref=None, dataset_ref="datasets/example@1")
    with pytest.raises(ValueError, match="requires rollout inference"):
        replace(scenario, inference_ref=None)


def test_algorithm_qualification_cannot_shrink_below_ten_updates() -> None:
    scenario = scenario_by_id("automationbench-qwen35-08b-grpo-10")

    with pytest.raises(ValueError, match="below the acceptance minimum"):
        replace(scenario, update_budget=9)
    with pytest.raises(ValueError, match="at least ten"):
        replace(
            scenario,
            update_budget=10,
            acceptance=replace(
                scenario.acceptance,
                minimum_optimizer_updates=9,
            ),
        )


def test_evaluation_scenario_rejects_optimizer_budget() -> None:
    acceptance = QualificationAcceptance(
        minimum_optimizer_updates=0,
        minimum_complete_traces=16,
        require_reward_variance=False,
        require_nonzero_gradient=False,
        require_model_artifact=False,
    )

    with pytest.raises(ValueError, match="cannot select optimizer"):
        QualificationScenario(
            id="gsm8k-domain-eval",
            revision="1",
            job_kind="eval.domain",
            model_ref="models/qwen3.5-0.8b@bf16",
            environment_ref="gsm8k-test",
            dataset_ref=None,
            training_ref="evaluation/gsm8k@1",
            inference_ref="inference/qwen3.5-0.8b-vllm@1",
            update_budget=10,
            task_budget=16,
            rollouts_per_task=1,
            maximum_duration_seconds=3_600,
            acceptance=acceptance,
        )


def test_unknown_scenario_lists_valid_choices() -> None:
    with pytest.raises(ValueError, match="automationbench-qwen35-08b-grpo-10"):
        scenario_by_id("missing")
