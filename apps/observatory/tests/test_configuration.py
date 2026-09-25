"""Observatory reviews each run's recorded selections itself, including old runs."""

import asyncio

from posttrain.advisor import Architecture
from posttrain.tracking import MetricPoint, MetricSeries
from posttrain_observatory.configuration import configuration_review, configuration_review_sync, step_calibration

LFM = Architecture(
    parameters=2_600_000_000,
    hidden_size=2_048,
    layers=30,
    attention_layers=8,
    kv_heads=8,
    head_dim=64,
    recurrent_layers=22,
    max_position_embeddings=32_768,
)
TARGET = "targets/rtx-pro-6000"
# The shape a run recorded before the rules existed: no schedule, no acknowledgements.
SNAPSHOT = {
    "model": {
        "selection_id": "models/lfm2.5-2.6b@bf16",
        "resolved": {"artifact": {"repo_id": "LiquidAI/LFM2.5-2.6B", "revision": "abc"}, "weight_precision": "bf16"},
    },
    "settings": {
        "selection_id": "lfm/vortex",
        "resolved": {
            "max_steps": 20,
            "max_length": 24_576,
            "learning_rate": 1e-5,
            "num_prompts_per_step": 10,
            "num_generations": 4,
            "max_prompt_length": 20_480,
            "max_completion_length": 4_096,
            "active_sampling": {"max_candidate_batches": 10},
        },
    },
    "training": {
        "selection_id": "training/lfm",
        "resolved": {
            "backend": "trl@1.12.0",
            "parameter_update": {"kind": "lora", "rank": 4, "alpha": 8, "target_modules": "all-linear"},
        },
    },
    "rollout_inference": {
        "selection_id": "inference/lfm-rollout",
        "revision": "1",
        "resolved": {
            "model_variant_id": "models/lfm2.5-2.6b@bf16",
            "backend": "vllm@0.29.1.dev3",
            "engine": {"mode": "colocate", "enforce_eager": True, "max_model_len": 24_576, "max_num_seqs": 40},
            "sampling": {"temperature": 0.8, "max_tokens": 4_096},
            "target_id": TARGET,
            "purpose": ["rollout"],
        },
    },
    "execution_targets": {
        "targets": [
            {
                "selection_id": TARGET,
                "memory_gb": 96.0,
                "hardware": {"accelerator_model": "RTXPRO6000", "accelerator_count": 1},
            }
        ]
    },
}


def _series(name: str, values: list[float]) -> MetricSeries:
    return MetricSeries(
        name=name, points=tuple(MetricPoint(value=value, step=index) for index, value in enumerate(values))
    )


def test_rules_review_an_old_run_without_the_calculator() -> None:
    review = configuration_review_sync(SNAPSHOT, None)
    codes = {finding.code: finding for finding in review.findings}
    assert codes["VLLM_EAGER_DISABLES_CUDA_GRAPHS"].severity == "error"
    assert codes["VLLM_EAGER_DISABLES_CUDA_GRAPHS"].value is True
    assert codes["LORA_RL_LEARNING_RATE_BELOW_REFERENCE"].path == "settings.learning_rate"
    assert codes["LORA_RL_LEARNING_RATE_BELOW_REFERENCE"].source == "rule"
    # The schedule was not recorded, so no claim is made about it.
    assert "LR_SCHEDULE_DECAYS_WITHIN_SHORT_RUN" not in codes
    assert review.calculator == "disabled" and review.recommendations == ()


def test_calculator_step_options_are_calibrated_with_measured_step_times() -> None:
    calibration = step_calibration(
        {
            "train/rl/time/rollout_seconds": _series("train/rl/time/rollout_seconds", [150.0, 170.0]),
            "train/step_time_seconds": _series("train/step_time_seconds", [800.0, 900.0]),
        }
    )
    assert calibration is not None and calibration.rollout_seconds == 160.0 and calibration.steps == 2
    review = configuration_review_sync(SNAPSHOT, lambda repo, revision: LFM, calibration)
    (recommendation,) = review.recommendations
    assert recommendation.state == "available" and recommendation.step is not None
    current, recommended = recommendation.step.options[:2]
    assert current.estimated_step_seconds == 850.0 and current.estimated_relative_rows_per_second == 1.0
    assert recommended.estimated_step_seconds is not None and recommended.estimated_step_seconds > 850.0
    # Updates and scoring scale with rows, so calibrated throughput gains less than decode alone.
    assert recommended.estimated_relative_rows_per_second is not None
    assert recommended.estimated_relative_rows_per_second < recommended.relative_rows_per_second
    assert any(finding.source == "calculator" for finding in review.findings)


def test_async_review_uses_metric_series_for_calibration() -> None:
    review = asyncio.run(configuration_review(SNAPSHOT, None, (_series("train/rl/time/rollout_seconds", [100.0]),)))
    assert review.calibration is not None and review.calibration.step_seconds is None


def test_offline_calculator_is_reported_per_seat() -> None:
    def offline(repo: str, revision: str | None) -> Architecture:
        raise OSError("no network")

    (recommendation,) = configuration_review_sync(SNAPSHOT, offline).recommendations
    assert recommendation.state == "unavailable" and "no network" in (recommendation.unavailable_reason or "")


def test_waiting_time_in_a_round_does_not_grow_with_parallel_episodes() -> None:
    # Two rounds per step of 160 s, each decoding ~5.5K tokens per sequence: most of a
    # round is tool calls, prefill and the slowest episode, not bandwidth-bound decoding.
    calibration = step_calibration(
        {
            "train/rl/time/rollout_seconds": _series("train/rl/time/rollout_seconds", [160.0] * 4),
            "train/step_time_seconds": _series("train/step_time_seconds", [600.0, 600.0]),
            "train/rl/completion_tokens_mean": _series("train/rl/completion_tokens_mean", [5_500.0, 5_500.0]),
        }
    )
    assert calibration is not None and calibration.rounds_per_step == 2.0
    (recommendation,) = configuration_review_sync(SNAPSHOT, lambda repo, revision: LFM, calibration).recommendations
    assert recommendation.step is not None
    current, recommended = recommendation.step.options[:2]
    assert current.estimated_rollout_seconds == 320.0 and current.estimated_step_seconds == 600.0
    # Rollout grows far less than the decode-only factor; rows per second improves.
    assert recommended.estimated_rollout_seconds is not None
    assert recommended.estimated_rollout_seconds / 320.0 < recommended.relative_step_time
    assert (recommended.estimated_relative_rows_per_second or 0) > 1.3
