"""Golden contract tests for backend-neutral GRPO observations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from posttrain.train.grpo_observations import (
    GRPOObservationFeatures,
    assess_grpo_evidence,
    normalize_grpo_metrics,
    required_grpo_metrics,
)

_FIXTURES = Path(__file__).parent / "fixtures" / "grpo_metrics"


def _fixture(name: str) -> dict[str, Any]:
    payload = json.loads((_FIXTURES / name).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _full_features() -> GRPOObservationFeatures:
    return GRPOObservationFeatures(
        reference_kl_enabled=True,
        decoupled_rollout=True,
        mtp_rollout_enabled=True,
    )


def test_trl_and_verl_golden_records_normalize_to_the_same_logical_step() -> None:
    trl = _fixture("trl_step.json")
    verl = _fixture("verl_step.json")
    features = _full_features()

    trl_step = normalize_grpo_metrics(
        backend="trl",
        step=trl["step"],
        native=trl["native"],
        features=features,
    )
    verl_step = normalize_grpo_metrics(
        backend="verl",
        step=verl["step"],
        native=verl["native"],
        features=features,
    )

    assert trl_step.step == verl_step.step == 2
    assert trl_step.metrics == verl_step.metrics
    assert trl_step.attributes == {"training_backend": "trl"}
    assert verl_step.attributes == {"training_backend": "verl"}
    assert assess_grpo_evidence(trl_step.metrics, features).complete


def test_feature_flags_make_selected_runtime_evidence_required() -> None:
    features = GRPOObservationFeatures(
        reference_kl_enabled=True,
        clipping_enabled=True,
        decoupled_rollout=True,
        asynchronous_rollout=True,
        mtp_rollout_enabled=True,
        quantized_kv_cache=True,
        tool_environment=True,
    )

    required = required_grpo_metrics(features)

    assert {
        "train/rl/kl",
        "train/rl/clip_fraction",
        "train/rl/sampling_logp_delta_mean",
        "train/rl/policy_staleness_max",
        "serve/backend/speculative_acceptance_rate",
        "serve/backend/kv_cache_peak_usage_ratio",
        "train/rl/tool_failure_frequency",
    } <= required
    status = assess_grpo_evidence({}, features)
    assert not status.complete
    assert status.missing == required


def test_verl_async_metrics_use_the_shared_staleness_names() -> None:
    payload = _fixture("verl_async_step.json")
    features = GRPOObservationFeatures(asynchronous_rollout=True)

    step = normalize_grpo_metrics(
        backend="verl",
        step=payload["step"],
        native=payload["native"],
        features=features,
    )

    assert step.metrics == {
        "train/rl/policy_staleness_mean": 1.25,
        "train/rl/policy_staleness_max": 3.0,
        "train/rl/trajectory_version_span_mean": 1.5,
    }


def test_trl_kv_cache_runtime_metrics_use_backend_neutral_names() -> None:
    step = normalize_grpo_metrics(
        backend="trl",
        step=3,
        native={
            "rollout/kv_cache_capacity_tokens": 8192,
            "rollout/kv_cache_peak_usage_ratio": 0.625,
        },
        features=GRPOObservationFeatures(quantized_kv_cache=True),
    )

    assert step.metrics == {
        "serve/backend/kv_cache_capacity_tokens": 8192.0,
        "serve/backend/kv_cache_peak_usage_ratio": 0.625,
    }
    status = assess_grpo_evidence(step.metrics, GRPOObservationFeatures(quantized_kv_cache=True))
    assert "serve/backend/kv_cache_capacity_tokens" not in status.missing
    assert "serve/backend/kv_cache_peak_usage_ratio" not in status.missing


def test_trl_trainer_loss_is_the_grpo_policy_loss() -> None:
    step = normalize_grpo_metrics(
        backend="trl",
        step=1,
        native={"loss": 0.125},
        features=GRPOObservationFeatures(),
    )

    assert step.metrics == {"train/rl/policy_loss": 0.125}


def test_trl_dapo_native_dynamic_sampling_and_asymmetric_clip_metrics_are_preserved() -> None:
    step = normalize_grpo_metrics(
        backend="trl",
        step=4,
        native={
            "dynamic_sampling/candidate_batches": 2,
            "dynamic_sampling/retained_fraction": 0.875,
            "clip_ratio/low_mean": 0.0125,
            "clip_ratio/high_mean": 0.03125,
        },
        features=GRPOObservationFeatures(),
    )

    assert step.metrics == {
        "train/rl/dynamic_sampling_candidate_batches": 2.0,
        "train/rl/dynamic_sampling_retained_fraction": 0.875,
        "train/rl/clip_fraction_low": 0.0125,
        "train/rl/clip_fraction_high": 0.03125,
    }


def test_trl_olmo3_active_sampling_metrics_are_preserved() -> None:
    step = normalize_grpo_metrics(
        backend="trl",
        step=5,
        native={
            "active_sampling/generation_rounds": 3,
            "active_sampling/retained_fraction": 0.75,
            "active_sampling/generated_rows": 40,
            "active_sampling/candidate_groups_reserved": 128,
            "active_sampling/candidate_groups_generated": 40,
            "active_sampling/candidate_groups_retained": 32,
            "active_sampling/candidate_groups_unused": 88,
        },
        features=GRPOObservationFeatures(),
    )

    assert step.metrics == {
        "train/rl/active_sampling_generation_rounds": 3.0,
        "train/rl/active_sampling_retained_fraction": 0.75,
        "train/rl/active_sampling_generated_rows": 40.0,
        "train/rl/active_sampling_candidate_groups_reserved": 128.0,
        "train/rl/active_sampling_candidate_groups_generated": 40.0,
        "train/rl/active_sampling_candidate_groups_retained": 32.0,
        "train/rl/active_sampling_candidate_groups_unused": 88.0,
    }


def test_verl_runtime_totals_use_backend_neutral_names() -> None:
    step = normalize_grpo_metrics(
        backend="verl",
        step=1,
        native={
            "rollout/spec_num_verify_steps": 7,
            "rollout/spec_num_draft_tokens": 7,
            "rollout/spec_num_accepted_tokens": 4,
            "rollout/spec_accept_rate": 4 / 7,
            "rollout/spec_accept_length": 1 + 4 / 7,
        },
        features=GRPOObservationFeatures(mtp_rollout_enabled=True),
    )

    assert step.metrics == {
        "serve/backend/speculative_drafts": 7.0,
        "serve/backend/speculative_draft_tokens": 7.0,
        "serve/backend/speculative_accepted_tokens": 4.0,
        "serve/backend/speculative_acceptance_rate": 4 / 7,
        "serve/backend/speculative_accepted_length": 1 + 4 / 7,
    }


def test_verl_rollout_throughput_uses_completion_tokens_only() -> None:
    step = normalize_grpo_metrics(
        backend="verl",
        step=1,
        native={
            "perf/total_num_tokens": 772,
            "response_length/total": 586,
            "timing_s/gen": 18.25,
        },
        features=GRPOObservationFeatures(),
    )

    assert step.metrics["train/num_tokens"] == 772
    assert step.metrics["train/rl/completion_tokens_total"] == 586
    assert step.metrics["train/rl/rollout_tokens_per_second"] == pytest.approx(32.1096)


def test_verl_sampo_metrics_use_hierarchical_credit_names() -> None:
    step = normalize_grpo_metrics(
        backend="verl",
        step=2,
        native={
            "sampo/episode_advantage_mean": 0.0,
            "sampo/turn_advantage_mean": 0.125,
            "sampo/anchor_group_size_mean": 4,
            "sampo/sparse_reward_projection_fraction": 1.0,
        },
        features=GRPOObservationFeatures(),
    )

    assert step.metrics == {
        "train/rl/episode_advantage_mean": 0.0,
        "train/rl/turn_advantage_mean": 0.125,
        "train/rl/anchor_group_size_mean": 4.0,
        "train/rl/sparse_reward_projection_fraction": 1.0,
    }


@pytest.mark.parametrize(
    ("native", "message"),
    [
        ({"reward": float("nan")}, "non-finite"),
        ({"clip_ratio": 1.1}, "between zero and one"),
        ({"grad_norm": -0.1}, "cannot be negative"),
        (
            {"clip_ratio": 0.1, "clip_ratio/region_mean": 0.1},
            "both map to",
        ),
    ],
)
def test_invalid_native_metric_records_fail_before_tracking(
    native: dict[str, float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_grpo_metrics(
            backend="trl",
            step=1,
            native=native,
            features=GRPOObservationFeatures(),
        )


def test_partial_mtp_evidence_is_rejected() -> None:
    payload = _fixture("partial_mtp_step.json")

    with pytest.raises(ValueError, match="partial MTP evidence"):
        normalize_grpo_metrics(
            backend="trl",
            step=payload["step"],
            native=payload["native"],
            features=GRPOObservationFeatures(mtp_rollout_enabled=True),
        )


def test_mtp_evidence_is_rejected_when_the_feature_was_not_selected() -> None:
    payload = _fixture("mtp_step.json")

    with pytest.raises(ValueError, match="without MTP selected"):
        normalize_grpo_metrics(
            backend="trl",
            step=payload["step"],
            native=payload["native"],
            features=GRPOObservationFeatures(),
        )


def test_turboquant_fixture_makes_runtime_cache_evidence_required() -> None:
    payload = _fixture("turboquant_features.json")
    features = GRPOObservationFeatures(**payload["features"])

    assert set(payload["expected_required"]) <= required_grpo_metrics(features)


def test_unknown_metrics_are_ignored_instead_of_leaking_backend_vocabulary() -> None:
    step = normalize_grpo_metrics(
        backend="trl",
        step=1,
        native={"reward": 0.5, "some_new_trl_internal": 9.0, "debug": True},
        features=GRPOObservationFeatures(),
    )

    assert step.metrics == {"train/rl/reward_mean": 0.5}


def test_trl_advantage_diagnostics_normalize_to_backend_neutral_names() -> None:
    step = normalize_grpo_metrics(
        backend="trl",
        step=2,
        native={
            "advantages/mean": 0.0,
            "advantages/std": 0.82,
            "advantages/abs_mean": 0.70,
            "advantages/positive_fraction": 0.5,
            "advantages/negative_fraction": 0.5,
            "advantages/zero_fraction": 0.0,
            "advantages/scorable_fraction": 0.96875,
            "advantages/truncated_fraction": 0.03125,
            "group_reward_std/mean": 0.41,
            "sampling/importance_sampling_ratio/clamped_fraction": 0.02,
        },
        features=GRPOObservationFeatures(),
    )

    assert step.metrics == {
        "train/rl/advantage_mean": 0.0,
        "train/rl/advantage_std": 0.82,
        "train/rl/advantage_abs_mean": 0.70,
        "train/rl/advantage_positive_fraction": 0.5,
        "train/rl/advantage_negative_fraction": 0.5,
        "train/rl/advantage_zero_fraction": 0.0,
        "train/rl/advantage_scorable_fraction": 0.96875,
        "train/rl/advantage_truncated_fraction": 0.03125,
        "train/rl/group_reward_std_mean": 0.41,
        "train/rl/importance_sampling_ratio_clamped_fraction": 0.02,
    }


def test_trl_raw_policy_parity_evidence_is_persisted_separately_from_sampling_delta() -> None:
    step = normalize_grpo_metrics(
        backend="trl",
        step=1,
        native={
            "sampling/sampling_logp_difference/mean": 0.17,
            "sampling/sampling_logp_difference/max": 1.2,
            "sampling/policy_parity_logp_difference/mean": 0.003,
            "sampling/policy_parity_logp_difference/max": 0.021,
            "sampling/policy_parity_logp_difference/token_count": 16384,
        },
        features=GRPOObservationFeatures(decoupled_rollout=True),
    )

    assert step.metrics == {
        "train/rl/sampling_logp_delta_mean": 0.17,
        "train/rl/sampling_logp_delta_max": 1.2,
        "train/rl/policy_parity_logp_delta_mean": 0.003,
        "train/rl/policy_parity_logp_delta_max": 0.021,
        "train/rl/policy_parity_token_count": 16384.0,
    }
