"""The review runs the rules and the calculator on a recorded snapshot."""

from typing import Any

from posttrain.advisor import Architecture, review

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


def _load(repo_id: str, revision: str | None) -> Architecture:
    assert (repo_id, revision) == ("LiquidAI/LFM2.5-2.6B", "abc")
    return LFM


def _snapshot(snap, engine, **settings):  # noqa: ANN001, ANN003, ANN202
    return {
        "model": snap.model(),
        "training": snap.training(),
        "settings": snap.settings(**settings),
        "environment": snap.environment(),
        "rollout_inference": snap.rollout(engine),
        "execution_targets": snap.targets(),
    }


def test_rules_need_no_model_config(snap) -> None:
    result = review(_snapshot(snap, {**snap.TUNED, "enforce_eager": True, "request_mode": "async"}))
    assert [issue.code for issue in result.findings] == ["VLLM_EAGER_DISABLES_CUDA_GRAPHS"]
    assert result.recommendations == {}


def test_calculator_recommends_engine_settings_and_step_options(snap) -> None:
    engine = {**snap.TUNED, "max_num_seqs": 40, "max_num_batched_tokens": 16_384, "gpu_memory_utilization": 0.14}
    result = review(_snapshot(snap, engine), _load)
    recommendation: Any = result.recommendations["rollout_inference"]
    rows = {row["key"]: row for row in recommendation["settings"]}
    assert rows["max_num_seqs"]["suggested"] == 40 and rows["max_num_seqs"]["changed"] is False
    step = recommendation["step"]
    assert step["step_sequences"] == 40 and step["waves"] == 1
    labels = [option["label"] for option in step["options"]]
    assert labels[:2] == ["current", "recommended"]
    recommended = step["options"][1]
    # Oversampling takes at most 20% of the recommended step; the rest is prompts.
    assert recommended["oversample_groups"] * 4 <= 0.2 * recommended["rows"]
    assert recommended["relative_step_time"] > 1 and recommended["relative_rows_per_second"] > 1
    codes = {issue.code for issue in result.findings}
    assert "CALCULATOR_STEP_BELOW_USEFUL_CONCURRENCY" in codes
    step_issue = next(issue for issue in result.findings if issue.code == "CALCULATOR_STEP_BELOW_USEFUL_CONCURRENCY")
    assert step_issue.path == "settings.num_prompts_per_step"


def test_shortfalls_are_warnings(snap) -> None:
    engine = {**snap.TUNED, "max_num_seqs": 16, "max_num_batched_tokens": 8_192, "kv_cache_memory_bytes": 1 << 30}
    codes = {issue.code: issue.severity for issue in review(_snapshot(snap, engine), _load).findings}
    assert codes["CALCULATOR_MAX_NUM_SEQS_BELOW_CONCURRENCY"] == "warning"
    assert codes["CALCULATOR_BATCHED_TOKENS_BELOW_RECOMMENDED"] == "warning"
    assert codes["CALCULATOR_KV_CACHE_BELOW_CONCURRENCY"] == "warning"


def test_a_step_larger_than_fits_is_collected_in_waves(snap) -> None:
    snapshot = _snapshot(snap, dict(snap.TUNED), prompts=64, prompt=3_072, completion=1_024)
    snapshot["execution_targets"] = snap.targets(memory_gb=8.0, accelerator="RTX3070TI")
    snapshot["rollout_inference"]["resolved"]["engine"] = {**snap.TUNED, "mode": "server", "max_model_len": 4_096}
    result = review(snapshot, _load)
    step: Any = result.recommendations["rollout_inference"]["step"]
    assert step["waves"] > 1
    assert [option["label"] for option in step["options"]] == ["current", "one wave"]
    assert "CALCULATOR_STEP_COLLECTS_IN_WAVES" in {issue.code for issue in result.findings}


def test_missing_inputs_make_the_recommendation_unavailable_not_an_error(snap) -> None:
    snapshot = _snapshot(snap, dict(snap.TUNED))
    del snapshot["execution_targets"]
    result = review(snapshot, _load)
    unavailable: Any = result.recommendations["rollout_inference"]["unavailable"]
    assert "not recorded" in unavailable

    def offline(repo_id: str, revision: str | None) -> Architecture:
        raise OSError("offline")

    result = review(_snapshot(snap, dict(snap.TUNED)), offline)
    assert result.recommendations["rollout_inference"] == {"unavailable": "OSError: offline"}
