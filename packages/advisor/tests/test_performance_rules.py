"""Performance rules flag recorded snap.settings that undo measured optimizations."""

from typing import Any

from posttrain.advisor import performance_findings


def _codes(snapshot: dict[str, Any]) -> dict[str, str]:
    return {issue.code: issue.severity for issue in performance_findings(snapshot)}


def test_tuned_rollout_binding_has_no_findings(snap) -> None:
    assert _codes({"training": snap.training(), "rollout_inference": snap.rollout(snap.TUNED)}) == {}


def test_eager_override_is_an_error_with_measured_cost(snap) -> None:
    (issue,) = performance_findings({"rollout_inference": snap.rollout({**snap.TUNED, "enforce_eager": True})})
    assert issue.code == "VLLM_EAGER_DISABLES_CUDA_GRAPHS"
    assert issue.severity == "error"
    assert issue.path == "rollout_inference.engine.enforce_eager"
    assert "2.32x" in issue.message
    assert "performance_acknowledgements" in (issue.hint or "")


def test_acknowledged_finding_becomes_information_with_its_reason(snap) -> None:
    reason = "CUDA-graph pools must be released during colocated backward on this target"
    binding = snap.rollout(
        {**snap.TUNED, "enforce_eager": True}, acknowledgements={"VLLM_EAGER_DISABLES_CUDA_GRAPHS": reason}
    )
    (issue,) = performance_findings({"rollout_inference": binding})
    assert issue.severity == "info"
    assert reason in issue.message and "inference/lfm-rollout@1" in issue.message


def test_prefix_caching_disabled_fails_only_when_explicit(snap) -> None:
    assert _codes({"rollout_inference": snap.rollout({**snap.TUNED, "enable_prefix_caching": False})}) == {
        "VLLM_PREFIX_CACHING_DISABLED": "error"
    }
    omitted = {key: value for key, value in snap.TUNED.items() if key != "enable_prefix_caching"}
    assert _codes({"rollout_inference": snap.rollout(omitted)}) == {}


def test_draft_free_speculation_fails_only_while_sampling(snap) -> None:
    engine = {**snap.TUNED, "speculative_config": {"method": "ngram_gpu", "num_speculative_tokens": 4}}
    assert _codes({"rollout_inference": snap.rollout(engine)}) == {"SPECULATIVE_DRAFT_FREE_WITH_SAMPLING": "error"}
    greedy = snap.rollout(engine, sampling={"max_tokens": 4_096, "temperature": 0.0})
    assert _codes({"rollout_inference": greedy}) == {}


def test_keys_the_trl_rollout_engine_ignores_are_reported(snap) -> None:
    binding = snap.rollout({**snap.TUNED, "free_cache_engine": True, "load_format": "dummy"})
    (issue,) = performance_findings({"training": snap.training(), "rollout_inference": binding})
    assert issue.code == "TRL_ROLLOUT_ENGINE_KEYS_IGNORED"
    assert "free_cache_engine" in issue.message and "load_format" in issue.message
    # Without a TRL trainer the keys may belong to another backend.
    assert performance_findings({"rollout_inference": binding}) == ()


def test_rollout_context_must_hold_the_episode_budget(snap) -> None:
    binding = snap.rollout(snap.TUNED)  # max_model_len 24,576
    assert _codes({"settings": snap.settings(prompt=20_480), "rollout_inference": binding}) == {}
    assert _codes({"settings": snap.settings(prompt=24_576), "rollout_inference": binding}) == {
        "ROLLOUT_CONTEXT_BELOW_EPISODE_BUDGET": "error"
    }


def test_kv_offload_in_rollouts_is_an_error(snap) -> None:
    binding = snap.rollout({**snap.TUNED, "kv_offloading_size": 32, "kv_offloading_backend": "native"})
    assert _codes({"rollout_inference": binding})["VLLM_KV_OFFLOAD_IN_ROLLOUT"] == "error"


def test_float16_on_a_recorded_bf16_checkpoint_is_an_error_except_for_turboquant(snap) -> None:
    seats = {"model": snap.model()}
    assert _codes({**seats, "rollout_inference": snap.rollout({**snap.TUNED, "dtype": "float16"})}) == {
        "VLLM_FLOAT16_ON_BF16_CHECKPOINT": "error"
    }
    turboquant = {**snap.TUNED, "dtype": "float16", "kv_cache_dtype": "turboquant_k8v4"}
    assert _codes({**seats, "rollout_inference": snap.rollout(turboquant)}) == {}
    # Without the snap.model recorded the precision is unknown, so nothing is claimed.
    assert _codes({"rollout_inference": snap.rollout({**snap.TUNED, "dtype": "float16"})}) == {}


def test_hosted_and_non_generating_bindings_are_out_of_scope(snap) -> None:
    hosted: dict[str, Any] = {"selection_id": "judge", "resolved": {"hosted_model_id": "x", "purpose": ["judge"]}}
    embed = snap.rollout({**snap.TUNED, "enforce_eager": True}, purpose=("judge",))
    assert _codes({"judge": hosted, "embed": embed}) == {}
