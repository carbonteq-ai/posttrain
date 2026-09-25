"""LoRA RL snap.settings are compared with the Tinker / LoRA Without Regret recipes."""

from typing import Any

from posttrain.advisor import training_findings


def _codes(snapshot: dict[str, Any]) -> dict[str, str]:
    return {issue.code: issue.severity for issue in training_findings(snapshot)}


def test_vortex_v1_settings_warn_about_rate_and_schedule(snap) -> None:
    codes = _codes({"settings": snap.settings(1e-5, "linear"), "training": snap.training()})
    assert codes == {
        "LORA_RL_LEARNING_RATE_BELOW_REFERENCE": "warning",
        "LR_SCHEDULE_DECAYS_WITHIN_SHORT_RUN": "warning",
    }


def test_findings_point_at_recorded_fields(snap) -> None:
    issue = next(iter(training_findings({"settings": snap.settings(1e-5), "training": snap.training()})))
    assert issue.path == "settings.learning_rate"
    assert issue.related_paths == ("training.parameter_update.alpha",)


def test_reference_range_scales_with_alpha(snap) -> None:
    assert _codes({"settings": snap.settings(2e-4), "training": snap.training()}) == {}
    assert _codes({"settings": snap.settings(4e-5), "training": snap.training(alpha=32)}) == {}
    assert "LORA_RL_LEARNING_RATE_BELOW_REFERENCE" in _codes(
        {"settings": snap.settings(5e-6), "training": snap.training(alpha=32)}
    )
    assert _codes({"settings": snap.settings(1e-3), "training": snap.training()}) == {
        "LORA_RL_LEARNING_RATE_ABOVE_REFERENCE": "warning"
    }


def test_long_runs_may_decay_and_partial_targets_are_recommendations(snap) -> None:
    assert _codes({"settings": snap.settings(2e-4, "linear", steps=200), "training": snap.training()}) == {}
    assert _codes({"settings": snap.settings(2e-4), "training": snap.training(targets="q_proj,v_proj")}) == {
        "LORA_PARTIAL_TARGET_MODULES": "recommendation"
    }


def test_runs_recorded_before_the_schedule_was_snapshotted_are_not_flagged(snap) -> None:
    assert _codes({"settings": snap.settings(2e-4, None), "training": snap.training()}) == {}


def test_full_parameter_training_is_out_of_scope(snap) -> None:
    assert training_findings({"settings": snap.settings(1e-6), "training": snap.training(kind="full")}) == ()
