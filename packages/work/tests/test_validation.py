"""Tests for provider-free job validation reports."""

from posttrain.common import SettingOrigin
from posttrain.work import JobValidationReport, ValidationCheck


def test_job_validation_report_is_stable_and_bound_to_resolved_inputs() -> None:
    origins = (SettingOrigin("policy.reasoning_mode", "off", "default", "qwen-renderer@1"),)
    checks = (ValidationCheck("static-configuration", "passed", "static checks passed"),)

    first = JobValidationReport.for_resolved_inputs(
        {"b": 2, "a": {"value": 1}}, origins=origins, checks=checks
    )
    reordered = JobValidationReport.for_resolved_inputs(
        {"a": {"value": 1}, "b": 2}, origins=origins, checks=checks
    )
    changed = JobValidationReport.for_resolved_inputs(
        {"a": {"value": 1}, "b": 3}, origins=origins, checks=checks
    )

    assert first.resolved_input_digest == reordered.resolved_input_digest
    assert first.resolved_input_digest != changed.resolved_input_digest
    assert first.as_dict()["origins"][0]["kind"] == "default"
