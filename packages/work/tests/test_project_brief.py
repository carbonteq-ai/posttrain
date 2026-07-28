from __future__ import annotations

from pathlib import Path

import pytest
from posttrain.common import ContractError
from posttrain.work import (
    ProjectBrief,
    ServingRequirements,
    load_project_brief,
    project_brief_digest,
    project_brief_snapshot,
)


def test_project_brief_loads_strict_serving_requirements_and_stable_digest(tmp_path: Path) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text(
        """
schema_version: 1
objective: Select a model that fits the serving envelope.
serving:
  required_context_tokens: 32768
  min_sustained_output_tokens_per_second: 50
  max_p95_ttft_ms: 1000
  max_p95_tpot_ms: 30
  max_failure_rate: 0.01
""".strip()
        + "\n",
        encoding="utf-8",
    )
    second.write_text(
        """
serving:
  max_failure_rate: 0.01
  max_p95_tpot_ms: 30
  max_p95_ttft_ms: 1000
  min_sustained_output_tokens_per_second: 50
  required_context_tokens: 32768
objective: Select a model that fits the serving envelope.
schema_version: 1
""".strip()
        + "\n",
        encoding="utf-8",
    )

    first_brief = load_project_brief(first)
    second_brief = load_project_brief(second)

    assert first_brief == second_brief
    assert first_brief.serving is not None
    assert first_brief.serving.required_context_tokens == 32_768
    assert project_brief_digest(first_brief) == project_brief_digest(second_brief)
    snapshot = project_brief_snapshot(first_brief)
    assert snapshot["digest"] == project_brief_digest(first_brief)
    assert snapshot["serving"]["min_sustained_output_tokens_per_second"] == 50.0  # type: ignore[index]


@pytest.mark.parametrize(
    "field,value",
    (
        ("required_context_tokens", 0),
        ("min_sustained_output_tokens_per_second", 0),
        ("max_p95_ttft_ms", 0),
        ("max_p95_tpot_ms", 0),
        ("max_failure_rate", 1.01),
    ),
)
def test_serving_requirements_reject_invalid_thresholds(field: str, value: float) -> None:
    payload = {
        "required_context_tokens": 32_768,
        "min_sustained_output_tokens_per_second": 50,
        "max_p95_ttft_ms": 1000,
        "max_p95_tpot_ms": 30,
        "max_failure_rate": 0.01,
        field: value,
    }

    with pytest.raises(ValueError):
        ServingRequirements.model_validate(payload)


def test_project_brief_rejects_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "project.yaml"
    path.write_text(
        "schema_version: 1\nobjective: Test strict loading.\nunknown: true\n",
        encoding="utf-8",
    )

    with pytest.raises(ContractError, match="invalid post-training project brief"):
        load_project_brief(path)


def test_project_brief_without_serving_requirements_is_valid() -> None:
    brief = ProjectBrief(objective="Train a domain model.")

    assert brief.serving is None
