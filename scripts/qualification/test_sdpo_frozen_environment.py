import json
from pathlib import Path

import pytest

from sdpo_frozen_environment import (
    DOMAINS,
    LIMITS,
    TASK_OPTIONS,
    TOOL_ADDITIONS,
    taskset_config,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _names(fixture):
    data = json.loads((FIXTURES / fixture).read_text())
    rows = data.get("tasks") or data.get("task_names")
    return [row if isinstance(row, str) else row.get("name") or row.get("task") for row in rows]


def test_frozen_options_match_plan():
    assert TASK_OPTIONS["toolset"] == "limited_zapier"
    assert "turn_budget" not in TASK_OPTIONS
    assert TASK_OPTIONS["manifest_assessments"] and TASK_OPTIONS["capture_actions"]
    assert LIMITS["max_turns"] == 50


def test_additions_are_scoped_to_the_population():
    config = taskset_config(["finance.sales_tax_remittance", "hr.offboarding_automation"])
    assert config["additional_tools_by_task"] == {
        "finance.sales_tax_remittance": ["gmail_find_email", "gmail_get_email_by_id"]
    }
    assert "additional_tools_by_task" not in taskset_config(["hr.offboarding_automation"])


@pytest.mark.parametrize(
    "fixture",
    ["automationbench_heldout62_v1.json", "automationbench_train_pool_luna07_v1.json"],
)
def test_study_populations_are_accepted(fixture):
    names = _names(fixture)
    config = taskset_config(names)
    assert config["task_names"] == names
    assert set(config.get("additional_tools_by_task", {})) <= set(TOOL_ADDITIONS)


def test_every_addition_targets_a_study_task():
    names = set(_names("automationbench_heldout62_v1.json")) | set(
        _names("automationbench_train_pool_luna07_v1.json")
    )
    assert set(TOOL_ADDITIONS) <= names


def test_rejects_bad_populations():
    with pytest.raises(ValueError):
        taskset_config([])
    with pytest.raises(ValueError):
        taskset_config(["hr.a", "hr.a"])
    with pytest.raises(ValueError):
        taskset_config(["unknown.task"])
    assert "simple" in DOMAINS
