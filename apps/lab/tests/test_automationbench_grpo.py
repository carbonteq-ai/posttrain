"""AutomationBench environment policy tests (no parallel training bridges)."""

from __future__ import annotations

from dataclasses import replace

import pytest
from posttrain_lab.catalog import AUTOMATIONBENCH_ZAPIER_GRPO
from posttrain_lab.environments import automationbench_grpo
from pydantic import ValidationError


def test_training_parameters_accept_catalog_controls() -> None:
    parameters = automationbench_grpo.AutomationBenchTrainingParameters.model_validate(
        dict(AUTOMATIONBENCH_ZAPIER_GRPO.parameters)
    )

    assert parameters.domains == ("simple",)
    assert parameters.sampling_seed == 17
    assert parameters.toolset == "zapier"
    assert parameters.search_top_k == 20


def test_environment_rejects_exact_task_ids_as_an_unknown_authoring_control() -> None:
    environment = replace(
        AUTOMATIONBENCH_ZAPIER_GRPO,
        parameters={**AUTOMATIONBENCH_ZAPIER_GRPO.parameters, "task_indices": [194, 198]},
    )

    with pytest.raises(ValidationError, match="task_indices"):
        automationbench_grpo.AutomationBenchTrainingParameters.model_validate(dict(environment.parameters))
