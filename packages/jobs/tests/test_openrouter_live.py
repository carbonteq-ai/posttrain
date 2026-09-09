"""Paid, opt-in OpenRouter contract check for the exact default judge route."""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import pytest
from posttrain.catalog import open_catalog
from posttrain.common import CatalogRef, HostedInferenceBinding, NullObserver, RunContext
from posttrain.jobs import ExternalInferenceServiceRequest
from posttrain.jobs.providers.openrouter import OpenRouterResolver


@pytest.mark.network
def test_default_openrouter_judge_model_and_provider_are_live(tmp_path: Path) -> None:
    if not os.environ.get("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY is unavailable; paid OpenRouter probe was not attempted")
    binding = cast(
        HostedInferenceBinding,
        open_catalog(scope="openrouter-live")
        .resolve(CatalogRef("hosted-inference", "hosted-inference/deepseek-v4-flash-openrouter-judge@1"))
        .value,
    )
    context = RunContext(
        project_id="openrouter-live",
        work_package_id="qualify/openrouter",
        run_id="openrouter-live-contract",
        job_kind="qualify.judge",
        job_definition_version="qualify/openrouter-live@1",
        workspace=tmp_path,
        observer=NullObserver(),
    )

    with OpenRouterResolver(timeout_seconds=60)(
        context,
        "judge/quality",
        ExternalInferenceServiceRequest(binding),
    ) as resolved:
        identity = resolved.trace_identity()
        assert binding.model.model == "deepseek/deepseek-v4-flash-0731"
        assert binding.provider == "open-inference/fp8"
        assert resolved.provider["requested_provider"] == binding.provider
        assert resolved.provider["endpoint_tag"] == binding.provider
        assert resolved.provider["route"] == {
            "order": [binding.provider],
            "allow_fallbacks": False,
            "require_parameters": True,
            "zdr": False,
            "data_collection": "allow",
        }
        assert resolved.provider["capability_probe"]
        assert "artifact_digest" not in identity
        assert os.environ["OPENROUTER_API_KEY"] not in str(identity)
