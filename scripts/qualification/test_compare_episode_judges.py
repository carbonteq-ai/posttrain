"""The replay harness preserves explicit lifecycle and task-owned judge config."""

import pytest
from compare_episode_judges import LAB_CATALOG, judge_config, service_request
from posttrain.catalog import open_catalog
from posttrain.common import CatalogRef, HostedInferenceBinding, InferenceBinding
from posttrain.environment import EnvironmentBinding
from posttrain.jobs import ExternalInferenceServiceRequest, ManagedInferenceService


def test_hosted_binding_maps_to_external_request_with_explicit_model_and_provider():
    catalog = open_catalog(scope="posttrain-lab", overlays=(LAB_CATALOG,))
    binding = catalog.resolve(
        CatalogRef("hosted-inference", "hosted-inference/deepseek-v4-flash-openrouter-judge@1")
    ).value

    assert isinstance(binding, HostedInferenceBinding)
    request = service_request(binding, port=8123)
    assert isinstance(request, ExternalInferenceServiceRequest)
    assert request.binding.model.model == "deepseek/deepseek-v4-flash-0731"
    assert request.binding.provider == "open-inference/fp8"


def test_managed_binding_maps_to_owned_service_request():
    catalog = open_catalog(scope="posttrain-lab", overlays=(LAB_CATALOG,))
    binding = catalog.resolve(CatalogRef("inference", "inference/gemma4-12b-vllm-automationbench-judge-mtp2@1")).value

    assert isinstance(binding, InferenceBinding)
    request = service_request(binding, port=9123)
    assert isinstance(request, ManagedInferenceService)
    assert request.request.port == 9123
    assert request.request.inference is binding


def test_environment_remains_authority_for_judge_rubric_and_sampling():
    pytest.importorskip("automationbench_v1")
    catalog = open_catalog(scope="posttrain-lab", overlays=(LAB_CATALOG,))
    environment = catalog.resolve(
        CatalogRef("environment", "automationbench-lfm26-train-mix-episode-openrouter-v1")
    ).value

    assert isinstance(environment, EnvironmentBinding)
    config = judge_config(environment, "quality")
    assert config.rubric.startswith(
        "You are an exacting, domain-general evaluator of one agent episode."
    )
    assert config.sampling.max_tokens == 16_384
