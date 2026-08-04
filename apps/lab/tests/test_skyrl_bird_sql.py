from __future__ import annotations

from pathlib import Path

import pytest
from posttrain.common import CatalogRef, InferenceBinding, ModelVariant
from posttrain.environment import EnvironmentBinding, ProjectPathEnvironmentSource
from posttrain.eval import EvaluationPlan
from posttrain.train import GRPOSettings, LoRAUpdate, TrainingBinding
from posttrain_lab.catalog import open_project_catalog
from posttrain_lab.cli import main
from posttrain_lab.gemma4 import GEMMA4_RENDERER_CONTRACT
from posttrain_lab.project import discover_project
from posttrain_lab.work_packages import load_work_package, resolve_work_package

WORKSPACE = Path(__file__).resolve().parents[3]
LAB = WORKSPACE / "apps" / "lab"


def _catalog():
    return open_project_catalog(discover_project(LAB, explicit_root=LAB), scope="posttrain-lab")


def test_lab_registers_pinned_text_only_gemma_composition() -> None:
    catalog = _catalog()
    model = catalog.resolve(CatalogRef("model", "models/gemma4-12b-it@bf16")).value
    environment = catalog.resolve(CatalogRef("environment", "skyrl-bird-sql-train-full")).value
    settings = catalog.resolve(CatalogRef("training", "gemma4-12b/skyrl-bird-grpo-full-v1")).value
    training = catalog.resolve(
        CatalogRef("training", "training/gemma4-12b-trl-skyrl-bird-grpo-lora@1")
    ).value
    inference = catalog.resolve(
        CatalogRef("inference", "inference/gemma4-12b-vllm-skyrl-bird-rollout@1")
    ).value

    assert isinstance(model, ModelVariant)
    assert model.base.repo_id == "google/gemma-4-12B-it"
    assert model.base.revision == "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
    assert model.renderer is GEMMA4_RENDERER_CONTRACT
    assert model.renderer.conversation.reasoning_mode("off").kwargs() == {"enable_thinking": False}
    assert model.renderer.conversation.tool_calls is None
    assert model.capabilities.modalities == ("text", "image", "audio")

    assert isinstance(environment, EnvironmentBinding)
    assert isinstance(environment.source, ProjectPathEnvironmentSource)
    assert environment.source.path == "environments/skyrl_bird_sql_v1"
    assert environment.num_tasks == 2_064
    assert environment.num_rollouts == 16

    assert isinstance(settings, GRPOSettings)
    assert settings.loop.max_steps == 516
    assert settings.num_prompts_per_step == 4
    assert settings.num_generations == 16
    assert settings.loop.per_device_batch_size * settings.loop.gradient_accumulation_steps == 64

    assert isinstance(training, TrainingBinding)
    assert isinstance(training.update, LoRAUpdate)
    assert training.update.rank == 32
    assert training.update.target_modules.startswith(".*[.]language_model[.]")
    assert training.renderer.reasoning_mode == "off"
    assert training.runtime.global_batch_size == 64

    assert isinstance(inference, InferenceBinding)
    assert inference.engine["text_only"] is True
    assert inference.engine["skip_mm_profiling"] is True
    assert inference.engine["weight_sync_mode"] == "lora"


def test_canary_reuses_full_run_geometry_and_only_changes_population_and_steps() -> None:
    catalog = _catalog()
    canary = catalog.resolve(CatalogRef("training", "gemma4-12b/skyrl-bird-grpo-canary-v1")).value
    full = catalog.resolve(CatalogRef("training", "gemma4-12b/skyrl-bird-grpo-full-v1")).value
    canary_environment = catalog.resolve(CatalogRef("environment", "skyrl-bird-sql-train-canary")).value

    assert isinstance(canary, GRPOSettings)
    assert isinstance(full, GRPOSettings)
    assert canary.loop.max_steps == 1
    assert full.loop.max_steps == 516
    assert canary.num_prompts_per_step == full.num_prompts_per_step == 4
    assert canary.num_generations == full.num_generations == 16
    assert canary.loop.gradient_accumulation_steps == full.loop.gradient_accumulation_steps == 64
    assert canary.max_prompt_length == full.max_prompt_length == 24_576
    assert canary.max_completion_length == full.max_completion_length == 8_192
    assert isinstance(canary_environment, EnvironmentBinding)
    assert canary_environment.num_tasks == 4
    assert canary_environment.num_rollouts == 16


@pytest.mark.parametrize(
    "filename,expected_seats",
    [
        (
            "gemma4_skyrl_bird_grpo_canary.yaml",
            {"model", "environment", "settings", "training", "rollout_inference"},
        ),
        (
            "gemma4_skyrl_bird_grpo_full.yaml",
            {"model", "environment", "settings", "training", "rollout_inference"},
        ),
        (
            "gemma4_skyrl_bird_base_eval.yaml",
            {"model", "evaluation_inference", "target", "evaluation_plan", "environment"},
        ),
    ],
)
def test_static_work_packages_resolve_all_declared_seats(filename: str, expected_seats: set[str]) -> None:
    package = load_work_package(LAB / ".posttrain" / "work_packages" / filename)
    resolved = resolve_work_package(_catalog(), package)

    assert set(package.bindings) == expected_seats
    assert expected_seats <= set(resolved.snapshot)


def test_heldout_plan_contains_exact_validation_population() -> None:
    plan = _catalog().resolve(CatalogRef("evaluation", "skyrl-bird-sql-heldout-v1")).value
    assert isinstance(plan, EvaluationPlan)
    environment = plan.environment("skyrl-bird-sql-validation")
    assert environment.num_tasks == 398
    assert environment.num_rollouts == 1
    assert environment.sampling.temperature == 0.0


def test_adapter_eval_requires_explicit_immutable_trackio_version() -> None:
    with pytest.raises(SystemExit, match="explicit --adapter-version"):
        main(
            [
                "gemma4-skyrl-bird-adapter-eval",
                "--project-root",
                str(LAB),
                "--project",
                "skyrl-bird-sql",
            ]
        )
    with pytest.raises(SystemExit, match="immutable Trackio version"):
        main(
            [
                "gemma4-skyrl-bird-adapter-eval",
                "--project-root",
                str(LAB),
                "--project",
                "skyrl-bird-sql",
                "--adapter-version",
                "latest",
            ]
        )
