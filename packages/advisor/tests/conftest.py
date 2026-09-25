"""Snapshot builders: the JSON a run records for its resolved selections."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

TARGET_ID = "targets/local-96gb"
TUNED: dict[str, Any] = {
    "mode": "colocate",
    "sleep_during_optimization": True,
    "max_model_len": 24_576,
    "enforce_eager": False,
    "enable_prefix_caching": True,
}


def rollout(
    engine: dict[str, Any],
    *,
    sampling: dict[str, Any] | None = None,
    acknowledgements: dict[str, str] | None = None,
    purpose: tuple[str, ...] = ("rollout",),
    backend: str = "vllm@0.29.1.dev3",
) -> dict[str, Any]:
    resolved: dict[str, Any] = {
        "model_variant_id": "models/lfm2.5-2.6b@bf16",
        "backend": backend,
        "engine": engine,
        "sampling": sampling if sampling is not None else {"max_tokens": 4_096, "temperature": 0.8},
        "target_id": TARGET_ID,
        "purpose": list(purpose),
    }
    if acknowledgements:
        resolved["performance_acknowledgements"] = acknowledgements
    return {"selection_id": "inference/lfm-rollout", "revision": "1", "resolved": resolved}


def model() -> dict[str, Any]:
    return {
        "selection_id": "models/lfm2.5-2.6b@bf16",
        "revision": "1",
        "resolved": {
            "artifact": {"kind": "huggingface", "repo_id": "LiquidAI/LFM2.5-2.6B", "revision": "abc"},
            "weight_precision": "bf16",
        },
    }


def training(*, alpha: int = 8, rank: int = 4, targets: str = "all-linear", kind: str = "lora") -> dict[str, Any]:
    return {
        "selection_id": "training/lfm",
        "revision": "1",
        "resolved": {
            "backend": "trl@1.12.0.post9",
            "parameter_update_kind": kind,
            "parameter_update": {"kind": kind, "rank": rank, "alpha": alpha, "dropout": 0.0, "target_modules": targets},
            "target_id": TARGET_ID,
        },
    }


def settings(
    learning_rate: float = 2e-4,
    schedule: str | None = "constant",
    *,
    steps: int = 20,
    prompt: int = 20_480,
    completion: int = 4_096,
    prompts: int = 10,
    generations: int = 4,
    active_sampling: bool = True,
) -> dict[str, Any]:
    resolved: dict[str, Any] = {
        "max_steps": steps,
        "max_length": prompt + completion,
        "learning_rate": learning_rate,
        "beta": 0.0,
        "num_prompts_per_step": prompts,
        "num_generations": generations,
        "max_prompt_length": prompt,
        "max_completion_length": completion,
        "active_sampling": {"max_candidate_batches": 10} if active_sampling else None,
        "dynamic_sampling": None,
    }
    if schedule is not None:
        resolved["lr_scheduler_type"] = schedule
    return {"selection_id": "lfm/vortex", "revision": "1", "resolved": resolved}


def environment(max_concurrent: int = 40) -> dict[str, Any]:
    return {
        "selection_id": "automationbench",
        "revision": "1",
        "resolved": {"max_concurrent": max_concurrent, "activation": {"kind": "verifiers-config"}},
    }


def targets(memory_gb: float = 96.0, accelerator: str = "RTXPRO6000") -> dict[str, Any]:
    return {
        "schema_version": 1,
        "targets": [
            {
                "selection_id": TARGET_ID,
                "memory_gb": memory_gb,
                "hardware": {"accelerator_count": 1, "accelerator_model": accelerator, "supports_bf16": True},
            }
        ],
    }


@pytest.fixture
def snap() -> SimpleNamespace:
    """Snapshot builders for tests (the suite imports modules by path, not by package)."""

    return SimpleNamespace(
        TUNED=TUNED,
        rollout=rollout,
        model=model,
        training=training,
        settings=settings,
        environment=environment,
        targets=targets,
    )
