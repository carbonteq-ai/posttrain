from __future__ import annotations

import subprocess
from argparse import Namespace
from pathlib import Path

import pytest
from posttrain.train import GRPOSettings, LoRAUpdate, TrainingLoop
from posttrain_lab.cli import (
    _grpo_settings_with_loop_overrides,
    _grpo_training_binding,
    _tracking_backend,
)
from posttrain_tracking_trackio import TrackioBackend
from posttrain_tracking_wandb import WandbBackend


def _args(**changes: object) -> Namespace:
    values = {
        "tracked": False,
        "tracking_backend": None,
        "project": "tests",
        "tracking_server_url": None,
        "wandb_entity": None,
        "wandb_base_url": None,
        "peft": "lora",
        "training_backend": "trl",
        "verl_python_executable": None,
        "verl_working_directory": None,
        "verl_source_revision": None,
        "verl_dependency_lock": None,
    }
    values.update(changes)
    return Namespace(**values)


def test_tracking_is_opt_in_and_trackio_is_the_default() -> None:
    assert _tracking_backend(_args()) is None
    assert isinstance(_tracking_backend(_args(tracked=True)), TrackioBackend)
    assert isinstance(_tracking_backend(_args(tracking_backend="trackio")), TrackioBackend)


def test_wandb_tracking_requires_an_entity() -> None:
    with pytest.raises(SystemExit, match="WANDB_ENTITY"):
        _tracking_backend(_args(tracking_backend="wandb"))
    assert isinstance(
        _tracking_backend(_args(tracking_backend="wandb", wandb_entity="team")),
        WandbBackend,
    )


def test_verl_grpo_binding_requires_and_records_immutable_local_runtime(tmp_path: Path) -> None:
    worktree = tmp_path / "verl"
    worktree.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=worktree, check=True)
    subprocess.run(["git", "config", "user.email", "tests@example.com"], cwd=worktree, check=True)
    subprocess.run(["git", "config", "user.name", "Tests"], cwd=worktree, check=True)
    tracked = worktree / "tracked.txt"
    tracked.write_text("qualified\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=worktree, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=worktree, check=True)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=worktree,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    lockfile = tmp_path / "uv.lock"
    lockfile.write_text("version = 1\n", encoding="utf-8")
    binding = _grpo_training_binding(
        _args(
            training_backend="verl",
            verl_python_executable="/opt/verl/bin/python",
            verl_working_directory=str(worktree),
            verl_source_revision=revision,
            verl_dependency_lock=lockfile,
        )
    )

    assert binding.backend == f"verl@{revision[:7]}"
    assert binding.backend_options["python_executable"] == "/opt/verl/bin/python"
    assert binding.backend_options["working_directory"] == str(worktree)
    assert binding.backend_options["source_revision"] == revision
    assert binding.backend_options["source_dirty"] is False
    assert binding.backend_options["dependency_lock_sha256"] == (
        "dbab12665d98aef021ba64953c61b0ed8a908cfb56a1c01e2fcb4b052b71a2a1"
    )
    assert isinstance(binding.update, LoRAUpdate)
    assert binding.update.target_modules == r".*[.](o_proj|down_proj)$"


def test_verl_grpo_binding_rejects_qlora() -> None:
    with pytest.raises(SystemExit, match="requires a LoRA update selection"):
        _grpo_training_binding(_args(training_backend="verl", peft="qlora"))


def test_grpo_loop_overrides_apply_three_matched_optimizer_steps() -> None:
    settings = GRPOSettings(
        "benchmark",
        TrainingLoop(max_steps=1, max_length=8192, per_device_batch_size=16),
        num_prompts_per_step=2,
        num_generations=8,
        max_prompt_length=2048,
        max_completion_length=6144,
    )

    benchmark = _grpo_settings_with_loop_overrides(
        settings,
        max_steps=3,
        max_length=None,
    )

    assert benchmark.loop.max_steps == 3
    assert benchmark.loop.max_length == 8192
