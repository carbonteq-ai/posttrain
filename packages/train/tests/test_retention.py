"""Run-scoped training retention tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from posttrain.train.backends import retention
from posttrain.train.backends.retention import finalize_training_outputs, validate_adapter_only_directory


def _checkpoint(root: Path, step: int, payload: bytes) -> Path:
    path = root / f"global_step_{step}"
    path.mkdir(parents=True)
    (path / "actor.pt").write_bytes(payload)
    return path


def _lora_model(root: Path) -> tuple[Path, Path]:
    model = root / "model"
    adapter = model / "lora_adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text('{"r": 8}\n', encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    (model / "config.json").write_text('{"model_type": "test"}\n', encoding="utf-8")
    return model, adapter


def test_adapter_only_validation_accepts_final_adapter_and_recovery_state(tmp_path: Path) -> None:
    _, adapter = _lora_model(tmp_path)
    recovery = tmp_path / "checkpoint-1"
    recovery.mkdir()
    (recovery / "adapter_config.json").write_text('{"r": 8}\n', encoding="utf-8")
    (recovery / "adapter_model.safetensors").write_bytes(b"adapter")
    (recovery / "trainer_state.json").write_text('{"global_step": 1}\n', encoding="utf-8")
    (recovery / "optimizer.pt").write_bytes(b"optimizer")
    (recovery / "scheduler.pt").write_bytes(b"scheduler")
    (recovery / "rng_state.pth").write_bytes(b"rng")

    validate_adapter_only_directory(adapter)
    validate_adapter_only_directory(recovery, require_recovery_state=True)


@pytest.mark.parametrize("name", ["model.safetensors", "pytorch_model.bin", "model-00001-of-00002.safetensors"])
def test_adapter_only_validation_rejects_full_base_weight_exports(tmp_path: Path, name: str) -> None:
    _, adapter = _lora_model(tmp_path)
    (adapter / name).write_bytes(b"base")

    with pytest.raises(RuntimeError, match="full base-model weights"):
        validate_adapter_only_directory(adapter)


def test_adapter_only_validation_requires_exact_resume_state(tmp_path: Path) -> None:
    _, adapter = _lora_model(tmp_path)

    with pytest.raises(RuntimeError, match="trainer_state.json, optimizer.pt, scheduler.pt, rng_state.*pth"):
        validate_adapter_only_directory(adapter, require_recovery_state=True)


def test_lora_finalizer_removes_base_weights_and_superseded_checkpoints(tmp_path: Path) -> None:
    workspace = tmp_path.resolve()
    model, adapter = _lora_model(workspace)
    (model / "model.safetensors").write_bytes(b"duplicate-base")
    (model / "model-00001-of-00002.safetensors").write_bytes(b"duplicate-shard")
    (model / "model.safetensors.index.json").write_text("{}\n", encoding="utf-8")
    checkpoints = workspace / "checkpoints"
    first = _checkpoint(checkpoints, 1, b"old")
    latest = _checkpoint(checkpoints, 2, b"latest")

    result = finalize_training_outputs(
        workspace=workspace,
        model_dir=model,
        checkpoint_root=checkpoints,
        recovery_checkpoint=latest,
        update_kind="lora",
        checkpoint_limit=1,
    )

    assert result.model_dir == adapter
    assert result.recovery_checkpoint == latest
    assert result.reclaimed_bytes > len(b"duplicate-base") + len(b"old")
    assert not first.exists()
    assert latest.is_dir()
    assert (adapter / "adapter_model.safetensors").read_bytes() == b"adapter"
    assert (model / "config.json").is_file()
    assert not (model / "model.safetensors").exists()
    assert not (model / "model-00001-of-00002.safetensors").exists()
    assert not (model / "model.safetensors.index.json").exists()
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["checkpoint_limit"] == 1
    assert manifest["retained_model"] == "model/lora_adapter"
    assert manifest["retained_recovery_checkpoint"] == "checkpoints/global_step_2"
    assert {entry["reason"] for entry in manifest["planned_removals"]} == {
        "duplicate-immutable-base-weight",
        "superseded-recovery-checkpoint",
    }


def test_retention_finalizer_is_idempotent_after_completion(tmp_path: Path) -> None:
    workspace = tmp_path.resolve()
    model, _ = _lora_model(workspace)
    (model / "model.safetensors").write_bytes(b"base")
    checkpoints = workspace / "checkpoints"
    latest = _checkpoint(checkpoints, 1, b"latest")
    first = finalize_training_outputs(
        workspace=workspace,
        model_dir=model,
        checkpoint_root=checkpoints,
        recovery_checkpoint=latest,
        update_kind="lora",
        checkpoint_limit=1,
    )
    manifest_before = first.manifest_path.read_bytes()

    second = finalize_training_outputs(
        workspace=workspace,
        model_dir=model,
        checkpoint_root=checkpoints,
        recovery_checkpoint=latest,
        update_kind="lora",
        checkpoint_limit=1,
    )

    assert second == first
    assert second.manifest_path.read_bytes() == manifest_before


def test_retention_finalizer_rejects_changed_output_after_completion(tmp_path: Path) -> None:
    workspace = tmp_path.resolve()
    model, adapter = _lora_model(workspace)
    checkpoints = workspace / "checkpoints"
    latest = _checkpoint(checkpoints, 1, b"latest")
    finalize_training_outputs(
        workspace=workspace,
        model_dir=model,
        checkpoint_root=checkpoints,
        recovery_checkpoint=latest,
        update_kind="lora",
        checkpoint_limit=1,
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"changed-adapter")

    with pytest.raises(ValueError, match="digest"):
        finalize_training_outputs(
            workspace=workspace,
            model_dir=model,
            checkpoint_root=checkpoints,
            recovery_checkpoint=latest,
            update_kind="lora",
            checkpoint_limit=1,
        )


def test_retention_finalizer_validates_adapter_before_deleting(tmp_path: Path) -> None:
    workspace = tmp_path.resolve()
    model = workspace / "model"
    adapter = model / "lora_adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    duplicate = model / "model.safetensors"
    duplicate.write_bytes(b"base")
    checkpoints = workspace / "checkpoints"
    old = _checkpoint(checkpoints, 1, b"old")
    latest = _checkpoint(checkpoints, 2, b"latest")

    with pytest.raises(RuntimeError, match="LoRA update is incomplete"):
        finalize_training_outputs(
            workspace=workspace,
            model_dir=model,
            checkpoint_root=checkpoints,
            recovery_checkpoint=latest,
            update_kind="lora",
            checkpoint_limit=1,
        )

    assert duplicate.is_file()
    assert old.is_dir()
    assert latest.is_dir()
    assert not (workspace / "retention-manifest.json").exists()


def test_retention_finalizer_leaves_planned_manifest_when_cleanup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path.resolve()
    model, _ = _lora_model(workspace)
    duplicate = model / "model.safetensors"
    duplicate.write_bytes(b"base")
    checkpoints = workspace / "checkpoints"
    old = _checkpoint(checkpoints, 1, b"old")
    latest = _checkpoint(checkpoints, 2, b"latest")

    def fail_cleanup(root: Path, path: Path) -> None:
        del root, path
        raise OSError("simulated cleanup failure")

    monkeypatch.setattr(retention, "_remove_inside", fail_cleanup)
    with pytest.raises(OSError, match="simulated cleanup failure"):
        finalize_training_outputs(
            workspace=workspace,
            model_dir=model,
            checkpoint_root=checkpoints,
            recovery_checkpoint=latest,
            update_kind="lora",
            checkpoint_limit=1,
        )

    manifest = json.loads((workspace / "retention-manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "planned"
    assert duplicate.is_file()
    assert old.is_dir()
    assert latest.is_dir()


def test_full_update_retains_weights_but_rotates_recovery_state(tmp_path: Path) -> None:
    workspace = tmp_path.resolve()
    model = workspace / "model"
    model.mkdir()
    weights = model / "model.safetensors"
    weights.write_bytes(b"trained-weights")
    checkpoints = workspace / "checkpoints"
    old = _checkpoint(checkpoints, 1, b"old")
    latest = _checkpoint(checkpoints, 2, b"latest")

    result = finalize_training_outputs(
        workspace=workspace,
        model_dir=model,
        checkpoint_root=checkpoints,
        recovery_checkpoint=latest,
        update_kind="full",
        checkpoint_limit=1,
    )

    assert result.model_dir == model
    assert weights.read_bytes() == b"trained-weights"
    assert not old.exists()
    assert latest.is_dir()
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert [entry["reason"] for entry in manifest["planned_removals"]] == ["superseded-recovery-checkpoint"]


def test_finalizer_allows_absent_checkpoint_root_without_recovery_state(tmp_path: Path) -> None:
    workspace = tmp_path.resolve()
    model, adapter = _lora_model(workspace)
    checkpoints = workspace / "checkpoints"

    result = finalize_training_outputs(
        workspace=workspace,
        model_dir=model,
        checkpoint_root=checkpoints,
        recovery_checkpoint=None,
        update_kind="lora",
        checkpoint_limit=1,
    )

    assert result.model_dir == adapter
    assert result.recovery_checkpoint is None
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["checkpoint_root"] == "checkpoints"
    assert manifest["retained_recovery_checkpoint"] is None


def test_retention_finalizer_rejects_paths_outside_workspace(tmp_path: Path) -> None:
    workspace = (tmp_path / "workspace").resolve()
    workspace.mkdir()
    outside, _ = _lora_model((tmp_path / "outside").resolve())
    checkpoints = workspace / "checkpoints"
    latest = _checkpoint(checkpoints, 1, b"latest")

    with pytest.raises(ValueError, match="inside the training workspace"):
        finalize_training_outputs(
            workspace=workspace,
            model_dir=outside,
            checkpoint_root=checkpoints,
            recovery_checkpoint=latest,
            update_kind="lora",
            checkpoint_limit=1,
        )

    assert (outside / "lora_adapter" / "adapter_model.safetensors").is_file()
