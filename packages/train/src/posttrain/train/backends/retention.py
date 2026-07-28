"""Run-scoped retention finalization shared by training backends."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

_CHECKPOINT = re.compile(r"global_step_(\d+)")
_SHARDED_SAFETENSORS = re.compile(r"model-\d+-of-\d+\.safetensors")
_SHARDED_PYTORCH = re.compile(r"pytorch_model-\d+-of-\d+\.bin")


@dataclass(frozen=True, slots=True)
class TrainingRetentionResult:
    """Verified retained outputs after terminal run cleanup."""

    model_dir: Path
    recovery_checkpoint: Path | None
    manifest_path: Path
    reclaimed_bytes: int


def finalize_training_outputs(
    *,
    workspace: Path,
    model_dir: Path,
    checkpoint_root: Path,
    recovery_checkpoint: Path | None,
    update_kind: Literal["full", "lora"],
    checkpoint_limit: int,
    manifest_path: Path | None = None,
) -> TrainingRetentionResult:
    """Retain the result and bounded recovery state, then record removed bytes.

    The finalizer is deliberately run-scoped. It only removes known full-weight
    exports directly inside a LoRA model directory and superseded
    ``global_step_*`` directories under the selected checkpoint root.
    """

    if checkpoint_limit < 1:
        raise ValueError("checkpoint_limit must be at least one")
    root = _absolute_directory(workspace, "training retention workspace")
    model_root = _inside(root, model_dir, "model directory")
    checkpoints_root = _inside(root, checkpoint_root, "checkpoint root")
    manifest = _inside(
        root,
        manifest_path or root / "retention-manifest.json",
        "retention manifest",
        allow_missing=True,
    )
    if manifest.exists():
        existing = _read_manifest(manifest)
        if existing.get("status") == "completed":
            _validate_existing_policy(
                existing,
                root=root,
                model_root=model_root,
                checkpoints_root=checkpoints_root,
                recovery_checkpoint=recovery_checkpoint,
                update_kind=update_kind,
                checkpoint_limit=checkpoint_limit,
            )
            retained_model = _manifest_path(root, existing["retained_model"])
            retained_recovery_value = existing.get("retained_recovery_checkpoint")
            retained_recovery = (
                _manifest_path(root, retained_recovery_value) if isinstance(retained_recovery_value, str) else None
            )
            _validate_retained_output(retained_model, update_kind)
            retained_digest = existing.get("retained_model_digest")
            if not isinstance(retained_digest, str) or _digest_tree(retained_model) != retained_digest:
                raise ValueError("retained model digest does not match completed manifest")
            if retained_recovery is not None and not retained_recovery.is_dir():
                raise FileNotFoundError(retained_recovery)
            return TrainingRetentionResult(
                retained_model,
                retained_recovery,
                manifest,
                int(existing["reclaimed_bytes"]),
            )

    if not model_root.is_dir():
        raise FileNotFoundError(model_root)
    retained_model = model_root / "lora_adapter" if update_kind == "lora" else model_root
    _validate_retained_output(retained_model, update_kind)

    checkpoints = _checkpoint_directories(checkpoints_root)
    retained_recovery = None
    retained_checkpoints: tuple[Path, ...] = ()
    if recovery_checkpoint is not None:
        retained_recovery = _inside(root, recovery_checkpoint, "recovery checkpoint")
        if not retained_recovery.is_dir():
            raise FileNotFoundError(retained_recovery)
        if retained_recovery.parent != checkpoints_root:
            raise ValueError("recovery checkpoint must be a direct child of checkpoint root")
        if retained_recovery not in checkpoints:
            raise ValueError("recovery checkpoint must use the global_step_<n> convention")
        retained_checkpoints = checkpoints[-checkpoint_limit:]
        if retained_recovery != checkpoints[-1] or retained_recovery not in retained_checkpoints:
            raise ValueError("recovery checkpoint must be the latest retained checkpoint")
    elif checkpoints:
        raise ValueError("checkpoint directories exist without a selected recovery checkpoint")

    removals: list[tuple[Path, str]] = [
        (path, "superseded-recovery-checkpoint") for path in checkpoints if path not in retained_checkpoints
    ]
    if update_kind == "lora":
        removals.extend(
            (path, "duplicate-immutable-base-weight")
            for path in sorted(model_root.iterdir())
            if path.is_file() and _is_full_weight_export(path.name)
        )

    before_bytes = _logical_bytes(root)
    planned = {
        "schema_version": 1,
        "status": "planned",
        "created_at": datetime.now(UTC).isoformat(),
        "workspace": ".",
        "update_kind": update_kind,
        "checkpoint_limit": checkpoint_limit,
        "model_root": _relative(root, model_root),
        "checkpoint_root": _relative(root, checkpoints_root),
        "retained_model": _relative(root, retained_model),
        "retained_model_digest": _digest_tree(retained_model),
        "retained_recovery_checkpoint": (_relative(root, retained_recovery) if retained_recovery is not None else None),
        "planned_removals": [
            {
                "path": _relative(root, path),
                "reason": reason,
                "logical_bytes": _logical_bytes(path),
            }
            for path, reason in removals
        ],
        "workspace_bytes_before": before_bytes,
    }
    _write_manifest(manifest, planned)

    for path, _ in removals:
        _remove_inside(root, path)

    _validate_retained_output(retained_model, update_kind)
    if retained_recovery is not None and not retained_recovery.is_dir():
        raise FileNotFoundError(retained_recovery)
    reclaimed_bytes = sum(int(entry["logical_bytes"]) for entry in planned["planned_removals"])
    completed = {
        **planned,
        "status": "completed",
        "completed_at": datetime.now(UTC).isoformat(),
        "reclaimed_bytes": reclaimed_bytes,
        "workspace_bytes_after": _logical_bytes(root),
    }
    _write_manifest(manifest, completed)
    return TrainingRetentionResult(
        retained_model.resolve(),
        retained_recovery.resolve() if retained_recovery is not None else None,
        manifest.resolve(),
        reclaimed_bytes,
    )


def _absolute_directory(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    resolved = path.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    return resolved


def _inside(
    root: Path,
    path: Path,
    label: str,
    *,
    allow_missing: bool = False,
) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    resolved = path.resolve(strict=not allow_missing)
    if not resolved.is_relative_to(root):
        raise ValueError(f"{label} must remain inside the training workspace")
    return resolved


def _checkpoint_directories(root: Path) -> tuple[Path, ...]:
    if not root.exists():
        return ()
    if not root.is_dir():
        raise NotADirectoryError(root)
    values: list[tuple[int, Path]] = []
    for path in root.iterdir():
        match = _CHECKPOINT.fullmatch(path.name)
        if match is not None and path.is_dir():
            values.append((int(match.group(1)), path.resolve()))
    return tuple(path for _, path in sorted(values))


def _validate_retained_output(path: Path, update_kind: Literal["full", "lora"]) -> None:
    if not path.is_dir():
        raise FileNotFoundError(path)
    if update_kind == "full":
        weights = tuple(path.glob("*.safetensors")) + tuple(path.glob("pytorch_model*.bin"))
        if not weights:
            raise RuntimeError(f"full update has no model weights under {path}")
        return
    config = path / "adapter_config.json"
    weights = tuple(path.glob("adapter_model*.safetensors")) + tuple(path.glob("adapter_model*.bin"))
    if not config.is_file() or not weights:
        raise RuntimeError(f"LoRA update is incomplete under {path}")


def _is_full_weight_export(name: str) -> bool:
    return (
        name == "model.safetensors"
        or name == "model.safetensors.index.json"
        or name == "pytorch_model.bin"
        or name == "pytorch_model.bin.index.json"
        or _SHARDED_SAFETENSORS.fullmatch(name) is not None
        or _SHARDED_PYTORCH.fullmatch(name) is not None
    )


def _logical_bytes(path: Path) -> int:
    if path.is_file() or path.is_symlink():
        return path.lstat().st_size
    return sum(child.lstat().st_size for child in path.rglob("*") if child.is_file() or child.is_symlink())


def _digest_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        with child.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _manifest_path(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError("retention manifest path escapes the training workspace")
    return path


def _remove_inside(root: Path, path: Path) -> None:
    resolved = _inside(root, path, "retention removal")
    if resolved == root:
        raise ValueError("retention finalizer cannot remove its workspace")
    if resolved.is_symlink() or resolved.is_file():
        resolved.unlink()
    elif resolved.is_dir():
        shutil.rmtree(resolved)


def _read_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError(f"unsupported retention manifest at {path}")
    return value


def _validate_existing_policy(
    manifest: dict[str, Any],
    *,
    root: Path,
    model_root: Path,
    checkpoints_root: Path,
    recovery_checkpoint: Path | None,
    update_kind: str,
    checkpoint_limit: int,
) -> None:
    expected_recovery = (
        _relative(root, _inside(root, recovery_checkpoint, "recovery checkpoint"))
        if recovery_checkpoint is not None
        else None
    )
    expected = {
        "update_kind": update_kind,
        "checkpoint_limit": checkpoint_limit,
        "model_root": _relative(root, model_root),
        "checkpoint_root": _relative(root, checkpoints_root),
        "retained_recovery_checkpoint": expected_recovery,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"completed retention manifest does not match {key}")


def _write_manifest(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


__all__ = ["TrainingRetentionResult", "finalize_training_outputs"]
