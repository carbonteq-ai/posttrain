"""Tests for checkpoint publication contracts."""

from datetime import UTC, datetime

import pytest
from posttrain.common import ContractError
from posttrain.common.variants import FOUNDATION_VARIANTS
from posttrain.train import (
    CheckpointComponent,
    CheckpointProjectionCapability,
    CheckpointPublicationPolicy,
    CheckpointSelector,
    CheckpointSnapshotId,
    CheckpointSnapshotManifest,
    inspect_checkpoint_artifacts,
    resolve_checkpoint_artifacts,
)


def _manifest(*, complete: bool = True) -> CheckpointSnapshotManifest:
    return CheckpointSnapshotManifest(
        snapshot_id=CheckpointSnapshotId("run/example", 25),
        created_at=datetime(2026, 8, 9, tzinfo=UTC),
        training_backend="trl",
        backend_revision="trl@revision",
        technique="grpo",
        parameter_update_kind="lora",
        base_model=FOUNDATION_VARIANTS["qwen3.5-2b"].base,
        renderer_id="qwen3.5-tools@1",
        tokenizer_fingerprint="a" * 64,
        trainer_checkpoint_schema="transformers@5",
        components=(CheckpointComponent("adapter", "adapter_model.safetensors", 10, "b" * 64),),
        complete=complete,
    )


def test_manifest_has_stable_snapshot_identity_and_size() -> None:
    manifest = _manifest()

    assert manifest.checkpoint_snapshot_id == "run/example/step-00000025"
    assert manifest.global_step == 25
    assert manifest.total_bytes == 10


def test_manifest_rejects_incomplete_paths_and_duplicate_components() -> None:
    with pytest.raises(ContractError, match="relative safe path"):
        CheckpointComponent("adapter", "../adapter.safetensors", 1, "a" * 64)

    with pytest.raises(ContractError, match="component paths must be unique"):
        CheckpointSnapshotManifest(
            snapshot_id=CheckpointSnapshotId("run/example", 25),
            created_at=datetime(2026, 8, 9, tzinfo=UTC),
            training_backend="trl",
            backend_revision="trl@revision",
            technique="grpo",
            parameter_update_kind="lora",
            base_model=FOUNDATION_VARIANTS["qwen3.5-2b"].base,
            renderer_id="qwen3.5-tools@1",
            tokenizer_fingerprint=None,
            trainer_checkpoint_schema="transformers@5",
            components=(
                CheckpointComponent("adapter", "adapter.safetensors", 1, "a" * 64),
                CheckpointComponent("adapter", "adapter.safetensors", 1, "b" * 64),
            ),
            complete=False,
        )


def test_publication_policy_is_independent_from_local_save_cadence() -> None:
    policy = CheckpointPublicationPolicy(milestone_steps=(25, 50), publish_terminal=True)

    assert policy.selects_step(25)
    assert policy.selects_step(100, terminal=True)
    assert not policy.selects_step(26)


def test_projection_capability_rejects_hidden_model_transform() -> None:
    with pytest.raises(ContractError, match="cannot also require a transform"):
        CheckpointProjectionCapability("full", recovery=True, model_view=True, requires_transform=True)


def _artifact(step: int, view: str) -> dict[str, object]:
    kind = "training-checkpoint" if view == "recovery" else "model-adapter"
    return {
        "logical_name": f"training/qwen/checkpoints/step-{step:08d}/{view}",
        "kind": kind,
        "metadata": {
            "checkpoint_snapshot_id": f"run/example/step-{step:08d}",
            "checkpoint_step": step,
            "checkpoint_view": view,
        },
        "artifact": {
            "provider": "trackio",
            "namespace": "project",
            "name": f"checkpoint-{step}-{view}",
            "version": "v1",
            "digest": "sha256:" + ("a" if view == "recovery" else "b") * 64,
            "provider_metadata": {},
        },
    }


def test_checkpoint_resolver_selects_latest_complete_pair() -> None:
    records = [_artifact(1, "recovery"), _artifact(1, "model"), _artifact(2, "recovery")]
    inspections = inspect_checkpoint_artifacts(records)
    assert [item.step for item in inspections] == [2, 1]
    assert inspections[0].ready is False

    selected = resolve_checkpoint_artifacts(
        records,
        CheckpointSelector("run/example", "latest-complete", "model"),
    )
    assert selected.step == 1
    assert selected.artifact_kind == "model-adapter"


def test_checkpoint_resolver_rejects_duplicate_views() -> None:
    with pytest.raises(ContractError, match="duplicate model views"):
        inspect_checkpoint_artifacts([_artifact(1, "model"), _artifact(1, "model")])


def test_inspection_keeps_digestless_views_visible_but_does_not_resolve_them() -> None:
    record = _artifact(3, "model")
    record["artifact"] = {**record["artifact"], "digest": None}  # type: ignore[index]

    inspections = inspect_checkpoint_artifacts([record])
    assert inspections[0].model_ready
    assert not inspections[0].model.has_digest  # type: ignore[union-attr]
    with pytest.raises(ContractError, match="no model checkpoint"):
        resolve_checkpoint_artifacts(
            [record],
            CheckpointSelector("run/example", 3, "model"),
        )
