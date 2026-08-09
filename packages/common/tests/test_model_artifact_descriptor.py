"""Tests for checkpoint-derived model descriptors."""

import pytest
from posttrain.common import ContractError, ModelArtifactDescriptor, StoredArtifactRef
from posttrain.common.variants import FOUNDATION_VARIANTS


def test_descriptor_round_trips_model_facts_and_reconstructs_adapter_view() -> None:
    foundation = FOUNDATION_VARIANTS["qwen3.5-2b"]
    descriptor = ModelArtifactDescriptor.from_model_variant(
        foundation,
        source_run_id="run/example",
        checkpoint_step=25,
        checkpoint_snapshot_id="run/example/step-00000025",
    )
    adapter = ModelArtifactDescriptor(
        form="adapter",
        weight_precision=descriptor.weight_precision,
        family=descriptor.family,
        parameters=descriptor.parameters,
        instruction_tuned=descriptor.instruction_tuned,
        renderer=descriptor.renderer,
        capabilities=descriptor.capabilities,
        base=descriptor.base,
        source_run_id=descriptor.source_run_id,
        checkpoint_step=descriptor.checkpoint_step,
        checkpoint_snapshot_id=descriptor.checkpoint_snapshot_id,
    )

    model = adapter.to_model_variant(
        StoredArtifactRef("trackio", "project", "adapter", "v1", "sha256:" + "a" * 64),
        variant_id="models/qwen35/step-00000025/adapter",
        kind="model-adapter",
    )

    assert model.form == "adapter"
    assert model.artifact_uri == "trackio://project/adapter@v1"
    assert model.provenance["checkpoint_step"] == 25


def test_descriptor_rejects_model_weights_for_adapter_form() -> None:
    foundation = FOUNDATION_VARIANTS["qwen3.5-2b"]
    descriptor = ModelArtifactDescriptor(
        form="adapter",
        weight_precision=foundation.weight_precision,
        family=foundation.family,
        parameters=foundation.parameters,
        instruction_tuned=foundation.instruction_tuned,
        renderer=foundation.renderer,
        capabilities=foundation.capabilities,
        base=foundation.base,
    )

    with pytest.raises(ContractError, match="cannot use an adapter model form"):
        descriptor.to_model_variant(
            StoredArtifactRef("trackio", "project", "adapter", "v1"),
            variant_id="models/qwen35/adapter",
            kind="model-weights",
        )
