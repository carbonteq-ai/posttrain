"""Training-checkpoint catalog contract tests."""

from posttrain.common import CatalogRef
from posttrain.train import TrainingCheckpoint
from posttrain.train.catalog_schema import decode_training_selection


def test_training_checkpoint_decodes_with_content_integrity() -> None:
    digest = "8" * 64
    checkpoint = decode_training_selection(
        CatalogRef("training", "checkpoints/opd-step-64"),
        {
            "selection_type": "training-checkpoint",
            "id": "checkpoints/opd-step-64",
            "revision": "1",
            "artifact": {
                "provider": "trackio",
                "namespace": "policy-prism-scope-opd-e2b-12b",
                "name": "training-checkpoint-step-0064",
                "version": "v0",
                "content_digest": digest,
            },
        },
        {},
    )

    assert isinstance(checkpoint, TrainingCheckpoint)
    assert checkpoint.content_digest == digest
    assert checkpoint.artifact.provider == "trackio"
    assert checkpoint.artifact.provider_metadata == {
        "posttrain_content_digest": digest,
        "posttrain_content_digest_kind": "tree",
    }
