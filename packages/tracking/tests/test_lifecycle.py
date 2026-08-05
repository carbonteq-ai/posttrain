from __future__ import annotations

from datetime import UTC, datetime

import pytest
from posttrain.common import ContractError
from posttrain.tracking import (
    TrackingArtifactPurge,
    TrackingProjectDeletePlan,
    TrackingPurgePlan,
)


def test_tracking_purge_contract_validates_exact_artifact_scope() -> None:
    now = datetime.now(UTC)
    artifact = TrackingArtifactPurge(
        version_id="version-1",
        name="model",
        version="v1",
        digest="sha256:" + "a" * 64,
        logical_bytes=64,
        consumer_run_ids=("consumer-1",),
    )
    plan = TrackingPurgePlan(
        provider="trackio",
        project="purge-fixture",
        provider_run_ids=("provider-1",),
        run_ids=("producer-1",),
        artifacts=(artifact,),
        blockers=(),
        digest="sha256:" + "b" * 64,
        created_at=now,
    )

    assert plan.artifacts[0].consumer_run_ids == ("consumer-1",)


def test_tracking_purge_rejects_duplicate_artifact_versions() -> None:
    artifact = TrackingArtifactPurge(
        version_id="version-1",
        name="model",
        version="v1",
        digest=None,
        logical_bytes=0,
    )

    with pytest.raises(ContractError, match="artifact versions must be unique"):
        TrackingPurgePlan(
            provider="trackio",
            project="purge-fixture",
            provider_run_ids=("provider-1",),
            run_ids=("producer-1",),
            artifacts=(artifact, artifact),
            blockers=(),
            digest="sha256:" + "b" * 64,
            created_at=datetime.now(UTC),
        )


def test_project_delete_plan_rejects_negative_counts() -> None:
    with pytest.raises(ContractError, match="cannot be negative"):
        TrackingProjectDeletePlan(
            provider="trackio",
            project="purge-fixture",
            exists=True,
            runs=-1,
            artifacts=0,
            artifact_versions=0,
            logical_bytes=0,
            storage_bytes=0,
            blockers=(),
            digest="sha256:" + "c" * 64,
            created_at=datetime.now(UTC),
        )
