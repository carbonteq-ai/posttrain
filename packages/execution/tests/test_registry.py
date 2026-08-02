from __future__ import annotations

from datetime import UTC, datetime

import pytest
from posttrain.common import ContractError
from posttrain.execution import RegistryManifestDeletePlan, RegistryManifestRef


def test_registry_ref_requires_digest_and_rejects_tags() -> None:
    ref = RegistryManifestRef.parse(
        "registry.lan:5000/carbonteq/posttrain-job@sha256:" + "a" * 64
    )
    assert ref.value.endswith("@sha256:" + "a" * 64)
    with pytest.raises(ContractError, match="repository@sha256"):
        RegistryManifestRef.parse("registry.lan/carbonteq/posttrain-job:latest")
    with pytest.raises(ContractError, match="repository@sha256"):
        RegistryManifestRef.parse("registry.lan/carbonteq/posttrain-job@sha256:abc")


def test_registry_plan_is_digest_bound_and_content_addressed() -> None:
    reference = RegistryManifestRef("registry.lan/posttrain-job", "sha256:" + "b" * 64)
    first = RegistryManifestDeletePlan.build(
        reference,
        exists=True,
        created_at=datetime(2026, 8, 2, tzinfo=UTC),
    )
    second = RegistryManifestDeletePlan.build(
        reference,
        exists=True,
        created_at=datetime(2026, 8, 3, tzinfo=UTC),
    )
    assert first.digest == second.digest
    with pytest.raises(ContractError):
        RegistryManifestDeletePlan(
            reference=reference,
            exists=True,
            eligible=True,
            blockers=(),
            digest="sha256:" + "c" * 64,
            created_at=datetime(2026, 8, 2, tzinfo=UTC),
        )
