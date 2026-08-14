from __future__ import annotations

from types import SimpleNamespace

import pytest
from posttrain.common import ContractError
from posttrain.execution import RegistryManifestRef
from posttrain_execution_buildkit import (
    DistributionRegistryContentReader,
    DistributionRegistryLifecycleAdmin,
    RegistryPurgeActionExecutor,
)


def test_distribution_admin_revalidates_exact_digest() -> None:
    expected = RegistryManifestRef("registry.lan/carbonteq/posttrain-job", "sha256:" + "a" * 64)

    class Transport:
        def __init__(self) -> None:
            self.present = True
            self.deleted: list[str] = []

        def head(self, reference: RegistryManifestRef) -> bool:
            assert reference == expected
            return self.present

        def delete(self, reference: RegistryManifestRef) -> bool:
            self.deleted.append(reference.value)
            self.present = False
            return True

    transport = Transport()
    admin = DistributionRegistryLifecycleAdmin(transport)
    plan = admin.plan_manifest_delete(expected)
    receipt = admin.delete_manifest(plan)
    assert receipt.deleted is True
    assert transport.deleted == [expected.value]

    with pytest.raises(ContractError, match="presence changed"):
        admin.delete_manifest(plan)


def test_registry_action_executor_binds_preview_to_action() -> None:
    expected = RegistryManifestRef("registry.lan/carbonteq/posttrain-job", "sha256:" + "c" * 64)

    class Admin:
        def plan_manifest_delete(self, reference: RegistryManifestRef):
            from posttrain.execution import RegistryManifestDeletePlan

            assert reference == expected
            return RegistryManifestDeletePlan.build(reference, exists=True)

        def delete_manifest(self, plan):
            assert plan.reference == expected

    action = SimpleNamespace(
        action_id="registry:run-1",
        target={"reference": expected.value},
    )
    executor = RegistryPurgeActionExecutor(Admin())  # type: ignore[arg-type]
    executor.revalidate(action)
    executor.apply(action)


def test_registry_content_reader_counts_only_destination_missing_blobs() -> None:
    source = RegistryManifestRef("registry.lan/carbonteq/posttrain-kind", "sha256:" + "a" * 64)
    child = RegistryManifestRef(source.repository, "sha256:" + "b" * 64)

    class Transport:
        def manifest(self, reference: RegistryManifestRef) -> bytes | None:
            if reference == source:
                return (
                    '{"manifests": [{"digest": "sha256:'
                    + "b" * 64
                    + '"}]}'
                ).encode()
            if reference == child:
                return (
                    '{"config": {"digest": "sha256:'
                    + "c" * 64
                    + '", "size": 10}, "layers": ['
                    + '{"digest": "sha256:'
                    + "d" * 64
                    + '", "size": 100}, {"digest": "sha256:'
                    + "e" * 64
                    + '", "size": 200}]}'
                ).encode()
            raise AssertionError(reference)

        def blob_exists(self, repository: str, digest: str) -> bool:
            assert repository == "mirror.lan/team/posttrain-kind"
            return digest in {"sha256:" + "c" * 64, "sha256:" + "d" * 64}

    reader = DistributionRegistryContentReader(Transport())
    inventory = reader.inventory(source)

    assert inventory.logical_bytes == 310
    assert reader.missing_blob_bytes(source, "mirror.lan/team/posttrain-kind") == 200
