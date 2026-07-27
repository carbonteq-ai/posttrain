from __future__ import annotations

import json
from pathlib import Path

import pytest
from posttrain.common import ContractError
from posttrain.execution import JobPackageManifest, RuntimeImageRef
from posttrain.execution_pack import (
    ImagePublicationSpec,
    JobImagePublicationRequest,
)
from posttrain_execution_buildkit import (
    BuildKitJobImagePublisher,
    RemoteImageNotFoundError,
)


class FakeBuildx:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.digest = "d" * 64
        self.observed_digest: str | None = None
        self.image_missing = False

    def invoke(self, arguments):
        call = tuple(arguments)
        self.calls.append(call)
        if "--metadata-file" in call:
            path = Path(call[call.index("--metadata-file") + 1])
            path.write_text(
                json.dumps(
                    {
                        "posttrain-job": {
                            "containerimage.digest": f"sha256:{self.digest}",
                        }
                    }
                ),
                encoding="utf-8",
            )
            return ""
        if call[:2] == ("imagetools", "inspect"):
            if self.image_missing:
                self.image_missing = False
                raise RemoteImageNotFoundError("manifest unknown")
            return json.dumps(f"sha256:{self.observed_digest or self.digest}")
        return ""


def _manifest() -> JobPackageManifest:
    return JobPackageManifest(
        project_id="foundation-models",
        work_package_id="train/qwen-grpo",
        job_id="grpo",
        job_definition_id="train/trl-grpo@1",
        job_kind="train.grpo",
        resolved_inputs_digest="a" * 64,
        framework_source_digest="b" * 64,
        project_source_digest="c" * 64,
        runtime_dependencies_digest="1" * 64,
        code_requirements_digest="2" * 64,
        resolved_config_digest="3" * 64,
        project_config_digest="6" * 64,
        universal_image=RuntimeImageRef(f"registry.lan/posttrain/base@sha256:{'4' * 64}"),
        kind_image=RuntimeImageRef(f"registry.lan/posttrain/online-rl@sha256:{'5' * 64}"),
        runtime_variant="online-rl-trl-py312",
        expected_artifact_roles=("model", "summary"),
    )


def _definition(tmp_path: Path) -> Path:
    root = tmp_path / "definition"
    root.mkdir()
    (root / "Dockerfile").write_text(
        "ARG POSTTRAIN_KIND_IMAGE\nFROM ${POSTTRAIN_KIND_IMAGE}\n",
        encoding="utf-8",
    )
    bake = root / "docker-bake.hcl"
    bake.write_text(
        'target "posttrain-job" {}\ntarget "posttrain-job-smoke" {}\n',
        encoding="utf-8",
    )
    return bake.resolve()


def _request(tmp_path: Path) -> JobImagePublicationRequest:
    context = tmp_path / "staged"
    context.mkdir(parents=True, exist_ok=True)
    return JobImagePublicationRequest(
        manifest=_manifest(),
        staged_context=context.resolve(),
        publication=ImagePublicationSpec(
            "registry.lan/carbonteq/posttrain-job",
        ),
    )


def test_publisher_checks_smokes_pushes_verifies_and_reuses_receipt(
    tmp_path: Path,
) -> None:
    gateway = FakeBuildx()
    request = _request(tmp_path)
    publisher = BuildKitJobImagePublisher(
        bake_file=_definition(tmp_path),
        receipt_root=(tmp_path / "receipts").resolve(),
        gateway=gateway,
        builder="posttrain-builder",
    )

    first = publisher.publish(request)
    second = publisher.publish(request)

    assert not first.cache_hit
    assert second.cache_hit
    assert first.image == second.image
    assert first.image.value == (f"{request.publication.repository}@sha256:{gateway.digest}")
    assert first.receipt.stat().st_mode & 0o777 == 0o600
    build_calls = [call for call in gateway.calls if "--metadata-file" in call]
    assert len(build_calls) == 1
    build = build_calls[0]
    assert build[-2:] == ("posttrain-job-smoke", "posttrain-job")
    assert (
        "posttrain-job.output=type=image,push=true,compression=zstd,"
        "compression-level=3,force-compression=true,oci-mediatypes=true"
    ) in build
    assert ("--provenance", "mode=max") == (
        build[build.index("--provenance")],
        build[build.index("--provenance") + 1],
    )
    assert ("--sbom", "true") == (
        build[build.index("--sbom")],
        build[build.index("--sbom") + 1],
    )
    for variable in (
        f"POSTTRAIN_KIND_IMAGE={request.manifest.kind_image.value}",
        f"PACKAGE_KEY={request.package_key}",
        f"IMAGE_TAG={request.publication_key}",
        f"JOB_KIND={request.manifest.job_kind}",
        f"RUNTIME_VARIANT={request.manifest.runtime_variant}",
        "RUNTIME_DEPENDENCIES_DIGEST=" + "1" * 64,
        "CODE_REQUIREMENTS_DIGEST=" + "2" * 64,
        "RESOLVED_CONFIG_DIGEST=" + "3" * 64,
        "PROJECT_CONFIG_DIGEST=" + "6" * 64,
    ):
        assert variable in build
    assert sum(call[:2] == ("imagetools", "inspect") for call in gateway.calls) == 2


def test_publication_identity_excludes_local_context_and_bake_paths(
    tmp_path: Path,
) -> None:
    first = _request(tmp_path / "first")
    second = _request(tmp_path / "second")

    assert first.package_key == second.package_key
    assert first.publication_key == second.publication_key


def test_missing_cached_remote_image_is_rebuilt(tmp_path: Path) -> None:
    gateway = FakeBuildx()
    request = _request(tmp_path)
    publisher = BuildKitJobImagePublisher(
        bake_file=_definition(tmp_path),
        receipt_root=(tmp_path / "receipts").resolve(),
        gateway=gateway,
    )
    publisher.publish(request)
    gateway.image_missing = True

    rebuilt = publisher.publish(request)

    assert not rebuilt.cache_hit
    assert sum("--metadata-file" in call for call in gateway.calls) == 2
    assert rebuilt.receipt.is_file()


def test_receipt_with_unsafe_permissions_is_never_reused(
    tmp_path: Path,
) -> None:
    gateway = FakeBuildx()
    request = _request(tmp_path)
    publisher = BuildKitJobImagePublisher(
        bake_file=_definition(tmp_path),
        receipt_root=(tmp_path / "receipts").resolve(),
        gateway=gateway,
    )
    result = publisher.publish(request)
    result.receipt.chmod(0o644)

    with pytest.raises(ContractError, match="group or other"):
        publisher.publish(request)


def test_remote_digest_mismatch_never_writes_a_receipt(
    tmp_path: Path,
) -> None:
    gateway = FakeBuildx()
    gateway.observed_digest = "e" * 64
    request = _request(tmp_path)
    publisher = BuildKitJobImagePublisher(
        bake_file=_definition(tmp_path),
        receipt_root=(tmp_path / "receipts").resolve(),
        gateway=gateway,
    )

    with pytest.raises(RuntimeError, match="digest mismatch"):
        publisher.publish(request)

    assert not (tmp_path / "receipts" / f"{request.publication_key}.json").exists()


def test_changed_build_definition_cannot_reuse_receipt(
    tmp_path: Path,
) -> None:
    gateway = FakeBuildx()
    request = _request(tmp_path)
    bake = _definition(tmp_path)
    receipt_root = (tmp_path / "receipts").resolve()
    BuildKitJobImagePublisher(
        bake_file=bake,
        receipt_root=receipt_root,
        gateway=gateway,
    ).publish(request)
    (bake.parent / "Dockerfile").write_text(
        "ARG POSTTRAIN_KIND_IMAGE\nFROM ${POSTTRAIN_KIND_IMAGE}\nRUN true\n",
        encoding="utf-8",
    )

    with pytest.raises(ContractError, match="build definition"):
        BuildKitJobImagePublisher(
            bake_file=bake,
            receipt_root=receipt_root,
            gateway=gateway,
        ).publish(request)


def test_publication_requires_attestations() -> None:
    with pytest.raises(ContractError, match="provenance and an SBOM"):
        ImagePublicationSpec(
            "registry.lan/carbonteq/posttrain-job",
            provenance=False,
        )
