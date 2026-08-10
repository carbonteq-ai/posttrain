from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from posttrain.common import ContractError
from posttrain.execution import EnvironmentActivationLock, EnvironmentPackageLock, JobPackageManifest, RuntimeImageRef
from posttrain.execution_pack import (
    ImagePublicationSpec,
    JobImagePublicationRequest,
    JobImageResolutionRequest,
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
        for argument in call:
            if isinstance(argument, str) and "output=type=oci,dest=" in argument:
                destination = Path(argument.rsplit("dest=", 1)[1].split(",", 1)[0])
                destination.mkdir(parents=True)
                (destination / "index.json").write_text("{}\n", encoding="utf-8")
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


def test_deferred_qualification_waiver_is_forwarded_without_changing_publication_identity(tmp_path: Path) -> None:
    gateway = FakeBuildx()
    request = _request(tmp_path)
    waived = JobImagePublicationRequest(
        manifest=request.manifest,
        staged_context=request.staged_context,
        publication=request.publication,
        allow_deferred_qualification=True,
    )
    publisher = BuildKitJobImagePublisher(
        bake_file=_definition(tmp_path),
        receipt_root=(tmp_path / "receipts").resolve(),
        gateway=gateway,
    )

    publisher.publish_local(waived)

    assert waived.publication_key == request.publication_key
    assert any("ALLOW_DEFERRED_QUALIFICATION=1" in call for call in gateway.calls)


def test_deferred_qualification_is_rejected_before_cache_or_build_without_waiver(tmp_path: Path) -> None:
    gateway = FakeBuildx()
    request = _request(tmp_path)
    activation = {"kind": "verifiers-config", "config": {"taskset": {"id": "network-backed"}}}
    deferred = EnvironmentActivationLock(
        environment_id="network-backed",
        package="network-env",
        kind="verifiers-config",
        digest=hashlib.sha256(json.dumps(activation, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        config={"taskset": {"id": "network-backed"}},
        qualification="deferred",
    )
    package = EnvironmentPackageLock(
        package="network-env",
        repository=None,
        revision=None,
        subdirectory=None,
        tree_digest="b" * 64,
        wheel_filename="network_env-1.0.0-py3-none-any.whl",
        wheel_digest="c" * 64,
        wheel_size_bytes=1,
        source_kind="project-path",
        project_path="environments/network_env",
    )
    guarded = JobImagePublicationRequest(
        manifest=replace(
            request.manifest,
            environment_packages=(package,),
            environment_activations=(deferred,),
        ),
        staged_context=request.staged_context,
        publication=request.publication,
    )
    publisher = BuildKitJobImagePublisher(
        bake_file=_definition(tmp_path),
        receipt_root=(tmp_path / "receipts").resolve(),
        gateway=gateway,
    )

    with pytest.raises(ContractError, match="--allow-deferred-qualification"):
        publisher.publish_local(guarded)

    assert gateway.calls == []


def test_publishes_a_verified_local_oci_layout(tmp_path: Path) -> None:
    gateway = FakeBuildx()
    publisher = BuildKitJobImagePublisher(
        bake_file=_definition(tmp_path),
        receipt_root=(tmp_path / "receipts").resolve(),
        gateway=gateway,
    )

    result = publisher.publish_local(_request(tmp_path))

    assert result.layout.joinpath("index.json").is_file()
    assert result.receipt.is_file()
    assert result.tag.startswith("posttrain-local:")
    published_check = next(call for call in gateway.calls if call[-1] == "posttrain-job" and "--call" in call)
    assert "posttrain-job.output=type=cacheonly" in published_check
    local_export = next(
        argument
        for call in gateway.calls
        for argument in call
        if isinstance(argument, str) and "output=type=oci,dest=" in argument
    )
    assert local_export.endswith(",tar=false")
    assert not any("push=true" in argument for call in gateway.calls for argument in call)


def test_loads_a_single_platform_image_into_the_local_daemon(tmp_path: Path) -> None:
    gateway = FakeBuildx()
    publisher = BuildKitJobImagePublisher(
        bake_file=_definition(tmp_path),
        receipt_root=(tmp_path / "receipts").resolve(),
        gateway=gateway,
    )

    result = publisher.publish_local_daemon(_request(tmp_path))

    assert result.image.value == f"registry.lan/carbonteq/posttrain-job@sha256:{gateway.digest}"
    assert result.tag.startswith("posttrain-local:")
    assert result.receipt.name.endswith(".local-daemon.json")
    daemon_output = next(
        argument
        for call in gateway.calls
        for argument in call
        if isinstance(argument, str) and argument.endswith("output=type=docker")
    )
    daemon_build = next(call for call in gateway.calls if daemon_output in call)
    assert f"posttrain-job.tags={result.tag}" in daemon_build
    assert daemon_build[daemon_build.index("--provenance") + 1] == "false"
    assert daemon_build[daemon_build.index("--sbom") + 1] == "false"
    assert not any("output=type=oci" in argument for call in gateway.calls for argument in call)


def test_local_daemon_loading_rejects_multi_platform_publication(tmp_path: Path) -> None:
    gateway = FakeBuildx()
    request = _request(tmp_path)
    request = replace(
        request,
        publication=replace(request.publication, platforms=("linux/amd64", "linux/arm64")),
    )
    publisher = BuildKitJobImagePublisher(
        bake_file=_definition(tmp_path),
        receipt_root=(tmp_path / "receipts").resolve(),
        gateway=gateway,
    )

    with pytest.raises(ContractError, match="single publication platform"):
        publisher.publish_local_daemon(request)

    assert gateway.calls == []


def test_local_layout_can_use_a_rebuildable_root_separate_from_receipts(tmp_path: Path) -> None:
    gateway = FakeBuildx()
    receipt_root = (tmp_path / "durable-receipts").resolve()
    local_root = (tmp_path / "cache" / "local-layouts").resolve()
    publisher = BuildKitJobImagePublisher(
        bake_file=_definition(tmp_path),
        receipt_root=receipt_root,
        local_layout_root=local_root,
        gateway=gateway,
    )

    result = publisher.publish_local(_request(tmp_path))

    assert result.layout.is_relative_to(local_root)
    assert result.receipt.is_relative_to(receipt_root)
    assert not (receipt_root / "local-layouts").exists()


def test_explicit_local_output_is_user_owned_and_outside_project_cache(tmp_path: Path) -> None:
    gateway = FakeBuildx()
    receipt_root = (tmp_path / "receipts").resolve()
    output = (tmp_path / "exports" / "job-image.oci").resolve()
    request = _request(tmp_path)
    request = replace(request, local_output=output)
    publisher = BuildKitJobImagePublisher(
        bake_file=_definition(tmp_path),
        receipt_root=receipt_root,
        gateway=gateway,
    )

    result = publisher.publish_local(request)

    assert result.layout == output
    assert result.layout.joinpath("index.json").is_file()
    assert result.receipt == output.parent / ".job-image.oci.posttrain-receipt.json"
    assert result.receipt.is_file()
    assert not (receipt_root / "local-layouts").exists()
    assert not (tmp_path / "leases").exists()


def test_credential_free_python_index_is_forwarded_as_a_build_variable(tmp_path: Path) -> None:
    gateway = FakeBuildx()
    publisher = BuildKitJobImagePublisher(
        bake_file=_definition(tmp_path),
        receipt_root=(tmp_path / "receipts").resolve(),
        gateway=gateway,
        python_index_url="https://pypi.example.test/simple/",
    )

    publisher.publish_local(_request(tmp_path))

    assert any("PYTHON_INDEX_URL=https://pypi.example.test/simple/" in call for call in gateway.calls)


@pytest.mark.parametrize(
    "url",
    [
        "https://user:secret@example.test/simple/",
        "https://example.test/simple/?token=secret",
    ],
)
def test_secret_bearing_python_index_is_rejected(tmp_path: Path, url: str) -> None:
    with pytest.raises(ContractError, match="credential-free"):
        BuildKitJobImagePublisher(
            bake_file=_definition(tmp_path),
            receipt_root=(tmp_path / "receipts").resolve(),
            python_index_url=url,
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
    assert build[-1] == "posttrain-job"
    assert f"posttrain-job.contexts.job-context={request.staged_context}" in build
    smoke_calls = [
        call for call in gateway.calls if call and call[-1] == "posttrain-job-smoke" and "--call" not in call
    ]
    assert len(smoke_calls) == 1
    assert gateway.calls.index(smoke_calls[0]) < gateway.calls.index(build)
    assert "--no-cache" in smoke_calls[0]
    assert "--no-cache" not in build
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


def test_publisher_resolves_a_receipt_without_a_staged_context(tmp_path: Path) -> None:
    gateway = FakeBuildx()
    request = _request(tmp_path)
    publisher = BuildKitJobImagePublisher(
        bake_file=_definition(tmp_path),
        receipt_root=(tmp_path / "receipts").resolve(),
        gateway=gateway,
    )
    published = publisher.publish(request)
    resolved = publisher.resolve(
        JobImageResolutionRequest(
            manifest=request.manifest,
            publication=request.publication,
            publication_key=request.publication_key,
        )
    )

    assert resolved is not None
    assert resolved.cache_hit
    assert resolved.image == published.image
    assert sum("--metadata-file" in call for call in gateway.calls) == 1


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


def test_every_bake_call_may_read_the_context_and_the_definitions(tmp_path: Path) -> None:
    """Build definitions ship as package data, outside the project directory.

    buildx refuses to read outside the working directory without an explicit
    entitlement. A checkout hides this, because the definitions sit under the
    same tree as everything else; a wheel install puts them in site-packages,
    where every bake call fails until both directories are granted.
    """
    bake = _definition(tmp_path)
    request = _request(tmp_path)
    gateway = FakeBuildx()
    publisher = BuildKitJobImagePublisher(
        bake_file=bake,
        receipt_root=tmp_path / "receipts",
        gateway=gateway,
    )
    publisher.publish(request)

    bakes = [call for call in gateway.calls if call and call[0] == "bake"]
    assert bakes, "no bake call was made"
    for call in bakes:
        granted = {call[index + 1] for index, item in enumerate(call) if item == "--allow"}
        assert f"fs.read={request.staged_context}" in granted
        assert f"fs.read={bake.parent}" in granted
