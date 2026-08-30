from __future__ import annotations

from pathlib import Path, PurePosixPath

import pytest
from posttrain.common import ContractError
from posttrain.execution import JobPackageManifest, RuntimeImageRef
from posttrain.execution_pack import (
    ContextFile,
    ImagePublicationSpec,
    JobBuilderCapabilities,
    JobContextManifest,
    JobContextTransferPlan,
    JobContextTransferReceipt,
    JobPublicationPlanRequest,
    JobPublicationState,
    PackedJobContext,
    publication_key_for,
)


def _context(tmp_path: Path) -> PackedJobContext:
    root = (tmp_path / "context").resolve()
    root.mkdir(exist_ok=True)
    (root / "b.txt").write_text("b", encoding="utf-8")
    (root / "a.txt").write_text("a", encoding="utf-8")
    return PackedJobContext(
        root=root,
        manifest=JobPackageManifest(
            project_id="project",
            work_package_id="package",
            job_id="job",
            job_definition_id="job-definition",
            job_kind="eval.general",
            resolved_inputs_digest="a" * 64,
            framework_source_digest="b" * 64,
            project_source_digest="c" * 64,
            runtime_dependencies_digest="d" * 64,
            code_requirements_digest="e" * 64,
            resolved_config_digest="f" * 64,
            project_config_digest="7" * 64,
            universal_image=RuntimeImageRef("registry.example/base@sha256:" + "1" * 64),
            kind_image=RuntimeImageRef("registry.example/kind@sha256:" + "2" * 64),
            runtime_variant="eval-py313",
        ),
        context_digest="8" * 64,
        publication_key="9" * 64,
    )


def test_context_manifest_is_canonical_and_content_addressed(tmp_path: Path) -> None:
    manifest = JobContextManifest.from_packed_context(_context(tmp_path))

    assert [file.path.as_posix() for file in manifest.files] == ["a.txt", "b.txt"]
    assert manifest.total_bytes == 2
    assert manifest.digest == JobContextManifest.from_packed_context(_context(tmp_path)).digest
    assert b'"schema":"posttrain.job-context-manifest.v1"' in manifest.to_bytes()


def test_context_manifest_rejects_unsafe_paths_and_links(tmp_path: Path) -> None:
    with pytest.raises(ContractError, match="safe relative"):
        ContextFile(PurePosixPath("../secret"), "a" * 64, 1, 0o600)

    context = _context(tmp_path)
    (context.root / "outside").symlink_to(tmp_path)
    with pytest.raises(ContractError, match="symbolic"):
        JobContextManifest.from_packed_context(context)


def test_remote_transfer_values_bind_the_existing_package_identity(tmp_path: Path) -> None:
    context = _context(tmp_path)
    manifest = JobContextManifest.from_packed_context(context)
    publication = ImagePublicationSpec("registry.example/posttrain-projects/alice/project/posttrain-job")
    request = JobPublicationPlanRequest(
        manifest=context.manifest,
        publication=publication,
        context=JobContextManifest(
            package_key=manifest.package_key,
            publication_key=publication_key_for(context.manifest, publication),
            context_digest=manifest.context_digest,
            files=manifest.files,
        ),
        release_manifest_digest="a" * 64,
        build_definition_digest="b" * 64,
        allow_deferred_qualification=True,
    )

    assert request.package_key == context.manifest.package_key
    assert request.project_id == "project"
    assert JobPublicationPlanRequest.from_payload(request.to_payload()).allow_deferred_qualification
    plan = JobContextTransferPlan(
        publication_key=request.publication_key,
        state=JobPublicationState.UPLOAD_REQUIRED,
        missing_blobs=request.context.files,
    )
    assert plan.expected_upload_bytes == 2
    receipt = JobContextTransferReceipt(
        publication_key=request.publication_key,
        state=JobPublicationState.PUBLISHED,
        context_manifest_digest=request.context.digest,
        source_context_digest=request.context.context_digest,
        declared_file_count=2,
        declared_bytes=2,
        uploaded_blob_count=1,
        uploaded_bytes=1,
        reused_blob_count=1,
        reused_bytes=1,
        receipt_digest="c" * 64,
    )
    assert receipt.uploaded_bytes + receipt.reused_bytes == receipt.declared_bytes


def test_remote_transfer_values_reject_mismatched_identity_or_accounting(tmp_path: Path) -> None:
    context = _context(tmp_path)
    manifest = JobContextManifest.from_packed_context(context)
    with pytest.raises(ContractError, match="publication key"):
        JobPublicationPlanRequest(
            manifest=context.manifest,
            publication=ImagePublicationSpec("registry.example/posttrain-projects/alice/project/posttrain-job"),
            context=manifest,
            release_manifest_digest="a" * 64,
            build_definition_digest="b" * 64,
        )
    with pytest.raises(ContractError, match="counters exceed"):
        JobContextTransferReceipt(
            publication_key="a" * 64,
            state=JobPublicationState.PUBLISHED,
            context_manifest_digest="b" * 64,
            source_context_digest="c" * 64,
            declared_file_count=1,
            declared_bytes=1,
            uploaded_blob_count=1,
            uploaded_bytes=1,
            reused_blob_count=1,
            reused_bytes=1,
        )
    capabilities = JobBuilderCapabilities(
        api_versions=("v1",),
        release_manifest_digests=("a" * 64,),
        build_definition_digests=("b" * 64,),
        platforms=("linux/amd64",),
        max_context_bytes=1,
        max_file_count=1,
        max_blob_bytes=1,
        queue_available=True,
    )
    assert capabilities.queue_available
