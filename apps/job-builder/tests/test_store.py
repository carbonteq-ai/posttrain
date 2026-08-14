from __future__ import annotations

import hashlib
import io
import stat
from pathlib import Path

import pytest
from posttrain.common import ContractError
from posttrain.execution import JobPackageManifest, RuntimeImageRef
from posttrain.execution_pack import (
    ImagePublicationSpec,
    JobBuilderCapabilities,
    JobContextManifest,
    JobPublicationImage,
    JobPublicationPlanRequest,
    JobPublicationState,
    PackedJobContext,
    publication_key_for,
)
from posttrain_job_builder import FileSystemJobContextStore


def _request(tmp_path: Path) -> JobPublicationPlanRequest:
    root = (tmp_path / "context").resolve()
    root.mkdir()
    (root / "a.txt").write_bytes(b"alpha")
    (root / "b.txt").write_bytes(b"bravo")
    manifest = JobPackageManifest(
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
    )
    publication = ImagePublicationSpec("registry.example/posttrain-projects/alice/project/posttrain-job")
    source_context = PackedJobContext(root, manifest, "8" * 64, publication_key_for(manifest, publication))
    return JobPublicationPlanRequest(
        manifest,
        publication,
        JobContextManifest.from_packed_context(source_context),
        "9" * 64,
        "a" * 64,
    )


def _store(tmp_path: Path) -> FileSystemJobContextStore:
    capabilities = JobBuilderCapabilities(
        api_versions=("v1",),
        release_manifest_digests=("9" * 64,),
        build_definition_digests=("a" * 64,),
        platforms=("linux/amd64",),
        max_context_bytes=1024,
        max_file_count=10,
        max_blob_bytes=1024,
        queue_available=True,
    )
    return FileSystemJobContextStore(root=(tmp_path / "store").resolve(), capabilities=capabilities)


def test_filesystem_store_admits_only_missing_blobs_and_seals_a_context(tmp_path: Path) -> None:
    store = _store(tmp_path)
    request = _request(tmp_path)

    plan = store.plan(principal="alice", project_id="project", request=request)

    assert plan.state is JobPublicationState.UPLOAD_REQUIRED
    assert plan.expected_upload_bytes == 10
    for descriptor in plan.missing_blobs:
        payload = (tmp_path / "context" / descriptor.path).read_bytes()
        store.put_blob(
            principal="alice",
            project_id="project",
            publication_key=request.publication_key,
            digest=descriptor.sha256,
            content=io.BytesIO(payload),
            content_length=len(payload),
        )
    receipt = store.seal(principal="alice", project_id="project", publication_key=request.publication_key)

    assert receipt.state is JobPublicationState.QUEUED
    assert receipt.context_manifest_digest == request.context.digest
    assert receipt.uploaded_blob_count == 2
    assert receipt.uploaded_bytes == 10
    record = store.get(principal="alice", project_id="project", publication_key=request.publication_key)
    assert record is not None and record.state is JobPublicationState.QUEUED
    record_path = tmp_path / "store" / "requests" / "alice" / "project" / request.publication_key / "record.json"
    assert stat.S_IMODE(record_path.stat().st_mode) == 0o600
    receipt_path = tmp_path / "store" / "receipts" / "alice" / "project" / f"{request.publication_key}.json"
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o600


def test_filesystem_store_rejects_undeclared_or_mismatched_blob_uploads(tmp_path: Path) -> None:
    store = _store(tmp_path)
    request = _request(tmp_path)
    plan = store.plan(principal="alice", project_id="project", request=request)
    descriptor = plan.missing_blobs[0]

    with pytest.raises(ContractError, match="not declared"):
        store.put_blob(
            principal="alice",
            project_id="project",
            publication_key=request.publication_key,
            digest="f" * 64,
            content=io.BytesIO(b"bad"),
            content_length=3,
        )
    with pytest.raises(ContractError, match="content length"):
        store.put_blob(
            principal="alice",
            project_id="project",
            publication_key=request.publication_key,
            digest=descriptor.sha256,
            content=io.BytesIO(b"bad"),
            content_length=3,
        )
    assert hashlib.sha256((tmp_path / "context" / descriptor.path).read_bytes()).hexdigest() == descriptor.sha256


def test_store_claims_materializes_and_completes_one_sealed_publication(tmp_path: Path) -> None:
    store = _store(tmp_path)
    request = _request(tmp_path)
    plan = store.plan(principal="alice", project_id="project", request=request)
    for descriptor in plan.missing_blobs:
        content = (tmp_path / "context" / descriptor.path).read_bytes()
        store.put_blob(
            principal="alice",
            project_id="project",
            publication_key=request.publication_key,
            digest=descriptor.sha256,
            content=io.BytesIO(content),
            content_length=len(content),
        )
    store.seal(principal="alice", project_id="project", publication_key=request.publication_key)

    claim = store.claim_next()

    assert claim is not None
    staged = store.materialize(claim, (tmp_path / "reconstructed").resolve())
    assert (staged / "a.txt").read_bytes() == b"alpha"
    assert (staged / "b.txt").read_bytes() == b"bravo"
    assert (staged / "wheels" / "framework").is_dir()
    completed = store.complete(
        claim,
        JobPublicationImage(
            RuntimeImageRef("registry.example/posttrain-projects/alice/project/posttrain-job@sha256:" + "3" * 64),
            request.manifest.kind_image,
            cache_hit=False,
        ),
    )

    assert completed.state is JobPublicationState.PUBLISHED
    assert completed.image is not None
    assert store.claim_next() is None
