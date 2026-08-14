from __future__ import annotations

import hashlib
import io
from pathlib import Path

from posttrain.execution import JobPackageManifest, RuntimeImageRef
from posttrain.execution_pack import (
    ImagePublicationSpec,
    JobBuilderCapabilities,
    JobContextManifest,
    JobPublicationPlanRequest,
    JobPublicationState,
    PackedJobContext,
    PublishedJobImage,
    publication_key_for,
)
from posttrain_job_builder import FileSystemJobContextStore, JobBuildWorker


def _request(tmp_path: Path) -> JobPublicationPlanRequest:
    root = (tmp_path / "context").resolve()
    root.mkdir()
    (root / "source.py").write_bytes(b"print('hello')\n")
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
    packed = PackedJobContext(root, manifest, "8" * 64, publication_key_for(manifest, publication))
    return JobPublicationPlanRequest(
        manifest,
        publication,
        JobContextManifest.from_packed_context(packed),
        "9" * 64,
        "a" * 64,
    )


def _store(tmp_path: Path) -> FileSystemJobContextStore:
    return FileSystemJobContextStore(
        root=(tmp_path / "store").resolve(),
        capabilities=JobBuilderCapabilities(
            api_versions=("v1",),
            release_manifest_digests=("9" * 64,),
            build_definition_digests=("a" * 64,),
            platforms=("linux/amd64",),
            max_context_bytes=1024,
            max_file_count=10,
            max_blob_bytes=1024,
            queue_available=True,
        ),
    )


class FakePublisher:
    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.seen_context: Path | None = None

    def publish(self, request):
        self.seen_context = request.staged_context
        assert (request.staged_context / "source.py").read_bytes() == b"print('hello')\n"
        assert (request.staged_context / "wheels" / "framework").is_dir()
        receipt = (self.tmp_path / "publisher-receipt.json").resolve()
        receipt.write_text("{}\n", encoding="utf-8")
        return PublishedJobImage(
            package_key=request.package_key,
            publication_key=request.publication_key,
            image=RuntimeImageRef("registry.example/posttrain-projects/alice/project/posttrain-job@sha256:" + "3" * 64),
            kind_image=request.manifest.kind_image,
            receipt=receipt,
            cache_hit=False,
        )

    def resolve(self, request):
        return None


def test_worker_builds_only_a_sealed_context_and_retains_safe_result(tmp_path: Path) -> None:
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
    publisher = FakePublisher(tmp_path)

    result = JobBuildWorker(store, publisher, (tmp_path / "staging").resolve()).run_one()

    assert result is not None and result.state is JobPublicationState.PUBLISHED
    assert result.image is not None and result.image.image.value.endswith("3" * 64)
    assert publisher.seen_context is not None and not publisher.seen_context.exists()
    assert hashlib.sha256(b"print('hello')\n").hexdigest() == request.context.files[0].sha256
