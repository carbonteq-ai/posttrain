from __future__ import annotations

import json
from pathlib import Path

import httpx
from posttrain.execution import JobPackageManifest, RuntimeImageRef
from posttrain.execution_pack import (
    ImagePublicationSpec,
    JobContextManifest,
    JobContextTransferPlan,
    JobImagePublicationRequest,
    JobPublicationState,
    PackedJobContext,
)
from posttrain_execution_job_builder import RemoteJobBuilderConfig, RemoteJobImagePublisher


def _request(tmp_path: Path) -> JobImagePublicationRequest:
    context = (tmp_path / "context").resolve()
    context.mkdir()
    (context / "source.py").write_bytes(b"print('hello')\n")
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
    packed = PackedJobContext(context, manifest, "8" * 64, "9" * 64)
    return JobImagePublicationRequest(
        manifest=manifest,
        staged_context=context,
        publication=publication,
        source_context_digest=packed.context_digest,
    )


def test_remote_publisher_plans_uploads_seals_polls_and_writes_a_local_receipt(tmp_path: Path) -> None:
    request = _request(tmp_path)
    context = JobContextManifest.from_packed_context(
        PackedJobContext(
            request.staged_context, request.manifest, request.source_context_digest or "", request.publication_key
        )
    )
    calls: list[str] = []

    def handler(http_request: httpx.Request) -> httpx.Response:
        calls.append(f"{http_request.method} {http_request.url.path}")
        if http_request.url.path == "/v1/job-publications:plan":
            return httpx.Response(
                200,
                json=JobContextTransferPlan(
                    request.publication_key,
                    JobPublicationState.UPLOAD_REQUIRED,
                    context.files,
                ).to_payload(),
            )
        if "/blobs/" in http_request.url.path:
            assert http_request.content == b"print('hello')\n"
            return httpx.Response(204)
        if http_request.url.path.endswith(":seal"):
            return httpx.Response(200, json={})
        if http_request.url.path.startswith("/v1/job-publications/"):
            return httpx.Response(
                200,
                json={
                    "schema": "posttrain.job-publication-status.v1",
                    "publication_key": request.publication_key,
                    "state": "published",
                    "safe_error_code": None,
                    "image": {
                        "image": "registry.example/posttrain-projects/alice/project/posttrain-job@sha256:" + "3" * 64,
                        "kind_image": request.manifest.kind_image.value,
                        "cache_hit": False,
                    },
                },
            )
        raise AssertionError(http_request.url.path)

    publisher = RemoteJobImagePublisher(
        RemoteJobBuilderConfig(
            endpoint="https://job-builder.example",
            token="redacted",
            release_manifest_digest="a" * 64,
            build_definition_digest="b" * 64,
            receipt_root=(tmp_path / "receipts").resolve(),
            poll_interval_seconds=0.001,
        ),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    result = publisher.publish(request)

    assert result.publication_key == request.publication_key
    assert result.kind_image == request.manifest.kind_image
    assert result.receipt.is_file()
    assert json.loads(result.receipt.read_text(encoding="utf-8"))["image"] == result.image.value
    assert calls == [
        "POST /v1/job-publications:plan",
        f"PUT /v1/job-publications/{request.publication_key}/blobs/{context.files[0].sha256}",
        f"POST /v1/job-publications/{request.publication_key}:seal",
        f"GET /v1/job-publications/{request.publication_key}",
    ]
