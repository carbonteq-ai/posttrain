from __future__ import annotations

import hashlib
from pathlib import Path

from fastapi.testclient import TestClient
from posttrain.execution import JobPackageManifest, RuntimeImageRef
from posttrain.execution_pack import (
    ImagePublicationSpec,
    JobBuilderCapabilities,
    JobContextManifest,
    JobPublicationPlanRequest,
    PackedJobContext,
    publication_key_for,
)
from posttrain_job_builder import (
    BearerTokenAuthorizer,
    FileSystemJobContextStore,
    PrincipalGrant,
    ProjectRepositoryPolicy,
    create_http_app,
)


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
    context = PackedJobContext(root, manifest, "8" * 64, publication_key_for(manifest, publication))
    return JobPublicationPlanRequest(
        manifest,
        publication,
        JobContextManifest.from_packed_context(context),
        "9" * 64,
        "a" * 64,
    )


def _client(tmp_path: Path) -> TestClient:
    token = "test-token"
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
    app = create_http_app(
        store=FileSystemJobContextStore(root=(tmp_path / "store").resolve(), capabilities=capabilities),
        capabilities=capabilities,
        authorizer=BearerTokenAuthorizer(
            {hashlib.sha256(token.encode()).hexdigest(): PrincipalGrant("alice", frozenset({"project"}))}
        ),
        repositories=ProjectRepositoryPolicy("registry.example/posttrain-projects"),
    )
    return TestClient(app)


def test_authenticated_manifest_first_upload_flow(tmp_path: Path) -> None:
    client = _client(tmp_path)
    request = _request(tmp_path)
    headers = {"Authorization": "Bearer test-token", "X-Posttrain-Project": "project"}

    assert client.get("/v1/capabilities").status_code == 401
    assert client.get("/v1/capabilities", headers=headers).json()["api_versions"] == ["v1"]
    plan = client.post("/v1/job-publications:plan", headers=headers, json=request.to_payload())
    assert plan.status_code == 200
    assert plan.json()["state"] == "upload-required"
    for descriptor in request.context.files:
        content = (tmp_path / "context" / descriptor.path).read_bytes()
        response = client.put(
            f"/v1/job-publications/{request.publication_key}/blobs/{descriptor.sha256}",
            headers=headers,
            content=content,
        )
        assert response.status_code == 204
    sealed = client.post(f"/v1/job-publications/{request.publication_key}:seal", headers=headers)
    assert sealed.status_code == 200
    assert sealed.json()["state"] == "queued"
    status = client.get(f"/v1/job-publications/{request.publication_key}", headers=headers)
    assert status.json()["state"] == "queued"
    cancelled = client.post(f"/v1/job-publications/{request.publication_key}:cancel", headers=headers)
    assert cancelled.json()["state"] == "cancelled"


def test_plan_rejects_an_unscoped_repository_before_upload(tmp_path: Path) -> None:
    client = _client(tmp_path)
    request = _request(tmp_path)
    publication = ImagePublicationSpec("registry.example/arbitrary-repository")
    unscoped = JobPublicationPlanRequest(
        request.manifest,
        publication,
        JobContextManifest(
            request.context.package_key,
            publication_key_for(request.manifest, publication),
            request.context.context_digest,
            request.context.files,
        ),
        request.release_manifest_digest,
        request.build_definition_digest,
    )

    response = client.post(
        "/v1/job-publications:plan",
        headers={"Authorization": "Bearer test-token", "X-Posttrain-Project": "project"},
        json=unscoped.to_payload(),
    )

    assert response.status_code == 403
