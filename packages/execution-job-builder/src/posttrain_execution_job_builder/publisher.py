"""HTTP implementation of the existing actual-job publication port."""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from posttrain.common import ContractError
from posttrain.execution import RuntimeImageRef
from posttrain.execution_pack import (
    JobContextManifest,
    JobContextTransferPlan,
    JobImagePublicationRequest,
    JobImageResolutionRequest,
    JobPublicationImage,
    JobPublicationPlanRequest,
    JobPublicationState,
    PublishedJobImage,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class RemoteJobBuilderConfig:
    """Machine-local connection settings for one trusted builder service."""

    endpoint: str
    token: str = field(repr=False)
    release_manifest_digest: str
    build_definition_digest: str
    receipt_root: Path
    request_timeout_seconds: float = 30.0
    poll_timeout_seconds: float = 60.0 * 60.0
    poll_interval_seconds: float = 2.0
    upload_concurrency: int = 1

    def __post_init__(self) -> None:
        if not self.endpoint.startswith(("http://", "https://")) or self.endpoint.rstrip("/") != self.endpoint:
            raise ContractError("remote job builder endpoint must be an absolute URL without a trailing slash")
        if not self.token or self.token != self.token.strip():
            raise ContractError("remote job builder token is invalid")
        if any(_SHA256.fullmatch(value) is None for value in (self.release_manifest_digest, self.build_definition_digest)):
            raise ContractError("remote job builder definition digests must be SHA-256")
        if not self.receipt_root.is_absolute():
            raise ContractError("remote job builder receipt root must be absolute")
        if min(self.request_timeout_seconds, self.poll_timeout_seconds, self.poll_interval_seconds) <= 0:
            raise ContractError("remote job builder timeouts must be positive")
        if self.upload_concurrency <= 0:
            raise ContractError("remote job builder upload concurrency must be positive")


class RemoteJobImagePublisher:
    """Plans before upload and implements ``JobImagePublisher`` over HTTP."""

    def __init__(self, config: RemoteJobBuilderConfig, *, client: httpx.Client | None = None) -> None:
        self._config = config
        self._client = client or httpx.Client(timeout=config.request_timeout_seconds)

    def publish(self, request: JobImagePublicationRequest) -> PublishedJobImage:
        if request.local_output is not None or request.local_tag is not None:
            raise ContractError("remote job builder cannot produce a local OCI image")
        if request.source_context_digest is None:
            raise ContractError("remote job builder requires the packed context digest")
        context = JobContextManifest.from_packed_context(
            _packed_context_view(request, request.source_context_digest)
        )
        plan_request = JobPublicationPlanRequest(
            request.manifest,
            request.publication,
            context,
            self._config.release_manifest_digest,
            self._config.build_definition_digest,
        )
        plan = self._plan(plan_request)
        if plan.publication_key != request.publication_key:
            raise ContractError("remote job builder returned a mismatched publication key")
        if plan.state is JobPublicationState.BLOCKED:
            raise ContractError("remote job builder rejected the publication plan")
        if plan.state is JobPublicationState.UPLOAD_REQUIRED:
            self._upload_missing(request, plan, plan_request.project_id)
            self._seal(request.publication_key, plan_request.project_id)
        return self._wait_for_publication(request, plan_request.project_id)

    def resolve(self, request: JobImageResolutionRequest) -> PublishedJobImage | None:
        status = self._status(request.publication_key, request.manifest.project_id)
        if status is None:
            return None
        return self._published_from_status(
            status,
            package_key=request.package_key,
            publication_key=request.publication_key,
            expected_kind_image=request.manifest.kind_image,
        )

    def _plan(self, request: JobPublicationPlanRequest) -> JobContextTransferPlan:
        response = self._client.post(
            f"{self._config.endpoint}/v1/job-publications:plan",
            headers=self._headers(request.project_id),
            json=request.to_payload(),
        )
        self._raise_for_status(response, "remote job builder plan failed")
        return JobContextTransferPlan.from_payload(response.json())

    def _upload_missing(
        self,
        request: JobImagePublicationRequest,
        plan: JobContextTransferPlan,
        project_id: str,
    ) -> None:
        with ThreadPoolExecutor(max_workers=self._config.upload_concurrency) as executor:
            futures = tuple(
                executor.submit(self._upload_blob, request, descriptor, project_id)
                for descriptor in plan.missing_blobs
            )
            for future in futures:
                future.result()

    def _upload_blob(self, request: JobImagePublicationRequest, descriptor, project_id: str) -> None:
        """Upload one admitted blob, retrying only safe pre-seal requests."""

        path = request.staged_context.joinpath(*descriptor.path.parts)
        if not path.is_file() or path.is_symlink() or path.stat().st_size != descriptor.size_bytes:
            raise ContractError("remote job builder source context differs from its sealed manifest")
        for attempt in range(3):
            try:
                with path.open("rb") as stream:
                    response = self._client.put(
                        f"{self._config.endpoint}/v1/job-publications/{request.publication_key}/blobs/{descriptor.sha256}",
                        headers={**self._headers(project_id), "Content-Length": str(descriptor.size_bytes)},
                        content=stream,
                    )
                if not response.is_error:
                    return
                if response.status_code < 500 or attempt == 2:
                    self._raise_for_status(response, "remote job builder blob upload failed")
            except httpx.TransportError:
                if attempt == 2:
                    raise ContractError("remote job builder blob upload could not reach the service") from None
            time.sleep(0.25 * (attempt + 1))

    def _seal(self, publication_key: str, project_id: str) -> None:
        response = self._client.post(
            f"{self._config.endpoint}/v1/job-publications/{publication_key}:seal",
            headers=self._headers(project_id),
        )
        self._raise_for_status(response, "remote job builder seal failed")

    def _wait_for_publication(self, request: JobImagePublicationRequest, project_id: str) -> PublishedJobImage:
        deadline = time.monotonic() + self._config.poll_timeout_seconds
        while True:
            status = self._status(request.publication_key, project_id)
            if status is not None:
                published = self._published_from_status(
                    status,
                    package_key=request.package_key,
                    publication_key=request.publication_key,
                    expected_kind_image=request.manifest.kind_image,
                )
                if published is not None:
                    return published
                state = status.get("state")
                if state in {JobPublicationState.BLOCKED.value, JobPublicationState.CANCELLED.value, JobPublicationState.FAILED.value}:
                    raise ContractError("remote job builder did not publish the requested image")
            if time.monotonic() >= deadline:
                raise TimeoutError("remote job builder publication timed out")
            time.sleep(self._config.poll_interval_seconds)

    def _status(self, publication_key: str, project_id: str) -> dict[str, object] | None:
        response = self._client.get(
            f"{self._config.endpoint}/v1/job-publications/{publication_key}",
            headers=self._headers(project_id),
        )
        if response.status_code == httpx.codes.NOT_FOUND:
            return None
        self._raise_for_status(response, "remote job builder status lookup failed")
        payload = response.json()
        if not isinstance(payload, dict):
            raise ContractError("remote job builder returned invalid publication status")
        return payload

    def _published_from_status(
        self,
        status: dict[str, object],
        *,
        package_key: str,
        publication_key: str,
        expected_kind_image: RuntimeImageRef,
    ) -> PublishedJobImage | None:
        state = status.get("state")
        image_payload = status.get("image")
        if state not in {JobPublicationState.PUBLISHED.value, JobPublicationState.REUSED.value}:
            return None
        image = JobPublicationImage.from_payload(image_payload)
        if image.kind_image != expected_kind_image:
            raise ContractError("remote job builder returned the wrong kind-image parent")
        return PublishedJobImage(
            package_key=package_key,
            publication_key=publication_key,
            image=image.image,
            kind_image=image.kind_image,
            receipt=self._write_receipt(package_key, publication_key, image),
            cache_hit=image.cache_hit,
        )

    def _write_receipt(self, package_key: str, publication_key: str, image: JobPublicationImage) -> Path:
        self._config.receipt_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._config.receipt_root.chmod(0o700)
        destination = self._config.receipt_root / f"{publication_key}.remote.json"
        payload = {
            "schema": "posttrain.remote-job-image-publication-receipt.v1",
            "package_key": package_key,
            "publication_key": publication_key,
            **image.to_payload(),
        }
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{publication_key}.", dir=self._config.receipt_root)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            temporary.chmod(0o600)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    def _headers(self, project_id: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._config.token}",
            "X-Posttrain-Project": project_id,
        }

    @staticmethod
    def _raise_for_status(response: httpx.Response, message: str) -> None:
        if response.is_error:
            raise ContractError(f"{message}: HTTP {response.status_code}")


def _packed_context_view(request: JobImagePublicationRequest, source_context_digest: str):
    """Adapt the publisher request without extending its stable public shape."""

    from posttrain.execution_pack import PackedJobContext

    return PackedJobContext(
        root=request.staged_context,
        manifest=request.manifest,
        context_digest=source_context_digest,
        publication_key=request.publication_key,
    )


__all__ = ["RemoteJobBuilderConfig", "RemoteJobImagePublisher"]
