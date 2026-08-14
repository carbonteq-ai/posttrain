"""Narrow authenticated HTTP admission API for the developer builder.

The API transfers only a sealed context manifest and declared blobs.  It never
accepts build definitions, arbitrary image repositories, BuildKit options, or
client registry credentials.
"""

from __future__ import annotations

import hashlib
import hmac
import io
from collections.abc import Mapping
from dataclasses import dataclass

from fastapi import FastAPI, Header, HTTPException, Request, Response, status
from posttrain.common import ContractError
from posttrain.execution_pack import JobBuilderCapabilities, JobPublicationPlanRequest

from .store import FileSystemJobContextStore


@dataclass(frozen=True, slots=True)
class PrincipalGrant:
    """One opaque bearer token's permitted developer-builder scope."""

    principal: str
    project_ids: frozenset[str]

    def __post_init__(self) -> None:
        if not self.principal or "/" in self.principal or "\\" in self.principal:
            raise ValueError("job builder principal is invalid")
        if not self.project_ids or any(not value or "/" in value or "\\" in value for value in self.project_ids):
            raise ValueError("job builder project grants are invalid")


class BearerTokenAuthorizer:
    """Matches only SHA-256 token digests stored in protected service config."""

    def __init__(self, token_grants: Mapping[str, PrincipalGrant]) -> None:
        if not token_grants:
            raise ValueError("job builder requires at least one token grant")
        if any(len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest) for digest in token_grants):
            raise ValueError("job builder token digest is invalid")
        self._token_grants = dict(token_grants)

    def authenticate(self, authorization: str | None) -> PrincipalGrant:
        if authorization is None or not authorization.startswith("Bearer "):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing bearer token")
        token = authorization.removeprefix("Bearer ")
        if not token or token != token.strip():
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid bearer token")
        observed = hashlib.sha256(token.encode()).hexdigest()
        for expected, grant in self._token_grants.items():
            if hmac.compare_digest(expected, observed):
                return grant
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid bearer token")


class ProjectRepositoryPolicy:
    """Derives the only project-scoped OCI repository a principal may publish."""

    def __init__(self, repository_prefix: str) -> None:
        normalized = repository_prefix.rstrip("/")
        if not normalized or "://" in normalized or "@" in normalized:
            raise ValueError("job builder repository prefix is invalid")
        self._prefix = normalized

    def repository_for(self, grant: PrincipalGrant, project_id: str) -> str:
        if project_id not in grant.project_ids:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "project scope is not authorized")
        return f"{self._prefix}/{grant.principal}/{project_id}/posttrain-job"


def create_http_app(
    *,
    store: FileSystemJobContextStore,
    capabilities: JobBuilderCapabilities,
    authorizer: BearerTokenAuthorizer,
    repositories: ProjectRepositoryPolicy,
) -> FastAPI:
    """Create the v1 API; a separate composition root supplies configuration."""

    app = FastAPI(title="Posttrain developer job builder", version="1")

    @app.get("/health/live")
    def live() -> dict[str, object]:
        return {"status": "live"}

    @app.get("/health/ready")
    def ready() -> Response:
        if not capabilities.queue_available:
            return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.get("/v1/capabilities")
    def get_capabilities(authorization: str | None = Header(default=None)) -> dict[str, object]:
        authorizer.authenticate(authorization)
        return capabilities.to_payload()

    @app.post("/v1/job-publications:plan")
    async def plan(request: Request, authorization: str | None = Header(default=None)) -> dict[str, object]:
        grant = authorizer.authenticate(authorization)
        try:
            publication = JobPublicationPlanRequest.from_payload(await request.json())
        except (ContractError, ValueError) as error:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "invalid job publication plan") from error
        expected_repository = repositories.repository_for(grant, publication.project_id)
        if publication.publication.repository != expected_repository:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "publication repository is not authorized")
        try:
            result = store.plan(principal=grant.principal, project_id=publication.project_id, request=publication)
        except ContractError as error:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "job publication was not admitted") from error
        return result.to_payload()

    @app.put("/v1/job-publications/{publication_key}/blobs/{digest}")
    async def put_blob(
        publication_key: str,
        digest: str,
        request: Request,
        authorization: str | None = Header(default=None),
        content_length: int | None = Header(default=None),
        x_posttrain_project: str | None = Header(default=None),
    ) -> Response:
        grant = authorizer.authenticate(authorization)
        if x_posttrain_project is None:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "project scope is required")
        repositories.repository_for(grant, x_posttrain_project)
        if content_length is None or content_length < 0:
            raise HTTPException(status.HTTP_411_LENGTH_REQUIRED, "content length is required")
        if content_length > capabilities.max_blob_bytes:
            raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "blob exceeds the admitted size limit")
        try:
            store.put_blob(
                principal=grant.principal,
                project_id=x_posttrain_project,
                publication_key=publication_key,
                digest=digest,
                content=io.BytesIO(await request.body()),
                content_length=content_length,
            )
        except ContractError as error:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "job publication blob was rejected") from error
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.post("/v1/job-publications/{publication_key}:seal")
    def seal(
        publication_key: str,
        authorization: str | None = Header(default=None),
        x_posttrain_project: str | None = Header(default=None),
    ) -> dict[str, object]:
        grant = authorizer.authenticate(authorization)
        if x_posttrain_project is None:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "project scope is required")
        repositories.repository_for(grant, x_posttrain_project)
        try:
            return store.seal(
                principal=grant.principal,
                project_id=x_posttrain_project,
                publication_key=publication_key,
            ).to_payload()
        except ContractError as error:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "job publication could not be sealed") from error

    @app.get("/v1/job-publications/{publication_key}")
    def get_publication(
        publication_key: str,
        authorization: str | None = Header(default=None),
        x_posttrain_project: str | None = Header(default=None),
    ) -> dict[str, object]:
        grant = authorizer.authenticate(authorization)
        if x_posttrain_project is None:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "project scope is required")
        repositories.repository_for(grant, x_posttrain_project)
        try:
            record = store.get(principal=grant.principal, project_id=x_posttrain_project, publication_key=publication_key)
        except ContractError as error:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "job publication lookup was rejected") from error
        if record is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "job publication does not exist")
        return {
            "schema": "posttrain.job-publication-status.v1",
            "publication_key": record.request.publication_key,
            "state": record.state.value,
            "safe_error_code": record.safe_error_code,
            "image": record.image.to_payload() if record.image is not None else None,
        }

    @app.post("/v1/job-publications/{publication_key}:cancel")
    def cancel(
        publication_key: str,
        authorization: str | None = Header(default=None),
        x_posttrain_project: str | None = Header(default=None),
    ) -> dict[str, object]:
        grant = authorizer.authenticate(authorization)
        if x_posttrain_project is None:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "project scope is required")
        repositories.repository_for(grant, x_posttrain_project)
        try:
            record = store.cancel(principal=grant.principal, project_id=x_posttrain_project, publication_key=publication_key)
        except ContractError as error:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "job publication could not be cancelled") from error
        return {
            "schema": "posttrain.job-publication-status.v1",
            "publication_key": record.request.publication_key,
            "state": record.state.value,
            "safe_error_code": record.safe_error_code,
            "image": record.image.to_payload() if record.image is not None else None,
        }

    return app


__all__ = ["BearerTokenAuthorizer", "PrincipalGrant", "ProjectRepositoryPolicy", "create_http_app"]
