"""Production composition for the isolated developer job-build service.

This module deliberately accepts only protected, server-owned configuration.
The client supplies a sealed context through :mod:`posttrain_job_builder.http`;
it never chooses a Dockerfile, registry credential, or BuildKit setting.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI
from posttrain.common import ContractError
from posttrain.execution_pack import JobBuilderCapabilities
from posttrain.runtime_images import JOB_BAKE_FILE, cached_definition_root, published_manifest_digest
from posttrain_execution_buildkit import BuildKitJobImagePublisher, job_build_definition_digest

from .http import BearerTokenAuthorizer, InfrastructureGrant, ProjectRepositoryPolicy, create_http_app
from .store import FileSystemJobContextStore
from .worker import JobBuildWorker


@dataclass(frozen=True, slots=True)
class ServiceConfig:
    """The server-owned paths, limits, credentials, and BuildKit selection."""

    store_root: Path
    staging_root: Path
    receipt_root: Path
    repository_prefix: str
    infrastructure_grants: Mapping[str, InfrastructureGrant]
    builder: str
    python_index_url: str | None
    max_context_bytes: int = 64 * 1024 * 1024
    max_file_count: int = 10_000
    max_blob_bytes: int = 16 * 1024 * 1024
    poll_seconds: float = 0.25


def load_config(environ: Mapping[str, str] | None = None) -> ServiceConfig:
    """Load the small protected JSON document selected by the service unit."""

    values = os.environ if environ is None else environ
    path_value = values.get("POSTTRAIN_JOB_BUILDER_CONFIG")
    if path_value is None:
        raise ContractError("POSTTRAIN_JOB_BUILDER_CONFIG is required")
    path = Path(path_value)
    if not path.is_absolute() or not path.is_file() or path.stat().st_mode & 0o077:
        raise ContractError("job builder configuration must be a protected absolute file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError("job builder configuration is invalid") from error
    if not isinstance(payload, dict):
        raise ContractError("job builder configuration is invalid")
    required = {
        "store_root",
        "staging_root",
        "receipt_root",
        "repository_prefix",
        "infrastructure_grants",
        "builder",
    }
    optional = {"python_index_url", "max_context_bytes", "max_file_count", "max_blob_bytes", "poll_seconds"}
    if set(payload) - required - optional or not required.issubset(payload):
        raise ContractError("job builder configuration has unsupported fields")
    grants_raw = payload["infrastructure_grants"]
    if not isinstance(grants_raw, dict):
        raise ContractError("job builder infrastructure grants are invalid")
    grants: dict[str, InfrastructureGrant] = {}
    for digest, grant in grants_raw.items():
        if not isinstance(digest, str) or not isinstance(grant, dict) or set(grant) != {"principal"}:
            raise ContractError("job builder infrastructure grants are invalid")
        principal = grant["principal"]
        if not isinstance(principal, str):
            raise ContractError("job builder infrastructure grants are invalid")
        try:
            grants[digest] = InfrastructureGrant(principal)
        except ValueError as error:
            raise ContractError("job builder infrastructure grants are invalid") from error
    return ServiceConfig(
        store_root=_absolute_path(payload["store_root"], "store_root"),
        staging_root=_absolute_path(payload["staging_root"], "staging_root"),
        receipt_root=_absolute_path(payload["receipt_root"], "receipt_root"),
        repository_prefix=_string(payload["repository_prefix"], "repository_prefix"),
        infrastructure_grants=grants,
        builder=_string(payload["builder"], "builder"),
        python_index_url=_optional_string(payload.get("python_index_url"), "python_index_url"),
        max_context_bytes=_positive_int(payload.get("max_context_bytes", 64 * 1024 * 1024), "max_context_bytes"),
        max_file_count=_positive_int(payload.get("max_file_count", 10_000), "max_file_count"),
        max_blob_bytes=_positive_int(payload.get("max_blob_bytes", 16 * 1024 * 1024), "max_blob_bytes"),
        poll_seconds=_positive_float(payload.get("poll_seconds", 0.25), "poll_seconds"),
    )


def create_app(config: ServiceConfig | None = None) -> FastAPI:
    """Compose the HTTP admission surface and one background BuildKit worker."""

    config = config or load_config()
    definition_root = cached_definition_root()
    bake_file = definition_root / JOB_BAKE_FILE
    capabilities = JobBuilderCapabilities(
        api_versions=("v1",),
        release_manifest_digests=(published_manifest_digest(),),
        build_definition_digests=(job_build_definition_digest(bake_file),),
        platforms=("linux/amd64",),
        max_context_bytes=config.max_context_bytes,
        max_file_count=config.max_file_count,
        max_blob_bytes=config.max_blob_bytes,
        queue_available=True,
    )
    store = FileSystemJobContextStore(root=config.store_root, capabilities=capabilities)
    publisher = BuildKitJobImagePublisher(
        bake_file=bake_file,
        receipt_root=config.receipt_root,
        builder=config.builder,
        python_index_url=config.python_index_url,
    )
    worker = JobBuildWorker(store=store, publisher=publisher, staging_root=config.staging_root)
    app = create_http_app(
        store=store,
        capabilities=capabilities,
        authorizer=BearerTokenAuthorizer(config.infrastructure_grants),
        repositories=ProjectRepositoryPolicy(config.repository_prefix),
    )
    app.router.lifespan_context = _worker_lifespan(worker, config.poll_seconds)
    return app


def _worker_lifespan(worker: JobBuildWorker, poll_seconds: float):
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        stopping = threading.Event()
        thread = threading.Thread(target=_run_worker, args=(worker, stopping, poll_seconds), daemon=True)
        thread.start()
        try:
            yield
        finally:
            stopping.set()
            thread.join(timeout=max(2.0, poll_seconds * 4))

    return lifespan


def _run_worker(worker: JobBuildWorker, stopping: threading.Event, poll_seconds: float) -> None:
    while not stopping.is_set():
        if worker.run_one() is None:
            stopping.wait(poll_seconds)


def _absolute_path(value: object, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ContractError(f"job builder {name} is invalid")
    path = Path(value)
    if not path.is_absolute():
        raise ContractError(f"job builder {name} must be absolute")
    return path


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"job builder {name} is invalid")
    return value


def _optional_string(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _string(value, name)


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ContractError(f"job builder {name} must be positive")
    return value


def _positive_float(value: object, name: str) -> float:
    if not isinstance(value, (float, int)) or isinstance(value, bool) or value <= 0:
        raise ContractError(f"job builder {name} must be positive")
    return float(value)
