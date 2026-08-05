"""BuildKit adapter for framework-owned actual-job OCI images."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from posttrain.common import ContractError, JsonValue
from posttrain.execution import RuntimeImageRef
from posttrain.execution_pack import (
    JobImagePublicationRequest,
    LocalPublishedJobImage,
    PublishedJobImage,
)

from .builder import (
    BuildxCli,
    BuildxGateway,
    RemoteImageNotFoundError,
)

_SCHEMA = "posttrain.job-image-publication-receipt.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_BUILDER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/@-]*$")
_PUBLISHED_TARGET = "posttrain-job"
_SMOKE_TARGET = "posttrain-job-smoke"
_URL_USERINFO = re.compile(r"https?://[^/\s:@]+(?::[^@/\s]*)?@")
_SENSITIVE_QUERY = re.compile(
    r"[?&](?:access[_-]?token|api[_-]?key|auth|credential|password|secret|token)=",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class _PublicationReceipt:
    package_key: str
    publication_key: str
    image: RuntimeImageRef
    kind_image: RuntimeImageRef
    repository: str
    platforms: tuple[str, ...]
    compression: str
    compression_level: int
    provenance: bool
    sbom: bool
    build_definition_digest: str
    path: Path


class BuildKitJobImagePublisher:
    """Publish and remotely verify one content-addressed actual-job image."""

    def __init__(
        self,
        *,
        bake_file: Path,
        receipt_root: Path,
        gateway: BuildxGateway | None = None,
        builder: str | None = None,
        python_index_url: str | None = None,
    ) -> None:
        if not bake_file.is_absolute() or not bake_file.is_file():
            raise ValueError("job image Bake file must be an existing absolute path")
        if not (bake_file.parent / "Dockerfile").is_file():
            raise ValueError("job image Bake file must be beside a Dockerfile")
        if not receipt_root.is_absolute():
            raise ValueError("job image receipt root must be absolute")
        if builder is not None and not _SAFE_BUILDER.fullmatch(builder):
            raise ContractError("job image BuildKit builder name is invalid")
        if python_index_url is not None and (
            not python_index_url.startswith(("http://", "https://"))
            or _URL_USERINFO.search(python_index_url)
            or _SENSITIVE_QUERY.search(python_index_url)
        ):
            raise ContractError("job image Python index URL must be credential-free HTTP(S)")
        self._bake_file = bake_file
        self._definition_root = bake_file.parent
        self._definition_digest = _build_definition_digest(bake_file.parent)
        self._receipt_root = receipt_root
        self._gateway = gateway or BuildxCli()
        self._builder = builder
        self._python_index_url = python_index_url or ""

    def publish(
        self,
        request: JobImagePublicationRequest,
    ) -> PublishedJobImage:
        _enforce_qualification_policy(request)
        receipt_path = self._receipt_root / f"{request.publication_key}.json"
        if receipt_path.is_file():
            receipt = self._load_receipt(receipt_path)
            self._ensure_receipt_matches(request, receipt)
            try:
                self._verify_remote(receipt.image)
            except RemoteImageNotFoundError:
                # An immutable digest that the registry garbage-collected no
                # longer proves a runnable publication. Remove only that exact
                # stale cache entry and reproduce it from the same inputs.
                receipt_path.unlink()
            else:
                return _published(receipt, cache_hit=True)

        self._check_definition(request)
        self._receipt_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        metadata = self._receipt_root / f".metadata-{uuid.uuid4().hex}.json"
        try:
            # Build the smoke target separately and uncached. Affected BuildKit
            # releases can reuse a stale local named-context source across
            # content-addressed directories, even when PACKAGE_KEY and the COPY
            # destination both change. The verified smoke build establishes the
            # current context layers; the publication build can then reuse those
            # exact layers without rebuilding the package a second time.
            self._gateway.invoke(self._smoke_arguments(request))
            self._gateway.invoke(self._build_arguments(request, metadata))
            image = RuntimeImageRef(f"{request.publication.repository}@sha256:{_metadata_digest(metadata)}")
            self._verify_remote(image)
            receipt = _PublicationReceipt(
                package_key=request.package_key,
                publication_key=request.publication_key,
                image=image,
                kind_image=request.manifest.kind_image,
                repository=request.publication.repository,
                platforms=request.publication.platforms,
                compression=request.publication.compression,
                compression_level=request.publication.compression_level,
                provenance=request.publication.provenance,
                sbom=request.publication.sbom,
                build_definition_digest=self._definition_digest,
                path=receipt_path,
            )
            self._write_receipt(receipt)
            return _published(receipt, cache_hit=False)
        finally:
            metadata.unlink(missing_ok=True)

    def publish_local(self, request: JobImagePublicationRequest) -> LocalPublishedJobImage:
        """Build the verified actual-job image into an OCI layout, never a registry."""

        _enforce_qualification_policy(request)
        layout = self._receipt_root / "local-layouts" / request.publication_key
        receipt = self._receipt_root / f"{request.publication_key}.local.json"
        tag = f"posttrain-local:{request.publication_key}"
        if layout.joinpath("index.json").is_file() and receipt.is_file():
            return LocalPublishedJobImage(request.package_key, request.publication_key, layout, tag, receipt, True)
        self._check_definition(request, local=True)
        self._gateway.invoke(self._smoke_arguments(request))
        layout.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = layout.parent / f".{request.publication_key}-{uuid.uuid4().hex}"
        try:
            self._gateway.invoke(
                [
                    "bake",
                    "--file",
                    str(self._bake_file),
                    *self._entitlement_arguments(request),
                    *self._builder_arguments(),
                    "--progress",
                    "plain",
                    *self._context_arguments(request),
                    "--set",
                    f"{_PUBLISHED_TARGET}.output=type=oci,dest={temporary},tar=false",
                    *self._variable_arguments(request),
                    _PUBLISHED_TARGET,
                ]
            )
            if not temporary.joinpath("index.json").is_file():
                raise ContractError("local OCI publication did not produce index.json")
            _remove_local_output(layout)
            temporary.replace(layout)
            receipt.write_text(
                json.dumps(
                    {
                        "package_key": request.package_key,
                        "publication_key": request.publication_key,
                        "layout": str(layout),
                        "tag": tag,
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            receipt.chmod(0o600)
            return LocalPublishedJobImage(request.package_key, request.publication_key, layout, tag, receipt, False)
        finally:
            _remove_local_output(temporary)

    def _check_definition(self, request: JobImagePublicationRequest, *, local: bool = False) -> None:
        for target in (_PUBLISHED_TARGET, _SMOKE_TARGET):
            local_output = (
                ("--set", f"{_PUBLISHED_TARGET}.output=type=cacheonly") if local and target == _PUBLISHED_TARGET else ()
            )
            self._gateway.invoke(
                (
                    "bake",
                    "--file",
                    str(self._bake_file),
                    *self._entitlement_arguments(request),
                    *self._builder_arguments(),
                    *self._context_arguments(request),
                    *local_output,
                    *self._variable_arguments(request),
                    "--call",
                    "check",
                    target,
                )
            )

    def _build_arguments(
        self,
        request: JobImagePublicationRequest,
        metadata: Path,
    ) -> list[str]:
        output = (
            "type=image,push=true,"
            f"compression={request.publication.compression},"
            f"compression-level={request.publication.compression_level},"
            "force-compression=true,oci-mediatypes=true"
        )
        platforms = ",".join(request.publication.platforms)
        return [
            "bake",
            "--file",
            str(self._bake_file),
            *self._entitlement_arguments(request),
            *self._builder_arguments(),
            "--progress",
            "plain",
            "--metadata-file",
            str(metadata),
            *self._context_arguments(request),
            "--set",
            f"{_PUBLISHED_TARGET}.output={output}",
            "--set",
            f"{_PUBLISHED_TARGET}.platform={platforms}",
            "--set",
            f"{_SMOKE_TARGET}.platform={platforms}",
            "--provenance",
            "mode=max",
            "--sbom",
            "true",
            *self._variable_arguments(request),
            _PUBLISHED_TARGET,
        ]

    def _smoke_arguments(
        self,
        request: JobImagePublicationRequest,
    ) -> list[str]:
        platforms = ",".join(request.publication.platforms)
        return [
            "bake",
            "--file",
            str(self._bake_file),
            *self._entitlement_arguments(request),
            *self._builder_arguments(),
            "--progress",
            "plain",
            "--no-cache",
            *self._context_arguments(request),
            "--set",
            f"{_SMOKE_TARGET}.platform={platforms}",
            *self._variable_arguments(request),
            _SMOKE_TARGET,
        ]

    def _builder_arguments(self) -> list[str]:
        return ["--builder", self._builder] if self._builder is not None else []

    def _entitlement_arguments(self, request: JobImagePublicationRequest) -> list[str]:
        """Grant read access to both directories the build reads from.

        The staged context holds the job. The build definitions ship as package
        data, so when the framework is installed rather than checked out they
        sit outside the project entirely, and buildx refuses to read them
        without being told to.
        """
        return [
            "--allow",
            f"fs.read={request.staged_context}",
            "--allow",
            f"fs.read={self._definition_root}",
        ]

    def _context_arguments(self, request: JobImagePublicationRequest) -> list[str]:
        return [
            "--set",
            f"{_PUBLISHED_TARGET}.context={self._definition_root}",
            "--set",
            f"{_SMOKE_TARGET}.context={self._definition_root}",
            "--set",
            f"{_PUBLISHED_TARGET}.contexts.job-context={request.staged_context}",
            "--set",
            f"{_SMOKE_TARGET}.contexts.job-context={request.staged_context}",
        ]

    def _variable_arguments(
        self,
        request: JobImagePublicationRequest,
    ) -> list[str]:
        manifest = request.manifest
        variables: Mapping[str, str] = {
            "ALLOW_DEFERRED_QUALIFICATION": "1" if request.allow_deferred_qualification else "0",
            "CODE_REQUIREMENTS_DIGEST": manifest.code_requirements_digest,
            "FRAMEWORK_SOURCE_DIGEST": manifest.framework_source_digest,
            "IMAGE_REPOSITORY": request.publication.repository,
            "IMAGE_TAG": request.publication_key,
            "JOB_KIND": manifest.job_kind,
            "PACKAGE_KEY": manifest.package_key,
            "POSTTRAIN_KIND_IMAGE": manifest.kind_image.value,
            "PYTHON_INDEX_URL": self._python_index_url,
            "PROJECT_CONFIG_DIGEST": manifest.project_config_digest,
            "PROJECT_SOURCE_DIGEST": manifest.project_source_digest,
            "RESOLVED_CONFIG_DIGEST": manifest.resolved_config_digest,
            "RESOLVED_INPUTS_DIGEST": manifest.resolved_inputs_digest,
            "RUNTIME_DEPENDENCIES_DIGEST": (manifest.runtime_dependencies_digest),
            "RUNTIME_VARIANT": manifest.runtime_variant,
        }
        arguments: list[str] = []
        for name, value in sorted(variables.items()):
            arguments.extend(("--var", f"{name}={value}"))
        return arguments

    def _verify_remote(self, image: RuntimeImageRef) -> None:
        output = self._gateway.invoke(
            (
                "imagetools",
                "inspect",
                image.value,
                "--format",
                "{{json .Manifest.Digest}}",
            )
        )
        try:
            observed = json.loads(output)
        except json.JSONDecodeError as error:
            raise RuntimeError("Buildx returned invalid remote actual-job image metadata") from error
        expected = image.value.rsplit("@", 1)[1]
        if observed != expected:
            raise RuntimeError(f"published actual-job digest mismatch: expected {expected}, observed {observed}")

    def _write_receipt(self, receipt: _PublicationReceipt) -> None:
        payload: dict[str, JsonValue] = {
            "schema": _SCHEMA,
            "package_key": receipt.package_key,
            "publication_key": receipt.publication_key,
            "image": receipt.image.value,
            "kind_image": receipt.kind_image.value,
            "repository": receipt.repository,
            "platforms": list(receipt.platforms),
            "compression": receipt.compression,
            "compression_level": receipt.compression_level,
            "provenance": receipt.provenance,
            "sbom": receipt.sbom,
            "build_definition_digest": receipt.build_definition_digest,
        }
        encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
        temporary = self._receipt_root / f".receipt-{uuid.uuid4().hex}.tmp"
        descriptor = os.open(
            temporary,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
        try:
            os.write(descriptor, encoded)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.link(temporary, receipt.path)
        except FileExistsError:
            existing = self._load_receipt(receipt.path)
            if existing != receipt:
                raise ContractError("job image publication receipt conflicts with an existing publication") from None
        finally:
            temporary.unlink(missing_ok=True)

    def _load_receipt(self, path: Path) -> _PublicationReceipt:
        if path.stat().st_mode & 0o077:
            raise ContractError(f"job image publication receipt must not be accessible by group or other: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ContractError(f"job image publication receipt is invalid: {path}") from error
        fields = {
            "schema",
            "package_key",
            "publication_key",
            "image",
            "kind_image",
            "repository",
            "platforms",
            "compression",
            "compression_level",
            "provenance",
            "sbom",
            "build_definition_digest",
        }
        if not isinstance(payload, dict) or set(payload) != fields or payload.get("schema") != _SCHEMA:
            raise ContractError(f"job image publication receipt schema is unsupported: {path}")
        try:
            platforms = payload["platforms"]
            if not isinstance(platforms, list) or not all(isinstance(value, str) for value in platforms):
                raise TypeError("platforms")
            receipt = _PublicationReceipt(
                package_key=str(payload["package_key"]),
                publication_key=str(payload["publication_key"]),
                image=RuntimeImageRef(str(payload["image"])),
                kind_image=RuntimeImageRef(str(payload["kind_image"])),
                repository=str(payload["repository"]),
                platforms=tuple(platforms),
                compression=str(payload["compression"]),
                compression_level=int(payload["compression_level"]),
                provenance=_exact_bool(payload["provenance"]),
                sbom=_exact_bool(payload["sbom"]),
                build_definition_digest=str(payload["build_definition_digest"]),
                path=path,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ContractError(f"job image publication receipt fields are invalid: {path}") from error
        if (
            not _SHA256.fullmatch(receipt.package_key)
            or not _SHA256.fullmatch(receipt.publication_key)
            or not _SHA256.fullmatch(receipt.build_definition_digest)
        ):
            raise ContractError(f"job image publication receipt digests are invalid: {path}")
        return receipt

    def _ensure_receipt_matches(
        self,
        request: JobImagePublicationRequest,
        receipt: _PublicationReceipt,
    ) -> None:
        publication = request.publication
        if (
            receipt.package_key != request.package_key
            or receipt.publication_key != request.publication_key
            or receipt.kind_image != request.manifest.kind_image
            or receipt.repository != publication.repository
            or receipt.image.value.rsplit("@", 1)[0] != publication.repository
            or receipt.platforms != publication.platforms
            or receipt.compression != publication.compression
            or receipt.compression_level != publication.compression_level
            or receipt.provenance != publication.provenance
            or receipt.sbom != publication.sbom
            or receipt.build_definition_digest != self._definition_digest
        ):
            raise ContractError(
                "job image publication receipt does not match the requested "
                "package, publication settings, or build definition"
            )


def _remove_local_output(path: Path) -> None:
    """Remove only the exact temporary or prior local-export target."""

    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _enforce_qualification_policy(request: JobImagePublicationRequest) -> None:
    deferred = sorted(
        lock.environment_id for lock in request.manifest.environment_activations if lock.qualification == "deferred"
    )
    if deferred and not request.allow_deferred_qualification:
        raise ContractError(
            "offline qualification is deferred for "
            + ", ".join(repr(environment_id) for environment_id in deferred)
            + "; pass --allow-deferred-qualification to make the live job the Taskset.load gate"
        )


def _published(
    receipt: _PublicationReceipt,
    *,
    cache_hit: bool,
) -> PublishedJobImage:
    return PublishedJobImage(
        package_key=receipt.package_key,
        publication_key=receipt.publication_key,
        image=receipt.image,
        kind_image=receipt.kind_image,
        receipt=receipt.path,
        cache_hit=cache_hit,
    )


def _metadata_digest(path: Path) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata = payload[_PUBLISHED_TARGET]
        digest = metadata["containerimage.digest"]
    except (
        FileNotFoundError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as error:
        raise RuntimeError("Buildx did not return a published actual-job image digest") from error
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        raise RuntimeError("Buildx returned an invalid actual-job image digest")
    value = digest.removeprefix("sha256:")
    if not _SHA256.fullmatch(value):
        raise RuntimeError("Buildx returned an invalid actual-job image digest")
    return value


def _build_definition_digest(root: Path) -> str:
    entries = []
    for name in ("Dockerfile", "docker-bake.hcl"):
        content = (root / name).read_bytes()
        entries.append(
            {
                "path": name,
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    return hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _exact_bool(value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError("boolean")
    return value
