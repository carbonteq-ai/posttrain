"""Provider-neutral port for publishing one materialized actual-job image."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from posttrain.common import ContractError
from posttrain.execution import JobPackageManifest, RuntimeImageRef

from .planning import ImagePublicationSpec

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class JobImagePublicationRequest:
    """Exact package and registry inputs supplied to an image publisher."""

    manifest: JobPackageManifest
    staged_context: Path
    publication: ImagePublicationSpec
    allow_deferred_qualification: bool = False
    local_output: Path | None = None
    local_tag: str | None = None
    source_context_digest: str | None = None

    def __post_init__(self) -> None:
        if not self.staged_context.is_absolute() or not self.staged_context.is_dir():
            raise ContractError("job image staged context must be an existing absolute directory")
        if self.local_output is not None and not self.local_output.is_absolute():
            raise ContractError("local OCI output must be an absolute path")
        if self.local_tag is not None and (
            not self.local_tag.strip()
            or "@" in self.local_tag
            or any(character.isspace() for character in self.local_tag)
        ):
            raise ContractError("local daemon image tag is invalid")
        if self.source_context_digest is not None and _SHA256.fullmatch(self.source_context_digest) is None:
            raise ContractError("job image source context digest must be SHA-256")

    @property
    def package_key(self) -> str:
        return self.manifest.package_key

    @property
    def publication_key(self) -> str:
        """Portable identity that deliberately excludes local filesystem paths."""

        return publication_key_for(self.manifest, self.publication)


@dataclass(frozen=True, slots=True)
class JobImageResolutionRequest:
    """Receipt-only lookup for a previously published immutable image."""

    manifest: JobPackageManifest
    publication: ImagePublicationSpec
    publication_key: str
    allow_deferred_qualification: bool = False

    @property
    def package_key(self) -> str:
        return self.manifest.package_key

    def __post_init__(self) -> None:
        expected = publication_key_for(self.manifest, self.publication)
        if self.publication_key != expected:
            raise ContractError("job image resolution publication key is invalid")


def publication_key_for(manifest: JobPackageManifest, publication: ImagePublicationSpec) -> str:
    """Return the stable publication identity without a local build context."""

    payload = {
        "package_key": manifest.package_key,
        "publication": publication.to_payload(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class PublishedJobImage:
    """Remotely verified immutable image and its local publication evidence."""

    package_key: str
    publication_key: str
    image: RuntimeImageRef
    kind_image: RuntimeImageRef
    receipt: Path
    cache_hit: bool

    def __post_init__(self) -> None:
        if not _SHA256.fullmatch(self.package_key):
            raise ContractError("published job package key must be SHA-256")
        if not _SHA256.fullmatch(self.publication_key):
            raise ContractError("published job publication key must be SHA-256")
        if not self.receipt.is_absolute():
            raise ContractError("published job receipt path must be absolute")


@dataclass(frozen=True, slots=True)
class LocalPublishedJobImage:
    """A content-addressed OCI layout for local Docker-compatible execution.

    This intentionally is not a ``RuntimeImageRef`` and cannot be passed to a
    remote execution provider until an explicit OCI publication occurs.
    """

    package_key: str
    publication_key: str
    layout: Path
    tag: str
    receipt: Path
    cache_hit: bool

    def __post_init__(self) -> None:
        if not _SHA256.fullmatch(self.package_key) or not _SHA256.fullmatch(self.publication_key):
            raise ContractError("local job image keys must be SHA-256")
        if not self.layout.is_absolute() or not self.receipt.is_absolute():
            raise ContractError("local job image paths must be absolute")
        if (
            not self.tag.startswith("posttrain-local:")
            or not self.tag
            or "@" in self.tag
            or any(character.isspace() for character in self.tag)
        ):
            raise ContractError("local job image tag is invalid")


@dataclass(frozen=True, slots=True)
class LocalDaemonJobImage:
    """A verified single-platform image loaded into the local Docker daemon.

    ``image`` remains the immutable identity used by the framework. ``tag`` is
    only a machine-local transport handle and is never valid for a remote
    provider. Keeping both values makes retries and cleanup explicit without
    weakening the digest-pinned execution contract.
    """

    package_key: str
    publication_key: str
    image: RuntimeImageRef
    tag: str
    receipt: Path
    cache_hit: bool

    def __post_init__(self) -> None:
        if not _SHA256.fullmatch(self.package_key) or not _SHA256.fullmatch(self.publication_key):
            raise ContractError("local daemon image keys must be SHA-256")
        if not self.receipt.is_absolute():
            raise ContractError("local daemon image receipt path must be absolute")
        if (
            not self.tag.startswith("posttrain-local:")
            or not self.tag
            or "@" in self.tag
            or any(character.isspace() for character in self.tag)
        ):
            raise ContractError("local daemon image tag is invalid")


class LocalJobImagePublisher(Protocol):
    """Publish an actual-job image to a local OCI layout or daemon store."""

    def publish_local(self, request: JobImagePublicationRequest) -> LocalPublishedJobImage: ...

    def publish_local_daemon(self, request: JobImagePublicationRequest) -> LocalDaemonJobImage: ...


class JobImagePublisher(Protocol):
    """Application-facing port implemented by BuildKit or another OCI builder."""

    def publish(
        self,
        request: JobImagePublicationRequest,
    ) -> PublishedJobImage: ...

    def resolve(self, request: JobImageResolutionRequest) -> PublishedJobImage | None: ...
