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

    def __post_init__(self) -> None:
        if not self.staged_context.is_absolute() or not self.staged_context.is_dir():
            raise ContractError("job image staged context must be an existing absolute directory")

    @property
    def package_key(self) -> str:
        return self.manifest.package_key

    @property
    def publication_key(self) -> str:
        """Portable identity that deliberately excludes local filesystem paths."""

        payload = {
            "package_key": self.package_key,
            "publication": self.publication.to_payload(),
        }
        return hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()


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


class JobImagePublisher(Protocol):
    """Application-facing port implemented by BuildKit or another OCI builder."""

    def publish(
        self,
        request: JobImagePublicationRequest,
    ) -> PublishedJobImage: ...
