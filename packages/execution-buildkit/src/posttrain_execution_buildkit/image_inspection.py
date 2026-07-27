"""Read the identity a published image actually carries.

Image identity is only useful if it can be checked against the registry rather
than trusted from configuration. A job-kind image whose recorded lock digest
disagrees with the framework that is about to use it invalidates every
qualification run made against it, so that disagreement has to be observable.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

from .builder import BuildxCli, BuildxGateway, RemoteImageNotFoundError

LOCK_DIGEST_LABEL = "org.carbonteq.posttrain.lock-digest"
"""SHA-256 of the dependency lock the image's environment was built from."""

IMAGE_LEVEL_LABEL = "org.carbonteq.posttrain.image-level"
"""Which level of the hierarchy this image is: base, job-kind, or actual-job.

Load-bearing, not decorative. Every level of the hierarchy is built from the
same workspace lock, so the lock digest alone cannot distinguish a job-kind
image from the base image it was derived from.
"""

REVISION_LABEL = "org.opencontainers.image.revision"
"""Framework commit the image was built from."""

VERSION_LABEL = "org.opencontainers.image.version"
"""Composite `rev-<revision>-lock-<lock digest>` identity."""

@dataclass(frozen=True, slots=True)
class RemoteImageFacts:
    """What the registry reports about one published image."""

    reference: str
    digest: str
    labels: Mapping[str, str]

    @property
    def lock_digest(self) -> str | None:
        return self.labels.get(LOCK_DIGEST_LABEL)

    @property
    def image_level(self) -> str | None:
        return self.labels.get(IMAGE_LEVEL_LABEL)

    @property
    def revision(self) -> str | None:
        return self.labels.get(REVISION_LABEL)

    @property
    def version(self) -> str | None:
        return self.labels.get(VERSION_LABEL)


def _config_labels(payload: object) -> dict[str, str]:
    """Extract labels from `imagetools inspect --format '{{json .Image}}'`.

    A single-platform image reports one config object; a multi-platform index
    reports a mapping of platform to config. Labels are identical across
    platforms for these images, so the first config that carries any is used.
    """
    if isinstance(payload, dict) and "config" in payload:
        config = payload.get("config")
        if isinstance(config, dict):
            labels = config.get("Labels")
            if isinstance(labels, dict):
                return {str(k): str(v) for k, v in labels.items()}
        return {}
    if isinstance(payload, dict):
        for value in payload.values():
            labels = _config_labels(value)
            if labels:
                return labels
    return {}


class ImageInspector(Protocol):
    """Reads what a registry holds for a reference.

    A protocol rather than a concrete type so callers can be exercised without
    a registry: identity checks are the part most worth testing, and they must
    not require the network to be reachable.
    """

    def inspect(self, reference: str) -> RemoteImageFacts: ...


class RuntimeImageInspector:
    """Read published digests and labels without pulling image layers."""

    def __init__(self, gateway: BuildxGateway | None = None) -> None:
        self._gateway = gateway or BuildxCli()

    def inspect(self, reference: str) -> RemoteImageFacts:
        """Return what the registry holds for `reference`.

        Raises `RemoteImageNotFoundError` when the registry confirms absence,
        which callers distinguish from drift: one is a missing publication, the
        other is a publication that no longer matches.
        """
        digest_output = self._gateway.invoke(
            ("imagetools", "inspect", reference, "--format", "{{json .Manifest.Digest}}")
        )
        try:
            digest = json.loads(digest_output)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"Buildx returned invalid remote-image metadata for {reference}"
            ) from error
        if not isinstance(digest, str) or not digest.startswith("sha256:"):
            raise RuntimeError(f"Buildx reported no manifest digest for {reference}")

        label_output = self._gateway.invoke(
            ("imagetools", "inspect", reference, "--format", "{{json .Image}}")
        )
        try:
            image_payload = json.loads(label_output)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"Buildx returned invalid image configuration for {reference}"
            ) from error

        return RemoteImageFacts(
            reference=reference,
            digest=digest,
            labels=MappingProxyType(_config_labels(image_payload)),
        )

    def copy(self, source: str, destination_tag: str) -> str:
        """Copy an image between registries, returning the destination digest.

        `destination_tag` must be a `repository:tag` reference. A digest cannot
        be pushed to: it is derived from content, not chosen, so naming the
        destination by digest is rejected. Content addressing still does the
        real work here — the copied image keeps the source's digest — but that
        digest is read back from the destination rather than requested, which
        is what makes the copy verifiable.
        """
        if "@" in destination_tag:
            raise ValueError(
                f"mirror destination must be a tag, not a digest: {destination_tag}"
            )
        self._gateway.invoke(("imagetools", "create", "--tag", destination_tag, source))
        return self.inspect(destination_tag).digest


__all__ = [
    "IMAGE_LEVEL_LABEL",
    "REVISION_LABEL",
    "VERSION_LABEL",
    "ImageInspector",
    "LOCK_DIGEST_LABEL",
    "RemoteImageFacts",
    "RemoteImageNotFoundError",
    "RuntimeImageInspector",
]
