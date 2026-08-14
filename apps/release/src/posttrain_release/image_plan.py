"""Plan release-image work from immutable desired inputs and registry facts.

The planner deliberately has no BuildKit or registry dependency.  Callers turn
remote inspection into :class:`ObservedImage` values first, then this module
answers the useful release question: *which immutable image nodes need work,
and why?*  Keeping that decision pure makes an interrupted release, a partial
mirror, and a cold local receipt cache testable without an OCI registry.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal

type ImageAction = Literal["reuse-remote", "copy", "build", "blocked"]
type ObservationStatus = Literal["present", "missing", "unreachable"]


@dataclass(frozen=True, slots=True)
class DesiredImage:
    """The immutable inputs that define one release image node.

    ``identity`` is derived rather than supplied so a caller cannot accidentally
    compare a hand-maintained label instead of the actual source/lock/parent
    selection.  ``parent`` names another planned node; only job-kind nodes use
    it today.
    """

    name: str
    repository: str
    source_digest: str
    lock_digest: str
    parent: str | None = None
    trust_bundle_digest: str | None = None
    backend_lock_digest: str | None = None
    backend_identity: tuple[str, str, str] | None = None
    forced: bool = False

    @property
    def identity(self) -> str:
        payload = {
            "source_digest": self.source_digest,
            "lock_digest": self.lock_digest,
            "parent": self.parent,
            "trust_bundle_digest": self.trust_bundle_digest,
            "backend_lock_digest": self.backend_lock_digest,
            "backend_identity": self.backend_identity,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class ObservedImage:
    """Registry evidence for one image name at one endpoint.

    ``identity`` is the semantic identity reconstructed from the release
    manifest/receipt that named ``digest``.  It is intentionally distinct from
    the OCI digest: a source-compatible image can be copied by OCI digest,
    while a changed desired identity must build a new digest.

    ``missing_blob_bytes`` is the destination-side descriptor set difference
    when a registry implementation can supply it.  ``logical_bytes`` is a
    conservative copy upper bound for older registries.  A build has no honest
    OCI byte estimate before BuildKit emits its descriptors, so the plan leaves
    it ``None`` rather than inventing a number.
    """

    status: ObservationStatus
    digest: str | None = None
    identity: str | None = None
    logical_bytes: int | None = None
    missing_blob_bytes: int | None = None
    detail: str = ""

    def __post_init__(self) -> None:
        if self.status == "present" and (self.digest is None or self.identity is None):
            raise ValueError("a present image observation needs both digest and identity")
        if self.status != "present" and (self.digest is not None or self.identity is not None):
            raise ValueError("only a present image observation may name a digest or identity")
        for field in ("logical_bytes", "missing_blob_bytes"):
            value = getattr(self, field)
            if value is not None and value < 0:
                raise ValueError(f"{field} cannot be negative")


@dataclass(frozen=True, slots=True)
class PlannedImage:
    """One executable image action with its evidence and byte estimate."""

    desired: DesiredImage
    action: ImageAction
    reason: str
    source: ObservedImage
    destination: ObservedImage
    expected_digest: str | None
    estimated_transfer_bytes: int | None


@dataclass(frozen=True, slots=True)
class ReleaseImagePlan:
    """A topologically ordered, immutable release-image plan."""

    nodes: tuple[PlannedImage, ...]

    @property
    def blocked(self) -> bool:
        return any(node.action == "blocked" for node in self.nodes)

    @property
    def known_transfer_bytes(self) -> int:
        return sum(node.estimated_transfer_bytes or 0 for node in self.nodes)

    @property
    def has_unknown_transfer_bytes(self) -> bool:
        return any(node.estimated_transfer_bytes is None for node in self.nodes)

    def node(self, name: str) -> PlannedImage:
        for candidate in self.nodes:
            if candidate.desired.name == name:
                return candidate
        raise KeyError(name)


def plan_release_images(
    desired: tuple[DesiredImage, ...],
    *,
    source: dict[str, ObservedImage],
    destination: dict[str, ObservedImage],
    allow_build: bool = True,
) -> ReleaseImagePlan:
    """Choose ``reuse-remote``, ``copy``, ``build``, or ``blocked`` per node.

    ``desired`` must be in parent-before-child order.  A parent that needs a
    new build necessarily makes every child a build too: it will receive a new
    OCI digest even if all of the child's other inputs are unchanged.  This is
    the explicit base-change fan-out that release UIs used to hide.
    """

    names = [node.name for node in desired]
    if len(names) != len(set(names)):
        raise ValueError("release image plan contains duplicate node names")
    known = set(names)
    completed: dict[str, PlannedImage] = {}
    nodes: list[PlannedImage] = []
    for node in desired:
        if node.parent is not None and node.parent not in known:
            raise ValueError(f"{node.name}: unknown parent {node.parent!r}")
        if node.parent is not None and node.parent not in completed:
            raise ValueError(f"{node.name}: parent {node.parent!r} must appear first")
        source_observation = source.get(node.name, ObservedImage("missing", detail="no source receipt"))
        destination_observation = destination.get(
            node.name, ObservedImage("missing", detail="not present at destination")
        )
        parent = completed.get(node.parent) if node.parent is not None else None
        planned = _plan_node(
            node,
            source=source_observation,
            destination=destination_observation,
            parent=parent,
            allow_build=allow_build,
        )
        completed[node.name] = planned
        nodes.append(planned)
    return ReleaseImagePlan(tuple(nodes))


def _plan_node(
    desired: DesiredImage,
    *,
    source: ObservedImage,
    destination: ObservedImage,
    parent: PlannedImage | None,
    allow_build: bool,
) -> PlannedImage:
    if parent is not None and parent.action == "build":
        return _build_or_block(
            desired,
            source,
            destination,
            allow_build,
            f"parent {parent.desired.name} requires a new immutable digest",
        )
    if desired.forced:
        return _build_or_block(desired, source, destination, allow_build, "explicitly selected for rebuild")
    if destination.status == "present" and destination.identity == desired.identity:
        return PlannedImage(
            desired,
            "reuse-remote",
            "destination already carries the requested immutable inputs",
            source,
            destination,
            destination.digest,
            0,
        )
    if source.status == "present" and source.identity == desired.identity:
        estimate = source.missing_blob_bytes
        if estimate is None:
            estimate = source.logical_bytes
        return PlannedImage(
            desired,
            "copy",
            "source registry carries the requested immutable inputs; destination does not",
            source,
            destination,
            source.digest,
            estimate,
        )
    reason = _build_reason(desired, source, destination)
    return _build_or_block(desired, source, destination, allow_build, reason)


def _build_reason(desired: DesiredImage, source: ObservedImage, destination: ObservedImage) -> str:
    if source.status == "unreachable":
        return "source registry is unreachable and cannot provide a compatible immutable image"
    if destination.status == "unreachable":
        return "destination registry is unreachable and cannot be verified"
    if source.status == "present" and source.identity != desired.identity:
        return "published source image was built from different immutable inputs"
    if destination.status == "present" and destination.identity != desired.identity:
        return "destination tag names an image built from different immutable inputs"
    return "no compatible published image is available"


def _build_or_block(
    desired: DesiredImage,
    source: ObservedImage,
    destination: ObservedImage,
    allow_build: bool,
    reason: str,
) -> PlannedImage:
    if allow_build:
        return PlannedImage(desired, "build", reason, source, destination, None, None)
    return PlannedImage(
        desired,
        "blocked",
        f"{reason}; building is disabled",
        source,
        destination,
        None,
        0,
    )


__all__ = [
    "DesiredImage",
    "ImageAction",
    "ObservedImage",
    "ObservationStatus",
    "PlannedImage",
    "ReleaseImagePlan",
    "plan_release_images",
]
