"""Storage admission for work-package execution.

Admission is intentionally independent of a concrete scheduler. Hosts supply an
observed filesystem capacity and the job's declared storage envelope before a
GPU is allocated.
"""

from __future__ import annotations

from dataclasses import dataclass

from posttrain.common import ContractError

GIB = 1024**3


@dataclass(frozen=True, slots=True)
class StorageCapacity:
    """Observed byte capacity for the filesystem that will hold job data."""

    total_bytes: int
    available_bytes: int

    def __post_init__(self) -> None:
        if self.total_bytes <= 0:
            raise ContractError("storage total bytes must be positive")
        if self.available_bytes < 0:
            raise ContractError("storage available bytes cannot be negative")
        if self.available_bytes > self.total_bytes:
            raise ContractError("storage available bytes cannot exceed total bytes")


@dataclass(frozen=True, slots=True)
class StorageRequirement:
    """Declared peak storage envelope for one execution attempt."""

    download_bytes: int = 0
    peak_workspace_bytes: int = 0
    retained_output_bytes: int = 0
    safety_margin_ratio: float = 0.15
    minimum_free_bytes: int = 30 * GIB

    def __post_init__(self) -> None:
        byte_values = (
            self.download_bytes,
            self.peak_workspace_bytes,
            self.retained_output_bytes,
            self.minimum_free_bytes,
        )
        if any(value < 0 for value in byte_values):
            raise ContractError("storage requirement byte values cannot be negative")
        if not 0 <= self.safety_margin_ratio < 1:
            raise ContractError("storage safety margin ratio must be in [0, 1)")

    @property
    def workload_bytes(self) -> int:
        return self.download_bytes + self.peak_workspace_bytes + self.retained_output_bytes


@dataclass(frozen=True, slots=True)
class StorageAdmission:
    """The explainable result of evaluating a storage requirement."""

    accepted: bool
    available_bytes: int
    workload_bytes: int
    reserve_bytes: int
    required_bytes: int
    shortfall_bytes: int

    @property
    def status(self) -> str:
        return "accepted" if self.accepted else "rejected"


def assess_storage(
    capacity: StorageCapacity,
    requirement: StorageRequirement,
) -> StorageAdmission:
    """Return a deterministic admission decision without mutating the host."""

    ratio_reserve = int(capacity.total_bytes * requirement.safety_margin_ratio)
    reserve_bytes = max(requirement.minimum_free_bytes, ratio_reserve)
    required_bytes = requirement.workload_bytes + reserve_bytes
    shortfall_bytes = max(0, required_bytes - capacity.available_bytes)
    return StorageAdmission(
        accepted=shortfall_bytes == 0,
        available_bytes=capacity.available_bytes,
        workload_bytes=requirement.workload_bytes,
        reserve_bytes=reserve_bytes,
        required_bytes=required_bytes,
        shortfall_bytes=shortfall_bytes,
    )


def require_storage(
    capacity: StorageCapacity,
    requirement: StorageRequirement,
) -> StorageAdmission:
    """Return admission or raise an actionable error before execution."""

    admission = assess_storage(capacity, requirement)
    if not admission.accepted:
        raise ContractError(
            "storage admission rejected: "
            f"required={admission.required_bytes} "
            f"available={admission.available_bytes} "
            f"shortfall={admission.shortfall_bytes}"
        )
    return admission


__all__ = [
    "GIB",
    "StorageAdmission",
    "StorageCapacity",
    "StorageRequirement",
    "assess_storage",
    "require_storage",
]
