"""Tests for provider-neutral storage admission."""

from __future__ import annotations

import pytest
from posttrain.common import ContractError
from posttrain.work import (
    GIB,
    StorageCapacity,
    StorageRequirement,
    assess_storage,
    require_storage,
)


def test_admission_preserves_larger_ratio_reserve() -> None:
    capacity = StorageCapacity(total_bytes=1000 * GIB, available_bytes=200 * GIB)
    requirement = StorageRequirement(
        download_bytes=10 * GIB,
        peak_workspace_bytes=20 * GIB,
        retained_output_bytes=5 * GIB,
    )

    admission = assess_storage(capacity, requirement)

    assert admission.accepted
    assert admission.workload_bytes == 35 * GIB
    assert admission.reserve_bytes == 150 * GIB
    assert admission.required_bytes == 185 * GIB
    assert admission.shortfall_bytes == 0


def test_admission_preserves_minimum_free_space_on_smaller_disk() -> None:
    capacity = StorageCapacity(total_bytes=100 * GIB, available_bytes=40 * GIB)
    requirement = StorageRequirement(peak_workspace_bytes=10 * GIB)

    admission = require_storage(capacity, requirement)

    assert admission.accepted
    assert admission.reserve_bytes == 30 * GIB
    assert admission.required_bytes == 40 * GIB


def test_rejection_explains_required_available_and_shortfall() -> None:
    capacity = StorageCapacity(total_bytes=100 * GIB, available_bytes=39 * GIB)
    requirement = StorageRequirement(peak_workspace_bytes=10 * GIB)

    admission = assess_storage(capacity, requirement)

    assert not admission.accepted
    assert admission.shortfall_bytes == GIB
    with pytest.raises(
        ContractError,
        match=r"required=42949672960 available=41875931136 shortfall=1073741824",
    ):
        require_storage(capacity, requirement)


@pytest.mark.parametrize(
    ("capacity", "message"),
    [
        (StorageCapacity(total_bytes=1, available_bytes=0), None),
    ],
)
def test_valid_capacity_boundary(capacity: StorageCapacity, message: str | None) -> None:
    assert capacity.available_bytes == 0
    assert message is None


@pytest.mark.parametrize(
    ("total_bytes", "available_bytes", "message"),
    [
        (0, 0, "total bytes must be positive"),
        (10, -1, "available bytes cannot be negative"),
        (10, 11, "available bytes cannot exceed total bytes"),
    ],
)
def test_capacity_rejects_invalid_values(total_bytes: int, available_bytes: int, message: str) -> None:
    with pytest.raises(ContractError, match=message):
        StorageCapacity(total_bytes=total_bytes, available_bytes=available_bytes)


@pytest.mark.parametrize("value", [-1])
def test_requirement_rejects_invalid_byte_values(value: int) -> None:
    with pytest.raises(ContractError):
        StorageRequirement(download_bytes=value)
    with pytest.raises(ContractError):
        StorageRequirement(peak_workspace_bytes=value)
    with pytest.raises(ContractError):
        StorageRequirement(retained_output_bytes=value)
    with pytest.raises(ContractError):
        StorageRequirement(minimum_free_bytes=value)


@pytest.mark.parametrize("ratio", [-0.01, 1.0])
def test_requirement_rejects_invalid_margin_ratio(ratio: float) -> None:
    with pytest.raises(ContractError):
        StorageRequirement(safety_margin_ratio=ratio)
