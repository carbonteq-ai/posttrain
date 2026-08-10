from __future__ import annotations

from pathlib import Path

import pytest
from posttrain.common import ContractError
from posttrain.execution_pack import CacheLease, has_active_lease


def test_cache_lease_is_visible_until_released(tmp_path: Path) -> None:
    key = "a" * 64
    lease = CacheLease.acquire(tmp_path / "leases", key, ttl_seconds=60)

    assert lease.active
    assert has_active_lease(tmp_path / "leases", key)

    lease.release()

    assert not lease.active
    assert not has_active_lease(tmp_path / "leases", key)


def test_expired_cache_lease_does_not_protect_an_object(tmp_path: Path) -> None:
    key = "b" * 64
    lease = CacheLease.acquire(tmp_path / "leases", key, ttl_seconds=1)

    assert not has_active_lease(tmp_path / "leases", key, now=lease.expires_at + 1)
    lease.expires_at = 0

    with pytest.raises(ContractError, match="expired"):
        lease.renew(ttl_seconds=60)
    lease.release()


def test_lease_heartbeat_replaces_json_atomically(tmp_path: Path) -> None:
    key = "c" * 64
    lease = CacheLease.acquire(tmp_path / "leases", key, ttl_seconds=1)
    previous = lease.expires_at

    lease.renew(ttl_seconds=60)

    assert lease.expires_at > previous
    assert has_active_lease(tmp_path / "leases", key)
    assert lease.path.stat().st_mode & 0o077 == 0
    lease.release()
