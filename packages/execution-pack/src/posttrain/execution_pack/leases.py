"""Short-lived leases that protect rebuildable pack material from pruning."""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from posttrain.common import ContractError, JsonValue

_SCHEMA = "posttrain.cache-lease.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(slots=True)
class CacheLease:
    """A renewable, expiring lease for one rebuildable cache object."""

    root: Path
    object_key: str
    lease_id: str
    path: Path
    expires_at: float

    @classmethod
    def acquire(
        cls,
        root: Path,
        object_key: str,
        *,
        ttl_seconds: float = 3600.0,
    ) -> CacheLease:
        if not root.is_absolute():
            raise ContractError("cache lease root must be absolute")
        if _SHA256.fullmatch(object_key) is None:
            raise ContractError("cache lease object key must be SHA-256")
        if ttl_seconds <= 0:
            raise ContractError("cache lease TTL must be positive")
        lease_root = root / object_key
        lease_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        lease_id = uuid.uuid4().hex
        path = lease_root / f"{lease_id}.json"
        expires_at = time.time() + ttl_seconds
        payload: dict[str, JsonValue] = {
            "schema": _SCHEMA,
            "object_key": object_key,
            "lease_id": lease_id,
            "expires_at": expires_at,
        }
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{lease_id}.", suffix=".tmp", dir=lease_root)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            temporary.chmod(0o600)
            os.link(temporary, path)
        except OSError as error:
            raise ContractError("cache lease could not be created") from error
        finally:
            temporary.unlink(missing_ok=True)
        return cls(root, object_key, lease_id, path, expires_at)

    @property
    def active(self) -> bool:
        return self.path.is_file() and not self.path.is_symlink() and self.expires_at > time.time()

    def renew(self, *, ttl_seconds: float = 3600.0) -> None:
        if ttl_seconds <= 0:
            raise ContractError("cache lease TTL must be positive")
        if not self.active:
            raise ContractError("cache lease is expired or released")
        self.expires_at = time.time() + ttl_seconds
        payload = {
            "schema": _SCHEMA,
            "object_key": self.object_key,
            "lease_id": self.lease_id,
            "expires_at": self.expires_at,
        }
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{self.lease_id}.", suffix=".tmp", dir=self.path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            temporary.chmod(0o600)
            os.replace(temporary, self.path)
        except OSError as error:
            raise ContractError("cache lease heartbeat could not be written") from error
        finally:
            temporary.unlink(missing_ok=True)

    def release(self) -> None:
        if self.path.is_symlink():
            raise ContractError("cache lease path cannot be a symlink")
        self.path.unlink(missing_ok=True)

    def __enter__(self) -> CacheLease:
        return self

    def __exit__(self, *_args: object) -> None:
        self.release()


def has_active_lease(root: Path, object_key: str, *, now: float | None = None) -> bool:
    """Return whether one valid, unexpired lease protects ``object_key``."""

    if not root.is_absolute() or _SHA256.fullmatch(object_key) is None:
        return False
    lease_root = root / object_key
    if not lease_root.is_dir() or lease_root.is_symlink():
        return False
    observed = time.time() if now is None else now
    for path in lease_root.iterdir():
        if path.suffix != ".json" or path.is_symlink() or not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if (
            isinstance(payload, dict)
            and payload.get("schema") == _SCHEMA
            and payload.get("object_key") == object_key
            and isinstance(payload.get("expires_at"), (int, float))
            and float(payload["expires_at"]) > observed
        ):
            return True
    return False


__all__ = ["CacheLease", "has_active_lease"]
