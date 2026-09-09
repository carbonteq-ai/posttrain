"""Fail-closed identity checks for recovery of structured-reward training."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any

from .requests import CAPORequest, GDPORequest, SAMPORequest
from .reward_projection import RewardProjection

FILENAME = "posttrain-reward-contract.json"


def reward_contract_digest(request: GDPORequest | CAPORequest | SAMPORequest) -> str:
    projection = getattr(request.bridge, "reward_projection", None)
    if not isinstance(projection, RewardProjection):
        raise ValueError("structured training bridge must declare its versioned reward_projection")
    settings = asdict(request.settings)
    # Extending a run's step budget is allowed; changing learning/credit semantics is not.
    settings["loop"].pop("max_steps", None)
    payload = {
        "schema": "posttrain.reward-contract.v1",
        "settings": settings,
        "projection": asdict(projection),
        "environment": request.environment,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, allow_nan=False, default=_identity_value).encode()
    ).hexdigest()


def _identity_value(value: Any) -> object:
    """Freeze activation and scoring configuration, not only a package revision label."""
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: getattr(value, item.name) for item in fields(value)}
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"reward recovery requires serializable environment identity, got {type(value).__name__}")


def validate_reward_recovery(checkpoint: Path, digest: str) -> None:
    try:
        retained = json.loads((checkpoint / FILENAME).read_text())
    except (OSError, ValueError) as error:
        raise ValueError("checkpoint lacks retained reward contract; refusing unverified structured resume") from error
    if retained != {"schema": "posttrain.reward-contract.v1", "sha256": digest}:
        raise ValueError("checkpoint reward contract differs from selected scoring or algorithm semantics")


def retain_reward_contract(checkpoint: Path, digest: str) -> None:
    path = checkpoint / FILENAME
    if path.exists():
        validate_reward_recovery(checkpoint, digest)
        return
    # A partially written marker is invalid on read and cannot authorize recovery.
    with path.open("x") as stream:
        json.dump({"schema": "posttrain.reward-contract.v1", "sha256": digest}, stream, sort_keys=True)
