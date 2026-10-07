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
    # An absent additive schedule must preserve pre-engine checkpoint identity.
    if settings.get("policy_updates") is None:
        settings.pop("policy_updates", None)
    # Extending a run's step budget is allowed; changing learning/credit semantics is not.
    settings["loop"].pop("max_steps", None)
    # The KL reference changes what is learned, so it is part of the contract. "start"
    # is what TRL did before the setting existed (a frozen copy of the starting
    # adapter), so it keeps the digest of checkpoints written before; "base" does not,
    # and a resume that would silently switch the reference is refused.
    if settings.get("kl_reference") == "start":
        settings.pop("kl_reference")
    # Oversampling changes how many prompt groups each round generates, not rewards,
    # credit, or how an update is assembled (the first target groups with reward
    # spread, in candidate order). Leaving it out keeps digests of checkpoints written
    # before the setting existed and lets a resumed run turn it on or off.
    active_sampling = settings.get("active_sampling")
    if isinstance(active_sampling, dict):
        active_sampling.pop("oversample", None)
        active_sampling.pop("oversample_refill", None)
        # Keeping surplus groups in candidate order is what runs did before the setting
        # existed; choosing them by learning signal changes what is trained on.
        if active_sampling.get("retain") == "first":
            active_sampling.pop("retain")
    # Goal-relative turn credit changes credit; its absence is what runs did before it existed.
    if settings.get("goal_credit") == "none":
        settings.pop("goal_credit")
    if settings.get("goal_credit_scale") == 1.0:
        settings.pop("goal_credit_scale")
    projection_identity = asdict(projection)
    for key in ("turn_goal_prefix", "turn_harm_key"):
        if projection_identity.get(key) is None:
            projection_identity.pop(key, None)
    payload = {
        "schema": "posttrain.reward-contract.v1",
        "settings": settings,
        "projection": projection_identity,
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
