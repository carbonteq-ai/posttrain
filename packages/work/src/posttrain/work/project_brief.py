"""Typed project policy loaded independently from catalog selections."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, cast

import yaml
from posttrain.common import ContractError, JsonValue
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ServingRequirements(_StrictModel):
    """Product constraints used to interpret serving-capacity evidence."""

    required_context_tokens: int = Field(gt=0)
    min_sustained_output_tokens_per_second: float = Field(gt=0)
    max_p95_ttft_ms: float = Field(gt=0)
    max_p95_tpot_ms: float | None = Field(default=None, gt=0)
    max_failure_rate: float = Field(ge=0, le=1)


class ProjectBrief(_StrictModel):
    """Versioned project objective and optional decision requirements."""

    schema_version: Literal[1] = 1
    objective: str = Field(min_length=1)
    serving: ServingRequirements | None = None

    @field_validator("objective")
    @classmethod
    def normalize_objective(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("project objective cannot be empty")
        return normalized


def load_project_brief(path: Path) -> ProjectBrief:
    """Load one strict project-owned YAML brief."""

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ContractError(f"post-training project brief not found: {path}") from error
    except yaml.YAMLError as error:
        raise ContractError(f"invalid post-training project brief {path}: {error}") from error
    try:
        return ProjectBrief.model_validate(payload)
    except ValidationError as error:
        raise ContractError(f"invalid post-training project brief {path}: {error}") from error


def project_brief_digest(brief: ProjectBrief) -> str:
    """Return the stable digest used to correlate historical decisions."""

    payload = json.dumps(
        brief.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def project_brief_snapshot(brief: ProjectBrief) -> dict[str, JsonValue]:
    """Serialize a brief with the digest retained beside its policy fields."""

    snapshot = cast(dict[str, JsonValue], brief.model_dump(mode="json"))
    return {"digest": project_brief_digest(brief), **snapshot}


__all__ = [
    "ProjectBrief",
    "ServingRequirements",
    "load_project_brief",
    "project_brief_digest",
    "project_brief_snapshot",
]
