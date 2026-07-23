"""Strict runtime configuration with safe local defaults."""

from __future__ import annotations

import json
import os
from typing import Annotated, Literal

from pydantic import Field, model_validator

from .models import ObservatoryModel


class FixtureSourceSettings(ObservatoryModel):
    provider: Literal["fixture"] = "fixture"
    source_id: str = Field(min_length=1)


class TrackioSourceSettings(ObservatoryModel):
    provider: Literal["trackio"] = "trackio"
    source_id: str = Field(min_length=1)
    project: str = Field(min_length=1)
    server_url: str | None = None


class WandbSourceSettings(ObservatoryModel):
    provider: Literal["wandb"] = "wandb"
    source_id: str = Field(min_length=1)
    entity: str = Field(min_length=1)
    project: str = Field(min_length=1)
    base_url: str | None = None


type ObservatorySourceSettings = Annotated[
    FixtureSourceSettings | TrackioSourceSettings | WandbSourceSettings,
    Field(discriminator="provider"),
]


class ObservatorySettings(ObservatoryModel):
    environment: Literal["local", "production"] = "local"
    host: str = "127.0.0.1"
    port: int = Field(default=7861, ge=1, le=65535)
    source: Literal["fixture", "trackio", "wandb"] = "trackio"
    source_id: str = "trackio-local"
    trackio_project: str = "posttrain"
    trackio_server_url: str | None = None
    wandb_entity: str | None = None
    wandb_project: str | None = None
    wandb_base_url: str | None = None
    auth_mode: Literal["none", "ingress"] = "none"
    cors_origins: tuple[str, ...] = ()
    semantic_provider: Literal["disabled", "fixture", "openai-compatible"] = "disabled"
    semantic_base_url: str | None = None
    semantic_model: str | None = None
    semantic_api_key: str | None = Field(default=None, repr=False)
    frontend_dir: str | None = None
    sources: tuple[ObservatorySourceSettings, ...] = ()

    @model_validator(mode="after")
    def validate_runtime(self) -> ObservatorySettings:
        if self.environment == "production" and self.host not in {"127.0.0.1", "::1", "localhost"}:
            if self.auth_mode == "none":
                raise ValueError("production Observatory on a non-loopback host requires an auth boundary")
        if self.source == "wandb" and (not self.wandb_entity or not self.wandb_project):
            raise ValueError("W&B source requires entity and project")
        source_ids = [source.source_id for source in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("Observatory source ids must be unique")
        if self.semantic_provider == "openai-compatible" and (
            not self.semantic_base_url or not self.semantic_model or not self.semantic_api_key
        ):
            raise ValueError("OpenAI-compatible semantic analysis requires base URL, model, and API key")
        return self

    def configured_sources(self) -> tuple[ObservatorySourceSettings, ...]:
        if self.sources:
            return self.sources
        if self.source == "trackio":
            return (
                TrackioSourceSettings(
                    source_id=self.source_id,
                    project=self.trackio_project,
                    server_url=self.trackio_server_url,
                ),
            )
        if self.source == "wandb":
            assert self.wandb_entity is not None and self.wandb_project is not None
            return (
                WandbSourceSettings(
                    source_id=self.source_id,
                    entity=self.wandb_entity,
                    project=self.wandb_project,
                    base_url=self.wandb_base_url,
                ),
            )
        return (FixtureSourceSettings(source_id=self.source_id),)

    @classmethod
    def from_env(cls) -> ObservatorySettings:
        origins = tuple(
            value.strip() for value in os.getenv("POSTTRAIN_OBSERVATORY_CORS", "").split(",") if value.strip()
        )
        raw_sources = os.getenv("POSTTRAIN_OBSERVATORY_SOURCES")
        sources: tuple[object, ...] = ()
        if raw_sources:
            try:
                decoded = json.loads(raw_sources)
            except json.JSONDecodeError as error:
                raise ValueError("POSTTRAIN_OBSERVATORY_SOURCES must be valid JSON") from error
            if not isinstance(decoded, list):
                raise ValueError("POSTTRAIN_OBSERVATORY_SOURCES must be a JSON array")
            sources = tuple(decoded)
        return cls.model_validate(
            {
                "environment": os.getenv("POSTTRAIN_OBSERVATORY_ENV", "local"),
                "host": os.getenv("POSTTRAIN_OBSERVATORY_HOST", "127.0.0.1"),
                "port": int(os.getenv("POSTTRAIN_OBSERVATORY_PORT", "7861")),
                "source": os.getenv("POSTTRAIN_OBSERVATORY_SOURCE", "trackio"),
                "source_id": os.getenv("POSTTRAIN_OBSERVATORY_SOURCE_ID", "trackio-local"),
                "trackio_project": os.getenv("POSTTRAIN_TRACKIO_PROJECT", "posttrain"),
                "trackio_server_url": os.getenv("POSTTRAIN_TRACKIO_SERVER_URL"),
                "wandb_entity": os.getenv("WANDB_ENTITY"),
                "wandb_project": os.getenv("POSTTRAIN_WANDB_PROJECT"),
                "wandb_base_url": os.getenv("WANDB_BASE_URL"),
                "auth_mode": os.getenv("POSTTRAIN_OBSERVATORY_AUTH", "none"),
                "cors_origins": origins,
                "semantic_provider": os.getenv("POSTTRAIN_OBSERVATORY_LLM_PROVIDER", "disabled"),
                "semantic_base_url": os.getenv("POSTTRAIN_OBSERVATORY_LLM_BASE_URL"),
                "semantic_model": os.getenv("POSTTRAIN_OBSERVATORY_LLM_MODEL"),
                "semantic_api_key": os.getenv("POSTTRAIN_OBSERVATORY_LLM_API_KEY"),
                "frontend_dir": os.getenv("POSTTRAIN_OBSERVATORY_FRONTEND_DIR"),
                "sources": sources,
            }
        )


__all__ = [
    "FixtureSourceSettings",
    "ObservatorySettings",
    "ObservatorySourceSettings",
    "TrackioSourceSettings",
    "WandbSourceSettings",
]
