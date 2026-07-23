"""The only module that selects concrete tracking and semantic adapters."""

from __future__ import annotations

from posttrain_tracking_trackio import TrackioDataSource
from posttrain_tracking_wandb import WandbDataSource, WandbSettings

from .fixtures import FixtureRunDataSource
from .semantic import FixtureSemanticSummaryProvider, OpenAICompatibleSemanticSummaryProvider
from .service import ObservatoryService
from .settings import (
    FixtureSourceSettings,
    ObservatorySettings,
    TrackioSourceSettings,
    WandbSourceSettings,
)


def create_service(settings: ObservatorySettings | None = None) -> ObservatoryService:
    settings = settings or ObservatorySettings.from_env()
    sources = {}
    for source_settings in settings.configured_sources():
        if isinstance(source_settings, TrackioSourceSettings):
            source = TrackioDataSource(
                source_settings.project,
                server_url=source_settings.server_url,
            )
        elif isinstance(source_settings, WandbSourceSettings):
            source = WandbDataSource(
                WandbSettings(
                    entity=source_settings.entity,
                    project=source_settings.project,
                    base_url=source_settings.base_url,
                    mode="online",
                )
            )
        elif isinstance(source_settings, FixtureSourceSettings):
            source = FixtureRunDataSource()
        else:  # pragma: no cover - discriminated settings make this unreachable
            raise TypeError(f"unsupported Observatory source settings: {type(source_settings).__name__}")
        sources[source_settings.source_id] = source

    semantic = None
    if settings.semantic_provider == "fixture":
        semantic = FixtureSemanticSummaryProvider()
    elif settings.semantic_provider == "openai-compatible":
        assert settings.semantic_base_url and settings.semantic_model and settings.semantic_api_key
        semantic = OpenAICompatibleSemanticSummaryProvider(
            base_url=settings.semantic_base_url,
            api_key=settings.semantic_api_key,
            model=settings.semantic_model,
        )
    return ObservatoryService(sources, semantic_provider=semantic)


__all__ = ["create_service"]
