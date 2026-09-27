"""The only module that selects concrete tracking and semantic adapters."""

from __future__ import annotations

from collections.abc import Callable

from posttrain.advisor import HubModelReader
from posttrain.tracking import RunDataSource, RunNoteStore
from posttrain_tracking_trackio import TrackioDataSource, TrackioProjectCatalog
from posttrain_tracking_wandb import WandbDataSource, WandbSettings

from .discovery import TrackioSourceDiscovery
from .fixtures import FixtureRunDataSource
from .run_cards import TemplateSet
from .semantic import FixtureSemanticSummaryProvider, OpenAICompatibleSemanticSummaryProvider
from .service import ObservatoryService
from .settings import (
    FixtureSourceSettings,
    ObservatorySettings,
    TrackioSourceSettings,
    WandbSourceSettings,
)
from .sources import RunSourceRegistry


def _note_stores(write_token: str | None) -> Callable[[str, RunDataSource], RunNoteStore | None]:
    def store(source_id: str, source: RunDataSource) -> RunNoteStore | None:
        return source.note_store(write_token=write_token) if isinstance(source, TrackioDataSource) else None

    return store


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

    registry = RunSourceRegistry(sources)
    discovery = None
    if settings.discover_trackio_projects:
        assert settings.trackio_server_url is not None
        server_url = settings.trackio_server_url
        discovery = TrackioSourceDiscovery(
            registry,
            TrackioProjectCatalog(server_url),
            lambda project: TrackioDataSource(project, server_url=server_url),
            interval_seconds=settings.trackio_discovery_interval_seconds,
        )

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
    return ObservatoryService(
        registry,
        semantic_provider=semantic,
        source_discovery=discovery,
        architecture_loader=HubModelReader() if settings.model_config_source == "hub" else None,
        note_store_factory=_note_stores(settings.trackio_write_token),
        note_templates=TemplateSet(settings.note_templates_dir),
        note_writes=settings.note_writes,
    )


__all__ = ["create_service"]
