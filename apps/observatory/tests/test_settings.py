"""Deployment safety tests."""

import pytest
from posttrain_observatory.settings import ObservatorySettings
from pydantic import ValidationError


def test_remote_production_requires_auth_boundary() -> None:
    with pytest.raises(ValidationError, match="auth boundary"):
        ObservatorySettings(environment="production", host="0.0.0.0")


def test_semantic_provider_requires_server_side_credentials() -> None:
    with pytest.raises(ValidationError, match="base URL, model, and API key"):
        ObservatorySettings(semantic_provider="openai-compatible")


def test_trackio_server_url_is_loaded_for_live_container_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POSTTRAIN_TRACKIO_SERVER_URL", "http://trackio:7860")

    assert ObservatorySettings.from_env().trackio_server_url == "http://trackio:7860"


def test_trackio_discovery_is_opt_in_and_loads_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTTRAIN_OBSERVATORY_DISCOVER_TRACKIO_PROJECTS", "true")
    monkeypatch.setenv("POSTTRAIN_TRACKIO_DISCOVERY_INTERVAL_SECONDS", "45")
    monkeypatch.setenv("POSTTRAIN_TRACKIO_SERVER_URL", "http://trackio:7860")

    settings = ObservatorySettings.from_env()

    assert settings.discover_trackio_projects is True
    assert settings.trackio_discovery_interval_seconds == 45
    assert settings.configured_sources() == ()


def test_trackio_discovery_requires_server_url() -> None:
    with pytest.raises(ValidationError, match="requires a Trackio server URL"):
        ObservatorySettings(discover_trackio_projects=True)


def test_trackio_discovery_rejects_invalid_environment_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTTRAIN_OBSERVATORY_DISCOVER_TRACKIO_PROJECTS", "sometimes")
    with pytest.raises(ValueError, match="must be a boolean"):
        ObservatorySettings.from_env()

    monkeypatch.setenv("POSTTRAIN_OBSERVATORY_DISCOVER_TRACKIO_PROJECTS", "false")
    monkeypatch.setenv("POSTTRAIN_TRACKIO_DISCOVERY_INTERVAL_SECONDS", "0")
    with pytest.raises(ValidationError):
        ObservatorySettings.from_env()


def test_trackio_discovery_preserves_explicit_sources() -> None:
    settings = ObservatorySettings.model_validate(
        {
            "discover_trackio_projects": True,
            "trackio_server_url": "http://trackio:7860",
            "sources": ({"provider": "fixture", "source_id": "manual"},),
        }
    )

    assert [source.source_id for source in settings.configured_sources()] == ["manual"]


def test_multiple_sources_are_loaded_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "POSTTRAIN_OBSERVATORY_SOURCES",
        '[{"provider":"trackio","source_id":"trackio-local","project":"demo","server_url":"http://trackio:7860"},'
        '{"provider":"wandb","source_id":"wandb-cloud","entity":"team","project":"demo"}]',
    )

    settings = ObservatorySettings.from_env()

    assert [source.source_id for source in settings.configured_sources()] == ["trackio-local", "wandb-cloud"]


def test_multiple_source_ids_must_be_unique() -> None:
    with pytest.raises(ValidationError, match="source ids must be unique"):
        ObservatorySettings.model_validate(
            {
                "sources": (
                    {"provider": "fixture", "source_id": "evidence"},
                    {"provider": "fixture", "source_id": "evidence"},
                )
            }
        )


def test_project_settings_select_trackio_project_and_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POSTTRAIN_TRACKIO_SERVER_URL", "http://trackio:7860")

    settings = ObservatorySettings.for_project("support-agent", "trackio", port=8787)

    assert settings.source == "trackio"
    assert settings.source_id == "trackio-support-agent"
    assert settings.trackio_project == "support-agent"
    assert settings.trackio_server_url == "http://trackio:7860"
    assert settings.port == 8787


def test_project_settings_select_wandb_project(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WANDB_ENTITY", "carbonteq")
    monkeypatch.setenv("WANDB_PROJECT", "shared-evidence")

    settings = ObservatorySettings.for_project("support-agent", "wandb")

    assert settings.source == "wandb"
    assert settings.source_id == "wandb-support-agent"
    assert settings.wandb_entity == "carbonteq"
    assert settings.wandb_project == "shared-evidence"
