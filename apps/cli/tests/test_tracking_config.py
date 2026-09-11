"""Tests for tracking-only execution configuration boundaries."""

from __future__ import annotations

from typing import cast

from posttrain.catalog import ProjectLayout
from posttrain_cli import tracking_config


def test_tracking_environment_does_not_validate_unrelated_runtime_images(monkeypatch) -> None:
    configuration = object()
    calls: list[bool] = []

    def load(_layout: ProjectLayout, *, verify_published_locks: bool = True) -> object:
        calls.append(verify_published_locks)
        return configuration

    monkeypatch.setattr(tracking_config, "load_local_execution_config", load)
    monkeypatch.setattr(
        tracking_config,
        "load_execution_environment",
        lambda value: {"POSTTRAIN_TRACKIO_SERVER_URL": "https://trackio.example"} if value is configuration else {},
    )

    environment = tracking_config.project_tracking_environment(cast(ProjectLayout, object()))

    assert calls == [False]
    assert environment["POSTTRAIN_TRACKIO_SERVER_URL"] == "https://trackio.example"
