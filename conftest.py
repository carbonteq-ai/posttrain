"""Repository-wide isolation from developer-machine configuration."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_machine_configuration(
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests must not inherit the machine running the repository suite."""

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path_factory.mktemp("config-home")))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path_factory.mktemp("state-home")))
