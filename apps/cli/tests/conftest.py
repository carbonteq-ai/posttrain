"""Shared isolation for CLI tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from posttrain_cli import execution_config


@pytest.fixture(autouse=True)
def _isolate_machine_trust(
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the machine's own certificate authority out of every test.

    Trust resolution falls back to a well-known path so that a configured
    machine needs no project configuration at all. That makes any test which
    resolves a provider depend on whether the machine running it happens to
    have an internal authority installed: the same test passes on a laptop and
    fails on a worker. Tests that care about the fallback point this somewhere
    they control.
    """
    absent = tmp_path_factory.mktemp("no-machine-trust") / "internal-ca.pem"
    monkeypatch.setattr(execution_config, "WELL_KNOWN_TRUST_BUNDLE", Path(absent))
    monkeypatch.delenv(execution_config.TRUST_BUNDLE_ENVIRONMENT_VARIABLE, raising=False)
