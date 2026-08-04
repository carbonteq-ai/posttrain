"""Framework source closure staged into actual-job images."""

from __future__ import annotations

from pathlib import Path

from posttrain_cli.execution_planning import _framework_source_request


def test_source_packed_runtime_includes_environment_package() -> None:
    repository = Path(__file__).resolve().parents[3]

    request = _framework_source_request(repository)

    assert request is not None
    assert "packages/environment" in request.includes
    assert "packages/environment" in request.install_roots
