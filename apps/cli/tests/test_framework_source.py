"""Framework source staged into actual-job images from a checkout."""

from pathlib import Path

from posttrain_cli.execution_planning import _framework_source_request

WORKSPACE = Path(__file__).resolve().parents[3]


def test_framework_source_includes_environment_contracts() -> None:
    request = _framework_source_request(WORKSPACE)

    assert request is not None
    assert "packages/environment" in request.includes
    assert "packages/environment" in request.install_roots
