from __future__ import annotations

import os
import sys
from pathlib import Path

from posttrain_cli.execution_config import DstackBinding, LocalExecutionConfig
from posttrain_cli.tracking_config import project_tracking_environment


def test_project_tracking_environment_uses_configured_internal_trust_bundle(
    tmp_path: Path,
    monkeypatch,
) -> None:
    trust_bundle = tmp_path / "internal-ca.pem"
    trust_bundle.write_text("test certificate\n", encoding="utf-8")
    configuration = LocalExecutionConfig(
        path=tmp_path / "execution.toml",
        dstack=DstackBinding(
            project="main",
            python=Path(sys.executable),
            trust_bundle=trust_bundle,
        ),
    )
    monkeypatch.setattr(
        "posttrain_cli.tracking_config.load_local_execution_config",
        lambda _layout: configuration,
    )
    monkeypatch.setattr(
        "posttrain_cli.tracking_config.load_execution_environment",
        lambda _configuration: {},
    )
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)

    assert project_tracking_environment(object()) == {}
    assert os.environ["SSL_CERT_FILE"] == str(trust_bundle)
    assert os.environ["REQUESTS_CA_BUNDLE"] == str(trust_bundle)
