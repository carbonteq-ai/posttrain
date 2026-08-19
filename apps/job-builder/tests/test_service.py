from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from posttrain.common import ContractError
from posttrain_job_builder.service import create_app, load_config


def _config_file(tmp_path: Path) -> tuple[Path, str]:
    token = "ambient-builder-token"
    path = (tmp_path / "job-builder.json").resolve()
    path.write_text(
        json.dumps(
            {
                "store_root": str((tmp_path / "store").resolve()),
                "staging_root": str((tmp_path / "staging").resolve()),
                "receipt_root": str((tmp_path / "receipts").resolve()),
                "repository_prefix": "registry.example/posttrain-projects",
                "infrastructure_grants": {
                    hashlib.sha256(token.encode()).hexdigest(): {
                        "principal": "hammad",
                    }
                },
                "builder": "posttrain-job-builder",
                "max_context_bytes": 4096,
                "max_file_count": 32,
                "max_blob_bytes": 2048,
                "poll_seconds": 0.01,
            }
        ),
        encoding="utf-8",
    )
    path.chmod(0o600)
    return path, token


def test_service_composes_authenticated_capabilities_and_liveness(tmp_path: Path) -> None:
    path, token = _config_file(tmp_path)
    config = load_config({"POSTTRAIN_JOB_BUILDER_CONFIG": str(path)})

    with TestClient(create_app(config)) as client:
        assert client.get("/health/live").json() == {"status": "live"}
        assert client.get("/health/ready").status_code == 204
        capabilities = client.get("/v1/capabilities", headers={"Authorization": f"Bearer {token}"})
        assert capabilities.status_code == 200
        assert capabilities.json()["platforms"] == ["linux/amd64"]


def test_service_rejects_an_unprotected_config_file(tmp_path: Path) -> None:
    path, _ = _config_file(tmp_path)
    path.chmod(0o644)

    with pytest.raises(ContractError, match="protected absolute file"):
        load_config({"POSTTRAIN_JOB_BUILDER_CONFIG": str(path)})


def test_service_rejects_legacy_project_scoped_token_grants(tmp_path: Path) -> None:
    path, _ = _config_file(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    grants = payload.pop("infrastructure_grants")
    payload["token_grants"] = {
        digest: {**grant, "project_ids": ["ambient-agent"]} for digest, grant in grants.items()
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ContractError, match="unsupported fields"):
        load_config({"POSTTRAIN_JOB_BUILDER_CONFIG": str(path)})


def test_service_rejects_project_ids_inside_an_infrastructure_grant(tmp_path: Path) -> None:
    path, _ = _config_file(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    digest, grant = next(iter(payload["infrastructure_grants"].items()))
    payload["infrastructure_grants"] = {digest: {**grant, "project_ids": ["ambient-agent"]}}
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ContractError, match="infrastructure grants are invalid"):
        load_config({"POSTTRAIN_JOB_BUILDER_CONFIG": str(path)})
