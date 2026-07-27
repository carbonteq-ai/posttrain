"""Compose protected tracking service bindings for CLI readers and Observatory."""

from __future__ import annotations

import os
import ssl
from pathlib import Path
from typing import Any

from posttrain.catalog import ProjectLayout
from posttrain.execution import ExecutionEvidenceSource

from .execution_config import (
    load_execution_environment,
    load_local_execution_config,
)


def project_tracking_environment(layout: ProjectLayout) -> dict[str, str]:
    """Load service bindings and install the host's verified CA path for SDKs."""

    local = load_local_execution_config(layout)
    environment = load_execution_environment(local)
    verify_paths = ssl.get_default_verify_paths()
    default_ca = next(
        (
            str(path)
            for value in (
                verify_paths.cafile,
                verify_paths.openssl_cafile,
                "/etc/ssl/certs/ca-certificates.crt",
            )
            if value and (path := Path(value)).is_file()
        ),
        None,
    )
    for name in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
        configured = environment.get(name)
        if configured:
            os.environ.setdefault(name, configured)
        elif default_ca:
            os.environ.setdefault(name, default_ca)
    return environment


def project_observatory_settings(
    layout: ProjectLayout,
    settings_type: Any,
    *,
    host: str | None = None,
    port: int | None = None,
    evidence_source: ExecutionEvidenceSource | None = None,
) -> Any:
    """Apply the same protected endpoint binding used by reconciliation."""

    tracking = evidence_source.provider if evidence_source is not None else layout.tracking
    if tracking not in {"trackio", "wandb"}:
        raise RuntimeError("Observatory requires Trackio or W&B evidence")
    settings = settings_type.for_project(
        layout.project_id,
        tracking,
        host=host,
        port=port,
    )
    environment = project_tracking_environment(layout)
    updates: dict[str, str | None] = {}
    if evidence_source is not None:
        updates["source_id"] = evidence_source.source_id
        if tracking == "trackio":
            updates["trackio_project"] = evidence_source.project
            updates["trackio_server_url"] = evidence_source.endpoint
        else:
            updates["wandb_entity"] = evidence_source.scope
            updates["wandb_project"] = evidence_source.project
            updates["wandb_base_url"] = evidence_source.endpoint
    elif tracking == "trackio":
        if server_url := environment.get("POSTTRAIN_TRACKIO_SERVER_URL"):
            updates["trackio_server_url"] = server_url
        updates["trackio_project"] = environment.get(
            "POSTTRAIN_TRACKIO_PROJECT",
            layout.project_id,
        )
    elif tracking == "wandb":
        if entity := environment.get("WANDB_ENTITY"):
            updates["wandb_entity"] = entity
        updates["wandb_project"] = environment.get(
            "POSTTRAIN_WANDB_PROJECT",
            layout.project_id,
        )
        if base_url := environment.get("WANDB_BASE_URL"):
            updates["wandb_base_url"] = base_url
    return settings.model_copy(update=updates) if updates else settings


__all__ = ["project_observatory_settings", "project_tracking_environment"]
