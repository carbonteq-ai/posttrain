"""Content-addressed active-collection evidence shared by native adapters.

Each snapshot is written once under its SHA-256 name, fsynced, and published as
one ``training-collection`` artifact, so earlier round evidence is never
overwritten and every backend reports candidate accounting identically.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from posttrain.common import LocalArtifactRef, ProducedArtifact, RunContext

from ..update_records import InvalidPolicyUpdate


def publish_collection_snapshot(context: RunContext, destination: Path, state: Mapping[str, Any]) -> str:
    """Durably retain one collection snapshot and publish it; returns its digest."""
    content = json.dumps(state, sort_keys=True, allow_nan=False).encode() + b"\n"
    digest = hashlib.sha256(content).hexdigest()
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{digest}.json"
    # Content-addressed snapshots never overwrite earlier round evidence.
    if not path.exists():
        descriptor, temporary = tempfile.mkstemp(prefix=".collection-", dir=destination)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                pass  # Concurrent publication must match the same bytes below.
        finally:
            os.unlink(temporary)
        descriptor = os.open(destination, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    if path.read_bytes() != content:
        raise InvalidPolicyUpdate("native collection snapshot differs from retained content")
    context.artifact(
        ProducedArtifact(
            f"training/collection/{digest}",
            "training-collection",
            LocalArtifactRef(path, digest),
            role="collection-evidence",
            metadata={"sampler_step": state["sampler_step"], "status": state["status"], "schema": state["schema"]},
        )
    )
    return digest
