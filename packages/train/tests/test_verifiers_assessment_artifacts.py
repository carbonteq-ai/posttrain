"""Native transient artifacts survive sealing without unsafe archive paths."""

import gzip
import hashlib
import json
import tarfile
import threading
from types import SimpleNamespace

import pytest
from posttrain.common import ContractError, LocalArtifactRef
from posttrain.train.integrations.verifiers import VerifiersEnvironmentRolloutBridge
from posttrain.train.integrations.verifiers_assessment_artifacts import (
    retain_episode_artifacts,
    seal_assessment_artifacts,
)


def _episode(artifacts):
    return SimpleNamespace(
        id="episode/unsafe",
        traces=[SimpleNamespace(id="trace", state=SimpleNamespace(artifacts=artifacts))],
    )


def test_retains_bytes_and_unavailable_sources_with_safe_deterministic_bundle(tmp_path):
    artifacts = {"../../private/source": b"runtime tar bytes", "/optional": None, "copy": b"runtime tar bytes"}
    episode = _episode(artifacts)
    before = artifacts.copy()
    manifest = retain_episode_artifacts(episode, tmp_path)
    assert manifest["episode_id"] == episode.id
    assert artifacts == before
    assert len(list((tmp_path / "blobs").iterdir())) == 1
    assert retain_episode_artifacts(episode, tmp_path) == manifest
    first = seal_assessment_artifacts(tmp_path, name="native/evidence")
    second = seal_assessment_artifacts(tmp_path, name="native/evidence")
    assert first == second
    assert first is not None and isinstance(first.reference, LocalArtifactRef)
    assert first.reference.digest == hashlib.sha256(first.reference.path.read_bytes()).hexdigest()
    assert first.metadata["unavailable_artifact_count"] == 1
    assert first.metadata["external_references_resolved"] is False
    with tarfile.open(first.reference.path) as archive:
        members = archive.getnames()
        assert len(members) == 2
        assert all(name.startswith(("manifests/", "blobs/")) and ".." not in name for name in members)
        for member in archive:
            stream = archive.extractfile(member)
            assert stream is not None
            value = stream.read()
            if member.name.startswith("blobs/"):
                assert value == b"runtime tar bytes"
            else:
                assert json.loads(value) == manifest
    assert not tuple(tmp_path.rglob(".retain-*"))
    assert not tuple(tmp_path.glob(".bundle-*"))


def test_no_artifacts_creates_no_bundle(tmp_path):
    assert retain_episode_artifacts(_episode({}), tmp_path)["artifacts"] == []
    assert seal_assessment_artifacts(tmp_path, name="evidence") is None
    assert not list(tmp_path.iterdir())


def test_corrupt_existing_bytes_are_rejected_without_overwrite(tmp_path):
    episode = _episode({"source": b"bytes"})
    retain_episode_artifacts(episode, tmp_path)
    blob = next((tmp_path / "blobs").iterdir())
    blob.write_bytes(b"changed")
    with pytest.raises(ContractError, match="modified"):
        retain_episode_artifacts(episode, tmp_path)
    with pytest.raises(ContractError, match="modified"):
        seal_assessment_artifacts(tmp_path, name="evidence")
    assert blob.read_bytes() == b"changed"


def test_missing_bytes_prevent_sealing_and_changed_state_retains_history(tmp_path):
    episode = _episode({"source": b"first"})
    retain_episode_artifacts(episode, tmp_path)
    episode.traces[0].state.artifacts["source"] = b"second"
    retain_episode_artifacts(episode, tmp_path)
    assert len(list((tmp_path / "manifests").iterdir())) == 2
    next((tmp_path / "blobs").iterdir()).unlink()
    with pytest.raises(ContractError, match="missing"):
        seal_assessment_artifacts(tmp_path, name="evidence")


def test_collector_publishes_native_json_and_retained_bytes_together(tmp_path):
    bridge = object.__new__(VerifiersEnvironmentRolloutBridge)
    bridge.trace_path = tmp_path / "traces.jsonl"
    bridge._write_lock = threading.Lock()
    object.__setattr__(bridge, "_dataset", SimpleNamespace(id="data", revision="v1"))
    bridge.technique = "sampo"
    bridge.environment_id = "test-env"
    episode = _episode({"source": b"evidence"})
    record = {
        "id": episode.id,
        "traces": [{"id": "trace"}],
        "assessment_batches": [{"source": {"source_json": "retained-input"}}],
    }
    episode.to_record = lambda: record
    encoded = bridge._preserve_episode(episode)
    assert json.loads(encoded.line) == record
    assert encoded.trace_ids == ("trace",) and encoded.assessment is not None
    artifacts = bridge.finalize()
    assert len(artifacts) == 2
    bundle, native = artifacts
    assert isinstance(bundle.reference, LocalArtifactRef)
    assert isinstance(native.reference, LocalArtifactRef)
    assert json.loads(gzip.decompress(native.reference.path.read_bytes())) == record
    with tarfile.open(bundle.reference.path) as archive:
        blob = next(member for member in archive if member.name.startswith("blobs/"))
        stream = archive.extractfile(blob)
        assert stream is not None and stream.read() == b"evidence"
    assert bundle.metadata["external_references_resolved"] is False
