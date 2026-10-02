"""Worker observation forwarding preserves the host identity and retry cursor."""

import hashlib
import json
from types import SimpleNamespace
from typing import cast

import pytest
from posttrain.common import LocalArtifactRef, Observer, ProducedArtifact, RunContext
from posttrain.train.backends.verl.contracts import VerlRunContext
from posttrain.train.backends.verl.policy_observer import ResolvedWorkerObserver, observation_tailer


def setup(tmp_path, emit):
    context = RunContext(
        "project",
        "work",
        "run",
        "train.grpo",
        "job@1",
        tmp_path,
        observer=cast(Observer, SimpleNamespace(artifact=emit, metrics=emit, event=emit)),
    )
    transported = VerlRunContext.model_validate({**context.identity_attributes, "workspace": context.workspace})
    path = tmp_path / "observations.jsonl"
    return context, path, ResolvedWorkerObserver(path, transported)


def test_worker_artifact_and_context_metrics_roundtrip(tmp_path):
    emitted = []
    context, path, worker = setup(tmp_path, emitted.append)
    source = tmp_path / "population.jsonl"
    source.write_bytes(b"native population\n")
    artifact = ProducedArtifact(
        "population",
        "evaluation-traces",
        LocalArtifactRef(source, hashlib.sha256(source.read_bytes()).hexdigest()),
        metadata={"replay_authority": True},
    )
    worker.artifact(artifact)
    worker_context = RunContext(**worker.context.model_dump(), observer=worker)
    worker_context.metrics({"train/rl/loss": 0.5}, step=1)
    worker_context.event("resolved_update", {"applied": 1})
    tailer = observation_tailer(context, path)
    assert tailer.poll().emitted_records == 3
    assert emitted[0] == artifact
    assert emitted[1].step == 1 and emitted[1].attributes["run_id"] == "run"
    assert emitted[2].name == "resolved_update"
    assert tailer.poll().emitted_records == 3


def test_observer_failure_retries_without_acknowledging_record(tmp_path):
    emitted = []
    fail = True

    def emit(value):
        if fail:
            raise RuntimeError("host artifact queue temporarily unavailable")
        emitted.append(value)

    context, path, worker = setup(tmp_path, emit)
    RunContext(**worker.context.model_dump(), observer=worker).event("update")
    tailer = observation_tailer(context, path)
    assert not tailer.poll().complete and tailer.offset == 0
    fail = False
    assert tailer.poll().complete and len(emitted) == 1


@pytest.mark.parametrize("change", ["run", "schema"])
def test_worker_transport_rejects_wrong_run_or_schema(tmp_path, change):
    emitted = []
    context, path, worker = setup(tmp_path, emitted.append)
    RunContext(**worker.context.model_dump(), observer=worker).event("update")
    record = json.loads(path.read_text())
    if change == "run":
        record["context"]["run_id"] = "another-run"
    else:
        record["schema"] = "unknown"
    path.write_text(json.dumps(record) + "\n")
    tailer = observation_tailer(context, path)
    assert not tailer.poll().complete and emitted == []
