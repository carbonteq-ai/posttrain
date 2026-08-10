"""Durable checkpoint publication tests for long TRL jobs."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

from posttrain.common import ProducedArtifact, RunContext
from posttrain.common.variants import GEMMA_4_E2B_IT
from posttrain.train import LoRAUpdate
from posttrain.train.backends.trl.common import (
    checkpoint_callback_type,
    restore_checkpoint_runtime_states,
)


@dataclass
class _Observer:
    artifacts: list[ProducedArtifact] = field(default_factory=list)

    def event(self, observation) -> None:
        del observation

    def metric(self, observation) -> None:
        del observation

    def metrics(self, observation) -> None:
        del observation

    def trace(self, observation) -> None:
        del observation

    def artifact(self, artifact: ProducedArtifact) -> None:
        self.artifacts.append(artifact)


class _Callback:
    pass


def _write_ledger(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS state (value TEXT NOT NULL)")
        connection.execute("DELETE FROM state")
        connection.execute("INSERT INTO state VALUES (?)", (value,))


def _read_ledger(path: Path) -> str:
    with sqlite3.connect(path) as connection:
        row = connection.execute("SELECT value FROM state").fetchone()
    assert row is not None
    return str(row[0])


def test_checkpoint_callback_snapshots_sqlite_state_and_publishes_artifact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    observer = _Observer()
    context = RunContext(
        project_id="project",
        work_package_id="work-package",
        run_id="run",
        job_kind="train.distill",
        job_definition_version="1",
        workspace=tmp_path.resolve(),
        observer=observer,
    )
    runtime_path = Path(".policy-prism/scope-opd-ledger.sqlite3")
    _write_ledger(runtime_path, "step-16")
    output = tmp_path / "trainer"
    checkpoint = output / "checkpoint-16"
    checkpoint.mkdir(parents=True)
    for name in (
        "adapter_config.json",
        "adapter_model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
    ):
        (checkpoint / name).write_bytes(name.encode())
    (checkpoint / "trainer_state.json").write_text(
        '{"global_step": 16}\n', encoding="utf-8"
    )

    callback = checkpoint_callback_type(
        context,
        {
            "TrainerCallback": _Callback,
            "get_last_checkpoint": lambda _: str(checkpoint),
        },
        model=GEMMA_4_E2B_IT,
        technique="distill",
        settings=SimpleNamespace(id="training/opd-test", revision="1"),
        update=LoRAUpdate(),
        workspace=tmp_path,
        runtime_state_paths=(runtime_path,),
    )()
    callback.on_save(
        SimpleNamespace(output_dir=str(output)),
        SimpleNamespace(global_step=16),
        object(),
    )

    assert len(observer.artifacts) == 2
    recovery = next(artifact for artifact in observer.artifacts if artifact.role == "recovery")
    model = next(artifact for artifact in observer.artifacts if artifact.role == "checkpoint-model")
    assert recovery.name.endswith("/checkpoint-00000016/recovery")
    assert recovery.kind == "training-checkpoint"
    assert model.name.endswith("/checkpoint-00000016/model")
    assert model.kind == "model-adapter"
    snapshot = checkpoint / "posttrain-runtime-state" / runtime_path
    assert _read_ledger(snapshot) == "step-16"

    _write_ledger(runtime_path, "corrupt-later-state")
    restore_checkpoint_runtime_states(checkpoint, (runtime_path,))
    assert _read_ledger(runtime_path) == "step-16"
