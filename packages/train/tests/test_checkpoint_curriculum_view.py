"""Each TRL checkpoint publishes a small adaptive-curriculum controller view when the controller saved one."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from posttrain.common.variants import QWEN_35_2B
from posttrain.train.adaptive_curriculum import CURRICULUM_SNAPSHOT_NAME
from posttrain.train.backends.trl.common import publish_checkpoint_views
from posttrain.train.bindings import FullParameterUpdate


class RecordingContext:
    run_id = "run-1"

    def __init__(self) -> None:
        self.artifacts: list[Any] = []
        self.events: list[tuple[str, dict[str, object]]] = []

    def artifact(self, artifact: Any) -> None:
        self.artifacts.append(artifact)

    def event(self, name: str, attributes: dict[str, object]) -> None:
        self.events.append((name, dict(attributes)))


def _checkpoint(tmp_path: Path, *, with_curriculum: bool) -> Path:
    checkpoint = tmp_path / "output" / "checkpoint-20"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 20}))
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    if with_curriculum:
        (checkpoint / CURRICULUM_SNAPSHOT_NAME).write_text(json.dumps({"version": 5, "active_step": 20}))
    return checkpoint


def _publish(context: RecordingContext, checkpoint: Path, workspace: Path) -> None:
    publish_checkpoint_views(
        cast(Any, context),
        checkpoint,
        model=QWEN_35_2B,
        technique="olmo3",
        settings=SimpleNamespace(id="settings", revision="1"),
        update=FullParameterUpdate(),
        workspace=workspace,
        interrupted=False,
    )


def test_checkpoint_publishes_a_curriculum_view_with_the_controller_snapshot(tmp_path: Path) -> None:
    context = RecordingContext()
    _publish(context, _checkpoint(tmp_path, with_curriculum=True), tmp_path / "workspace")

    views = {artifact.metadata["checkpoint_view"]: artifact for artifact in context.artifacts}
    curriculum = views["curriculum"]
    assert curriculum.kind == "adaptive-curriculum-state"
    assert curriculum.name.endswith("/olmo3/checkpoint-00000020/curriculum")
    assert curriculum.metadata["checkpoint_step"] == 20
    assert [path.name for path in curriculum.reference.path.iterdir()] == [CURRICULUM_SNAPSHOT_NAME]
    assert json.loads((curriculum.reference.path / CURRICULUM_SNAPSHOT_NAME).read_text())["active_step"] == 20
    assert dict(context.events)["checkpoint_saved"]["curriculum_view_published"] is True


def test_checkpoint_without_a_curriculum_publishes_no_curriculum_view(tmp_path: Path) -> None:
    context = RecordingContext()
    _publish(context, _checkpoint(tmp_path, with_curriculum=False), tmp_path / "workspace")

    assert {artifact.metadata["checkpoint_view"] for artifact in context.artifacts} == {"recovery"}
    assert dict(context.events)["checkpoint_saved"]["curriculum_view_published"] is False


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
