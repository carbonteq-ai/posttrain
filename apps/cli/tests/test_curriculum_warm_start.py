"""`--curriculum-from-run` binds another run's published adaptive-curriculum-state as a warm-start input."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
from posttrain.common import ContractError, JsonValue
from posttrain.tracking import ArtifactLink, RunSpec, StoredArtifact
from posttrain_cli.cli import main
from posttrain_cli.commands.work_package import _select_curriculum_output
from posttrain_cli.execution_config import ResolvedExecutionSettings
from posttrain_cli.execution_planning import (
    PlannedJobExecution,
    PlannedJobLaunch,
    PlannedJobPackage,
    with_curriculum_state,
    with_recovery_checkpoint,
)


def _planned(job_kind: str = "train.grpo", run_id: str = "new-run") -> PlannedJobExecution:
    spec = RunSpec(
        project_id="example",
        work_package_id="train/example",
        stage="train",
        job_kind=job_kind,
        job_definition_version="train/trl-grpo@1",
        run_id=run_id,
    )
    return PlannedJobExecution(
        package=cast(PlannedJobPackage, SimpleNamespace()),
        launch=PlannedJobLaunch(spec, cast(ResolvedExecutionSettings, SimpleNamespace()), ()),
    )


def _link(
    kind: str = "adaptive-curriculum-state",
    digest: str | None = "c" * 64,
    metadata: dict[str, JsonValue] | None = None,
    name: str = "training-model-olmo3-adaptive-curriculum-state",
) -> ArtifactLink:
    return ArtifactLink(
        direction="output",
        logical_name="training/model/olmo3/adaptive-curriculum-state",
        kind=kind,
        artifact=StoredArtifact(
            provider="trackio",
            namespace="example",
            name=name,
            version="v6",
            digest=digest,
            provider_metadata=metadata if metadata is not None else {"format": "queued-jsonl-with-snapshot"},
        ),
    )


def test_curriculum_state_binds_a_warm_start_input() -> None:
    rebound = with_curriculum_state(_planned(), source_run_id="vortex-run", artifact=_link())

    selected = rebound.launch.run_spec.artifacts["curriculum_state"]
    assert selected.kind == "adaptive-curriculum-state"
    assert selected.reference.version == "v6"
    assert rebound.launch.run_spec.resolved_inputs["curriculum_state"] == {
        "source_run_id": "vortex-run",
        "logical_name": "training/model/olmo3/adaptive-curriculum-state",
        "provider": "trackio",
        "namespace": "example",
        "name": "training-model-olmo3-adaptive-curriculum-state",
        "version": "v6",
        "digest": "c" * 64,
    }


def test_curriculum_state_rejects_resume_wrong_kinds_and_same_run() -> None:
    recovery = ArtifactLink(
        direction="output",
        logical_name="training/model/olmo3/recovery-checkpoint",
        kind="training-checkpoint",
        artifact=StoredArtifact(
            provider="trackio", namespace="example", name="checkpoint", version="v1", digest="a" * 64
        ),
    )
    resumed = with_recovery_checkpoint(_planned(), source_run_id="old-run", artifact=recovery)
    with pytest.raises(ContractError, match="restores its own curriculum"):
        with_curriculum_state(resumed, source_run_id="vortex-run", artifact=_link())
    with pytest.raises(ContractError, match="adaptive-curriculum-state or training-checkpoint artifact"):
        with_curriculum_state(_planned(), source_run_id="vortex-run", artifact=_link(kind="model-adapter"))
    with pytest.raises(ContractError, match="committed content digest"):
        with_curriculum_state(_planned(), source_run_id="vortex-run", artifact=_link(digest=None))
    with pytest.raises(ContractError, match="training jobs"):
        with_curriculum_state(_planned(job_kind="eval.general"), source_run_id="vortex-run", artifact=_link())
    with pytest.raises(ContractError, match="different run identity"):
        with_curriculum_state(_planned(run_id="same"), source_run_id="same", artifact=_link())


def test_job_run_exposes_curriculum_warm_start(capsys) -> None:
    assert main(["job", "run", "--help"]) == 0
    help_text = capsys.readouterr().out
    assert "--curriculum-from-run" in help_text
    assert "warm-start the adaptive curriculum" in help_text


def test_job_run_rejects_warm_start_together_with_resume(capsys) -> None:
    code = main(["job", "run", "missing.yaml", "--resume-from-run", "old-run", "--curriculum-from-run", "vortex-run"])
    assert code != 0
    assert "cannot be combined with --resume-from-run" in capsys.readouterr().err


def test_curriculum_selection_uses_the_final_state_or_the_state_at_a_checkpoint_step() -> None:
    final = _link(digest="f" * 64)
    view10 = _link(
        digest="a" * 64, metadata={"checkpoint_step": 10, "global_step": 10}, name="checkpoint-10-curriculum"
    )
    view20 = _link(
        digest="b" * 64, metadata={"checkpoint_step": 20, "global_step": 20}, name="checkpoint-20-curriculum"
    )
    recovery10 = _link(kind="training-checkpoint", digest="d" * 64, metadata={"checkpoint_step": 10, "global_step": 10})
    links = (final, view10, view20, recovery10)

    assert _select_curriculum_output(links, source_run_id="run", step=None) is final
    assert _select_curriculum_output(links, source_run_id="run", step=20) is view20
    assert _select_curriculum_output(links, source_run_id="run", step=10) is view10


def test_curriculum_selection_falls_back_to_the_recovery_checkpoint_for_older_runs() -> None:
    periodic = _link(kind="training-checkpoint", digest="e" * 64, metadata={"checkpoint_step": 20, "global_step": 20})
    final_recovery = _link(kind="training-checkpoint", digest="e" * 64, metadata={"global_step": 20})
    assert _select_curriculum_output((periodic, final_recovery), source_run_id="run", step=20) is periodic


def test_curriculum_selection_without_a_final_state_names_the_available_steps() -> None:
    view10 = _link(metadata={"checkpoint_step": 10}, name="checkpoint-10-curriculum")
    with pytest.raises(ContractError, match=r"0 final adaptive-curriculum-state outputs.*\[10\]"):
        _select_curriculum_output((view10,), source_run_id="interrupted-run", step=None)


def test_curriculum_checkpoint_step_requires_a_curriculum_source(capsys) -> None:
    assert main(["job", "run", "missing.yaml", "--curriculum-checkpoint-step", "20"]) != 0
    assert "--curriculum-checkpoint-step requires --curriculum-from-run" in capsys.readouterr().err


def test_a_verl_run_checkpoint_curriculum_view_is_selectable_and_warm_starts(tmp_path) -> None:
    """A veRL run publishes the same per-checkpoint views as TRL; step N warm-starts a new run."""

    from dataclasses import dataclass, field

    from posttrain.train.adaptive_curriculum import CURRICULUM_SNAPSHOT_NAME
    from posttrain.train.backends.verl.curriculum import CURRICULUM_JOURNAL_NAME, PosttrainCurriculumSelector
    from posttrain.train.backends.verl.curriculum import SelectorConfig as VerlSelectorConfig
    from posttrain.train.backends.verl.launcher import _publish_curriculum_state
    from posttrain.train.backends.verl.worker import CURRICULUM_CHECKPOINT_VIEWS_DIR, CURRICULUM_STATE_DIR

    rows = [{"example_id": f"task/{index}", "domain": "mail" if index % 2 else "crm"} for index in range(6)]
    settings: dict[str, JsonValue] = {"class_field": "domain", "policy": "yield_first", "seed": 3}
    output = tmp_path / "trainer"
    output.mkdir()
    VerlSelectorConfig(
        settings=settings,
        num_generations=2,
        state_dir=output / CURRICULUM_STATE_DIR,
        journal_path=output / CURRICULUM_JOURNAL_NAME,
        checkpoint_views_dir=output / CURRICULUM_CHECKPOINT_VIEWS_DIR,
    ).write(output / "selector.json")
    selector = PosttrainCurriculumSelector(rows, config_path=str(output / "selector.json"))
    for step in (1, 2):
        chosen = selector.select(2, global_steps=step, stage="active_sampling_refill", round_index=1)
        selector.observe([(chosen[0], [0.0, 1.0])], global_steps=step)
        checkpoint = output / "checkpoints" / f"global_step_{step}"
        checkpoint.mkdir(parents=True)
        selector.save_checkpoint(str(checkpoint))
    (output / CURRICULUM_STATE_DIR / CURRICULUM_SNAPSHOT_NAME).write_bytes(
        (checkpoint / CURRICULUM_SNAPSHOT_NAME).read_bytes()
    )
    selector.close()

    @dataclass
    class _Context:
        run_id: str = "verl-run"
        produced: list = field(default_factory=list)

        def artifact(self, artifact) -> None:
            self.produced.append(artifact)

        def event(self, name, attributes=None) -> None:
            pass

        def metrics(self, values, *, step, attributes=None) -> None:
            pass

        def metric(self, name, value, *, step, attributes=None) -> None:
            pass

    request = SimpleNamespace(
        policy=SimpleNamespace(id="lfm2.5-1.2b"),
        settings=SimpleNamespace(
            algorithm="olmo3",
            id="settings/vortex@1",
            revision="1",
            adaptive_curriculum=SimpleNamespace(class_field="domain"),
        ),
        training=SimpleNamespace(update=SimpleNamespace(kind="lora")),
    )
    context = _Context()
    _publish_curriculum_state(cast(object, context), cast(object, request), output)  # type: ignore[arg-type]

    links = tuple(
        ArtifactLink(
            direction="output",
            logical_name=artifact.name,
            kind=artifact.kind,
            artifact=StoredArtifact(
                provider="trackio",
                namespace="example",
                name=artifact.name.replace("/", "-"),
                version="v1",
                digest=artifact.reference.digest,
                provider_metadata=dict(artifact.metadata),
            ),
        )
        for artifact in context.produced
    )
    final = cast(ArtifactLink, _select_curriculum_output(links, source_run_id="verl-run", step=None))
    assert final.logical_name.endswith("adaptive-curriculum-state")
    chosen = cast(ArtifactLink, _select_curriculum_output(links, source_run_id="verl-run", step=1))
    assert chosen.logical_name == "training/lfm2.5-1.2b/olmo3/checkpoint-00000001/curriculum"
    (view,) = [item for item in context.produced if item.name == chosen.logical_name]
    import json

    # The step-1 view holds the controller as it was at step 1, not the final state.
    assert json.loads((view.reference.path / CURRICULUM_SNAPSHOT_NAME).read_text())["decision_index"] == 1
    assert view.metadata["checkpoint_view"] == "curriculum" and view.role == "checkpoint-curriculum"

    # The view is a directory holding the controller snapshot: a new veRL run warm-starts from it.
    warm = tmp_path / "warm"
    warm.mkdir()
    VerlSelectorConfig(
        settings=settings,
        num_generations=2,
        state_dir=warm / "state",
        journal_path=warm / "journal.jsonl",
        warm_start_state_dir=view.reference.path,
    ).write(warm / "selector.json")
    warmed = PosttrainCurriculumSelector(rows, config_path=str(warm / "selector.json"))
    warmed.select(2, global_steps=1, stage="active_sampling_refill", round_index=1)
    events = [
        line for line in (warm / "journal.jsonl").read_text().splitlines() if "adaptive_curriculum_started" in line
    ]
    assert '"warm_started":true' in events[0]
    warmed.close()
