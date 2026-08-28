from __future__ import annotations

from types import SimpleNamespace

from posttrain_cli.commands.work_package import _select_checkpoint_output


def _link(*, checkpoint_step: int | None) -> SimpleNamespace:
    metadata = {} if checkpoint_step is None else {"checkpoint_step": checkpoint_step}
    return SimpleNamespace(
        direction="output",
        kind="model-adapter",
        artifact=SimpleNamespace(provider_metadata=metadata),
    )


def test_unscoped_model_selection_prefers_the_terminal_artifact() -> None:
    checkpoint = _link(checkpoint_step=1)
    terminal = _link(checkpoint_step=None)

    selected = _select_checkpoint_output(
        (checkpoint, terminal),
        source_run_id="run",
        kinds=frozenset({"model-adapter"}),
        step=None,
    )

    assert selected is terminal


def test_step_scoped_model_selection_uses_the_checkpoint_artifact() -> None:
    checkpoint = _link(checkpoint_step=1)
    terminal = _link(checkpoint_step=None)

    selected = _select_checkpoint_output(
        (checkpoint, terminal),
        source_run_id="run",
        kinds=frozenset({"model-adapter"}),
        step=1,
    )

    assert selected is checkpoint
