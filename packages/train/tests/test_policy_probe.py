"""Post-step movement must not be confused with loss-time clipping."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, cast

import pytest
from posttrain.train.backends.trl.policy_probe import PostUpdateProbe, movement_metrics
from posttrain.train.backends.trl.policy_telemetry import actor_update_callback_type, actor_update_trainer_type


@pytest.mark.parametrize("precomputed", [False, True])
def test_precomputed_credit_does_not_log_discarded_grpo_advantages(precomputed: bool) -> None:
    class Parent:
        def __init__(self) -> None:
            self.model = SimpleNamespace(training=False)
            self.use_precomputed_advantages = precomputed
            self._metrics = {"eval": {"advantages/abs_mean": [0.87], "advantages/scorable_fraction": [1.0]}}

        def log(self, logs: Any, start_time: Any = None) -> None:
            del logs, start_time

    trainer = actor_update_trainer_type(Parent, cast(Any, SimpleNamespace()))()
    trainer.log({})
    assert ("advantages/abs_mean" in trainer._metrics["eval"]) is not precomputed
    assert trainer._metrics["eval"]["advantages/scorable_fraction"] == [1.0]


def test_actual_optimizer_moves_past_bound_when_loss_time_ratio_is_one() -> None:
    torch = pytest.importorskip("torch")
    theta = torch.nn.Parameter(torch.tensor(0.0))
    logp = torch.nn.functional.logsigmoid(theta)
    old = logp.detach().clone()
    assert float((logp - old).detach().exp()) == 1.0
    optimizer = torch.optim.SGD([theta], lr=0.02)
    (-((logp - old).exp())).backward()
    optimizer.step()
    delta = (torch.nn.functional.logsigmoid(theta).detach() - old).reshape(1, 1)
    metrics = movement_metrics([(delta, torch.ones(1, 1), torch.ones(1, 1))], 0.003, 0.004)
    assert metrics["ratio_max"] == pytest.approx(1.004999958, abs=1e-7)
    assert metrics["clip_fraction"] == 1.0
    assert metrics["old_policy_sampled_k3"] > 0


def test_masks_signs_and_sequence_dilution() -> None:
    torch = pytest.importorskip("torch")
    # The masked tool token would overflow; it must not enter scores or ratios.
    observations = [
        (
            torch.tensor([[0.01, -1000.0], [0.01, 0.0], [-0.01, 0.0], [0.0, 0.0]]),
            torch.tensor([[1, 0], [1, 1], [1, 1], [0, 0]]),
            torch.tensor([[-1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]]),
        )
    ]
    metrics = movement_metrics(observations, 0.003, 0.004)
    assert metrics["scored_tokens"] == 5
    assert metrics["scored_rows"] == 3
    assert metrics["outside_fraction"] == 1
    assert metrics["clip_fraction"] == pytest.approx(1 / 3)
    assert metrics["token_abs_log_ratio_max"] == pytest.approx(0.01)
    diluted = movement_metrics(
        [(torch.tensor([[0.01, 0.0, 0.0, 0.0]]), torch.ones(1, 4), torch.ones(1, 4))], 0.003, 0.004
    )
    assert diluted["outside_fraction"] == 0


@pytest.mark.parametrize("rows", [True, -1, 17, 1.5, "4"])
def test_invalid_limits_rejected(rows: Any) -> None:
    with pytest.raises(ValueError, match="post_update_probe_rows"):
        PostUpdateProbe(cast(Any, SimpleNamespace()), rows)


@pytest.mark.parametrize("skipped", [False, True])
def test_capture_and_callback_use_frozen_scores_and_release_rows(skipped: bool) -> None:
    torch = pytest.importorskip("torch")
    emitted: list[Any] = []
    context = SimpleNamespace(metrics=lambda values, **kw: emitted.append((values, kw)))
    probe = PostUpdateProbe(cast(Any, context), 1)

    class Parent:
        def __init__(self) -> None:
            self.model = torch.nn.Linear(1, 1, bias=False)
            self.model.weight.data.zero_()
            self.model_wrapped = self.model
            self.args = SimpleNamespace(use_liger_kernel=False)
            self.accelerator = SimpleNamespace(
                num_processes=1, device="cpu", autocast=nullcontext, optimizer_step_was_skipped=skipped
            )
            self.state = SimpleNamespace(global_step=0)
            self.epsilon_low, self.epsilon_high = 0.003, 0.004
            self.importance_sampling_level = "sequence"

        def _get_per_token_logps_and_entropies(self, *args: Any, **kwargs: Any) -> Any:
            rows = int(args[1].shape[0]) if args else 2
            return torch.nn.functional.logsigmoid(self.model.weight).expand(rows, 2), None, None

        def _compute_loss(self, model: Any, inputs: Any) -> Any:
            return self._get_per_token_logps_and_entropies()[0].sum()

    telemetry: Any = SimpleNamespace(complete=lambda step: None)
    trainer = actor_update_trainer_type(Parent, telemetry, probe)()
    inputs = {
        "prompt_ids": torch.ones(2, 1, dtype=torch.long),
        "prompt_mask": torch.ones(2, 1),
        "completion_ids": torch.ones(2, 2, dtype=torch.long),
        "completion_mask": torch.ones(2, 2),
        "tool_mask": torch.tensor([[1, 0], [1, 1]]),
        "advantages": torch.ones(2, 2),
    }
    trainer._compute_loss(trainer.model, inputs)
    assert probe.captured_rows == 1
    assert probe.seen_rows == 2
    inputs["completion_mask"].zero_()  # snapshot owns its masks
    trainer.model.weight.data.fill_(0.02)
    trainer.state.global_step = 1
    callback = actor_update_callback_type({"TrainerCallback": object}, telemetry, probe)()
    callback.on_step_end(None, trainer.state, None)
    metrics, tags = emitted[0]
    assert tags == {"step": 1}
    assert metrics["train/rl/post_update/rows"] == 1
    assert metrics["train/rl/post_update/seen_rows"] == 2
    assert metrics["train/rl/post_update/optimizer_skipped"] == float(skipped)
    if skipped:
        assert "train/rl/post_update/ratio_max" not in metrics
    else:
        assert metrics["train/rl/post_update/clip_fraction"] == 1
        assert metrics["train/rl/post_update/scored_tokens"] == 1
    assert probe.snapshots == []
    assert probe.seen_rows == 0
