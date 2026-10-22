"""Long-run authorization, LR boundaries and accumulated batch semantics."""

from types import SimpleNamespace

import pytest
from sdpo_agent_config import TRAIN
from sdpo_long_campaign_config import campaign_batch, campaign_learning_rate, distinct_campaign_batch, execution_profile


def configuration(**overrides):
    args = dict(run_profile="campaign250", steps=250, batch_size=32, online_rollouts=4,
                lr=1e-5, lr_end_ratio=.1, lr_schedule="hold10-decay20-flat",
                context_limit=16384, reply_limit=2048, thinking=False, qualify_capacity=False,
                shared_budget=None, teacher_feedback="solution-or-feedback", tool_parser_mode="schema-bound-v1",
                selection_mode="failure-frontier", alpha=1., topk=20, teacher_rate=.01)
    return SimpleNamespace(**(args | overrides))


def test_hold_cooldown_and_flat_applied_rate_boundaries():
    rates = [campaign_learning_rate(1e-5, i) for i in range(1, 251)]
    assert rates[:10] == [1e-5] * 10
    assert all(a > b for a, b in zip(rates[9:19], rates[10:20], strict=True))
    assert rates[19:] == pytest.approx([1e-6] * 231)
    assert len(set(rates[19:])) == 1
    assert campaign_learning_rate(1e-5, 15) == pytest.approx(5.5e-6)
    with pytest.raises(ValueError):
        campaign_learning_rate(1e-5, 251)


def test_new_campaign_does_not_lift_bounded_ablation_caps():
    with pytest.raises(ValueError):
        execution_profile(configuration(run_profile="ablation"))
    assert execution_profile(configuration()) == {
        "batch": 32, "evaluate_every": 25, "save_every": 25, "time_limit_s": 86200}
    assert execution_profile(configuration(run_profile="campaign-qualify", steps=2))["save_every"] == 1
    for override in ({"batch_size": 16}, {"steps": 249}, {"online_rollouts": 2},
                     {"thinking": True}, {"context_limit": 32768}, {"lr": 1e-6}):
        with pytest.raises(ValueError):
            execution_profile(configuration(**override))


def test_batch32_balances_same_eight_tasks_without_duplicate_actions():
    rows = [{"task": task, "episode_id": f"{task}-{replica}", "generation_index": replica,
             "terminal_text": replica > 1, "native_success": 1, "executed_tool_responses": 1}
            for task in TRAIN for replica in range(4)]
    batch = campaign_batch(rows, TRAIN)
    distinct_campaign_batch(batch, 32)
    assert all(sum(r["task"] == task for r in batch) == 4 for task in TRAIN)
    assert sum(r["terminal_text"] for r in batch) == 16
    with pytest.raises(ValueError):
        distinct_campaign_batch(batch[:31] + batch[:1], 32)


def test_accumulation32_is_mean_gradient_not_double_batch16_scale():
    import torch
    from trl.experimental.sdpo.loss_utils import compute_topk_self_distillation_loss

    torch.manual_seed(18)
    parameter = torch.nn.Parameter(torch.randn(2, 32))
    teachers = torch.randn(32, 2, 32)

    def loss(teacher):
        return compute_topk_self_distillation_loss(
            parameter, teacher, distillation_topk=20, distillation_alpha=1.,
            distillation_add_tail=True).mean()

    expected = torch.autograd.grad(torch.stack([loss(t) for t in teachers]).mean(), parameter)[0]
    for teacher in teachers:
        (loss(teacher) / 32).backward()
    torch.testing.assert_close(parameter.grad, expected)
