"""Verify task breadth, sealed split boundaries and no silent task substitution."""

import asyncio
from collections import Counter
from types import SimpleNamespace

import pytest
from sdpo_pool124_config import (
    collection_budget,
    known_frozen_scoring_errors,
    profile,
    step_tasks,
    task_batch,
    validate_pool,
)


def test_round_robin_covers124_and_balances_all50updates():
    tasks = [f"task-{i}" for i in range(124)]
    populations = [step_tasks(tasks, step) for step in range(1, 51)]
    assert all(len(p) == len(set(p)) == 32 for p in populations)
    assert set(sum(populations[:4], [])) == set(tasks)
    counts = Counter(sum(populations, []))
    assert set(counts.values()) == {12, 13}
    assert populations == [step_tasks(list(reversed(tasks)), step) for step in range(1, 51)]


def test_pool_authority_rejects_old_reserves_and_holdout_overlap():
    pool = {"task_names": [f"train-{i}" for i in range(124)]}
    heldout = {"task_names": [f"test-{i}" for i in range(62)]}
    cohort = {"train": pool["task_names"], "development": [], "reserve": []}
    assert len(validate_pool(cohort, pool, heldout)) == 124
    for changed in (dict(cohort, reserve=["train-0"]), dict(cohort, train=pool["task_names"][:32])):
        with pytest.raises(ValueError):
            validate_pool(changed, pool, heldout)
    with pytest.raises(ValueError):
        validate_pool(cohort, pool, {"task_names": ["train-0"] + heldout["task_names"][1:]})


def test_one_complete_transition_each_task_and_missing_task_fails():
    tasks = [f"task-{i}" for i in range(32)]
    rows = [{"task": task, "episode_id": f"{task}-{i}", "generation_index": i,
             "terminal_text": bool(i % 2), "native_success": 0,
             "executed_tool_responses": 1} for task in tasks for i in range(4)]
    batch = task_batch(rows, tasks, offset=3)
    assert Counter(row["task"] for row in batch) == Counter(tasks)
    with pytest.raises(ValueError):
        task_batch([r for r in rows if r["task"] != tasks[-1]], tasks, offset=3)


def test_profile_does_not_relax_update_batch_or_data_caps():
    args = dict(run_profile="pool124-50", steps=50, batch_size=32, online_rollouts=2,
                lr=1e-5, lr_end_ratio=.1, lr_schedule="hold10-decay20-flat",
                context_limit=16384, reply_limit=2048, thinking=False, qualify_capacity=False,
                shared_budget=None, teacher_feedback="solution-or-feedback",
                tool_parser_mode="schema-bound-v1", selection_mode="failure-frontier",
                alpha=1., topk=20, teacher_rate=.01)
    assert profile(SimpleNamespace(**args))["save_every"] == 10
    for override in ({"steps": 250}, {"online_rollouts": 4}, {"batch_size": 16}):
        with pytest.raises(ValueError):
            profile(SimpleNamespace(**(args | override)))


def test_eval_uses_full_frozen_budgets_not_training_budgets():
    assert collection_budget("heldout62") == {
        "context": 65536, "reply": 8192, "turns": 20, "episode": 32768, "replicas": 5}
    assert collection_budget("online_train")["context"] == 16384
    assert collection_budget("online_train")["replicas"] == 2


def test_only_known_frozen_finalization_errors_may_be_retained_in_eval():
    assert known_frozen_scoring_errors([{"type": "TaskError", "message": "PostContent extra article_url"}])
    assert not known_frozen_scoring_errors([{"type": "HarnessError", "message": "timeout"}])
    assert not known_frozen_scoring_errors([{"type": "TaskError", "message": "different error"}])
    assert not known_frozen_scoring_errors([])


def test_long_serving_gate_restores_training_admission_budget(tmp_path, monkeypatch):
    import sdpo_native_tool_gate
    from sdpo_pool124_runtime import qualify_long_rollout

    class Rollout:
        reply_limit = 2048

        async def generate(self, prefix, max_tokens):
            assert len(prefix) + max_tokens == 65536 and max_tokens == 8192
            assert self.reply_limit == 8192
            return [1], True, {"finish_reason": "stop"}

        async def batch_probe(self, prefix, maximum):
            assert len(prefix) == 57344 and maximum == 32
            return {"exact_batch_invariance": True, "records": [{"peak_in_flight": 32}]}

    async def gate(agent):
        assert agent.args.context_limit == 65536 and agent.args.reply_limit == 8192
        return {"qualified": True}

    monkeypatch.setattr(sdpo_native_tool_gate, "qualify", gate)
    args = SimpleNamespace(output=tmp_path, context_limit=16384, reply_limit=2048)
    agent = SimpleNamespace(args=args, rollout=Rollout(), tokenizer=SimpleNamespace(encode=lambda *a, **kw: [1, 2]))
    asyncio.run(qualify_long_rollout(agent))
    assert args.output == tmp_path and args.context_limit == 16384 and args.reply_limit == 2048
    assert agent.rollout.reply_limit == 2048
