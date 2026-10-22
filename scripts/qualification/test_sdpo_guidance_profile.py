"""New authorization never silently changes old run or task/update semantics."""

from types import SimpleNamespace

import pytest
from sdpo_guidance_data import select
from sdpo_long_campaign_config import execution_profile


def config():
    return SimpleNamespace(run_profile="guidance-five", steps=10, batch_size=32,
        online_rollouts=2, context_limit=32768, reply_limit=4096, thinking=True,
        lr=1e-6, lr_schedule="constant", alpha=1., topk=20, teacher_rate=.01,
        teacher_feedback="prepared-guidance", teacher_guidance="pinned.json",
        concurrency=40, speculation="mtp", qualify_capacity=True, tool_parser_mode="schema-bound-v1")


def test_ten_updates_and_exact_end_evaluation():
    result = execution_profile(config())
    assert result["batch"] == 32 and result["evaluate_every"] == 10


def test_explicit_user_plain_decoding_amendment_retains_learning_profile():
    args = config()
    args.speculation = "none"
    with pytest.raises(ValueError):
        execution_profile(args)
    args.plain_decoding_authorized = True
    assert execution_profile(args)["batch"] == 32


@pytest.mark.parametrize("key,value", [("steps", 11), ("concurrency", 32),
    ("speculation", "none"), ("teacher_guidance", None), ("qualify_capacity", False),
    ("batch_size", 16), ("thinking", False)])
def test_no_silent_budget_or_runtime_fallback(key, value):
    args = config()
    setattr(args, key, value)
    with pytest.raises(ValueError):
        execution_profile(args)


def test_task_selection_excludes_protected_and_is_stable():
    def row(name, tool):
        return {"task": name, "reference_actions": [{"call": {"name": tool}}]}

    rows = [row(f"sales.task{i}", "lookup") for i in range(9)]
    rows += [row("simple.protected", "rare"), row("simple.b", "create"), row("simple.a", "read")]
    first = select(rows, {"simple.protected"}, simple_count=2)
    second = select(list(reversed(rows)), {"simple.protected"}, simple_count=2)
    assert first == second and len(first) == 11
    assert "simple.protected" not in {r["task"] for r in first}
