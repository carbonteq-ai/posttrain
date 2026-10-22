"""Research protocol invariants; no GPU or environment dependencies."""

import json
from pathlib import Path

from sdpo_agent_config import (
    TRAIN,
    admission,
    aligned_messages,
    balanced_rows,
    cohorts,
    feedback_text,
    observed_tool_error,
    population_rollouts,
    successful_reference,
    update_learning_rate,
)

ROOT = Path(__file__).resolve().parents[2]


def test_first_error_variant_targets_earliest_observed_error_and_keeps_terminal():
    rows = [{"task": "t", "episode_id": "e", "generation_index": i,
             "native_success": 0, "executed_tool_responses": 1, "observed_tool_error": i in (1, 2),
             "terminal_text": i == 4} for i in range(1, 5)]
    old = balanced_rows(rows, 2, tasks=["t"], mode="failure-frontier")
    new = balanced_rows(rows, 2, tasks=["t"], mode="first-error-frontier")
    assert [r["generation_index"] for r in old] == [3, 4]
    assert [r["generation_index"] for r in new] == [1, 4]


def test_first_error_variant_preserves_existing_frontier_when_no_failed_error_exists():
    rows = [{"task": "t", "episode_id": "e", "generation_index": i,
             "native_success": 0, "executed_tool_responses": 1,
             "observed_tool_error": False, "terminal_text": i == 4} for i in range(1, 5)]
    assert balanced_rows(rows, 2, tasks=["t"], mode="first-error-frontier") == balanced_rows(
        rows, 2, tasks=["t"], mode="failure-frontier")
    # An error in a successfully recovered episode is not treated as a failed-task target.
    for row in rows:
        row.update(native_success=1, observed_tool_error=True)
    assert balanced_rows(rows, 2, tasks=["t"], mode="first-error-frontier") == balanced_rows(
        rows, 2, tasks=["t"], mode="failure-frontier")


def test_schedule_rates_are_applied_to_updates_with_exact_endpoints():
    import pytest

    values = [update_learning_rate(1e-5, step, 20, "cosine-decay", .1) for step in range(1, 21)]
    assert values[0] == 1e-5
    assert values[-1] == pytest.approx(1e-6)
    assert all(a > b for a, b in zip(values, values[1:], strict=False))
    assert sum(values) / 20 == pytest.approx(5.5e-6)
    assert all(update_learning_rate(1e-5, step, 20) == 1e-5 for step in range(1, 21))
    with pytest.raises(ValueError):
        update_learning_rate(1e-5, 0, 20, "cosine-decay")
    with pytest.raises(ValueError):
        update_learning_rate(1e-5, 1, 1, "cosine-decay")


def test_online_group_size_never_changes_greedy_comparison_population():
    for group in (2, 4, 8):
        assert population_rollouts("online_train", group) == group
        assert population_rollouts("train", group) == 2
        assert population_rollouts("development", group) == 2
        assert population_rollouts("reserve", group) == 2


def test_verified_reference_prefers_own_success_and_never_uses_failed_sibling():
    def episode(name, score):
        return {"id": name, "traces": [{"metrics": {"task_completed_correctly": score}}]}
    own, other, failed = episode("own", 1), episode("other", 1), episode("failed", 0)
    assert successful_reference(own, [other, own]) is own
    assert successful_reference(failed, [failed, other]) is other
    assert successful_reference(failed, [failed]) is None


def test_cohorts_exclude_previously_inspected_reserve_and_heldout():
    def read(p):
        return json.loads((ROOT / p).read_text())
    protocol = read("docs/research/evidence/self-vllm-20261017/task-protocol.json")
    pool = read("scripts/qualification/fixtures/automationbench_train_pool_luna07_v1.json")
    heldout = read("scripts/qualification/fixtures/automationbench_heldout62_v1.json")
    old = read("docs/research/evidence/self-vllm-20261017/adaptive-transfer-reserve24.json")
    result = cohorts(protocol, pool, heldout, old)
    assert len(result["reserve"]) == 24
    assert not set(result["reserve"]) & (set(old["task_names"]) | set(result["train"]) | set(result["development"]))


def test_balance_includes_native_terminal_behavior_without_inventing_it():
    rows = [{"task": t, "terminal_text": terminal} for t in TRAIN for terminal in (False, True)]
    batch = balanced_rows(rows)
    assert len(batch) == 16
    assert sum(r["terminal_text"] for r in batch) == 8
    assert all(sum(r["task"] == t for r in batch) == 2 for t in TRAIN)


def test_wider_training_cohort_rotates_without_changing_optimizer_batch():
    tasks = [f"train{i}" for i in range(32)]
    rows = [{"task": task, "terminal_text": terminal} for task in tasks for terminal in (False, True)]
    batches = [balanced_rows(rows, 16, offset=i, tasks=tasks) for i in range(4)]
    assert all(len(batch) == 16 for batch in batches)
    assert {r["task"] for batch in batches for r in batch} == set(tasks)
    assert not ({r["task"] for r in batches[0]} & {r["task"] for r in batches[1]})
    assert all(sum(r["terminal_text"] for r in batch) == 8 for batch in batches)
    assert all(a["terminal_text"] != b["terminal_text"]
               for a, b in zip(batches[0], batches[2], strict=True))


def test_sampling_treatments_repair_duplicate_actions_and_prioritize_actual_errors():
    rows = [{"task": task, "episode_id": task, "generation_index": i,
             "terminal_text": False, "observed_tool_error": i == 2}
            for task in TRAIN for i in range(3)]
    legacy = balanced_rows(rows)
    distinct = balanced_rows(rows, mode="distinct-balanced")
    priority = balanced_rows(rows, mode="native-error-priority")
    assert len({(r["task"], r["generation_index"]) for r in legacy}) == 8
    assert len({(r["task"], r["generation_index"]) for r in distinct}) == 16
    assert len({(r["task"], r["generation_index"]) for r in priority}) == 16
    assert all(r["observed_tool_error"] for r in priority[:8])


def test_transport_error_detection_does_not_label_empty_or_goal_failed_results():
    assert observed_tool_error([{"content": 'Error executing tool update: validation error'}])
    assert observed_tool_error([{"content": '{"success": false, "error": "unknown id"}'}])
    assert not observed_tool_error([{"content": '{"success": true, "results": []}'}])
    assert not observed_tool_error([{"content": '{"success": true, "due": "March 5, 2026"}'}])


def test_failed_episode_frontier_selects_late_executed_action_not_successful_search():
    rows = [{"task": task, "episode_id": task, "generation_index": i,
             "terminal_text": i == 3, "native_success": 0,
             "executed_tool_responses": 0 if i == 2 else 1}
            for task in TRAIN for i in range(4)]
    selected = balanced_rows(rows, mode="failure-frontier")
    assert all(r["generation_index"] == 1 for r in selected[:8])
    assert all(r["terminal_text"] for r in selected[8:])
    assert len({(r["episode_id"], r["generation_index"]) for r in selected}) == 16


def test_risk_frontier_selects_silent_creation_risk_without_relabelling_tool_success():
    rows = [{"task": task, "episode_id": task, "generation_index": i,
             "terminal_text": i == 2, "native_success": 0,
             "executed_tool_responses": int(i < 2), "observed_tool_error": False,
             "observed_action_risks": [{"kind": "undeclared_arguments"}] if i == 0 else []}
            for task in TRAIN for i in range(3)]
    old = balanced_rows(rows, mode="first-error-frontier")
    new = balanced_rows(rows, mode="first-risk-frontier")
    assert all(r["generation_index"] == 1 for r in old[:8])
    assert all(r["generation_index"] == 0 for r in new[:8])
    assert all(not r["observed_tool_error"] for r in new)
    assert all(r["terminal_text"] for r in new[8:])
    for row in rows:
        row["native_success"] = 1
    retained = balanced_rows(rows, offset=1, mode="first-risk-frontier")
    assert all(r["generation_index"] == 1 for r in retained[:8])


def test_no_training_truncation():
    row = {"complete": True, "action_ids": [1] * 2048,
           "student_prefix_ids": [2] * 100, "teacher_prefix_ids": [3] * 14337}
    assert admission(row) == "overlong_whole_transition"
    assert len(row["action_ids"]) == 2048
    row["complete"] = False
    assert admission(row) == "incomplete_generation"


def test_alignment_is_explicit_and_does_not_mutate_native_evidence():
    messages = [{"role": "system", "content": "You have a budget of ~50 tool-using turns — favor parallel calls."}]
    result = aligned_messages(messages)
    assert "at most 8 assistant turns" in result[0]["content"]
    assert "~50" in messages[0]["content"]


def test_teacher_feedback_is_factual_not_a_guessed_correction():
    text = feedback_text([{"type": "email_sent", "passed": False, "excluded": False, "params": {"to": "known"}}],
                         [{"response": '{"messages": [], "result_count": 0}'}])
    assert "result_count" in text and "original decision point" in text
    assert "known" in text and "Correct solution" not in text


def test_pinned_sdpo_loss_moves_probability_toward_teacher():
    import torch
    from trl.experimental.sdpo.loss_utils import compute_topk_self_distillation_loss

    for alpha in (1., .5):
        student = torch.tensor([[2., 0., -1.]], requires_grad=True)
        teacher = torch.tensor([[0., 3., -1.]])
        loss = compute_topk_self_distillation_loss(student, teacher,
            distillation_topk=3, distillation_alpha=alpha, distillation_add_tail=True).mean()
        loss.backward()
        assert torch.isfinite(loss)
        assert student.grad[0, 0] > 0  # gradient descent lowers the teacher-rejected token
        assert student.grad[0, 1] < 0  # and raises the teacher-preferred alternative
        assert teacher.grad is None


def test_pinned_tail_bucket_preserves_normalized_mass():
    import torch
    from trl.experimental.sdpo.loss_utils import add_tail_bucket

    result = add_tail_bucket(torch.tensor([[.2, .3]]).log())
    torch.testing.assert_close(result.exp().sum(-1), torch.ones(1))
    torch.testing.assert_close(result.exp()[0, -1], torch.tensor(.5))
