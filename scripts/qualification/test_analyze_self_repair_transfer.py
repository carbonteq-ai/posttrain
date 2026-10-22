import json

import pytest
from analyze_self_repair_transfer import audit
from self_campaign_config import summarize


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def experiment(tmp_path):
    import hashlib

    names = [f"task-{i}" for i in range(24)]
    write(tmp_path / "population.json", {"tasks": names, "sorted_names_sha256":
          hashlib.sha256(json.dumps(names, separators=(",", ":")).encode()).hexdigest()})
    write(tmp_path / "receipt.json", {"completed": True, "optimizer_updates": 0, "candidates": 3})
    write(tmp_path / "candidate-provenance.json", [{"name": n, "tensor_equality_verified": True,
                                                  "exported_sha256": n}
          for n in ["checkpoint64", "argument_sft", "argument_self_hybrid"]])
    for step in range(3):
        folder = tmp_path / f"step-{step:04d}" / "repair_transfer"
        episodes = [{"id": f"{step}-{name}-{rep}", "task": {"data": {"name": name}},
                     "traces": [{"stop_condition": "agent_completed", "metrics":
                     {"task_completed_correctly": float(step > 0)},
                     "rewards": {"partial_credit": {"score": float(step > 0)}}}]}
                    for name in names for rep in range(2)]
        (folder / "native").mkdir(parents=True)
        (folder / "native/traces.jsonl").write_text("\n".join(map(json.dumps, episodes)))
        write(folder / "outcomes.json", summarize(episodes, names, 2))
        write(folder / "training-admission.json", {"training_allowed": False,
              "truncated_training_sequences": 0})
        (folder / "generation-calls.jsonl").write_text(json.dumps({"policy_version": step,
                                                                  "adapter_sha256": ["checkpoint64", "argument_sft", "argument_self_hybrid"][step],
                                                                  "finish_reason": "stop"}))
    return tmp_path


def test_all_three_pairings_and_native_identity(experiment):
    result = audit(experiment)
    assert result["episodes"] == 144
    assert len(result["comparisons"]) == 3
    assert result["comparisons"]["argument_sft_to_argument_self_hybrid"][
        "task_completed_correctly"]["paired"]["delta"] == 0


def test_summary_must_match_native_evidence(experiment):
    path = experiment / "step-0001/repair_transfer/outcomes.json"
    summary = json.loads(path.read_text())
    summary["metrics"]["task_completed_correctly"]["full_population_task_mean"] = 0
    write(path, summary)
    with pytest.raises(ValueError, match="disagrees"):
        audit(experiment)


def test_stale_policy_rejected(experiment):
    path = experiment / "step-0002/repair_transfer/generation-calls.jsonl"
    write(path, {"policy_version": 0, "finish_reason": "stop"})
    with pytest.raises(ValueError, match="stale policy"):
        audit(experiment)


def test_correct_version_with_wrong_adapter_rejected(experiment):
    path = experiment / "step-0002/repair_transfer/generation-calls.jsonl"
    write(path, {"policy_version": 2, "adapter_sha256": "wrong-weights", "finish_reason": "stop"})
    with pytest.raises(ValueError, match="stale policy"):
        audit(experiment)


def test_missing_score_remains_missing(experiment):
    folder = experiment / "step-0002/repair_transfer"
    path = folder / "native/traces.jsonl"
    episodes = [json.loads(line) for line in path.read_text().splitlines()]
    del episodes[0]["traces"][0]["metrics"]["task_completed_correctly"]
    path.write_text("\n".join(map(json.dumps, episodes)))
    names = json.loads((experiment / "population.json").read_text())["tasks"]
    write(folder / "outcomes.json", summarize(episodes, names, 2))
    result = audit(experiment)
    metric = result["candidates"][2]["summary"]["metrics"]["task_completed_correctly"]
    assert metric["full_population_task_mean"] is None
    assert metric["episode_coverage"] == 47
    paired = result["comparisons"]["argument_sft_to_argument_self_hybrid"]["task_completed_correctly"]
    assert paired["excluded_tasks"] == ["task-0"]
    assert paired["paired"]["tasks"] == 23


def test_pre_generation_context_rejection_retained_without_invented_adapter(experiment):
    path = experiment / "step-0002/repair_transfer/generation-calls.jsonl"
    with path.open("a") as stream:
        stream.write("\n" + json.dumps({"policy_version": 2, "limit_reason": "context_limit",
                                       "complete": False, "action_ids": [], "raw": ""}))
    assert audit(experiment)["candidates"][2]["context_rejected_calls_without_generation"] == 1


def test_generated_tokens_cannot_use_context_rejection_exception(experiment):
    path = experiment / "step-0002/repair_transfer/generation-calls.jsonl"
    write(path, {"policy_version": 2, "limit_reason": "context_limit", "complete": False,
                 "action_ids": [1], "raw": "generated"})
    with pytest.raises(ValueError, match="stale policy"):
        audit(experiment)
