import json

import pytest
from analyze_sdpo_guidance_five import audit, runtime_evidence
from self_campaign_config import summarize


def fixture(tmp_path):
    inputs = dict(run_profile="guidance-five", steps=10, batch_size=32, online_rollouts=2,
        context_limit=32768, reply_limit=4096, thinking=True, lr=1e-6, lr_schedule="constant",
        alpha=1., topk=20, teacher_rate=.01, teacher_feedback="prepared-guidance",
        teacher_guidance="/sealed/guidance.json", concurrency=40, speculation="mtp",
        qualify_capacity=True, tool_parser_mode="schema-bound-v1")
    cohort, heldout = {"train": [f"t{i}" for i in range(20)]}, {"task_names": [f"h{i}" for i in range(62)]}
    (tmp_path / "inputs.json").write_text(json.dumps(inputs))
    folder = tmp_path / "step-0000/train"
    (folder / "native").mkdir(parents=True)
    episodes = [{"id": f"e{i}-{replica}", "task": {"data": {"name": name}}, "traces": [
        {"stop_condition": "agent_completed", "metrics": {"task_completed_correctly": 1},
         "rewards": {"partial_credit": {"score": 1}}, "nodes": []}]} for i, name in enumerate(cohort["train"]) for replica in range(2)]
    (folder / "outcomes.json").write_text(json.dumps(summarize(episodes, cohort["train"], 2)))
    (folder / "native/traces.jsonl").write_text("\n".join(json.dumps(e) for e in episodes))
    (folder / "generation-calls.jsonl").write_text("")
    return cohort, heldout, folder, episodes


def test_baseline_population_does_not_imply_training_completion(tmp_path):
    cohort, heldout, _, _ = fixture(tmp_path)
    result = audit(tmp_path, cohort, heldout)
    assert result["unique_native_episodes"] == 40
    assert result["execution_complete"] is False and result["updates"] == []


def test_duplicate_native_episode_cannot_pass_population_audit(tmp_path):
    cohort, heldout, folder, episodes = fixture(tmp_path)
    episodes[1]["id"] = episodes[0]["id"]
    (folder / "native/traces.jsonl").write_text("\n".join(json.dumps(e) for e in episodes))
    with pytest.raises(ValueError, match="duplicated"):
        audit(tmp_path, cohort, heldout)


def test_failed_runtime_is_explicit_even_with_complete_scores(tmp_path):
    from sdpo_greedy_repeatability import audit as repeatability_audit

    cohort, heldout, folder, _ = fixture(tmp_path)
    inputs = json.loads((tmp_path / "inputs.json").read_text())
    inputs["native_repeatability"] = True
    (tmp_path / "inputs.json").write_text(json.dumps(inputs))
    common = {"complete": True, "temperature": 0., "prefix_ids": [1],
              "requested_completion_budget": 8192, "adapter_sha256": "base", "policy_version": 0,
              "sampled_logprobs": [-.1]}
    calls = [{**common, "action_ids": [2]}, {**common, "action_ids": [3]}]
    (folder / "generation-calls.jsonl").write_text("\n".join(json.dumps(x) for x in calls))
    report = repeatability_audit(calls)
    (folder / "native-greedy-repeatability.json").write_text(json.dumps(report))
    result = audit(tmp_path, cohort, heldout)
    assert result["benchmark_readouts_runtime_qualified"] is False
    assert result["unqualified_native_readouts"][0]["runtime"]["qualified"] is False
    assert result["populations"][0]["summary"]["metrics"]["task_completed_correctly"]["episode_coverage"] == 40
    report["qualified"] = True
    (folder / "native-greedy-repeatability.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="contradicts original calls"):
        runtime_evidence(folder, True)


def test_missing_runtime_gate_stays_unknown(tmp_path):
    _, _, folder, _ = fixture(tmp_path)
    evidence = runtime_evidence(folder, True)
    assert evidence["qualified"] is None
    assert evidence["persisted_gate"] is False
