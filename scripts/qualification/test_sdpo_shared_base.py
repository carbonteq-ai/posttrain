import copy
import json
import shutil

import pytest
from sdpo_greedy_repeatability import audit as repeatability_audit
from sdpo_shared_base import RUNTIME_FILES, export, load, student_code
from self_campaign_config import summarize

SOURCE = """
class SamplingRollout:
    async def generate(self, prefix, max_tokens=2048):
        return prefix
class Agent:
    async def completions(self, request):
        return request
    async def collect(self, step, cohort, names, temperature=0.):
        self.context = 65536
        return names
"""


def fixture(tmp_path):
    root, driver = tmp_path / "run", tmp_path / "driver"
    root.mkdir()
    driver.mkdir()
    for name in RUNTIME_FILES:
        (driver / name).write_text("# pinned runtime\n")
    (driver / "sdpo_agent_config.py").write_text(
        "def aligned_messages(messages): return messages\ndef population_rollouts(cohort, n): return n\n")
    (driver / "sdpo_agent_ablation.py").write_text(SOURCE)
    inputs = dict(model="/hf/pinned-base", model_family="qwen3.5", run_profile="guidance-five",
                  run_name="source-base", thinking=True, concurrency=40, speculation="mtp",
                  mtp_tokens=2, tool_parser_mode="schema-bound-v1", disable_prefix_caching=True)
    (root / "inputs.json").write_text(json.dumps(inputs))
    names = [f"t{i}" for i in range(62)]
    heldout = tmp_path / "heldout.json"
    heldout.write_text(json.dumps({"task_names": names}))
    pop = root / "step-0000/heldout62"
    (pop / "native").mkdir(parents=True)
    episodes = [{"id": f"{name}-{rep}", "task": {"data": {"name": name}}, "traces": [{
        "stop_condition": "agent_completed", "metrics": {"task_completed_correctly": 1},
        "rewards": {"partial_credit": {"score": 1}}, "nodes": []}]} for name in names for rep in range(5)]
    # Preserve one missing score to check that sharing never zero-fills it.
    episodes[0]["traces"][0]["metrics"] = {}
    (pop / "native/traces.jsonl").write_text("\n".join(json.dumps(e) for e in episodes))
    (pop / "outcomes.json").write_text(json.dumps(summarize(episodes, names, 5)))
    call = dict(action_ids=[1], adapter_sha256="base-sha", policy_version=0, temperature=0.,
                thinking_enabled=True, complete=True, finish_reason="stop", prefix_ids=[0],
                requested_completion_budget=8192, sampled_logprobs=[-.1])
    calls = [{**call, "prefix_ids": [i//5]} for i in range(310)]
    (pop / "generation-calls.jsonl").write_text("\n".join(json.dumps(c) for c in calls))
    (pop / "native-greedy-repeatability.json").write_text(json.dumps(repeatability_audit(calls)))
    return root, driver, inputs, heldout


def test_shared_control_keeps_native_ids_missingness_and_origin_explicit(tmp_path):
    root, driver, inputs, heldout = fixture(tmp_path)
    destination = tmp_path / "shared"
    export(root, driver, heldout, destination)
    result = load(destination / "reference.json", driver, inputs, "base-sha", heldout)
    assert result["reused"] and result["origin_run"] == "source-base"
    assert len(result["episode_ids"]) == 310
    assert result["summary"]["metrics"]["task_completed_correctly"]["episode_coverage"] == 309
    assert result["summary"]["metrics"]["task_completed_correctly"]["full_population_task_mean"] is None


def test_wrong_weights_or_decoding_or_changed_raw_trace_rejects_reuse(tmp_path):
    root, driver, inputs, heldout = fixture(tmp_path)
    destination = tmp_path / "shared"
    export(root, driver, heldout, destination)
    for key, value in [("thinking", False), ("disable_prefix_caching", False), ("concurrency", 32),
                       ("speculation", "none")]:
        changed = copy.deepcopy(inputs)
        changed[key] = value
        with pytest.raises(ValueError, match="differs"):
            load(destination / "reference.json", driver, changed, "base-sha", heldout)
    with pytest.raises(ValueError, match="differs"):
        load(destination / "reference.json", driver, inputs, "trained-sha", heldout)
    (destination / "traces.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="file identity"):
        load(destination / "reference.json", driver, inputs, "base-sha", heldout)


def test_complete_scores_do_not_make_a_failed_repeatability_baseline_reusable(tmp_path):
    root, driver, _, heldout = fixture(tmp_path)
    p = root / "step-0000/heldout62/native-greedy-repeatability.json"
    report = json.loads(p.read_text())
    report["qualified"] = False
    p.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="unqualified native repeatability"):
        export(root, driver, heldout, tmp_path / "rejected-control")


def test_observation_only_counter_instrumentation_is_distinct_from_budget_changes():
    added = SOURCE.replace("        self.context = 65536", """        mtp_before = None
        if getattr(self.args, 'capture_mtp_metrics', False):
            mtp_before = snapshot()
        self.context = 65536
        if mtp_before is not None:
            write_metrics(mtp_before)
""")
    assert student_code(SOURCE) == student_code(added)
    assert student_code(SOURCE) != student_code(SOURCE.replace("65536", "32768"))


def test_run_audit_separates_own_episodes_from_shared_control_and_checks_own_weights(tmp_path):
    from analyze_sdpo_guidance_five import audit

    producer, driver, inputs, heldout = fixture(tmp_path)
    capsule = tmp_path / "consumer"
    capsule.mkdir()
    shutil.copytree(driver, capsule / "driver")
    shutil.copyfile(heldout, capsule / "heldout62.json")
    export(producer, driver, heldout, capsule / "baseline")
    inputs.update(steps=10, batch_size=32, online_rollouts=2, context_limit=32768,
                  reply_limit=4096, lr=1e-6, lr_schedule="constant", alpha=1., topk=20,
                  teacher_rate=.01, teacher_feedback="prepared-guidance", qualify_capacity=True,
                  teacher_guidance=str(capsule / "guidance.json"),
                  heldout_base_reference=str(capsule / "baseline/reference.json"))
    root = capsule / "evidence/run"
    root.mkdir(parents=True)
    (root / "inputs.json").write_text(json.dumps(inputs))
    reference = root / "step-0000/heldout62"
    reference.mkdir(parents=True)
    shared = load(capsule / "baseline/reference.json", capsule / "driver", inputs,
                  "base-sha", capsule / "heldout62.json")
    (reference / "shared-base.json").write_text(json.dumps({
        **shared, "source_reference": inputs["heldout_base_reference"]}))
    names = [f"training-{i}" for i in range(20)]
    pop = root / "step-0000/train"
    (pop / "native").mkdir(parents=True)
    episodes = [{"id": f"own-{name}-{rep}", "task": {"data": {"name": name}}, "traces": [{
        "stop_condition": "agent_completed", "metrics": {"task_completed_correctly": 1},
        "rewards": {"partial_credit": {"score": 1}}, "nodes": []}]} for name in names for rep in range(2)]
    (pop / "native/traces.jsonl").write_text("\n".join(json.dumps(e) for e in episodes))
    (pop / "outcomes.json").write_text(json.dumps(summarize(episodes, names, 2)))
    call = dict(action_ids=[1], adapter_sha256="base-sha", policy_version=0, complete=True,
                finish_reason="stop")
    calls = [copy.deepcopy(call) for _ in episodes]
    path = pop / "generation-calls.jsonl"
    path.write_text("\n".join(json.dumps(c) for c in calls))
    result = audit(root, {"train": names}, json.loads(heldout.read_text()))
    assert not result["execution_complete"]
    assert result["native_episodes_written_by_this_run"] == 40
    assert result["referenced_control_native_episodes"] == 310
    assert result["unique_native_episodes"] == 350
    calls[0]["adapter_sha256"] = "another-policy"
    path.write_text("\n".join(json.dumps(c) for c in calls))
    with pytest.raises(ValueError, match="identity missing or mixed"):
        audit(root, {"train": names}, json.loads(heldout.read_text()))
