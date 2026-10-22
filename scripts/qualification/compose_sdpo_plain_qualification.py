"""Bind closed zero-update native evidence to an independent matched capacity gate."""

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

from sdpo_greedy_repeatability import audit
from sdpo_shared_base import RUNTIME_FILES
from sdpo_speculation import validate_qualification_request
from self_campaign_config import summarize

IMAGE = "registry.lan/carbonteq/posttrain-lab/posttrain-lab/posttrain-job@sha256:e976e727786135f851f4cc96356ebff1f57c75ca5f2d36a2c6aacbf6b2822337"
MODEL = "/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_manifest(root):
    lines = (root / "source-sha256.txt").read_text().splitlines()
    require(bool(lines), "source manifest empty")
    for line in lines:
        digest, name = line.split("  ", 1)
        file = (root / name).resolve()
        require(file.is_relative_to(root.resolve()) and sha(file) == digest, "source manifest differs")
    return sha(root / "source-sha256.txt")


def compose(native, capacity, native_inspect, capacity_inspect):
    manifests = {"native": verify_manifest(native), "capacity": verify_manifest(capacity)}
    for inspect, exits in ((native_inspect, (0, 124)), (capacity_inspect, (0,))):
        require(not inspect["State"]["Running"] and not inspect["State"]["OOMKilled"]
                and inspect["State"]["ExitCode"] in exits, "component is not an eligible terminal attempt")
        require(inspect["Config"]["Image"] == IMAGE and MODEL in inspect["Config"]["Cmd"],
                "component image/model pin differs")
    sources = {}
    required = set(RUNTIME_FILES) | {"sdpo_agent_ablation.py", "sdpo_capacity.py", "sdpo_agent_config.py"}
    require(all((native / "driver" / name).is_file() for name in required), "runtime source missing")
    for source in (native / "driver").rglob("*.py"):
        name = source.relative_to(native / "driver")
        other = capacity / "driver" / name
        require(other.is_file() and sha(source) == sha(other), "runtime source differs")
        sources[str(name)] = sha(source)
    n, c = native / "evidence/mtp", capacity / "evidence/mtp"
    for folder in (n, c):
        validate_qualification_request(read(folder / "requested.json"), 2, method="none",
            prefix_caching=False, chunked_prefill=False, enforce_eager=False)
    nq, cq = read(native / "qualification.json"), read(capacity / "qualification.json")
    for key in ("method", "gdn_decode_kernel", "experiment_deadline_unix", "budget_window_started_unix",
                "budget_authorization", "logical_experiment"):
        require(nq[key] == cq[key], "component operating point/window differs")
    weights = sha(n / "initial-student/adapter_model.safetensors")
    require(weights == sha(c / "initial-student/adapter_model.safetensors"), "initial adapter differs")
    cap = read(c / "receipt.json")
    require(cap.get("qualified") and cap.get("component") == "plain-capacity-only-v1"
            and cap.get("method") == "none" and cap.get("optimizer_updates") == 0
            and cap.get("initial_adapter_sha256") == weights, "capacity component is not qualified")
    actor, rollout = cap["actor_capacity"], cap["rollout_capacity"]
    require(actor.get("qualified") and actor.get("weights_unchanged")
            and actor.get("finite_nonzero_student_gradients") and actor.get("optimizer_updates") == 0
            and actor.get("student_tokens") == 32768 and actor.get("teacher_tokens") == 32768
            and actor.get("scored_target_tokens") == 4096 and math.isfinite(actor["loss"]),
            "full-context backward qualification missing")
    require(rollout["context_limit"] == 32768 and rollout["reply_limit"] == 4096
            and rollout["generation_prompt_tokens"] == 28672 and rollout["generated_tokens"] > 0
            and rollout["parity"]["max_abs_delta"] <= .15
            and rollout["batch_invariance"]["exact_batch_invariance"]
            and max(r["peak_in_flight"] for r in rollout["batch_invariance"]["records"]) == 40,
            "full-context rollout qualification missing")
    names = read(native / "cohorts.json")["train"]
    require(len(names) == len(set(names)) == 20, "native cohort differs")
    seen, populations, greedy = set(), {}, None
    for group, temperature in (("train", 0.), ("online_train", 1.)):
        folder = n / "step-0000" / group
        episodes = [json.loads(line) for line in (folder / "native/traces.jsonl").read_text().splitlines() if line]
        require(len(episodes) == 40 and not seen.intersection(e["id"] for e in episodes), "native IDs overlap/incomplete")
        seen.update(e["id"] for e in episodes)
        summary = summarize(episodes, names, 2)
        require(not summary["errors"] and all(summary["metrics"][key]["episode_coverage"] == 40
                for key in ("task_completed_correctly", "partial_credit")),
                "native errors/missing scores")
        saved = read(folder / "outcomes.json")
        require(all(saved[k] == summary[k] for k in ("episodes", "tasks", "errors", "stop_conditions", "metrics")),
                "native saved summary differs")
        calls = [json.loads(line) for line in (folder / "generation-calls.jsonl").read_text().splitlines() if line]
        require(calls and all(call.get("policy_version") == 0 and call.get("adapter_sha256") == weights
                and call.get("temperature") == temperature
                and len(call["sampled_logprobs"]) == len(call["action_ids"])
                and all(math.isfinite(x) for x in call["sampled_logprobs"]) for call in calls),
                "native policy/weights/temperature/logprobs differ")
        if group == "train":
            greedy = audit(calls)
            require(greedy["qualified"] and greedy == read(n / "native-greedy-repeatability.json"),
                    "native repeatability differs or failed")
        populations[group] = {"episodes": 40, "native_sha256": sha(folder / "native/traces.jsonl"),
                              "calls_sha256": sha(folder / "generation-calls.jsonl"), "summary": summary}
    tool, batch = read(n / "native-tool-qualification/receipt.json"), read(n / "batch-invariance-v0.json")
    require(tool["qualified"] and batch["exact_batch_invariance"], "native tool/batch gate missing")
    return {"qualified": True, "method": "none", "optimizer_updates": 0,
        "composition": "closed-native-plus-matched-capacity-v1", "prefix_caching": False,
        "chunked_prefill": False, "enforce_eager": False, "native": tool, "batch_invariance": batch,
        "actor_capacity": actor, "stochastic_native": {"qualified": True, "episodes": 40,
            "temperature": 1., "tasks": names, "optimizer_updates": 0, "native_greedy_repeatability": greedy},
        "components": {"native_capsule": str(native), "capacity_capsule": str(capacity),
            "native_terminal_state": native_inspect["State"], "capacity_terminal_state": capacity_inspect["State"],
            "source_manifests": manifests, "matched_runtime_sources": sources,
            "initial_adapter_sha256": weights, "populations": populations,
            "capacity_receipt_sha256": sha(c / "receipt.json")},
        "scope": "Explicit finite-case component qualification. Native sub-timeout remains preserved; not full-heldout repeatability, learning or reward evidence."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--capacity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "retained composition exists")
    states = [json.loads(subprocess.check_output(["docker", "inspect", read(p / "qualification.json")["run_name"]]))[0]
              for p in (args.native, args.capacity)]
    result = compose(args.native, args.capacity, *states)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
