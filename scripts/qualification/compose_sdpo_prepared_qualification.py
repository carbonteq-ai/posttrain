"""Compose unchanged runtime capacity and fresh changed-cohort native evidence.

Run in the pinned offline CPU runtime after preparation exits. Never relabel an
old population, or use this receipt as evidence that fitting/evaluation finished.
"""

import argparse
import copy
import hashlib
import inspect
import json
from pathlib import Path

PREPARATION_ONLY = frozenset({"prepare_sdpo_guidance_five.py", "sdpo_checkwise_preparation.py",
    "sdpo_effect_brief.py",
    "prepare_sdpo_guidance20.py",
    "qualify_sdpo_prepared_tasks.py", "replay_sdpo_training_candidates.py",
    "compose_sdpo_prepared_qualification.py"})


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def runtime_sources(old, new):
    from sdpo_shared_base import RUNTIME_FILES

    old, new = Path(old), Path(new)
    before = {str(p.relative_to(old)): sha(p) for p in old.rglob("*.py")}
    after = {str(p.relative_to(new)): sha(p) for p in new.rglob("*.py")}
    changed = {name for name in before.keys() | after.keys() if before.get(name) != after.get(name)}
    if changed - PREPARATION_ONLY:
        raise ValueError(f"runtime source differs: {sorted(changed - PREPARATION_ONLY)}")
    required = {*RUNTIME_FILES, "sdpo_capacity.py", "sdpo_agent_ablation.py", "sdpo_agent_config.py"}
    if not required <= before.keys() or any(name not in after for name in before):
        raise ValueError("inherited runtime files missing")
    return {"identical_inherited_files": len(before) - len(changed & before.keys()),
            "preparation_only_changes": sorted(changed),
            "old_sources_sha256": hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest(),
            "new_sources_sha256": hashlib.sha256(json.dumps(after, sort_keys=True).encode()).hexdigest()}


def compose(old_gate, native, tasks):
    if (not old_gate["qualified"] or old_gate["method"] != "none" or old_gate["optimizer_updates"] != 0
            or old_gate["prefix_caching"] or old_gate["chunked_prefill"]
            or not old_gate["actor_capacity"]["qualified"]
            or old_gate["actor_capacity"]["student_tokens"] != 32768
            or old_gate["actor_capacity"]["scored_target_tokens"] != 4096):
        raise ValueError("inherited operating point or capacity is unqualified")
    if (len(tasks) != len(set(tasks)) or len(tasks) != 20 or not native["qualified"]
            or native["optimizer_updates"] != 0 or native["policy_version"] != 0
            or native["initial_adapter_sha256"] !=
            "e71dae81f1b4723c83aacd130bfbf60b072b3580212133a4fc45dcb9f403421c"
            or set(native["tasks"]) != set(tasks) or native["native_episodes"] != 80
            or native["greedy_episodes"] != 40 or native["stochastic_episodes"] != 40):
        raise ValueError("fresh new-cohort evidence differs")
    stochastic = native["stochastic_native"]
    if (not stochastic["qualified"] or stochastic["episodes"] != 40
            or stochastic["temperature"] != 1. or stochastic["optimizer_updates"] != 0
            or set(stochastic["tasks"]) != set(tasks)
            or not stochastic["native_greedy_repeatability"]["qualified"]
            or stochastic["native_greedy_repeatability"] != native["native_greedy_repeatability"]
            or stochastic["native_greedy_repeatability"]["compared_pairs"] <= 0):
        raise ValueError("fresh stochastic or exact greedy qualification missing")
    result = copy.deepcopy(old_gate)
    result["stochastic_native"] = copy.deepcopy(stochastic)
    result["scope"] = ("Unchanged inherited pinned model/runtime capacity plus separately replay-audited "
                       "fresh80native episodes for the changed20-task cohort. Not new capacity, updates or learning.")
    return result


def run(args):
    import automationbench.schema.google_sheets.base as sheets
    from analyze_sdpo_pool124 import bounded_scores, comparable_summary
    from automationbench_v1.world_codec import load_world
    from qualify_sdpo_prepared_tasks import digest, normalization
    from sdpo_greedy_repeatability import audit
    from sdpo_guidance_state_binding import bind

    old, preparation = args.old_capsule, args.preparation
    source = runtime_sources(old / "driver", preparation / "driver")
    root = preparation / "evidence/preparation"
    native_root = root / "task-qualification"
    cohort, materials = read(preparation / "cohorts.json"), read(preparation / "training-materials.json")
    task_receipt = read(native_root / "receipt.json")
    prep_receipt, guidance = read(root / "receipt.json"), read(root / "guidance.json")
    request = read(root / "decoding-request.json")
    if (not prep_receipt["qualified"] or guidance["rejected"]
            or set(guidance["guidance"]) != set(cohort["train"])
            or request["speculative_config"] is not None or request["context"] != 65536
            or request["concurrency"] != 40 or request["prefix_caching"] or request["chunked_prefill"]
            or request["enforce_eager"] or request["optimizer_updates"] != 0
            or task_receipt["cohorts_sha256"] != sha(preparation / "cohorts.json")
            or task_receipt["materials_sha256"] != sha(preparation / "training-materials.json")
            or prep_receipt["native_task_qualification"] != task_receipt
            or prep_receipt["guidance_sha256"] != digest(guidance["guidance"])):
        raise ValueError("preparation/task/source receipt binding differs")
    seen, native_files, greedy, states = set(), {}, None, None
    normalizer_sha = sha(Path(inspect.getsourcefile(sheets)))
    for group, temperature in (("train", 0.), ("online_train", 1.)):
        folder = native_root / "step-0000" / group
        episodes = [json.loads(line) for line in (folder / "native/traces.jsonl").read_text().splitlines() if line]
        ids = {episode["id"] for episode in episodes}
        if seen & ids:
            raise ValueError("new native populations reuse episode identities")
        seen |= ids
        normalized = normalization(materials, episodes, load_world, normalizer_sha)
        summary, bounds = bounded_scores(episodes, cohort["train"], 2)
        if (summary["errors"] or any(b["missing"] for b in bounds.values())
                or comparable_summary(summary) != comparable_summary(read(folder / "outcomes.json"))):
            raise ValueError("fresh native task scores/errors/coverage disagree")
        calls = [json.loads(line) for line in (folder / "generation-calls.jsonl").read_text().splitlines() if line]
        generated = [c for c in calls if c["action_ids"]]
        if (not generated or any(c["policy_version"] != 0 for c in calls)
                or any(c["temperature"] != temperature
                       or c["adapter_sha256"] != task_receipt["initial_adapter_sha256"] for c in generated)
                or any(not c["action_ids"] and c.get("limit_reason") != "context_limit" for c in calls)):
            raise ValueError("fresh task qualification used wrong policy or sampling")
        if group == "train":
            greedy, states = audit(calls), normalized
            if greedy != task_receipt["native_greedy_repeatability"]:
                raise ValueError("retained exact greedy evidence differs")
        native_files[group] = {"traces_sha256": sha(folder / "native/traces.jsonl"),
                               "calls_sha256": sha(folder / "generation-calls.jsonl"), "episode_ids": sorted(ids)}
    if len(seen) != 80 or states != read(native_root / "state-normalization.json"):
        raise ValueError("native normalization or population changed after qualification")
    result = compose(read(old / "mtp-gate-receipt.json"), task_receipt, cohort["train"])
    result["changed_cohort_composition"] = {"old_capsule": str(old), "preparation": str(preparation),
        "inherited_gate_sha256": sha(old / "mtp-gate-receipt.json"), "runtime_sources": source,
        "new_native_receipt_sha256": sha(native_root / "receipt.json"), "native_files": native_files,
        "old_native_population_reused_as_new": False}
    result["changed_cohort_composition"]["native_population_origin"] = task_receipt.get("referenced_from")
    result["changed_cohort_composition"]["native_episodes_generated_this_preparation"] = (
        0 if task_receipt.get("referenced_from") else 80)
    args.output.mkdir(exist_ok=False)
    (args.output / "mtp-gate-receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "guidance.json").write_text(json.dumps(bind(guidance, materials, states), indent=2) + "\n")
    (args.output / "state-normalization.json").write_text(json.dumps(states, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-capsule", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
