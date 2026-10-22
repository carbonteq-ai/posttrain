"""Reconcile the preselected zero-update reserve comparison from native evidence."""

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from analyze_self_repair_run import native_tool_observations
from self_campaign_config import comparisons, summarize


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def score_bounds(episodes):
    """Identification bounds for missing bounded scores, never replacement scores."""
    output = {}
    for metric in ["task_completed_correctly", "partial_credit"]:
        known = []
        for e in episodes:
            t = e["traces"][0]
            value = (t.get("rewards", {}).get("partial_credit", {}).get("score") if metric == "partial_credit"
                     else t.get("metrics", {}).get(metric))
            if isinstance(value, (int, float)) and math.isfinite(value):
                if not 0 <= value <= 1:
                    raise ValueError("bounded native score outside0..1")
                known.append(value)
        missing, total = len(episodes) - len(known), len(episodes)
        output[metric] = {"observed": len(known), "missing": missing,
                          "identification_bounds": [sum(known) / total, (sum(known) + missing) / total],
                          "scope": "Missing-score identification bounds, not confidence intervals or zero-filled estimates."}
    return output


def audit(root):
    selection = read(root / "selection-before-outcomes.json")
    if sha(root / "selection-before-outcomes.json") != read(root / "selection-sha256.json")["sha256"]:
        raise ValueError("preselected candidates changed")
    expected = selection["candidates"]
    names = read(root / "population.json")["tasks"]
    population = read(root / "population.json")
    if (len(names) != 24 or len(set(names)) != 24 or
            hashlib.sha256(json.dumps(names, separators=(",", ":")).encode()).hexdigest()
            != population["sorted_names_sha256"]):
        raise ValueError("reserved task identity mismatch")
    receipt = read(root / "receipt.json")
    if (not receipt["completed"] or receipt["optimizer_updates"] != 0 or receipt["external_calls"] != 0
            or receipt["candidates"] != len(expected) or receipt["elapsed_s"] > 7200
            or receipt["primary"] != selection["primary"] or not 3 <= len(expected) <= 7):
        raise ValueError("incomplete or over-budget comparison")
    inputs = read(root / "inputs.json")
    contract = {"context_limit": 16384, "reply_limit": 2048, "episode_output_limit": 12288,
                "turn_limit": 8, "server_sequences": 32, "episode_concurrency": 32,
                "kv_cache_gib": 16, "temperature": 0., "thinking": False, "optimizer_updates": 0}
    if any(inputs[k] != v for k, v in contract.items()):
        raise ValueError("evaluation operating point differs")
    gate = read(root / "zero-base-gate.json")
    if (not all(gate[k] for k in ["all_B_zero", "only_finite_A_B", "configuration_checked", "actual_loaded_state_verified"])
            or gate["active_vs_disabled_max_abs_delta"] > 1e-5):
        raise ValueError("fresh-base qualification missing")
    provenance = read(root / "candidate-provenance.json")
    if [x["name"] for x in provenance] != [x["name"] for x in expected]:
        raise ValueError("candidate order differs from preselection")
    phases, summaries, all_ids, reference_states = [], {}, set(), None
    for step, (item, source) in enumerate(zip(expected, provenance, strict=True)):
        if (not source["tensor_equality_verified"] or source["candidate_sha256"] != source["exported_sha256"]
                or (step and item["adapter_sha256"] != source["candidate_sha256"])):
            raise ValueError("selected/exported adapter parity missing")
        parity = read(root / f"parity-v{step}.json")
        if parity["policy_version"] != step or not 0 <= parity["max_abs_delta"] <= .15:
            raise ValueError("actor/rollout parity missing")
        folder = root / f"step-{step:04d}" / "reserve_transfer"
        path = folder / "native/traces.jsonl"
        episodes = [json.loads(line) for line in path.read_text().splitlines() if line]
        ids = {e["id"] for e in episodes}
        if all_ids & ids:
            raise ValueError("episode identities reused")
        all_ids |= ids
        summary = summarize(episodes, names, 2)
        if summary != read(folder / "outcomes.json") or summary["errors"]:
            raise ValueError("native outcomes mismatch or errors")
        states = defaultdict(Counter)
        for e in episodes:
            data = e["task"]["data"]
            states[data["name"]][hashlib.sha256(canonical(data["initial_state"]).encode()).hexdigest()] += 1
        if reference_states is None:
            reference_states = states
        elif states != reference_states:
            raise ValueError("candidate initial states differ")
        admission = read(folder / "evaluation-admission.json")
        if admission["training_allowed"] or admission["optimizer_updates"]:
            raise ValueError("reserve was used for updates")
        calls = [json.loads(line) for line in (folder / "generation-calls.jsonl").read_text().splitlines() if line]
        rejected = 0
        for c in calls:
            if c["policy_version"] != step:
                raise ValueError("stale policy version")
            if c.get("limit_reason") == "context_limit" and not c.get("action_ids") and not c.get("raw") and not c["complete"]:
                rejected += 1
            elif c.get("adapter_sha256") != source["exported_sha256"]:
                raise ValueError("wrong served adapter")
        if rejected != admission["context_rejected_calls"] or sum(not c["complete"] for c in calls) != admission["incomplete_generations"]:
            raise ValueError("admission counts differ from calls")
        summaries[item["name"]] = summary
        phases.append({"candidate": item["name"], "summary": summary, "native_sha256": sha(path),
                       "missing_score_bounds": score_bounds(episodes),
                       "generation_calls": len(calls), "finish_reasons": dict(Counter(str(c.get("finish_reason")) for c in calls)),
                       "context_rejections": rejected, "tool_observations": native_tool_observations(episodes),
                       "admission": admission, "actor_rollout_max_abs_delta": parity["max_abs_delta"], "retry_count": None})
    if len(all_ids) != 48 * len(expected):
        raise ValueError("incomplete native population")
    pairs = [(expected[0]["name"], x["name"]) for x in expected[1:]]
    pairs += [(expected[1]["name"], x["name"]) for x in expected[2:]]
    pairs += [("balanced_sft07_step10", "full_span06_step10")]
    paired, changed = {}, {}
    for before, after in pairs:
        if before not in summaries or after not in summaries:
            continue
        key = f"{before}_to_{after}"
        paired[key] = comparisons(summaries[before], summaries[after])
        changed[key] = {}
        for metric in ["task_completed_correctly", "partial_credit"]:
            a = summaries[before]["metrics"][metric]["complete_task_means"]
            b = summaries[after]["metrics"][metric]["complete_task_means"]
            changed[key][metric] = {n: {"before": a[n], "after": b[n], "delta": b[n] - a[n]}
                                    for n in sorted(a.keys() & b.keys()) if a[n] != b[n]}
    bounds = {p["candidate"]: p["missing_score_bounds"] for p in phases}
    primary_bounds = {}
    for metric in ["task_completed_correctly", "partial_credit"]:
        before = bounds["fresh_base"][metric]["identification_bounds"]
        after = bounds[selection["primary"]][metric]["identification_bounds"]
        primary_bounds[metric] = {"gain_identification_bounds": [after[0] - before[1], after[1] - before[0]],
                                  "scope": "Missing-score gain identification bounds, separate from paired sampling confidence intervals."}
    return {"receipt": receipt, "population": population, "episodes": len(all_ids), "selection": selection,
            "zero_base_gate": gate, "candidate_provenance": provenance, "candidates": phases,
            "comparisons": paired, "changed_tasks": changed,
            "primary_comparison": f"fresh_base_to_{selection['primary']}",
            "primary_gain_identification_bounds": primary_bounds,
            "initial_state_identity_verified": True,
            "scope": "Preselected exploratory primary; reserved research transfer, not heldout62. "
                     "Paired task bootstrap preserves both greedy repetitions; absolute intervals separate. "
                     "Missing scores remain missing, no zero fill. Native preservation audit is separate; "
                     "The legacy summary goals_and_harms string refers to the separate assessment path, not native assertions. "
                     "Absence of assessment batches is not harm-free evidence. Other arms descriptive, not selection."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(audit(args.root), indent=2) + "\n")
