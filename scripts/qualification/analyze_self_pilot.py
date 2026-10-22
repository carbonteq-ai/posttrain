"""Audit native SELF pilot evidence; run CPU-only in the exact retained runtime."""

import argparse
import ast
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace


def quantile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    return ordered[lower] + (ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]) * (index - lower)


def paired_bootstrap(before, after, repeats=10000):
    if set(before) != set(after) or not before:
        raise ValueError("paired bootstrap requires identical nonempty complete-task cohorts")
    names = sorted(before)
    rng = random.Random(20261017)
    samples = [[], [], []]
    for _ in range(repeats):
        indices = rng.choices(names, k=len(names))
        a = sum(before[name] for name in indices) / len(indices)
        b = sum(after[name] for name in indices) / len(indices)
        for values, value in zip(samples, (a, b, b - a), strict=True):
            values.append(value)
    a, b = sum(before.values()) / len(names), sum(after.values()) / len(names)
    return {"tasks": len(names), "before": a, "after": b, "delta": b - a,
            "before_absolute_ci95": [quantile(samples[0], q) for q in (0.025, 0.975)],
            "after_absolute_ci95": [quantile(samples[1], q) for q in (0.025, 0.975)],
            "paired_delta_ci95": [quantile(samples[2], q) for q in (0.025, 0.975)],
            "bootstrap_repeats": repeats, "unit": "task, preserving its two episodes"}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def phase_audit(folder, expected, tokenizer, version):
    from self_feedback_native import collect_transitions
    from self_pilot_config import admit_sequences

    episodes = read_jsonl(folder / "native/traces.jsonl")
    names = Counter(e["task"]["data"]["name"] for e in episodes)
    if set(names) != set(expected) or set(names.values()) != {2} or len(episodes) != 64:
        raise ValueError("native task identities/counts differ from 32x2 protocol")
    if len({e["id"] for e in episodes}) != len(episodes):
        raise ValueError("duplicate episode identities")
    calls = read_jsonl(folder / "generation-calls.jsonl")
    if any(c["policy_version"] != version for c in calls):
        raise ValueError("stale or unexpected rollout policy version")
    errors, stops, transport_ids, signatures = [], Counter(), {}, defaultdict(Counter)
    metrics = defaultdict(lambda: defaultdict(list))
    assessment_batches = 0
    for episode in episodes:
        task = episode["task"]["data"]["name"]
        for key in ("errors", "credit_errors", "assessment_errors"):
            errors.extend(episode.get(key, []))
        assessment_batches += len(episode.get("assessment_batches", []))
        if len(episode["traces"]) != 1:
            raise ValueError("expected one native trace per episode")
        for trace in episode["traces"]:
            errors.extend(trace.get("errors", []))
            stops[trace["stop_condition"]] += 1
            raw = dict(trace["metrics"])
            raw["partial_credit"] = trace.get("rewards", {}).get("partial_credit", {}).get("score")
            for key, value in raw.items():
                if isinstance(value, (int, float)) and math.isfinite(value):
                    metrics[key][task].append(value)
            for node in trace["nodes"]:
                message = node.get("message") or {}
                if message.get("role") == "tool":
                    transport_ids[message["tool_call_id"]] = episode["id"]
                for call in message.get("tool_calls", []):
                    transport_ids[call["id"]] = episode["id"]
    if errors:
        raise ValueError(f"native/scoring errors retained: {errors[:3]}")
    for call in calls:
        if call["complete"] and len(call["sampled_logprobs"]) != len(call["action_ids"]):
            raise ValueError("sampled log probabilities do not align to original action tokens")
        for tool in call["message"].get("tool_calls", []):
            function = dict(tool["function"])
            if isinstance(function["arguments"], str):
                function["arguments"] = json.loads(function["arguments"])
            signatures[transport_ids[tool["id"]]][json.dumps(function, sort_keys=True)] += 1

    exclusions = []
    proxies = [SimpleNamespace(model_dump=lambda value=e, **kwargs: value) for e in episodes]
    rebuilt, terminal = collect_transitions(calls, proxies, tokenizer, f"round{version}", exclusions)
    stored = json.loads((folder / "transitions.json").read_text())
    if version == 0:
        admitted, overlong = admit_sequences(rebuilt, 16384)
        admission = json.loads((folder / "training-admission.json").read_text())
        if stored != admitted or admission["incomplete_generations"] != exclusions or admission["overlong_transitions"] != overlong:
            raise ValueError("stored training sequences/admission differ from exact native reconstruction")
        if json.loads((folder / "all-complete-transitions.json").read_text()) != rebuilt:
            raise ValueError("retained complete transitions differ from native reconstruction")
    else:
        overlong = []
        if stored != rebuilt:
            raise ValueError("post-update transitions differ from native reconstruction")
        if json.loads((folder / "incomplete-generations.json").read_text()) != exclusions:
            raise ValueError("post-update incomplete-generation evidence differs")
    # Rebuilding includes exact prompt rendering and feedback tokenization. Equality
    # therefore checks more than decoded text or just action/episode identity.
    summaries, complete_task_means = {}, {}
    for key, tasks in metrics.items():
        values = [value for group in tasks.values() for value in group]
        summaries[key] = {"episode_coverage": len(values), "observed_mean": sum(values) / len(values)}
        complete_task_means[key] = {name: sum(group) / 2 for name, group in tasks.items() if len(group) == 2}
    return {
        "episodes": len(episodes), "tasks": len(names), "episodes_per_task": 2,
        "generations": len(calls), "stop_conditions": dict(stops),
        "finish_reasons": dict(Counter(c.get("finish_reason", c.get("limit_reason")) for c in calls)),
        "incomplete_generation_exclusions": len(exclusions), "overlong_training_exclusions": len(overlong),
        "context_limit_events": sum(c.get("limit_reason") == "context_limit" for c in calls),
        "terminal_unexecuted_actions": len(terminal), "transitions": len(stored),
        "eligible_trajectories": len({r["trajectory_id"] for r in stored}),
        "original_action_history_feedback_and_teacher_tokens_reconstructed_exactly": True,
        "max_identical_tool_call_repetition_within_episode": max((max(v.values()) for v in signatures.values()), default=0),
        "native_and_scoring_errors": [], "assessment_batches": assessment_batches,
        "goals_and_harms": "unavailable: no goal/harm metrics or assessment findings in this run; do not infer zero harms",
        "retry_count": None, "retry_note": "No authoritative transport-attempt counter retained.",
        "metrics": summaries,
        "traces_sha256": hashlib.sha256((folder / "native/traces.jsonl").read_bytes()).hexdigest(),
    }, complete_task_means


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    protocol = json.loads(args.protocol.read_text())
    expected = protocol["cohorts"]["pilot"]["task_names"]
    heldout = set(json.loads(args.heldout.read_text())["task_names"])
    if set(expected) & (heldout | set(protocol["cohorts"]["development"]["task_names"])):
        raise ValueError("heldout/development leakage")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    before, a = phase_audit(args.input, expected, tokenizer, 0)
    after, b = phase_audit(args.input / "round1", expected, tokenizer, 1)
    receipt = json.loads((args.input / "native-receipt.json").read_text())
    if receipt["applied_steps"] != 32 or not receipt["ema_verified"] or receipt["external_model_calls"] != 0:
        raise ValueError("training receipt failed required update/EMA/offline gates")
    checks = receipt["refreshed_policy_validation"]
    if checks["selected_token_logprob_movement"] <= 0.01:
        raise ValueError("updated policy not demonstrated in serving")
    for version in (0, 1):
        if not json.loads((args.input / f"batch-invariance-v{version}.json").read_text())["exact_batch_invariance"]:
            raise ValueError("native runtime batch-invariance gate failed")
        if json.loads((args.input / f"parity-v{version}.json").read_text())["max_abs_delta"] > 0.15:
            raise ValueError("native actor/rollout score gate failed")
    losses = []
    for line in args.logs.read_text().replace("\r", "\n").splitlines():
        if line.startswith("{'loss':"):
            losses.append(ast.literal_eval(line))
    if len(losses) != 32 or not all(math.isfinite(float(row["loss"])) for row in losses):
        raise ValueError("32 finite logged optimizer losses required")
    comparisons = {}
    for key in ("task_completed_correctly", "partial_credit", "tool_mistakes", "tool_unknown_id"):
        if set(a[key]) != set(expected) or set(b[key]) != set(expected):
            raise ValueError(f"incomplete scoring coverage for {key}; no headline imputation permitted")
        comparisons[key] = paired_bootstrap(a[key], b[key])
    changed = {name: {key: {"before": a[key][name], "after": b[key][name]}
                      for key in ("task_completed_correctly", "partial_credit", "tool_mistakes")
                      if a[key][name] != b[key][name]}
               for name in expected}
    result = {"run_id": "self-qwen35-4b-pilot32-20261017-r3", "before": before, "after": after,
              "comparisons": comparisons, "changed_tasks": {k: v for k, v in changed.items() if v},
              "training": {"updates": 32, "logged_total_losses": [float(row["loss"]) for row in losses],
                           "mean_total_loss": receipt["train_metrics"]["train_loss"],
                           "last_microbatch_components": receipt["last_losses"],
                           "ema_verified": True, "training_and_save_s": receipt["training_wall_s"],
                           "truncated_training_sequences": 0, "adapter_refresh": checks["refreshes"]},
              "interpretation": "Training-task diagnostics only, one seed and one collection round; no heldout generalization or goal/harm result. Losses across different samples are not a learning curve on a fixed evaluation set.",
              "uncertainty": "Percentile paired task bootstrap over32 tasks and10000 draws; absolute intervals are separate. This does not estimate variability across training seeds.",
              "artifact_authority": str(args.input), "heldout62_overlap": [], "development24_overlap": []}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"comparisons": comparisons, "changed_tasks": result["changed_tasks"]}, indent=2))


if __name__ == "__main__":
    main()
