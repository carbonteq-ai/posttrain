"""Compile explicitly reviewed native solver episodes through the canonical adapter.

Never admit an episode from its scalar score alone. The review is separate,
episode-addressed evidence; raw native files remain unchanged.
"""

import argparse
import hashlib
import json
from pathlib import Path

from luna_ssr_sft import exclusions

REVIEW_CHECKS = ("reasoning_grounded", "tool_order_correct", "preservation_claims_supported",
                 "no_privileged_context_dependency", "complete_native_checks")


def selected_records(episodes, reviews, protected):
    selected, audit = [], []
    by_id = {str(row["episode_id"]): row for row in reviews}
    if len(by_id) != len(reviews):
        raise ValueError("duplicate episode review")
    ids = {str(e["id"]) for e in episodes}
    if len(ids) != len(episodes) or not set(by_id) <= ids:
        raise ValueError("duplicate native IDs or unknown reviewed episode")
    for episode in episodes:
        eid = str(episode["id"])
        task = episode["task"]["data"]["name"]
        review = by_id.get(eid, {})
        reasons = []
        if task in protected:
            reasons.append("protected_evaluation_task")
        if review.get("task") != task or not review.get("evidence"):
            reasons.append("missing_bound_review")
        reasons += [key for key in REVIEW_CHECKS if review.get(key) is not True]
        if episode.get("ok") is not True or episode.get("errors") or episode.get("assessment_errors"):
            reasons.append("native_episode_error")
        for trace in episode["traces"]:
            if (trace.get("ok") is not True or trace.get("errors") or trace.get("assessment_errors")
                    or trace.get("credit_errors") or trace.get("stop_condition") in ("max_turns", "error")):
                reasons.append("native_trace_error_or_incomplete")
            if trace.get("metrics", {}).get("task_completed_correctly") != 1:
                reasons.append("native_success_not_one")
            if trace.get("rewards", {}).get("partial_credit", {}).get("score") != 1:
                reasons.append("native_partial_not_one")
            if any(c.get("finish_reason") == "length" for c in trace.get("calls", [])):
                reasons.append("length_limited_call")
        admitted = not reasons
        audit.append({"episode_id": eid, "task": task, "admitted": admitted, "reasons": sorted(set(reasons))})
        if admitted:
            selected.append(episode)
    return selected, audit


def main(args):
    from posttrain.data import supervised_from_verifiers_jsonl

    episodes = [json.loads(line) for line in args.native.read_text().splitlines() if line.strip()]
    protected, exclusion_hashes = exclusions(args.project)
    chosen, audit = selected_records(episodes, json.loads(args.reviews.read_text()), protected)
    args.output.mkdir(exist_ok=False, parents=True)
    # Even an empty selection receives an honest audit; never manufacture a
    # nonempty dataset or silently train on rejected records.
    native = args.output / "admitted-native.jsonl"
    native.write_text("".join(json.dumps(e) + "\n" for e in chosen))
    if chosen:
        dataset = supervised_from_verifiers_jsonl(native, dataset_id="luna-solver-reviewed-sft", revision="v1")
        rows = [{"id": e.id, "messages": e.message_records(), "tools": e.tool_records(),
                 "trainable_message_indices": list(e.trainable_message_indices), "metadata": dict(e.metadata)}
                for e in dataset.examples]
        (args.output / "supervised.jsonl").write_text("".join(json.dumps(e) + "\n" for e in rows))
    (args.output / "admission.json").write_text(json.dumps({
        "source_sha256": hashlib.sha256(args.native.read_bytes()).hexdigest(),
        "review_sha256": hashlib.sha256(args.reviews.read_bytes()).hexdigest(),
        "exclusion_hashes": exclusion_hashes, "episodes": audit,
        "admitted_episodes": len(chosen), "tokenization_and_no_truncation_qualified": False,
        "training_ready": False,
    }, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[2])
    main(parser.parse_args())
