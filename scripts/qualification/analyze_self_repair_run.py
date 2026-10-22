"""Audit a completed ten-update repair experiment against retained native evidence."""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from self_campaign_config import comparisons, summarize

FAILURES = ["hr.offboarding_automation", "simple.email_sf_contact_mobile_update",
            "simple.gmail_reply_meeting_request", "simple.slack_ceo_announcement"]
RETENTION = ["simple.mailchimp_add_subscriber", "simple.sf_lead_revenue_update",
             "simple.trello_catering_contract", "finance.po_email_logging"]


def read(path):
    return json.loads(path.read_text())


def native_tool_observations(episodes):
    """Factual response observations, not replacement benchmark rewards."""
    totals, by_task = Counter(), {}
    for episode in episodes:
        counts, calls, empty_seen, signatures = Counter(), {}, set(), []
        for trace in episode["traces"]:
            for node in trace.get("nodes", []):
                message = node.get("message") or {}
                for call in message.get("tool_calls") or []:
                    function = call.get("function", call)
                    arguments = function.get("arguments", {})
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            pass
                    signature = json.dumps([function.get("name"), arguments], sort_keys=True)
                    calls[call["id"]] = signature
                    signatures.append(signature)
                if message.get("role") != "tool":
                    continue
                signature = calls.get(message.get("tool_call_id"))
                counts["unmatched_tool_responses"] += signature is None
                content = message.get("content", "")
                text_error = isinstance(content, str) and content.startswith("Error executing tool ")
                if text_error:
                    counts["textual_tool_error_responses"] += 1
                    counts["observed_tool_error_responses"] += 1
                try:
                    response = json.loads(content) if isinstance(content, str) else content
                except (json.JSONDecodeError, TypeError):
                    continue
                if not isinstance(response, dict):
                    continue
                counts["explicit_tool_error_responses"] += response.get("success") is False
                structured_error = bool(response.get("error")) and response.get("success") is not True
                counts["structured_error_field_responses"] += structured_error
                counts["observed_tool_error_responses"] += response.get("success") is False or structured_error
                empty = ((response.get("messages") == [] and response.get("result_count") == 0)
                         or (response.get("results") == [] and response.get("count") == 0))
                if empty:
                    counts["factual_empty_retrieval_responses"] += 1
                    counts["repeated_identical_empty_responses"] += signature is not None and signature in empty_seen
                    if signature is not None:
                        empty_seen.add(signature)
        counts["attempted_tool_calls"] = len(signatures)
        counts["repeated_exact_tool_calls"] = len(signatures) - len(set(signatures))
        totals.update(counts)
        name = episode["task"]["data"]["name"]
        by_task.setdefault(name, Counter()).update(counts)
    return {"totals": dict(totals), "by_task": {name: dict(v) for name, v in by_task.items()},
            "scope": "API-successful empty retrievals are observations, not universally incorrect actions; "
                     "unexecuted terminal calls are attempts, not observed empty responses. "
                     "Explicit-tool-error is the historical success=false count; observed-tool-error also includes "
                     "nonempty error fields without success=true and textual execution exceptions."}


def audit(root):
    receipt, inputs, updates = [read(root / name) for name in ["receipt.json", "inputs.json", "updates.json"]]
    if not receipt["completed"] or receipt["optimizer_updates"] != 10 or [u["step"] for u in updates] != list(range(1, 11)):
        raise ValueError("incomplete update population")
    teacher = inputs.get("repair_objective", "").startswith("teacher_forward_") or inputs.get("kl_weight", 0) > 0
    if any(u["batch_size"] != 16 or u["ema_updates"] != (u["step"] if teacher else 0) for u in updates):
        raise ValueError("batch/EMA update mismatch")
    provenance = read(root / "policy-provenance.json")
    if provenance["exported"] != provenance["source"] or set(provenance["exported"]) != {"0", "1", "2"}:
        raise ValueError("selected/exported student differs")
    ids, phases, baseline, terminal = set(), [], {}, {}
    for version, step in enumerate([0, 5, 10]):
        for cohort, names in [("failures", FAILURES), ("retention", RETENTION)]:
            folder = root / f"step-{step:04d}" / cohort
            path = folder / "native/traces.jsonl"
            episodes = [json.loads(line) for line in path.read_text().splitlines() if line]
            if ids & {e["id"] for e in episodes}:
                raise ValueError("reused native episode ID")
            ids |= {e["id"] for e in episodes}
            summary = summarize(episodes, names, 2)
            if summary != read(folder / "outcomes.json"):
                raise ValueError("native summary mismatch")
            calls = [json.loads(line) for line in (folder / "generation-calls.jsonl").read_text().splitlines() if line]
            rejected = 0
            for call in calls:
                if call["policy_version"] != version:
                    raise ValueError("stale native policy version")
                if (call.get("limit_reason") == "context_limit" and not call.get("raw")
                        and not call.get("action_ids") and not call.get("complete")):
                    rejected += 1
                elif call.get("adapter_sha256") != provenance["exported"][str(version)]:
                    raise ValueError("generation used wrong adapter")
            admission = read(folder / "training-admission.json")
            if admission["truncated_training_sequences"] or admission["training_allowed"]:
                raise ValueError("evaluation population used for training or truncated")
            if step == 0:
                baseline[cohort] = summary
            if step == 10:
                terminal[cohort] = summary
            phases.append({"step": step, "cohort": cohort, "summary": summary,
                           "native_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "generation_calls": len(calls), "context_rejections": rejected,
                           "length_calls": sum(c.get("finish_reason") == "length" for c in calls),
                           "tool_observations": native_tool_observations(episodes),
                           "admission": admission, "compared_to_step0": comparisons(baseline[cohort], summary)})
    if len(ids) != 48:
        raise ValueError("expected48 unique native episodes")
    before, after = [read(root / f"argument-margins-{step}.json") for step in [0, 10]]
    if len(before) != 8 or len(after) != 8:
        raise ValueError("missing repair-context margins")
    deltas = []
    for b, a in zip(before, after, strict=True):
        if (b["task"], b["history_tool_messages"]) != (a["task"], a["history_tool_messages"]):
            raise ValueError("margin context identity mismatch")
        deltas.append({"task": a["task"], "history_tool_messages": a["history_tool_messages"],
                       "before": b["correct_minus_bad_logit"], "after": a["correct_minus_bad_logit"],
                       "delta": a["correct_minus_bad_logit"] - b["correct_minus_bad_logit"],
                       "correct_rank_after": a["correct_token_rank"]})
    checks = {"all_first_context_margins_improve": all(x["delta"] > 0 for x in deltas[:4]),
              "native_errors_absent": all(not x["summary"]["errors"] for x in phases)}
    for cohort in ["failures", "retention"]:
        for metric in ["task_completed_correctly", "partial_credit"]:
            b = baseline[cohort]["metrics"][metric]
            a = terminal[cohort]["metrics"][metric]
            complete = len(b["complete_task_means"]) == len(a["complete_task_means"]) == 4
            checks[f"{cohort}_{metric}_complete"] = complete
            checks[f"{cohort}_{metric}_no_task_regression"] = complete and all(
                a["complete_task_means"][name] >= value for name, value in b["complete_task_means"].items())
            if cohort == "failures":
                checks[f"{metric}_improves"] = complete and a["full_population_task_mean"] > b["full_population_task_mean"]
    if not all(math.isfinite(x["delta"]) for x in deltas):
        raise ValueError("nonfinite margin")
    return {"receipt": receipt, "inputs": inputs, "episodes": len(ids), "phases": phases,
            "policy_provenance": provenance, "actual_update_margin_changes": deltas,
            "qualification_checks": checks, "qualified_local_repair": all(checks.values()),
            "scope": "Selected offline repair examples and four retention tasks only; not heldout generalization. "
                     "Token margins are diagnostics, not complete-call proof. No separate goal/harm assessment batch; "
                     "inspect native preservation assertions separately rather than infer harm-free behavior. Retry count unknown."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(audit(args.root), indent=2) + "\n")
