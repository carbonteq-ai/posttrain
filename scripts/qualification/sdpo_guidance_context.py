"""Teacher-only summarized context; original student actions remain untouched."""

import copy
import hashlib
import json
from collections import Counter


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def native_findings(episode, bundle):
    """Deduplicate completed native findings; absent coverage stays unknown."""
    roles = bundle.get("check_roles", {})
    signals = bundle["check_signals"]
    if set(signals) != set(roles) or len(set(signals.values())) != len(signals):
        raise ValueError("manifest must declare an unambiguous signal for every requirement")
    expected = {signals[key]: role for key, role in roles.items()}
    findings = {}
    for trace in episode["traces"]:
        for batch in trace.get("assessment_batches", []):
            if batch.get("run", {}).get("status") != "complete":
                continue
            for item in batch.get("assessments", []):
                signal = item["signal"]["signal_id"]
                if signal not in expected:
                    continue
                key = item["assessment_id"]
                if key in findings and findings[key] != item:
                    raise ValueError("conflicting native assessment identity")
                findings[key] = item
    rows = [{"assessment_id": a["assessment_id"], "signal": a["signal"]["signal_id"],
             "role": expected[a["signal"]["signal_id"]], "status": a["status"],
             "value": a.get("value"), "reason": a.get("reason")}
            for a in findings.values()]
    decided = [r for r in rows if r["status"] == "valid" and r["value"] in (0, 1)]
    failed = any((r["role"] == "goal" and r["value"] == 0)
                 or (r["role"] == "harm" and r["value"] == 1) for r in decided)
    summary = []
    for signal, role in expected.items():
        group = [r for r in rows if r["signal"] == signal]
        valid = [r for r in group if r["status"] == "valid" and r["value"] in (0, 1)]
        summary.append({"signal": signal, "role": role,
            "valid_instance_count": len(valid),
            "zero_values": sum(r["value"] == 0 for r in valid),
            "one_values": sum(r["value"] == 1 for r in valid),
            "other_status_counts": dict(Counter(r["status"] for r in group
                if not (r["status"] == "valid" and r["value"] in (0, 1)))),
            "reasons": dict(Counter(r.get("reason") for r in group if r.get("reason")))})
    return {"findings": rows, "summary": summary, "observed_obligation_failure": failed,
            "missing_signal_ids": sorted(set(expected) - {r["signal"] for r in decided}),
            "per_action_credit": False}


def teacher_messages(call, native_task, bundle, assertions, observations, findings=None, *, context_mode="taskwide"):
    """Bind approved preparation to the current task; never infer action credit."""
    if context_mode not in ("taskwide", "feedback-only", "feedback-status"):
        raise ValueError("unknown teacher guidance context mode")
    if (bundle["task"] != native_task["name"]
            or bundle["current_state_sha256"] != digest(native_task["initial_state"])
            or not bundle["review_approved"]
            or bundle["prepared_by_model"] != "Qwen/Qwen3.5-4B"
            or not bundle["instructions"]
            or not bundle["check_ids"]
            or set(bundle.get("check_roles", {})) != set(bundle["check_ids"])
            or set(bundle.get("check_signals", {})) != set(bundle["check_ids"])
            or not set(bundle["check_roles"].values()) <= {"goal", "harm"}):
        raise ValueError("teacher guidance lacks exact task binding or accepted4B preparation")
    if len(set(bundle["check_ids"])) != len(bundle["check_ids"]):
        raise ValueError("duplicate prepared requirement identity")
    from sdpo_checkwise_preparation import validate_instruction_text

    public = "\n\n".join(m["content"] for m in native_task["prompt"] if m["role"] == "user")
    if bundle.get("user_request_sha256") != digest(public):
        raise ValueError("teacher guidance/public task instruction binding differs")
    validate_instruction_text({"user_request": public, "tool_definitions": call.get("tools", [])},
                              bundle["instructions"], [])
    facts = {"terminal_native_assertions": assertions,
             "native_manifest_findings": None if findings is None else {
                 k: findings[k] for k in ("summary", "missing_signal_ids", "per_action_credit")},
             "current_action_observations": observations}
    if context_mode == "feedback-status":
        from sdpo_hindsight_projection import project

        projected = project(assertions, observations)
        facts = {**projected, "native_manifest_findings": facts["native_manifest_findings"]}
    text = (
        "Teacher-only task guidance follows. It is not an observation or an executed action. "
        "Historical procedures guide retrieval; obtain current values and IDs from real tools. "
        "Preserve required reads before dependent writes and do not invent extra requirements.\n"
        + json.dumps({"guidance": bundle["instructions"] if context_mode == "taskwide" else None}, ensure_ascii=False)
        + "\nFactual hindsight from this student episode:\n"
        + json.dumps(facts, ensure_ascii=False)
        + "\nTerminal findings describe the whole episode, not proof this particular action caused them. "
        "An empty successful retrieval is not evidence the desired item was found. "
        "An API error does not establish that the task goal is impossible. "
        "Correctly continue the original task at the original decision point using this evidence. "
        "Produce the next assistant action in the native reasoning/tool format, not a critique or grading report. "
        "Do not cite preparation labels or claim historical results were retrieved in this execution."
    )
    messages = copy.deepcopy(call["messages"])
    messages.append({"role": "user", "content": text})
    return messages, {"version": f"guidance-five-{context_mode}-v1", "task": bundle["task"],
        "guidance_sha256": digest(bundle), "student_receives_guidance": False,
        "context_mode": context_mode, "teacher_receives_guidance": context_mode == "taskwide",
        "hindsight_projection": "status-only-candidate-v1" if context_mode == "feedback-status" else None,
        "terminal_findings_are_per_action_credit": False}
