"""Replay real native SDPO messages with teacher-only guidance, no GPU/updates."""

import argparse
import hashlib
import json
from pathlib import Path


def qualification_bootstrap(args, names, episodes):
    path = getattr(args, "qualification_receipt", None)
    if path is None:
        return False
    if (args.prior.parent.name != "step-0000" or args.prior.name != "online_train" or args.step != 1
            or path.parent.resolve() != args.prior.parents[1].resolve()):
        raise ValueError("qualification receipt does not bind this bootstrap population")
    receipt = json.loads(path.read_text())
    native = receipt.get("stochastic_native") or {}
    if (not receipt.get("qualified") or receipt.get("optimizer_updates") != 0
            or not native.get("qualified") or native.get("temperature") != 1.
            or native.get("episodes") != episodes or episodes != 40 or len(names) != 20
            or set(native.get("tasks", [])) != names):
        raise ValueError("qualification bootstrap lacks the closed zero-update stochastic40 population")
    return True


def main(args):
    from sdpo_agent_ablation import associate
    from sdpo_luna_context import native_task_bindings
    from transformers import AutoTokenizer
    from verifiers.v1.episode import Episode

    guidance = json.loads(args.guidance.read_text())["guidance"]
    if args.task != "all" and args.task not in guidance:
        raise ValueError("preflight requires already generated4B guidance")
    retained = [json.loads(line) for line in (args.prior / "native/traces.jsonl").read_text().splitlines() if line]
    retained = [e for e in retained if args.task == "all" or e["task"]["data"]["name"] == args.task]
    if not retained:
        raise ValueError("retained native fixture task missing")
    names = {e["task"]["data"]["name"] for e in retained}
    if args.task == "all" and names != set(guidance):
        raise ValueError("full native preflight must cover every prepared task")

    def user(messages):
        return next(m["content"] for m in messages if m["role"] == "user")

    users = {user([n["message"] for n in e["traces"][0]["nodes"] if n.get("message")]) for e in retained}
    calls = [json.loads(line) for line in (args.prior / "generation-calls.jsonl").read_text().splitlines() if line]
    calls = [c for c in calls if user(c["messages"]) in users]
    wire = [Episode.model_validate(e).model_dump(mode="json") for e in retained]
    bound = native_task_bindings(wire, retained)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    risk_sampling = getattr(args, "selection_mode", "first-error-frontier") == "first-risk-frontier"
    rows, excluded = associate(calls, retained, tokenizer, "prepared-guidance", None, bound,
        32768, True, {name: guidance[name] for name in names}, args.guidance_context,
        action_risk_sampling=risk_sampling)
    if not rows:
        raise ValueError("no complete original transitions survived teacher context admission")
    if {row["task"] for row in rows} != names:
        raise ValueError("a prepared task has no admitted native transition")
    for row in rows:
        original = calls[row["generation_index"]]
        assert row["student_prefix_ids"] == original["prefix_ids"]
        assert row["action_ids"] == original["action_ids"]
        assert row["prepared_guidance_audit"]["student_receives_guidance"] is False
        assert row["prepared_guidance_audit"]["teacher_receives_guidance"] == (args.guidance_context == "taskwide")
        assert row["prepared_guidance_audit"]["terminal_findings_are_per_action_credit"] is False
        assert not row["offline_luna_reference_used"] and not row["solution_used"]
        assert max(len(row[k]) + len(row["action_ids"])
                   for k in ("student_prefix_ids", "teacher_prefix_ids")) <= 32768
    sampling = None
    bootstrap = qualification_bootstrap(args, names, len(retained))
    if getattr(args, "check_guidance_batch", False) and not risk_sampling:
        from sdpo_guidance_sampling import batch

        if args.task != "all" or args.cohorts is None:
            raise ValueError("batch admission preflight requires full frozen cohort")
        tasks = json.loads(args.cohorts.read_text())["train"]
        if set(tasks) != names:
            raise ValueError("batch admission task authority differs")
        selected = batch(rows, tasks, args.step, mode=args.selection_mode)
        sampling = {"mode": args.selection_mode, "step": args.step,
            "distinct_actions": len({(r["episode_id"], r["generation_index"]) for r in selected}),
            "task_coverage": len({r["task"] for r in selected}),
            "selected": [{k: r[k] for k in ("task", "episode_id", "generation_index", "terminal_text")}
                         for r in selected],
            "scope": "Offline selection from completed native qualification trajectories, not actual optimizer updates."}
    if risk_sampling:
        from sdpo_guidance_sampling import batch

        if (args.task != "all" or args.cohorts is None
                or not bootstrap and args.prior.parent.name != f"step-{args.step:04d}"):
            raise ValueError("risk sampling requires exact full training cohort and original online step")
        tasks = json.loads(args.cohorts.read_text())["train"]
        if set(tasks) != names:
            raise ValueError("risk preflight task order authority differs")
        ordinary, ordinary_excluded = associate(calls, retained, tokenizer, "prepared-guidance", None,
            bound, 32768, True, {name: guidance[name] for name in names}, args.guidance_context)
        if len(rows) != len(ordinary) or excluded != ordinary_excluded:
            raise ValueError("sampling cue changed transition admission")
        for current, prior in zip(rows, ordinary, strict=True):
            if any(current[key] != prior[key] for key in
                   ("student_prefix_ids", "teacher_prefix_ids", "action_ids", "teacher_feedback")):
                raise ValueError("sampling cue changed original student/teacher supervision tokens")
        selected = batch(rows, tasks, args.step, mode="first-risk-frontier")
        previous = batch(ordinary, tasks, args.step, mode="first-error-frontier")
        fields = ("task", "generation_index", "episode_id", "terminal_text", "observed_tool_error")
        sampling = {"mode": "first-risk-frontier", "step": args.step,
            "qualification_population_bootstrap": bootstrap,
            "student_teacher_tokens_and_admission_unchanged": True,
            "selected": [{**{k: r[k] for k in fields}, "risks": r["observed_action_risks"]} for r in selected],
            "matched_first_error_selection": [{k: r[k] for k in fields} for r in previous],
            "selected_risk_actions": sum(bool(r["observed_action_risks"]) for r in selected),
            "scope": "Selection change only; risk metadata is not a reward or teacher prompt change."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"qualified": True, "task": args.task,
        "episodes": len(retained), "calls": len(calls), "admitted": len(rows), "excluded": excluded,
        "tasks": sorted(names), "distinct_tasks": len(names),
        "guidance_context": args.guidance_context,
        "original_student_prefix_and_actions": True, "truncated_training_sequences": 0,
        "teacher_only_guidance": True, "exact_retained_native_binding": True,
        "guidance_sha256": hashlib.sha256(args.guidance.read_bytes()).hexdigest(),
        "maximum_teacher_context_and_target": max(len(r["teacher_prefix_ids"]) + len(r["action_ids"]) for r in rows),
        "optimizer_updates": 0, "model_calls": 0,
        "sampling": sampling,
        "scope": "Real native-protocol fixtures and exact teacher binding; not gradient or learning qualification"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--guidance", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--guidance-context", choices=("taskwide", "feedback-only", "feedback-status"), default="taskwide")
    parser.add_argument("--selection-mode", choices=("first-error-frontier", "first-risk-frontier"),
                        default="first-error-frontier")
    parser.add_argument("--cohorts", type=Path)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--check-guidance-batch", action="store_true")
    parser.add_argument("--qualification-receipt", type=Path)
    main(parser.parse_args())
