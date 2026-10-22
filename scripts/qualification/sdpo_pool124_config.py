"""Full-pool task coverage and one-task-per-example SDPO campaign rules."""

import hashlib

from sdpo_agent_config import balanced_rows


def validate_pool(cohort, pool, heldout):
    tasks = cohort["train"]
    if (len(tasks) != 124 or len(set(tasks)) != 124 or set(tasks) != set(pool["task_names"])
            or len(heldout["task_names"]) != 62 or set(tasks) & set(heldout["task_names"])
            or cohort["development"] or cohort["reserve"]):
        raise ValueError("exact124training/62heldout authority required; internal reserves retired")
    return tasks


def step_tasks(tasks, step):
    if len(tasks) != 124 or len(set(tasks)) != 124 or not 1 <= step <= 250:
        raise ValueError("invalid pool or update")
    ordered = sorted(tasks, key=lambda task: hashlib.sha256(
        ("sdpo-pool124-20261018-v1:" + task).encode()).hexdigest())
    start = (step - 1) * 32
    return [ordered[(start + i) % 124] for i in range(32)]


def task_batch(rows, tasks, offset):
    if len(tasks) != 32 or len(set(tasks)) != 32:
        raise ValueError("exactly32distinct tasks per update required")
    batch = balanced_rows(rows, 32, offset=offset, tasks=tasks, mode="failure-frontier")
    if len(batch) != 32 or {row["task"] for row in batch} != set(tasks):
        raise ValueError("task admission shortfall; never replace or duplicate tasks")
    return batch


def profile(args):
    qualification = args.run_profile == "pool124-qualify"
    if (args.run_profile not in ("pool124-qualify", "pool124-50")
            or args.steps != (4 if qualification else 50) or args.batch_size != 32
            or args.online_rollouts != 2 or args.lr != 1e-5 or args.lr_end_ratio != .1
            or args.lr_schedule != "hold10-decay20-flat" or args.context_limit != 16384
            or args.reply_limit != 2048 or args.thinking or args.qualify_capacity
            or args.shared_budget is not None or args.teacher_feedback != "solution-or-feedback"
            or args.tool_parser_mode != "schema-bound-v1" or args.selection_mode != "failure-frontier"
            or args.alpha != 1. or args.topk != 20 or args.teacher_rate != .01):
        raise ValueError("pool124 differs from authorized recipe")
    return {"batch": 32, "evaluate_every": 1 if qualification else 10,
            "save_every": 1 if qualification else 10,
            "time_limit_s": 7000 if qualification else 86200}


def collection_budget(cohort, run_profile=None):
    if cohort == "heldout62":
        return {"context": 65536, "reply": 8192, "turns": 20, "episode": 32768, "replicas": 5}
    if run_profile == "pool124-bounded":
        return {"context": 32768, "reply": 4096, "turns": 8, "episode": 12288, "replicas": 2}
    return {"context": 16384, "reply": 2048, "turns": 8, "episode": 12288, "replicas": 2}


def diagnostic_groups(tasks, heldout_names, run_profile, reference_names=None):
    if run_profile == "pool124-bounded":
        if (reference_names is None or len(reference_names) != 5 or len(set(reference_names)) != 5
                or not set(reference_names) <= set(tasks)):
            raise ValueError("five predeclared training-only reference diagnostic tasks required")
        return [("reference_train", reference_names), ("heldout62", heldout_names)]
    if run_profile not in ("pool124-qualify", "pool124-50", "lfm-pool124-20"):
        raise ValueError("unknown full-pool diagnostic profile")
    groups = [("train", step_tasks(tasks, 1))]
    if run_profile in ("pool124-50", "lfm-pool124-20"):
        groups.append(("heldout62", heldout_names))
    return groups


def known_frozen_scoring_errors(errors):
    """Only the previously identified LinkedIn extra-field finalization bug."""
    return bool(errors) and all(isinstance(error, dict) and error.get("type") == "TaskError"
        and "extra" in str(error).lower() and "article_" in str(error)
        and ("PostContent" in str(error) or "WorldState" in str(error)) for error in errors)
