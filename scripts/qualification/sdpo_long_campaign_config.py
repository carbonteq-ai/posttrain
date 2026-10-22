"""Explicit long-run authority; bounded ablations retain their original limits."""

import math

from sdpo_agent_config import balanced_rows


def campaign_learning_rate(peak, step, end_ratio=.1):
    """Hold updates1–10, cosine cooldown11–20, then a constant floor."""
    if not math.isfinite(peak) or peak <= 0 or not 1 <= step <= 250 or not 0 < end_ratio <= 1:
        raise ValueError("invalid campaign learning rate")
    if step <= 10:
        return peak
    if step >= 20:
        return peak * end_ratio
    fraction = (step - 10) / 10
    return peak * (end_ratio + (1 - end_ratio) * (1 + math.cos(math.pi * fraction)) / 2)


def execution_profile(args):
    """Never infer long-run permission by a larger numeric step count alone."""
    profile = getattr(args, "run_profile", "ablation")
    batch = getattr(args, "batch_size", 16)
    if profile == "guidance-five":
        if (args.steps != 10 or batch != 32 or args.online_rollouts != 2
                or args.context_limit != 32768 or args.reply_limit != 4096
                or not args.thinking or args.lr != 1e-6
                or args.lr_schedule != "constant" or args.alpha != 1.
                or args.topk != 20 or args.teacher_rate != .01
                or args.teacher_feedback != "prepared-guidance"
                or args.teacher_guidance is None or args.concurrency != 40
                or args.speculation not in ("mtp", "none")
                or (args.speculation == "none" and not getattr(args, "plain_decoding_authorized", False))
                or not args.qualify_capacity
                or getattr(args, "guidance_context", "taskwide") not in ("taskwide", "feedback-only")
                or args.tool_parser_mode != "schema-bound-v1"):
            raise ValueError("guidance-five differs from the frozen ten-update recipe")
        return {"batch": 32, "evaluate_every": 10, "save_every": 10, "time_limit_s": 7000}
    if profile == "lfm-pool124-20":
        from sdpo_lfm_config import profile as lfm_profile

        return lfm_profile(args)
    if profile == "pool124-bounded":
        from sdpo_pool124_bounded_config import profile as bounded_profile

        return bounded_profile(args)
    if profile in ("pool124-qualify", "pool124-50"):
        from sdpo_pool124_config import profile as pool_profile

        return pool_profile(args)
    if profile == "ablation":
        if not 1 <= args.steps <= 20 or batch != 16:
            raise ValueError("bounded ablation update/batch cap")
        return {"batch": 16, "evaluate_every": 5, "save_every": 1, "time_limit_s": 6500}
    if profile not in ("campaign-qualify", "campaign250"):
        raise ValueError("unknown execution profile")
    expected_steps = 2 if profile == "campaign-qualify" else 250
    if (args.steps != expected_steps or batch != 32 or args.online_rollouts != 4
            or args.lr != 1e-5 or args.lr_end_ratio != .1
            or args.lr_schedule != "hold10-decay20-flat"
            or args.context_limit != 16384 or args.reply_limit != 2048
            or args.thinking or args.qualify_capacity or args.shared_budget is not None
            or args.teacher_feedback != "solution-or-feedback"
            or args.tool_parser_mode != "schema-bound-v1"
            or args.selection_mode != "failure-frontier"
            or args.alpha != 1. or args.topk != 20 or args.teacher_rate != .01):
        raise ValueError("long campaign differs from authorized matched recipe")
    cadence = 1 if profile == "campaign-qualify" else 25
    return {"batch": 32, "evaluate_every": cadence, "save_every": cadence,
            "time_limit_s": 6500 if profile == "campaign-qualify" else 86200}


def distinct_campaign_batch(batch, size):
    keys = {(r["episode_id"], r["generation_index"]) for r in batch}
    if len(batch) != size or len(keys) != size:
        raise ValueError("not enough distinct complete training transitions; no silent duplication/refill")


def campaign_batch(rows, tasks, offset=0):
    """Two disjoint16-action selections preserve the prior tool/terminal balance."""
    first = balanced_rows(rows, 16, offset=offset, tasks=tasks, mode="failure-frontier")
    used = {(r["episode_id"], r["generation_index"]) for r in first}
    remaining = [r for r in rows if (r["episode_id"], r["generation_index"]) not in used]
    second = balanced_rows(remaining, 16, offset=offset, tasks=tasks, mode="failure-frontier")
    batch = first + second
    distinct_campaign_batch(batch, 32)
    return batch
