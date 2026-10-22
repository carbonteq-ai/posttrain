"""Explicit new LFM20 authorization, separate from the closed ten Qwen attempts."""

MODEL_REVISION = "654f9463ce32b05d0429d76fe1f580b27d4c1ac0"


def profile(args):
    if (args.run_profile != "lfm-pool124-20" or args.model_family != "lfm2.5"
            or args.steps != 20 or args.batch_size != 32 or args.online_rollouts != 2
            or args.lr != 1e-5 or args.lr_end_ratio != .1
            or args.lr_schedule != "hold10-decay20-flat"
            or args.context_limit != 16384 or args.reply_limit != 2048
            or not args.thinking or not args.qualify_capacity or args.shared_budget is not None
            or args.teacher_feedback != "solution-or-feedback"
            or args.tool_parser_mode != "lfm2-native" or args.selection_mode != "failure-frontier"
            or args.alpha != 1. or args.topk != 20 or args.teacher_rate != .01):
        raise ValueError("LFM20 differs from the declared model/data/budget recipe")
    return {"batch": 32, "evaluate_every": 10, "save_every": 1, "time_limit_s": 7100}
