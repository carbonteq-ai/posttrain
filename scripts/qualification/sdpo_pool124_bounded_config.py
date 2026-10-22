"""Remaining Luna/thinking ablations use broad tasks and explicit matched controls."""


def profile(args):
    if (args.run_profile != "pool124-bounded" or args.steps != 6 or args.batch_size != 32
            or args.online_rollouts != 2 or args.lr != 1e-5 or args.lr_end_ratio != .1
            or args.lr_schedule != "cosine-decay" or args.context_limit != 32768
            or args.reply_limit != 4096 or not args.qualify_capacity
            or args.teacher_feedback not in ("solution-or-feedback", "luna-or-solution")
            or (args.thinking and args.teacher_feedback != "solution-or-feedback")
            or args.tool_parser_mode != "schema-bound-v1" or args.selection_mode != "failure-frontier"
            or args.alpha != 1. or args.topk != 20 or args.teacher_rate != .01):
        raise ValueError("bounded full-pool ablation differs from authorized matched recipe")
    return {"batch": 32, "evaluate_every": 10, "save_every": 1, "time_limit_s": 6500}


def paired_commands(config, script, root, model, heldout, python):
    if (config["budget_slot"] != 9 or config["steps"] != 12 or config["training_tasks"] != 124
            or config["batch_size"] != 32 or config["context_limit"] != 32768
            or config["reply_limit"] != 4096 or config["wall_time_cap_seconds"] != 7200
            or config["selection_mode"] != "failure-frontier"
            or config["lr"] != 1e-5 or config["lr_schedule"] != "cosine-decay"
            or config["lr_end_ratio"] != .1 or config["alpha"] != 1
            or config["topk"] != 20 or config["teacher_rate"] != .01):
        raise ValueError("paired broad-corpus Luna authorization mismatch")
    common = [python, str(script), "--model", str(model), "--heldout", str(heldout),
              "--cohorts", str(root / "cohorts.json"), "--training-pool", str(root / "train-pool.json"),
              "--run-profile", "pool124-bounded", "--batch-size", "32", "--steps", "6",
              "--context-limit", "32768", "--reply-limit", "4096", "--online-rollouts", "2",
              "--lr", "1e-5", "--lr-schedule", "cosine-decay", "--lr-end-ratio", ".1",
              "--alpha", "1", "--topk", "20", "--teacher-rate", ".01",
              "--selection-mode", "failure-frontier", "--tool-parser-mode", "schema-bound-v1",
              "--qualify-capacity"]
    return [("control", common + ["--teacher-feedback", "solution-or-feedback"]),
            ("luna", common + ["--teacher-feedback", "luna-or-solution",
                               "--luna-references", str(root / "luna-reviewed.json")])]
