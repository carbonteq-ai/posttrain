"""Readiness for one shared64K inference engine with16K training admission."""

import json


async def qualify_long_rollout(agent):
    from sdpo_native_tool_gate import qualify

    args = agent.args
    original = (args.output, args.context_limit, args.reply_limit, agent.rollout.reply_limit)
    folder = args.output / "full-evaluation-budget-qualification"
    folder.mkdir(exist_ok=False)
    try:
        args.output, args.context_limit, args.reply_limit = folder, 65536, 8192
        agent.rollout.reply_limit = 8192
        tool_gate = await qualify(agent)
        seed = agent.tokenizer.encode("Neutral local long-context capacity evidence. ", add_special_tokens=False)
        length = 65536 - 8192
        prefix = (seed * (length // len(seed) + 1))[:length]
        action, complete, evidence = await agent.rollout.generate(prefix, max_tokens=8192)
        if not action:
            raise ValueError("long-context evaluation readiness returned no action")
        concurrency = getattr(args, "concurrency", 32)
        batch = await agent.rollout.batch_probe(prefix, maximum=concurrency)
        (folder / "receipt.json").write_text(json.dumps({
            "qualified": True, "context_limit": 65536, "reply_limit": 8192,
            "near_limit_prefix_tokens": len(prefix), "generated_tokens": len(action),
            "complete": complete, "generation": evidence, "batch_invariance": batch,
            "native_tool_continuation": tool_gate, "optimizer_updates": 0,
            "scope": f"64K serving, native tool/result/final and{concurrency}in-flight16token probes; "
                     "not simultaneous forced8K completions or64K training backward"
        }, indent=2) + "\n")
    finally:
        args.output, args.context_limit, args.reply_limit, agent.rollout.reply_limit = original
