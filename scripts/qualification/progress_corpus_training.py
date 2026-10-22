"""Gated ten-update full124 SDPO driver; no fallback task mix or context."""

import argparse
import asyncio
import copy
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from progress_teacher_context import compose_corpus, selected_observations, semantic_selection
from qualify_combined_teacher_contexts import native_messages, write

MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def gate(root, plan, heldout):
    """Runtime receipts cannot substitute for task-addressed semantic review."""
    evidence = root / "evidence/run"
    receipt = json.loads((evidence / "receipt.json").read_text())
    approval = json.loads((root / "evidence/qualification-gate.json").read_text())
    cohort = json.loads((root / "cohorts.json").read_text())["train"]
    required = {"source-sha256.txt", "experiment.json", "cohorts.json", "heldout62.json", "evidence/cpu/materials.json",
                "evidence/run/receipt.json", "evidence/native-audit.json", "evidence/teacher-token-bindings.json"}
    if not required <= set(approval.get("qualification_files_sha256", {})):
        raise ValueError("qualification approval lacks required evidence bindings")
    for line in (root / "source-sha256.txt").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("qualified source snapshot changed")
    for name in ("progress_teacher_context.py", "sdpo_compact_guidance.py", "sdpo_prompting.py"):
        if (Path(__file__).parent / name).read_bytes() != (root / "driver" / name).read_bytes():
            raise ValueError("training context implementation differs from qualification")
    if (heldout != json.loads((root / "heldout62.json").read_text())
            or len(cohort) != 124 or len(set(cohort)) != 124 or set(plan["corpus_tasks"]) != set(cohort)
            or set(cohort) & set(heldout["task_names"]) or len(heldout["task_names"]) != 62
            or len(plan["steps"]) != 10 or plan["batch_distinct_tasks"] != 32):
        raise ValueError("exact124/heldout62/ten-step32-task protocol required")
    if (not receipt["runtime_completed"] or receipt["optimizer_updates"] != 0
            or receipt["native_episodes"] != 496 or receipt["same_decision_generations"] != 372
            or not approval.get("qualified") or not approval.get("semantic_qualified")
            or not approval.get("native_execution_qualified") or not approval.get("exact_token_bindings_qualified")
            or approval.get("known_blockers") or approval.get("task_count") != 124):
        raise ValueError("full124 source/native/semantic/input gate has not passed")
    reviews = approval.get("task_reviews", [])
    if (len(reviews) != 124 or {r["task"] for r in reviews} != set(cohort)
            or any(r.get("decision") != "pass" or not r.get("evidence") for r in reviews)):
        raise ValueError("each task needs explicit evidence-backed guidance review")
    for relative, digest in approval["qualification_files_sha256"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("qualification source/evidence changed")
    if not approval["qualification_files_sha256"]:
        raise ValueError("unbound qualification approval")
    for row in plan["steps"]:
        if len(row["tasks"]) != 32 or len(set(row["tasks"])) != 32 or not set(row["tasks"]) <= set(cohort):
            raise ValueError("every update requires32distinct allocated tasks")
    return approval


def retained_population(wire, retained, names):
    if ({e["id"]: e["task"]["data"]["name"] for e in retained}
            != {e["id"]: e["task"]["data"]["name"] for e in wire}
            or len(retained) != 64 or len(wire) != 64 or len({e["id"] for e in retained}) != 64
            or Counter(e["task"]["data"]["name"] for e in retained) != Counter({name: 2 for name in names})):
        raise ValueError("retained native/wire population identity differs")
    return retained


def teacher_batch(rows, calls, episodes, materials, names, tokenizer):
    """One original complete action per allocated task, with full context."""
    from sdpo_prompting import prompt_ids
    from self_feedback_signals import authority_errors

    batch = copy.deepcopy(semantic_selection(rows, calls, episodes, names))
    by_episode = {e["id"]: e for e in episodes}
    for row in batch:
        call = calls[row["generation_index"]]
        episode = by_episode[row["episode_id"]]
        material = materials[row["task"]]
        data = episode["task"]["data"]
        if (data["assertions"] != material["raw_native_data"]["assertions"]
                or authority_errors(material["material"]["goal_manifest"], data["initial_state"], data["prompt"])):
            raise ValueError("fresh native task authority differs")
        trace = episode["traces"][0]
        observations = selected_observations(call, [node["message"] for node in trace["nodes"] if node.get("message")])
        supplement = compose_corpus(row["task"], call["messages"], material["compiled"]["sheet"],
                                   trace.get("info", {}).get("automationbench", {}).get("assertions", []), observations)
        prefix = prompt_ids(tokenizer, native_messages(call["messages"]) + [
            {"role": "user", "content": supplement}], call["tools"], thinking=True)
        if len(prefix) + len(row["action_ids"]) > 32768:
            raise ValueError(f"whole teacher transition overlong for {row['task']}; no truncation/replacement")
        if call["prefix_ids"] != row["student_prefix_ids"] or call["action_ids"] != row["action_ids"]:
            raise ValueError("original student/action identity changed")
        row["teacher_prefix_ids"], row["teacher_feedback"] = prefix, supplement
    if len(batch) != 32 or {r["task"] for r in batch} != set(names):
        raise ValueError("one complete action per allocated task required; no dropping tasks")
    return batch


async def train(args):
    import torch
    from aiohttp import web
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
    from sdpo_agent_ablation import Agent, SamplingRollout, associate, loss_for, score_sampled, teacher_logits
    from sdpo_capacity import qualify_actor, qualify_rollout
    from sdpo_native_tool_gate import qualify as qualify_native
    from sdpo_pool124_runtime import qualify_long_rollout
    from sdpo_schema_parser import parser_class
    from transformers import AutoModelForImageTextToText, AutoTokenizer, set_seed
    from verifiers_preinstalled import configure_preinstalled_runtime

    started = time.monotonic()
    plan = json.loads(args.plan.read_text())
    heldout = json.loads(args.heldout.read_text())
    approval = gate(args.qualification, plan, heldout)
    if Path(args.model).name != MODEL_REVISION:
        raise ValueError("model pin changed")
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "accepted-qualification.json", approval)
    os.environ.update(POSTTRAIN_VERIFIERS_PREINSTALLED="1", SELF_LOCAL_KEY="EMPTY")
    configure_preinstalled_runtime(interpreter=sys.executable)
    set_seed(18)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer._sdpo_model_family = "qwen3.5"
    model = get_peft_model(AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16,
        device_map={"": "cuda:0"}, attn_implementation="sdpa", local_files_only=True),
        LoraConfig(r=16, lora_alpha=32, lora_dropout=0., target_modules="all-linear",
                   exclude_modules=r".*visual.*", task_type="CAUSAL_LM"))
    model.save_pretrained(args.output / "initial-student", selected_adapters=["default"])
    model.load_adapter(args.output / "initial-student", adapter_name="sdpo_teacher", is_trainable=False)
    model.set_adapter("default")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-6, weight_decay=.01)
    rollout = await SamplingRollout.create(args.model, model, tokenizer, args.output, 65536, 40, 16,
                                           speculative_config=None, prefix_caching=False, chunked_prefill=False)
    if rollout.refreshes[0]["weights_sha256"] != "e71dae81f1b4723c83aacd130bfbf60b072b3580212133a4fc45dcb9f403421c":
        raise ValueError("fresh initial student differs")
    environment_profile = json.loads((args.qualification / "experiment.json").read_text())["environment_profile"]
    config = SimpleNamespace(output=args.output, context_limit=32768, reply_limit=4096, thinking=True,
        feedback_mode="minimal", model_family="qwen3.5", tool_parser_mode="schema-bound-v1",
        run_profile="guidance-five", concurrency=40, run_name=args.run_name, teacher_feedback="native-feedback",
        online_rollouts=2, native_repeatability=True, environment_profile=environment_profile,
        topk=20, alpha=1.)
    rollout.reply_limit = 4096
    agent = Agent(config)
    agent.rollout, agent.tokenizer, agent.model = rollout, tokenizer, model
    agent.schema_parser, parser_receipt = parser_class()
    write(args.output / "parser.json", parser_receipt)
    materials = {b["material"]["task"]: b for b in json.loads(
        (args.qualification / "evidence/cpu/materials.json").read_text())}
    app = web.Application(client_max_size=16 * 1024 * 1024)
    app.router.add_post("/v1/chat/completions", agent.completions)
    runner = web.AppRunner(app)
    await runner.setup()
    updates = []
    try:
        await web.TCPSite(runner, "127.0.0.1", 18774).start()
        probe = tokenizer.encode("Local readiness", add_special_tokens=False)
        target = tokenizer.encode(" verified.", add_special_tokens=False)
        await rollout.score_probe(model, probe, target)
        await rollout.batch_probe(probe, maximum=40)
        await qualify_native(agent)
        await qualify_long_rollout(agent)
        capacity = await qualify_rollout(config, model, tokenizer, rollout)
        prior_base = args.qualification / "evidence/run/step-0000/baseline_train10"
        baseline = json.loads((prior_base / "outcomes.json").read_text())
        if baseline["episodes"] != 248 or baseline["tasks"] != 124 or baseline["errors"]:
            raise ValueError("full124 qualification control incomplete")
        agent.baselines["train124"] = baseline
        reference = {"source": str(prior_base), "referenced_native_episodes": 248,
                     "initial_adapter_sha256": rollout.refreshes[0]["weights_sha256"],
                     "scope": "Already-qualified repaired-source full124 greedy control; count native IDs once, not as fresh training episodes."}
        write(args.output / "referenced-training-base.json", reference)
        agent.curve.append({"step": 0, "cohort": "train124", "summary": baseline,
                            "reference": reference})
        await agent.collect(0, "heldout62", heldout["task_names"])
        for allocation in plan["steps"]:
            step, names = allocation["step"], allocation["tasks"]
            if time.monotonic() - started >= 7170:
                raise TimeoutError("training window exhausted; preserve without automatic resume")
            wire_episodes, calls = await agent.collect(step, "online_train", names, temperature=1.)
            folder = args.output / f"step-{step:04d}/online_train"
            episodes = retained_population(wire_episodes,
                [json.loads(line) for line in (folder / "native/traces.jsonl").read_text().splitlines() if line], names)
            rows, excluded = associate(calls, episodes, tokenizer, "native-feedback", context_limit=32768, thinking=True)
            batch = teacher_batch(rows, calls, episodes, materials, names, tokenizer)
            write(folder / "training-admission.json", {"selected_task_population": names, "selected": batch,
                "exclusions": excluded, "truncated_training_sequences": 0, "distinct_tasks": 32,
                "actual_optimizer_updates_before_collection": step - 1,
                "collection_policy_versions": sorted({c["policy_version"] for c in calls}),
                "scope": "Pre-update stochastic training diagnostics on rotating tasks, not a fixed-cohort post-update reward curve"})
            await rollout.suspend_for_update()
            if step == 1:
                qualify_actor(config, model, tokenizer, capacity, teacher_logits, loss_for, score_sampled)
                if optimizer.state:
                    raise ValueError("zero-update capacity probe changed optimizer")
                torch.cuda.empty_cache()
            optimizer.zero_grad(set_to_none=True)
            losses, before, directions = [], [], []
            for row in batch:
                old = score_sampled(model, row)
                teacher = teacher_logits(model, row)
                loss, advantage = loss_for(model, row, config, teacher=teacher)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite SDPO loss")
                (loss / 32).backward()
                losses.append(float(loss.detach()))
                before.append((old, advantage))
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
            if not torch.isfinite(norm) or norm <= 0:
                raise ValueError("nonfinite/zero full-batch gradient")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            for row, (old, advantage) in zip(batch, before, strict=True):
                delta = score_sampled(model, row) - old
                directions.append({"task": row["task"], "episode_id": row["episode_id"],
                    "generation_index": row["generation_index"], "sampled_teacher_advantage": advantage,
                    "actual_adam_sampled_logprob_deltas": delta.tolist(),
                    "teacher_support_diagnostic": row["teacher_support_diagnostic"],
                    "scope": "Local token movement, not correctness or reward gain"})
            student = get_peft_model_state_dict(model, adapter_name="default")
            teacher = get_peft_model_state_dict(model, adapter_name="sdpo_teacher")
            teacher = {k: v.lerp(student[k].to(v.device), .01) for k, v in teacher.items()}
            set_peft_model_state_dict(model, teacher, adapter_name="sdpo_teacher")
            model.set_adapter("default")
            moments = sorted({int(s["step"]) for s in optimizer.state.values() if "step" in s})
            if moments != [step]:
                raise ValueError("actual optimizer moment/update mismatch")
            checkpoint = args.output / f"checkpoint-{step}"
            model.save_pretrained(checkpoint)
            torch.save(optimizer.state_dict(), checkpoint / "optimizer.pt")
            write(checkpoint / "campaign-checkpoint.json", {"step": step, "ema_updates": step,
                "batch": 32, "optimizer_moment_steps": moments, "resume_qualified": False,
                "files_sha256": {str(p.relative_to(checkpoint)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in checkpoint.rglob("*") if p.is_file()}})
            write(folder / "teacher-adam-directions.json", directions)
            record = {"step": step, "loss": sum(losses) / 32, "batch": 32, "learning_rate": 1e-6,
                      "grad_norm": float(norm), "optimizer_moment_steps": moments, "ema_updates": step,
                      "elapsed_s": time.monotonic() - started}
            updates.append(record)
            write(args.output / "updates.json", updates)
            print(json.dumps({"stage": "optimizer_update", **record}), flush=True)
            await rollout.refresh()
            await rollout.score_probe(model, probe, target)
            if step == 1:
                await rollout.batch_probe(probe, maximum=40)
                old_output = config.output
                config.output = args.output / "changed-adapter-qualification"
                config.output.mkdir()
                try:
                    await qualify_native(agent)
                finally:
                    config.output = old_output
        # Complete the primary matched evaluation before secondary training
        # diagnostics; both retain their full declared budgets and populations.
        await agent.collect(10, "heldout62", heldout["task_names"])
        await agent.collect(10, "train124", plan["corpus_tasks"])
        write(args.output / "receipt.json", {"completed": True, "optimizer_updates": len(updates),
            "ema_updates": len(updates), "training_tasks": 124, "distinct_tasks_per_update": 32,
            "expected_online_episodes": 640, "elapsed_s": time.monotonic() - started,
            "heldout_tuning": False, "external_model_calls": 0, "resume_qualified": False})
    finally:
        await runner.cleanup()
        await rollout.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    asyncio.run(train(parser.parse_args()))
