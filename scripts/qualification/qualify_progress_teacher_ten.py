"""Fresh ten-training-task native gate for system/progress-aware teacher context.

No optimizer update is allowed in this stage. Semantic review precedes a separate
paired diagnostic, so a malformed teacher proposal cannot silently start training.
"""

import argparse
import asyncio
import copy
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from progress_environment_profile import (
    ACCESS,
    DISCOVERY,
    FROZEN,
    REPAIRED,
    task_options,
    taskset_options,
    validate_native_source,
    validate_reference,
)
from progress_teacher_context import (
    balanced_batch,
    compose,
    compose_brief,
    compose_corpus,
    digest,
    selected_observations,
    semantic_selection,
)
from qualify_combined_teacher_contexts import native_messages, parsed_message, replay, serving_identity, write


def cpu(args):
    from automationbench_v1.contracts.loader import load_task_contract
    from automationbench_v1.limited_tools import selected_tool_definitions
    from automationbench_v1.taskset import AutomationBenchConfig, AutomationBenchTaskConfig, AutomationBenchTaskset
    from automationbench_v1.world_codec import load_world
    from sdpo_compact_guidance import compile_guidance
    from self_feedback_signals import authority_errors

    cohort = json.loads(args.cohorts.read_text())
    names = cohort["train"]
    count = getattr(args, "expected_task_count", 10)
    if len(names) != count or len(set(names)) != count or cohort["protected_overlap"]:
        raise ValueError("requires the exact frozen task population with no heldout62 overlap")
    pool = json.loads(args.pool.read_text())["task_names"]
    if not set(names) <= set(pool):
        raise ValueError("not a subset of the frozen124pool")
    if count == 124 and set(names) != set(pool):
        raise ValueError("full-pool qualification cannot drop tasks")
    materials = []
    for name in names:
        task = AutomationBenchTaskset(AutomationBenchConfig(domains=[name.split(".")[0]], task_names=[name],
            **taskset_options(args.environment_profile, [name]),
            task=AutomationBenchTaskConfig(toolset="limited_zapier", turn_budget=8,
                                          capture_actions=True, deterministic_world=True,
                                          **task_options(args.environment_profile)))).load()[0]
        data = task.data.model_dump(mode="json")
        raw = copy.deepcopy(data)
        load_world(data["initial_state"])
        contract = load_task_contract(name).model_dump(mode="json", exclude_none=True)
        errors = authority_errors(contract, data["initial_state"], data["prompt"])
        if errors:
            raise ValueError(f"native task authority failed {name}: {errors}")
        material = {"task": name, "user_request": "\n\n".join(m["content"] for m in data["prompt"] if m["role"] == "user"),
                    "native_data": data, "tool_definitions": selected_tool_definitions(data["zapier_tools"]),
                    "goal_manifest": contract, "reference_actions": []}
        compiled = compile_guidance(material)
        materials.append({"material": material, "raw_native_data": raw, "compiled": compiled})
    write(args.output / "materials.json", materials)
    write(args.output / "receipt.json", {"cpu_qualified": True, "tasks": names,
        "environment_profile": args.environment_profile,
        "model_calls": 0, "optimizer_updates": 0, "luna_traces_used": False,
        "scope": "Pinned constructor/unchanged manifest binding and lossless compact contract only; not native behavior proof"})


async def run(args):
    import torch
    from aiohttp import web
    from peft import LoraConfig, get_peft_model
    from sdpo_agent_ablation import Agent, SamplingRollout, associate
    from sdpo_prompting import prompt_ids
    from sdpo_schema_parser import parser_class
    from transformers import AutoModelForImageTextToText, AutoTokenizer, set_seed
    from verifiers_preinstalled import configure_preinstalled_runtime

    names = json.loads(args.cohorts.read_text())["train"]
    count = len(names)
    if count != getattr(args, "expected_task_count", 10):
        raise ValueError("unexpected task count")
    source_reference = args.reference or args.native_reference
    if args.native_reference is not None:
        current = json.loads((args.cohorts.parent / "experiment.json").read_text())
        previous = json.loads((args.native_reference.parent.parent / "experiment.json").read_text())
        receipt = json.loads((args.native_reference / "receipt.json").read_text())
        validate_native_source(current, previous, receipt)
        if args.environment_profile not in (REPAIRED, ACCESS, DISCOVERY):
            raise ValueError("native reference profile differs")
    materials = json.loads(args.materials.read_text())
    by_name = {m["material"]["task"]: m for m in materials}
    if set(by_name) != set(names):
        raise ValueError("prepared task population differs")
    set_seed(18)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer._sdpo_model_family = "qwen3.5"
    model = get_peft_model(AutoModelForImageTextToText.from_pretrained(args.model,
        dtype=torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation="sdpa", local_files_only=True),
        LoraConfig(r=16, lora_alpha=32, lora_dropout=0., target_modules="all-linear",
                   exclude_modules=r".*visual.*", task_type="CAUSAL_LM"))
    rollout = await SamplingRollout.create(args.model, model, tokenizer, args.output,
        65536, 40, 16, speculative_config=None, prefix_caching=False, chunked_prefill=False)
    if rollout.refreshes[0]["weights_sha256"] != "e71dae81f1b4723c83aacd130bfbf60b072b3580212133a4fc45dcb9f403421c":
        raise ValueError("initial student adapter differs")
    rollout.reply_limit = 4096
    parser_type, parser_receipt = parser_class()
    write(args.output / "parser.json", parser_receipt)
    os.environ.update(POSTTRAIN_VERIFIERS_PREINSTALLED="1", SELF_LOCAL_KEY="EMPTY")
    configure_preinstalled_runtime(interpreter=sys.executable)
    agent = Agent(SimpleNamespace(output=args.output, context_limit=32768, reply_limit=4096,
        thinking=True, feedback_mode="minimal", model_family="qwen3.5", tool_parser_mode="schema-bound-v1",
        run_profile="guidance-five", concurrency=40, run_name=args.run_name,
        teacher_feedback="native-feedback", online_rollouts=2, native_repeatability=True))
    agent.args.environment_profile = args.environment_profile
    agent.rollout, agent.tokenizer, agent.model, agent.schema_parser = rollout, tokenizer, model, parser_type
    app = web.Application(client_max_size=16 * 1024 * 1024)
    app.router.add_post("/v1/chat/completions", agent.completions)
    runner = web.AppRunner(app)
    await runner.setup()
    try:
        await web.TCPSite(runner, "127.0.0.1", 18774).start()
        if source_reference is None:
            await agent.collect(0, "baseline_train10", names, temperature=0.)
            await agent.collect(0, "online_train", names, temperature=1.)
        else:
            if (source_reference / "rollout-policy-v0/adapter_model.safetensors").read_bytes() != (
                    args.output / "rollout-policy-v0/adapter_model.safetensors").read_bytes():
                raise ValueError("reference starting adapter differs")
            if serving_identity((source_reference / "engine-config.txt").read_text()) != serving_identity(
                    (args.output / "engine-config.txt").read_text()):
                raise ValueError("reference engine configuration differs")
            if args.cohorts.read_bytes() != (source_reference.parent.parent / "cohorts.json").read_bytes():
                raise ValueError("reference task cohort differs")
            write(args.output / "native-reference.json", {"root": str(source_reference),
                "native_episodes_referenced": 4 * count, "model_adapter_engine_cohort_and_environment_source_verified": True,
                "scope": "Completed fresh controls in the SAME repaired environment; not frozen-profile historical scores"})
        folder = (source_reference or args.output) / "step-0000/online_train"
        episodes = [json.loads(s) for s in (folder / "native/traces.jsonl").read_text().splitlines()]
        calls = (copy.deepcopy(agent.calls) if source_reference is None else
                 [json.loads(s) for s in (folder / "generation-calls.jsonl").read_text().splitlines() if s])
        if len(episodes) != 2 * count or len({e["id"] for e in episodes}) != 2 * count:
            raise ValueError("incomplete fresh stochastic population")
        if args.reference is None:
            rows, excluded = associate(calls, episodes, tokenizer, "native-feedback",
                                       context_limit=32768, thinking=True)
            primary = semantic_selection(rows, calls, episodes, names)
            batch = balanced_batch(rows, primary, names, max(32, count))
        else:
            batch = json.loads((args.reference / "selected-transitions.json").read_text())
            old_cases = json.loads((args.reference / "source-cases.json").read_text())["cases"]
            primary = [c["row"] for c in old_cases]
            rows, excluded = batch, []
            write(args.output / "reference.json", {"root": str(args.reference),
                "referenced_native_episodes": 40, "referenced_proposals": 20,
                "selected_transitions_sha256": digest(batch), "case_rows_sha256": digest(primary),
                "scope": "Exact completed source/population and unchanged control continuations; no new native collection or update"})
        write(args.output / "training-admission.json", {"admitted_rows": len(rows), "excluded": excluded,
            "truncated_sequences": 0, "primary": [{k: r[k] for k in ("task", "episode_id", "generation_index", "native_success")} for r in primary],
            "batch_unique_actions": len({(r["episode_id"], r["generation_index"]) for r in batch}),
            "tasks": names, "selection": "same semantic-primary selection for both potential update arms; no goal-credit claim"})
        write(args.output / "selected-transitions.json", batch)
        by_episode = {e["id"]: e for e in episodes}
        cases, alternatives = [], {}
        for row in batch:
            call = calls[row["generation_index"]]
            episode = by_episode[row["episode_id"]]
            material = by_name[row["task"]]
            # Contract/source proof must apply to the actual newly retained task.
            data = episode["task"]["data"]
            if data["assertions"] != material["raw_native_data"]["assertions"]:
                raise ValueError("native assertion/source mismatch")
            from self_feedback_signals import authority_errors

            errors = authority_errors(material["material"]["goal_manifest"], data["initial_state"], data["prompt"])
            if errors:
                raise ValueError(f"fresh native authority failed {row['task']}: {errors}")
            trace = episode["traces"][0]
            messages = [n["message"] for n in trace["nodes"] if n.get("message")]
            observed = selected_observations(call, messages)
            assertions = trace.get("info", {}).get("automationbench", {}).get("assertions", [])
            supplement = (compose_corpus(row["task"], call["messages"], material["compiled"]["sheet"], assertions, observed)
                          if args.context_version == "corpus-v6" else
                          compose_brief(row["task"], call["messages"], assertions, observed)
                          if args.context_version in ("brief-v2", "brief-v3", "brief-v4") else
                          compose(call["messages"], material["compiled"]["sheet"], assertions, observed))
            teacher_ids = prompt_ids(tokenizer, native_messages(call["messages"]) + [
                {"role": "user", "content": supplement}], call["tools"], thinking=True)
            if len(teacher_ids) + len(row["action_ids"]) > 32768:
                raise ValueError("revised teacher context overlong; no truncation")
            key = f"{row['episode_id']}-{row['generation_index']}"
            alternatives[key] = teacher_ids
            context_dir = args.output / "contexts" / key
            context_dir.mkdir(parents=True)
            (context_dir / "teacher-supplement.txt").write_text(supplement)
            (context_dir / "full-teacher-context.txt").write_text(tokenizer.decode(teacher_ids, skip_special_tokens=False))
            if row in primary:
                cases.append({"row": row, "call": call, "episode": episode,
                              "environment_profile": args.environment_profile,
                              "revised_teacher_prefix_ids": teacher_ids, "context_key": key})
        if len(cases) != count:
            raise ValueError("missing one semantic primary per task")
        write(args.output / "source-cases.json", {"cases": cases})
        write(args.output / "teacher-alternatives.json", alternatives)
        proposals = ([] if args.reference is None else [p for p in json.loads(
            (args.reference / "same-decision-proposals.json").read_text()) if p["mode"] != "revised_teacher"])
        if args.reference is not None and len(proposals) != 2 * count:
            raise ValueError("reference control proposal population incomplete")
        rollout.temperature = 0.
        for mode in (["student", "original_teacher", "revised_teacher"] if args.reference is None else ["revised_teacher"]):
            async def generate(case, mode=mode):
                row, call = case["row"], case["call"]
                prefix = row["student_prefix_ids"] if mode == "student" else row["teacher_prefix_ids"] if mode == "original_teacher" else case["revised_teacher_prefix_ids"]
                ids, complete, evidence = await rollout.generate(prefix, max_tokens=4096)
                raw = tokenizer.decode(ids, skip_special_tokens=True)
                result = {"mode": mode, "task": row["task"], "episode_id": row["episode_id"],
                          "generation_index": row["generation_index"], "prefix_ids": prefix,
                          "action_ids": ids, "raw": raw, "complete": complete, **evidence}
                if complete:
                    try:
                        result["message"] = parsed_message(raw, tokenizer, call["tools"], parser_type,
                                                           native_messages(call["messages"]))
                        result["native_replay"] = replay(case, result["message"])
                    except Exception as error:
                        result["native_error"] = {"type": type(error).__name__, "message": str(error)}
                return result
            proposals.extend(await asyncio.gather(*(generate(c) for c in cases)))
            write(args.output / "same-decision-proposals.json", proposals)
            print(json.dumps({"stage": "teacher_proposals", "mode": mode, "count": count}), flush=True)
        write(args.output / "receipt.json", {"runtime_completed": True, "semantic_qualified": None,
            "environment_profile": args.environment_profile,
            "optimizer_updates": 0, "native_episodes": 4 * count, "same_decision_generations": 3 * count,
            "native_episodes_referenced": 4 * count if source_reference is not None else 0,
            "generations_referenced": 2 * count if args.reference is not None else 0,
            "generations_new": count if args.reference is not None else 3 * count,
            "training_restarted": False, "tasks": names,
            "qualification": "Independent source/teacher review required before separate paired gradient/update diagnostic"})
    finally:
        await runner.cleanup()
        await rollout.aclose()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model")
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--cohorts", type=Path, required=True)
    p.add_argument("--materials", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--run-name", default="progress-teacher-ten")
    p.add_argument("--context-version", choices=["compact-v1", "brief-v2", "brief-v3", "brief-v4", "corpus-v6"], default="compact-v1")
    p.add_argument("--expected-task-count", type=int, choices=[10, 124], default=10)
    p.add_argument("--reference", type=Path)
    p.add_argument("--native-reference", type=Path)
    p.add_argument("--environment-profile", choices=[FROZEN, REPAIRED, ACCESS, DISCOVERY], default=FROZEN)
    p.add_argument("--cpu-only", action="store_true")
    args = p.parse_args()
    if args.reference is not None and args.native_reference is not None:
        raise ValueError("choose one reference mode")
    validate_reference(args.environment_profile, args.reference)
    args.output.mkdir(parents=True, exist_ok=False)
    if args.cpu_only:
        cpu(args)
    else:
        asyncio.run(run(args))
