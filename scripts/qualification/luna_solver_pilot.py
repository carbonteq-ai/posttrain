"""Two-task native solver pilot: matched control and compact reference hints."""

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from generate_luna_ssr_pilot import parsed_json
from luna_solver_prompts import (
    check_hint_literals,
    guided_messages,
    text_content,
    user_key,
)
from luna_solver_schema import instruction_schema
from luna_solver_stages import _sensitive_values, prepare_instructions


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


async def main(args):
    import httpx
    import torch
    from aiohttp import web
    from automationbench_v1.contracts.loader import load_task_contract
    from automationbench_v1.limited_tools import selected_tool_definitions
    from automationbench_v1.taskset import AutomationBenchConfig, AutomationBenchTaskConfig, AutomationBenchTaskset
    from live import environment_config
    from posttrain.common.cuda import activate_cuda_toolkit
    from self_feedback_signals import authority_errors
    from verifiers.v1.cli.eval.runner import run_eval
    from verifiers.v1.configs.cli.eval import EvalConfig
    from verifiers_preinstalled import configure_preinstalled_runtime

    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    activate_cuda_toolkit(torch)
    os.environ["POSTTRAIN_VERIFIERS_PREINSTALLED"] = "1"
    os.environ["SOLVER_LOCAL_KEY"] = "EMPTY"
    configure_preinstalled_runtime(interpreter=sys.executable)
    sources = json.loads(args.inputs.read_text())["tasks"]
    names = [row["task"] for row in sources]
    native = AutomationBenchTaskset(
        AutomationBenchConfig(
            domains=sorted({name.split(".", 1)[0] for name in names}),
            task_names=names,
            task=AutomationBenchTaskConfig(
                toolset="limited_zapier",
                turn_budget=12,
                capture_actions=True,
                deterministic_world=True,
            ),
        )
    ).load()
    source_by_name = {row["task"]: row for row in sources}
    materials = {}
    for task in native:
        data = task.data.model_dump(mode="json")
        source = source_by_name[task.key]
        users = [text_content(m["content"]) for m in data["prompt"] if m["role"] == "user"]
        if users != source["user_messages"]:
            raise ValueError("source/native user task mismatch")
        contract = load_task_contract(task.key).model_dump(mode="json", exclude_none=True)
        errors = authority_errors(contract, data["initial_state"], data["prompt"])
        if errors:
            raise ValueError(f"goal manifest does not bind native task: {task.key}: {errors}")
        materials[user_key(data["prompt"])] = {
            "task": task.key,
            "user_request": "\n\n".join(users),
            "tool_definitions": selected_tool_definitions(data["zapier_tools"]),
            "reference_actions": source["reference_actions"],
            "source": source["luna_source"],
            "native_data": data,
            "goal_manifest": contract,
        }
    write(args.output / "materials.json", materials)
    if args.preflight_only:
        write(
            args.output / "preflight-receipt.json",
            {
                "qualified": True,
                "tasks": names,
                "model_generations": 0,
                "source_manifest_sha256": hashlib.sha256(
                    (args.inputs.parent / "source-sha256.txt").read_bytes()
                ).hexdigest(),
            },
        )
        print("CPU_NATIVE_PREFLIGHT_OK", flush=True)
        return
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(args.model),
        "--served-model-name",
        "local-solver",
        "--host",
        "127.0.0.1",
        "--port",
        "18910",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "32768",
        "--max-num-seqs",
        "8",
        "--max-num-batched-tokens",
        "8192",
        "--kv-cache-memory-bytes",
        str(16 * 1024**3),
        "--block-size",
        "128",
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--limit-mm-per-prompt",
        '{"image":0,"video":0,"audio":0}',
        "--skip-mm-profiling",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_xml",
        "--reasoning-parser",
        "qwen3",
        "--generation-config",
        "vllm",
        "--seed",
        "17",
    ]
    write(args.output / "server-command.json", command)
    log = (args.output / "vllm-server.log").open("w")
    server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    runner = None
    client = httpx.AsyncClient(timeout=600)
    hints = {}
    proposal_rows = ([json.loads(line) for line in args.proposal_stages.read_text().splitlines() if line.strip()]
                     if args.proposal_stages else [])
    state = {"arm": "control", "calls": 0}
    try:
        for _ in range(480):
            if server.poll() is not None:
                raise RuntimeError("vLLM exited before readiness; inspect server log")
            try:
                response = await client.get("http://127.0.0.1:18910/health", timeout=1)
                if response.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.5)
        else:
            raise TimeoutError("vLLM startup exceeded240seconds")

        async def summarize(stage, system, data):
            if stage in ("requirements", "workflow", "assembly") and proposal_rows:
                task = next(m["task"] for m in materials.values() if m["user_request"] == data["public_request"])
                matches = [r for r in proposal_rows if r["task"] == task and r["stage"] == stage]
                if len(matches) != 1:
                    raise ValueError("missing or ambiguous saved 4B proposal")
                with (args.output / "reused-proposals.jsonl").open("a") as stream:
                    stream.write(json.dumps({"task": task, "stage": stage,
                                             "source_sha256": hashlib.sha256(args.proposal_stages.read_bytes()).hexdigest(),
                                             "source_model": "4b", "output": matches[0]["output"]}) + "\n")
                return json.loads(json.dumps(matches[0]["output"]))
            body = {
                "model": "local-solver",
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(data)}],
                "temperature": 0,
                "max_tokens": 2048,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            if args.instruction_variant.startswith(("structured", "scope-")):
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": stage, "strict": True, "schema": instruction_schema(stage, data)},
                }
            response = await client.post("http://127.0.0.1:18910/v1/chat/completions", json=body)
            response.raise_for_status()
            value = response.json()
            with (args.output / "hint-generations.jsonl").open("a") as stream:
                stream.write(json.dumps({"stage": stage, "request": body, "response": value}) + "\n")
            choice = value["choices"][0]
            if choice["finish_reason"] == "length":
                raise ValueError("hint generation incomplete; no silent clipping")
            return parsed_json(choice["message"]["content"])

        async def completions(request):
            clean = await request.json()
            key = user_key(clean["messages"])
            if key not in materials:
                raise ValueError("unknown native task request")
            body = {
                **clean,
                "model": "local-solver",
                "max_tokens": 4096,
                "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": True},
            }
            body.pop("max_completion_tokens", None)
            if state["arm"] == "guided":
                body["messages"] = guided_messages(clean["messages"], hints[key])
                if args.instruction_variant.endswith("v5"):
                    body["messages"][0]["content"] = (
                        body["messages"][0]["content"].replace(
                            "\n\nOperational checklist:\n", "\n\nTask execution requirements:\n"
                        )
                        + "\nExplain decisions using the user request and observed tool results, rather than repeating the execution plan."
                    )
                if "natural" in args.instruction_variant:
                    body["messages"][0]["content"] = body["messages"][0]["content"].replace(
                        "Task execution requirements:", "Execution guidance:"
                    ) + (
                        "\nGive concise decision reasoning grounded in the public task and the tool results "
                        "already observed. Explain why the chosen action follows from those facts. "
                        "Do not cite checklists, check IDs, step labels or goal/harm annotations as evidence. "
                        "Guidance is a fallible interpretation: an invented precondition must not override "
                        "an explicit action requested by the user. Do not invent identities, domains or results."
                    )
            state["calls"] += 1
            response = await client.post("http://127.0.0.1:18910/v1/chat/completions", json=body)
            response.raise_for_status()
            value = response.json()
            with (args.output / "solver-calls.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "arm": state["arm"],
                            "task": materials[key]["task"],
                            "clean_student_request": clean,
                            "actual_solver_request": body,
                            "response": value,
                        }
                    )
                    + "\n"
                )
            print(
                json.dumps(
                    {
                        "stage": "native_generation",
                        "arm": state["arm"],
                        "task": materials[key]["task"],
                        "finish": value["choices"][0]["finish_reason"],
                    }
                ),
                flush=True,
            )
            return web.json_response(value)

        app = web.Application(client_max_size=16 * 1024**2)
        app.router.add_post("/v1/chat/completions", completions)
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", 18911).start()
        outcomes = {}
        instruction_errors = {}
        review_rejected_tasks = set()
        for arm in (("guided",) if args.guided_only or args.preparation_only else ("control", "guided")):
            arm_names = names
            if arm == "guided":

                async def prepare_task(key, material):

                    def record(stage, value, task_name=material["task"]):
                        if stage == "review_rejected_diagnostic_only":
                            review_rejected_tasks.add(task_name)
                        with (args.output / "instruction-stages.jsonl").open("a") as stream:
                            stream.write(json.dumps({"task": task_name, "stage": stage, **value}) + "\n")

                    try:
                        hints[key] = await prepare_instructions(material, summarize, record, args.instruction_variant)
                    except (ValueError, KeyError, TypeError, IndexError) as error:
                        instruction_errors[material["task"]] = {"type": type(error).__name__, "message": str(error)}
                        write(args.output / "instruction-errors.json", instruction_errors)

                if args.prepared_guidance:
                    hints.update(json.loads(args.prepared_guidance.read_text()))
                    if set(hints) != set(materials):
                        raise ValueError("prepared guidance task coverage differs from solver population")
                    for key, lines in hints.items():
                        material = materials[key]
                        check_hint_literals(lines, material["user_request"] + json.dumps(material["tool_definitions"]),
                                            list(_sensitive_values({"manifest": material["goal_manifest"],
                                                                    "actions": material["reference_actions"]})))
                    write(args.output / "prepared-guidance-source.json", {
                        "sha256": hashlib.sha256(args.prepared_guidance.read_bytes()).hexdigest(),
                        "path": str(args.prepared_guidance), "author": "saved4B proposals,27B critic and bounded repair"})
                else:
                    await asyncio.gather(*(prepare_task(key, material) for key, material in materials.items()))
                write(args.output / "compressed-hints.json", hints)
                arm_names = [materials[key]["task"] for key in hints]
                if not arm_names or args.preparation_only:
                    outcomes[arm] = []
                    write(args.output / "outcomes.json", outcomes)
                    continue
            state["arm"] = arm
            env = environment_config(12, "limited_zapier")
            env["taskset"]["task"].update(
                turn_budget=12, capture_actions=True, deterministic_world=True, manifest_assessments=True
            )
            env["agent"]["max_output_tokens"] = 16384
            config = EvalConfig.model_validate(
                {
                    "env": env,
                    "model": "local-solver",
                    "client": {
                        "type": "eval",
                        "base_url": "http://127.0.0.1:18911/v1",
                        "api_key_var": "SOLVER_LOCAL_KEY",
                    },
                    "sampling": {"temperature": 0, "max_tokens": 4096},
                    "num_tasks": len(arm_names),
                    "task_keys": arm_names,
                    "num_rollouts": 1,
                    "max_concurrent": 2,
                    "shuffle": False,
                    "output_dir": args.output / arm / "native",
                    "run": {"name": args.run_name + "-" + arm, "dir": "."},
                    "push": False,
                    "rich": None,
                    "serve": None,
                }
            )
            episodes = await run_eval(config)
            rows = [episode.model_dump(mode="json") for episode in episodes]
            outcomes[arm] = [
                {
                    "id": episode["id"],
                    "task": episode["task"]["data"]["name"],
                    "errors": episode.get("errors"),
                    "assessment_errors": episode.get("assessment_errors"),
                    "traces": [
                        {
                            "metrics": trace["metrics"],
                            "rewards": trace["rewards"],
                            "errors": trace.get("errors"),
                            "stop_condition": trace.get("stop_condition"),
                        }
                        for trace in episode["traces"]
                    ],
                }
                for episode in rows
            ]
            write(args.output / "outcomes.json", outcomes)
        write(
            args.output / "receipt.json",
            {
                "completed": True,
                "instruction_variant": args.instruction_variant,
                "instruction_errors": instruction_errors,
                "all_instructions_admitted": not instruction_errors and not review_rejected_tasks,
                "review_rejected_diagnostic_tasks": sorted(review_rejected_tasks),
                "native_diagnostic_eligible_tasks": [materials[key]["task"] for key in hints],
                "outcomes": outcomes,
                "elapsed_seconds": time.monotonic() - started,
                "optimizer_updates": 0,
                "tasks": names,
                "maximum_native_episodes": 0 if args.preparation_only else (1 if args.guided_only else 2) * len(names),
                "guided_only": args.guided_only,
                "preparation_only": args.preparation_only,
                "saved_4b_proposals": bool(args.proposal_stages),
                "actor_mode": "fresh native tool execution, thinking, greedy",
                "instruction_pipeline": "4b-proposals-27b-critic-policy-retrieval-v1" if args.proposal_stages or args.prepared_guidance else "same-model-requirements-workflow-assembly-review-v1",
                "instruction_model": str(args.model),
                "instruction_reply_limit": 2048,
                "guided_context": "reviewed requirement-linked procedure; raw source/manifest withheld from actor",
                "context_limit": 32768,
                "reply_limit": 4096,
                "episode_output_limit": 16384,
                "turn_limit": 12,
                "sft_qualified": False,
                "scope": "Candidate generation; semantic/repetition/goal-harm inspection pending",
            },
        )
    finally:
        if runner:
            await runner.cleanup()
        await client.aclose()
        server.terminate()
        try:
            await asyncio.to_thread(server.wait, timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--guided-only", action="store_true")
    parser.add_argument("--preparation-only", action="store_true")
    parser.add_argument("--proposal-stages", type=Path)
    parser.add_argument("--prepared-guidance", type=Path)
    parser.add_argument("--instruction-variant", default="raw")
    asyncio.run(main(parser.parse_args()))
