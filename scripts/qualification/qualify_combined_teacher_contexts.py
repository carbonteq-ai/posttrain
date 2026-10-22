"""Matched one-update SDPO context diagnosis; never resumes a training campaign."""

import argparse
import asyncio
import copy
import gc
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

FAILURES = ["marketing.content_performance", "marketing.competitor_product_launch",
            "marketing.email_campaign_analytics", "sales.soft_matching"]
RETENTION = ["simple.email_sf_lead_rating_hot", "simple.airtable_find_update",
             "simple.trello_catering_contract", "simple.gmail_reply_complaint"]


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def json_value(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def retained_history_arguments(case, *, include_capture_matches=False):
    """Bind historical calls to actual captured execution, never guess aliases.

    The native MCP boundary may ignore an undeclared argument and supply the
    declared default. Its actual execution is different from raw model JSON.
    Equal full observed responses plus identical candidate arguments are
    required; conflicting matches are rejected. This is replay input binding,
    not proof of general action-credit alignment or a model-correctness claim.
    """
    def decoded(value):
        for _ in range(3):
            if not isinstance(value, str):
                break
            try:
                value = json.loads(value)
            except ValueError:
                break
        return value

    messages = case["call"]["messages"]
    expected = {m["tool_call_id"]: decoded(m["content"]) for m in messages if m["role"] == "tool"}
    events = case["episode"]["traces"][0].get("info", {}).get("automationbench_capture", {}).get("events", [])
    used, bound, audit = set(), {}, []
    for message in messages:
        for tool in message.get("tool_calls") or []:
            function = tool["function"]
            original = json_value(function["arguments"])
            matches = [(i, event) for i, event in enumerate(events) if i not in used
                       and event.get("tool_name") == function["name"] and event.get("status") == "returned"
                       and decoded(event.get("result_json")) == expected[tool["id"]]]
            compatible = [(i, e) for i, e in matches if all(
                k in json_value(e["arguments_json"]) and json_value(e["arguments_json"])[k] == v
                for k, v in original.items())]
            if compatible:
                matches = compatible
            candidates = {sha(json_value(e["arguments_json"])) for _, e in matches}
            if len(candidates) > 1:
                raise ValueError("ambiguous retained execution arguments for historical response")
            if matches:
                index, event = matches[0]
                used.add(index)
                arguments = json_value(event["arguments_json"])
            else:
                arguments, event = original, None
            bound[tool["id"]] = arguments
            if original != arguments or include_capture_matches:
                audit.append({"tool_call_id": tool["id"], "tool": function["name"],
                              "model_arguments": original, "actual_execution_arguments": arguments,
                              "ignored_model_fields": sorted(set(original) - set(arguments)),
                              "native_default_fields": sorted(set(arguments) - set(original)),
                              "retained_occurrence_index": event.get("occurrence_index") if event else None,
                              "capture_result_matched": event is not None,
                              "scope": "Native admission changed the effective call; not a declared semantic alias or a corrected model action."})
    return bound, audit


def arm_batch(original_rows, alternatives, mode):
    if mode not in ("original_teacher", "combined_teacher"):
        raise ValueError("unknown context arm")
    batch = copy.deepcopy(original_rows)
    changed = 0
    for row in batch:
        key = (row["episode_id"], row["generation_index"])
        if mode == "combined_teacher" and key in alternatives:
            row["teacher_prefix_ids"] = alternatives[key]
            changed += 1
    if changed != (4 if mode == "combined_teacher" else 0):
        raise ValueError("teacher treatment must change exactly four selected rows")
    return batch, changed


def bind_generated_threads(initial, messages):
    """Rehydrate only constructor-generated thread IDs missing in raw fixture.

    Stable message identity and every other observed field are still checked by
    exact replay. An explicitly seeded ID cannot be replaced.
    """
    observed = {}
    for message in messages:
        if message.get("role") != "tool":
            continue
        value = json_value(message["content"])
        for item in value.get("messages", []) if isinstance(value, dict) else []:
            identity, thread = item.get("id"), item.get("thread_id")
            if identity and thread:
                if identity in observed and observed[identity] != thread:
                    raise ValueError("conflicting observed thread identity")
                observed[identity] = thread
    bindings = []
    for item in initial.get("gmail", {}).get("messages", []):
        identity = item.get("id")
        if identity in observed:
            if item.get("thread_id") is not None and item["thread_id"] != observed[identity]:
                raise ValueError("explicit source thread identity differs")
            if item.get("thread_id") is None:
                item["thread_id"] = observed[identity]
                bindings.append({"message_id": identity, "thread_id": observed[identity],
                                 "reason": "raw fixture omitted constructor-generated field; bound from retained real observation"})
    return bindings


def native_messages(messages):
    """Restore the native wire reasoning alias before Qwen serialization."""
    result = copy.deepcopy(messages)
    for message in result:
        if message.get("role") == "assistant" and message.get("reasoning") is not None:
            if message.get("reasoning_content") not in (None, message["reasoning"]):
                raise ValueError("conflicting native reasoning aliases")
            message["reasoning_content"] = message["reasoning"]
    return result


def serving_identity(text):
    """Ignore only vLLM's per-process timestamp ID; all settings stay exact."""
    result, count = re.subn(r"\binstance_id='[0-9]+'", "instance_id='<process-id>'", text)
    if count != 1:
        raise ValueError("unexpected vLLM instance identifier shape")
    return result


def native_proposal_arguments(owner, toolset, name, arguments):
    """Use the actual pinned MCP SDK coercion/defaults, not Python truthiness."""
    from automationbench_v1.limited_tools import _ZAPIER_TOOLS
    from mcp.server.mcpserver.utilities.func_metadata import func_metadata

    wrapper = toolset._tool_wrapper(owner, name, _ZAPIER_TOOLS[name])
    return func_metadata(wrapper).validate_arguments(arguments)


def replay(case, proposed=None, *, exact_captured_history=False, diagnostic_differences=False, native_setup=False):
    """Replay native history and one decision through pinned invoke/capture."""
    import inspect

    import automationbench_v1.limited_tools as limited
    from automationbench_v1.capture import canonical_json, collect_local_evidence
    from automationbench_v1.simulation import simulated_setup, world_clock_base
    from automationbench_v1.taskset import score_world
    from automationbench_v1.tools import AutomationBenchState
    from automationbench_v1.world_codec import dump_world, load_world
    from progress_environment_profile import FROZEN, is_repaired, verified_rejection

    data = case["episode"]["task"]["data"]
    initial = copy.deepcopy(data["initial_state"])
    setup_initial = copy.deepcopy(case.get('setup_initial_state', initial)) if native_setup else initial
    thread_bindings = [] if native_setup else bind_generated_threads(initial, case["call"]["messages"])
    clock = world_clock_base(setup_initial)
    with simulated_setup(setup_initial, clock):
        loaded = load_world(setup_initial)
    if native_setup:
        from automationbench_v1.taskset import _allowed_services

        # Match the pinned native Task.setup, including fields used in the
        # simulator's whole-world deterministic seed. No volatile-field masking.
        loaded.meta.current_time = clock
        loaded.meta.allowed_services = _allowed_services(initial, data['assertions'], data['zapier_tools'])
        world = loaded.model_dump(mode='json')
        actual_digest = hashlib.sha256(canonical_json(world).encode()).hexdigest()
        expected_digest = case['episode']['traces'][0]['info']['automationbench_capture']['initial_digest']
        if actual_digest != expected_digest:
            raise ValueError(f'Native initial-world digest differs: {actual_digest} != {expected_digest}')
    else:
        world = dump_world(loaded)
    state = AutomationBenchState.model_construct(world=world, initial_state=initial,
        capture_actions=True, deterministic_world=True, world_clock=clock.isoformat())
    owner = SimpleNamespace(state=state, config=SimpleNamespace(allowed_tools=data["zapier_tools"]))
    toolset = next(v for v in vars(limited).values() if isinstance(v, type) and "invoke" in v.__dict__)
    expected = {m["tool_call_id"]: json_value(m["content"]) for m in case["call"]["messages"]
                if m["role"] == "tool"}
    history, decisions = [], []
    execution_arguments, argument_bindings = retained_history_arguments(case, include_capture_matches=exact_captured_history)
    captured_ids = {b['tool_call_id'] for b in argument_bindings if b.get('capture_result_matched')}
    with collect_local_evidence() as captured:
        for message in case["call"]["messages"]:
            for tool in message.get("tool_calls") or []:
                function = tool["function"]
                if is_repaired(case.get("environment_profile", FROZEN)):
                    declared = set(inspect.signature(limited._ZAPIER_TOOLS[function["name"]]).parameters) - {"world"}
                    if verified_rejection(function["name"], json_value(function["arguments"]),
                                          declared, expected[tool["id"]]):
                        history.append({"tool": function["name"], "result": expected[tool["id"]],
                                        "strict_admission_rejected": True, "executed": False})
                        continue
                if exact_captured_history:
                    if tool['id'] not in captured_ids:
                        raise ValueError('Historical action lacks exact retained capture; no guessed replay')
                    # Captured kwargs are the actual post-boundary execution.
                    # Validating them again can inject extra default keys and
                    # change the simulator's world+arguments deterministic seed.
                    validated_history = execution_arguments[tool['id']]
                else:
                    validated_history = native_proposal_arguments(owner, toolset, function["name"], execution_arguments[tool["id"]])
                before_native_digest = hashlib.sha256(canonical_json(owner.state.world).encode()).hexdigest()
                result = json_value(toolset.invoke(owner, function["name"], **validated_history))
                if result != expected[tool["id"]]:
                    if diagnostic_differences:
                        raise ValueError(json.dumps({'task':case['row']['task'],'tool':function['name'],
                            'expected':expected[tool['id']],'actual':result,'actual_before_digest':before_native_digest,
                            'world_clock':owner.state.world_clock,'world_revision':owner.state.world_revision,
                            'actual_execution_arguments':validated_history}))
                    raise ValueError(f"same-state historical replay differs: {case['row']['task']} {function['name']}")
                history.append({"tool": function["name"], "result": result})
        decision_state = sha(owner.state.world)
        message = case["call"]["message"] if proposed is None else proposed
        for tool in message.get("tool_calls") or []:
            function = tool["function"]
            if function["name"] not in data["zapier_tools"]:
                raise ValueError("proposed tool is not allowed")
            if is_repaired(case.get("environment_profile", FROZEN)):
                arguments = json_value(function["arguments"])
                declared = set(inspect.signature(limited._ZAPIER_TOOLS[function["name"]]).parameters) - {"world"}
                extra = set(arguments) - declared
                if extra:
                    error = (f"Unsupported arguments for {function['name']}: {', '.join(sorted(extra))}. "
                             "Use the declared tool parameter names. No action was executed.")
                    decisions.append({"tool": function["name"], "arguments": arguments,
                                      "result": error, "raised": None,
                                      "strict_admission_rejected": True, "executed": False})
                    continue
            try:
                from pydantic_core import to_jsonable_python

                validated = native_proposal_arguments(owner, toolset, function["name"], json_value(function["arguments"]))
                result = json_value(toolset.invoke(owner, function["name"], **validated))
                decisions.append({"tool": function["name"], "arguments": json_value(function["arguments"]),
                                  "native_validated_arguments": to_jsonable_python(validated),
                                  "result": result, "raised": None})
            except Exception as error:
                decisions.append({"tool": function["name"], "arguments": json_value(function["arguments"]),
                                  "raised": type(error).__name__, "error": str(error)})
        envelopes = copy.deepcopy(captured)
    scored = score_world(world=copy.deepcopy(owner.state.world), initial_state=copy.deepcopy(initial),
                         assertions=tuple(copy.deepcopy(data["assertions"])))
    return {"task": data["name"], "episode_id": case["row"]["episode_id"],
            "constructor_generated_thread_bindings": thread_bindings,
            "historical_native_argument_bindings": argument_bindings,
            "raw_initial_state_sha256": sha(data["initial_state"]),
            "decision_state_sha256": decision_state, "historical_calls": len(history),
            "decisions": decisions, "native_capture": envelopes,
            "scoring_after_one_decision": {k: getattr(scored, k) for k in [
                "partial_credit", "task_completed_correctly", "assertion_results"]},
            "scope": "exact-state native next-decision replay, not a complete new episode"}


def replay_source_bound(case, proposed=None):
    """Admit only a retained source form reproducing the full native world digest."""
    forms=[('retained_native_task_data',case['episode']['task']['data']['initial_state'])]
    if 'source_fixture_initial_state' in case:
        forms.append(('original_source_fixture',case['source_fixture_initial_state']))
    failures=[]
    for name,initial in forms:
        try:
            result=replay({**case,'setup_initial_state':initial},proposed,
                          exact_captured_history=True,native_setup=True)
            result['native_setup_source_form']=name
            result['native_initial_digest_verified']=True
            return result
        except ValueError as error:
            if not str(error).startswith('Native initial-world digest differs:'):
                raise
            failures.append({'source_form':name,'error':str(error)})
    raise ValueError(json.dumps({'documented_source_forms_failed':failures}))


def preflight(args):
    source = json.loads(args.cases.read_text())
    rows = json.loads(args.rows.read_text())
    if len(rows) != 32 or any(r["policy_version"] != 0 or not r["complete"] for r in rows):
        raise ValueError("requires exact complete historical batch32 at initial policy0")
    if [c["row"]["task"] for c in source["cases"]] != FAILURES:
        raise ValueError("four-case order/identity changed")
    audit = []
    for case in source["cases"]:
        row = case["row"]
        if rows.count(row) != 1 or row["native_success"] != 0:
            raise ValueError("case is not the unique real failed selected transition")
        if case["call"]["prefix_ids"] != row["student_prefix_ids"]:
            raise ValueError("native call/student token mismatch")
        audit.append(replay(case))
    write(args.output / "native-preflight.json", {"qualified": True, "cases": audit,
          "optimizer_updates": 0, "model_calls": 0})
    return source, rows


def parsed_message(raw, tokenizer, tools, parser_type, messages):
    from sdpo_prompting import split_reasoning
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    native = ChatCompletionRequest(model="local-self", messages=messages, tools=tools,
                                   chat_template_kwargs={"enable_thinking": True})
    reasoning, content, complete = split_reasoning(tokenizer, raw, native)
    if not complete:
        raise ValueError("incomplete native thinking/tool channels")
    parsed = parser_type(tokenizer, tools=native.tools).extract_tool_calls(content, native)
    message = {"role": "assistant", "content": parsed.content if parsed else content,
               "reasoning_content": reasoning}
    if parsed and parsed.tool_calls:
        message["tool_calls"] = [c.model_dump() for c in parsed.tool_calls]
    return message


def prefix_inputs(args, cases, tokenizer):
    from sdpo_prompting import prompt_ids

    alternatives = {}
    for case in cases:
        row = case["row"]
        name = row["task"]
        messages = native_messages(case["call"]["messages"])
        if prompt_ids(tokenizer, messages, case["call"]["tools"], thinking=True) != row["student_prefix_ids"]:
            raise ValueError(f"original student serialization differs: {name}")
        old = messages + [{"role": "user", "content": row["teacher_feedback"]}]
        if prompt_ids(tokenizer, old, case["call"]["tools"], thinking=True) != row["teacher_prefix_ids"]:
            raise ValueError(f"original teacher serialization differs: {name}")
        supplement = (args.contexts / name / "teacher-supplement.txt").read_text()
        guided = messages + [{"role": "user", "content": supplement}]
        ids = prompt_ids(tokenizer, guided, case["call"]["tools"], thinking=True)
        if tokenizer.decode(ids, skip_special_tokens=False) != (args.contexts / name / "full-teacher-context.txt").read_text():
            raise ValueError(f"native candidate serialization differs from reviewed artifact: {name}")
        if len(ids) + len(row["action_ids"]) > 32768:
            raise ValueError("overlong candidate; no truncation")
        alternatives[(row["episode_id"], row["generation_index"])] = ids
    write(args.output / "prefix-gate.json", {"qualified": True, "cases": len(alternatives), "no_truncation": True,
          "reasoning_wire_alias": "reasoning -> reasoning_content, exact original IDs required"})
    return alternatives


async def run(args):
    import torch
    from aiohttp import web
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file
    from sdpo_agent_ablation import Agent, SamplingRollout, loss_for, score_sampled
    from sdpo_schema_parser import parser_class
    from self_failure_diagnosis import logits
    from transformers import AutoModelForImageTextToText, AutoTokenizer, set_seed
    from verifiers_preinstalled import configure_preinstalled_runtime

    source, original_rows = preflight(args)
    cases = source["cases"]
    set_seed(18)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer._sdpo_model_family = "qwen3.5"
    alternatives = prefix_inputs(args, cases, tokenizer)
    model = get_peft_model(AutoModelForImageTextToText.from_pretrained(args.model,
        dtype=torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation="sdpa", local_files_only=True),
        LoraConfig(r=16, lora_alpha=32, lora_dropout=0., target_modules="all-linear",
                   exclude_modules=r".*visual.*", task_type="CAUSAL_LM"))
    initial = args.output / "initial-student"
    model.save_pretrained(initial, selected_adapters=["default"])
    model.load_adapter(initial, adapter_name="sdpo_teacher", is_trainable=False)
    model.set_adapter("default")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    if any(isinstance(m, torch.nn.Dropout) and m.p != 0 for m in model.modules()):
        raise ValueError("paired gradient probe requires zero dropout")
    snapshots = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if ".default." in n and "lora_" in n}
    params = [p for n, p in model.named_parameters() if n in snapshots and p.requires_grad]
    rollout = await SamplingRollout.create(args.model, model, tokenizer, args.output,
        65536, 40, 16, speculative_config=None, prefix_caching=False, chunked_prefill=False)
    if rollout.refreshes[0]["weights_sha256"] != "e71dae81f1b4723c83aacd130bfbf60b072b3580212133a4fc45dcb9f403421c":
        raise ValueError("initial adapter differs from original experiment03")
    rollout.reply_limit = 4096
    parser_type, parser_receipt = parser_class()
    write(args.output / "parser.json", parser_receipt)
    os.environ.update(POSTTRAIN_VERIFIERS_PREINSTALLED="1", SELF_LOCAL_KEY="EMPTY")
    configure_preinstalled_runtime(interpreter=sys.executable)
    agent = Agent(SimpleNamespace(output=args.output, context_limit=32768, reply_limit=4096,
        thinking=True, feedback_mode="minimal", model_family="qwen3.5", tool_parser_mode="schema-bound-v1",
        run_profile="guidance-five", concurrency=40, run_name=args.output.parent.name,
        teacher_feedback="native-feedback", online_rollouts=2, native_repeatability=True))
    agent.rollout, agent.tokenizer, agent.model = rollout, tokenizer, model
    agent.schema_parser = parser_type
    app = web.Application(client_max_size=16 * 1024 * 1024)
    app.router.add_post("/v1/chat/completions", agent.completions)
    runner = web.AppRunner(app)
    await runner.setup()
    try:
        await web.TCPSite(runner, "127.0.0.1", 18774).start()
        proposals = []
        referenced = args.reference is not None
        if referenced:
            reference_capsule = args.reference.parent.parent
            for filename in ["selected-transitions.json", "source-cases.json"]:
                current = args.rows if filename == "selected-transitions.json" else args.cases
                if current.read_bytes() != (reference_capsule / filename).read_bytes():
                    raise ValueError("reference input source changed")
            for case in cases:
                task = case["row"]["task"]
                if (args.contexts / task / "full-teacher-context.txt").read_bytes() != (
                        reference_capsule / "contexts" / task / "full-teacher-context.txt").read_bytes():
                    raise ValueError("reference teacher context changed")
            if (args.reference / "rollout-policy-v0/adapter_model.safetensors").read_bytes() != (
                    args.output / "rollout-policy-v0/adapter_model.safetensors").read_bytes():
                raise ValueError("reference initial adapter differs")
            if serving_identity((args.reference / "engine-config.txt").read_text()) != serving_identity(
                    (args.output / "engine-config.txt").read_text()):
                raise ValueError("reference serving configuration differs")
            proposals = json.loads((args.reference / "same-decision-proposals.json").read_text())
            if len(proposals) != 12 or not all(p["complete"] and "native_replay" in p for p in proposals):
                raise ValueError("reference proposal population incomplete")
            for case in cases:
                row = case["row"]
                for mode in ["student", "original_teacher", "combined_teacher"]:
                    proposal = next(p for p in proposals if p["task"] == row["task"] and p["mode"] == mode)
                    expected = (row["student_prefix_ids"] if mode == "student" else row["teacher_prefix_ids"]
                                if mode == "original_teacher" else alternatives[(row["episode_id"], row["generation_index"])])
                    if proposal["prefix_ids"] != expected or proposal["policy_version"] != 0:
                        raise ValueError("reference proposal input/policy differs")
            shutil.copy2(args.reference / "same-decision-proposals.json", args.output / "same-decision-proposals.json")
            for kind, names in [("failures", FAILURES), ("retention", RETENTION)]:
                folder = args.reference / "step-0000" / ("baseline_" + kind)
                episodes = [json.loads(s) for s in (folder / "native/traces.jsonl").read_text().splitlines()]
                counts = {n: sum(e["task"]["data"]["name"] == n for e in episodes) for n in names}
                summary = json.loads((folder / "outcomes.json").read_text())
                if len(episodes) != 8 or set(counts.values()) != {2} or summary["errors"]:
                    raise ValueError("reference native population incomplete")
                shutil.copytree(folder, args.output / "step-0000" / ("baseline_" + kind))
            write(args.output / "referenced-baseline.json", {"source": str(args.reference),
                  "same_decision_generations_referenced": 12, "native_episodes_referenced": 16,
                  "scope": "Complete unchanged baseline populations/proposals, verified source/initial-adapter/prefix/engine identity; not new generated episodes or resumed updates"})
        for mode in ([] if referenced else ["student", "original_teacher", "combined_teacher"]):
            async def generate(case, mode=mode):
                row = case["row"]
                key = (row["episode_id"], row["generation_index"])
                prefix = (row["student_prefix_ids"] if mode == "student" else
                          row["teacher_prefix_ids"] if mode == "original_teacher" else alternatives[key])
                ids, complete, evidence = await rollout.generate(prefix, max_tokens=4096)
                raw = tokenizer.decode(ids, skip_special_tokens=True)
                result = {"mode": mode, "task": row["task"], "episode_id": row["episode_id"],
                          "prefix_ids": prefix, "action_ids": ids, "raw": raw, "complete": complete, **evidence}
                if complete:
                    try:
                        result["message"] = parsed_message(raw, tokenizer, case["call"]["tools"], parser_type,
                                                           native_messages(case["call"]["messages"]))
                        result["native_replay"] = replay(case, result["message"])
                    except Exception as error:
                        result["native_error"] = {"type": type(error).__name__, "message": str(error)}
                return result
            proposals.extend(await asyncio.gather(*(generate(case) for case in cases)))
            write(args.output / "same-decision-proposals.json", proposals)
            print(json.dumps({"stage": "same_decision_generated", "mode": mode, "cases": 4}), flush=True)
        for cohort, names in ([] if referenced else [("baseline_failures", FAILURES), ("baseline_retention", RETENTION)]):
            await agent.collect(0, cohort, names, temperature=0.)
        outcome_paths = {"baseline": {name: str(args.output / "step-0000" / ("baseline_" + name) / "outcomes.json")
                                     for name in ["failures", "retention"]}}
        await rollout.suspend_for_update()
        if args.control_reference is not None:
            reference = args.control_reference
            reference_capsule = reference.parent.parent
            for filename, current in [("selected-transitions.json", args.rows), ("source-cases.json", args.cases)]:
                if current.read_bytes() != (reference_capsule / filename).read_bytes():
                    raise ValueError("completed control source differs")
            if (reference / "initial-student/adapter_model.safetensors").read_bytes() != (
                    args.output / "initial-student/adapter_model.safetensors").read_bytes():
                raise ValueError("completed control initial adapter differs")
            update = json.loads((reference / "original_teacher/update.json").read_text())
            if (update["optimizer_updates"] != 1 or update["optimizer_moment_steps"] != [1]
                    or update["batch"] != 32 or update["changed_teacher_rows"] != 0):
                raise ValueError("completed control optimizer receipt is invalid")
            saved_optimizer = torch.load(reference / "original_teacher/optimizer.pt", map_location="cpu", weights_only=True)
            if {int(v["step"].item()) for v in saved_optimizer["state"].values()} != {1}:
                raise ValueError("completed control saved optimizer moments invalid")
            del saved_optimizer
            folder = reference / "step-0001/original_teacher_failures"
            episodes = [json.loads(s) for s in (folder / "native/traces.jsonl").read_text().splitlines()]
            if (len(episodes) != 8 or len({e["id"] for e in episodes}) != 8
                    or {n: sum(e["task"]["data"]["name"] == n for e in episodes) for n in FAILURES} != {n: 2 for n in FAILURES}
                    or json.loads((folder / "outcomes.json").read_text())["errors"]):
                raise ValueError("completed control native failure readout incomplete")
            shutil.copytree(reference / "original_teacher", args.output / "original_teacher")
            shutil.copytree(folder, args.output / "step-0001/original_teacher_failures")
            state = load_file(args.output / "original_teacher/student/adapter_model.safetensors")
            set_peft_model_state_dict(model, state, adapter_name="default")
            del state
            model.set_adapter("default")
            gc.collect()
            torch.cuda.empty_cache()
            await rollout.refresh()
            expected_sha = hashlib.sha256((args.output / "original_teacher/student/adapter_model.safetensors").read_bytes()).hexdigest()
            if rollout.refreshes[-1]["weights_sha256"] != expected_sha:
                raise ValueError("restored control serving export differs from saved checkpoint")
            agent.baselines["original_teacher_retention"] = json.loads((args.output / "step-0000/baseline_retention/outcomes.json").read_text())
            await agent.collect(1, "original_teacher_retention", RETENTION, temperature=0.)
            outcome_paths["original_teacher"] = {kind: str(args.output / "step-0001" / ("original_teacher_" + kind) / "outcomes.json")
                                                  for kind in ["failures", "retention"]}
            write(args.output / "referenced-completed-control.json", {"source": str(reference),
                  "completed_optimizer_updates_referenced": 1, "native_failure_episodes_referenced": 8,
                  "restored_adapter_sha256": expected_sha, "optimizer_resumed": False,
                  "scope": "Completed control loaded for missing retention evaluation ONLY; no repeated or resumed control optimizer update"})
            await rollout.suspend_for_update()
        for mode in (["combined_teacher"] if args.control_reference is not None else ["original_teacher", "combined_teacher"]):
            arm_index = 1 if mode == "original_teacher" else 2
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if name in snapshots:
                        parameter.copy_(snapshots[name].to(parameter.device))
            model.set_adapter("default")
            model.train()
            optimizer = torch.optim.AdamW(params, lr=1e-6, weight_decay=.01)
            optimizer.zero_grad(set_to_none=True)
            batch, changed = arm_batch(original_rows, alternatives, mode)
            before = {}
            for case in cases:
                name = case["row"]["task"]
                proposal = next(p for p in proposals if p["task"] == name and p["mode"] == "combined_teacher")
                if proposal["complete"]:
                    row = {**case["row"], "action_ids": proposal["action_ids"]}
                    before[name] = float(score_sampled(model, row).mean())
            metrics = []
            for index, row in enumerate(batch):
                loss, advantages = loss_for(model, row, SimpleNamespace(topk=20, alpha=1.))
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite SDPO loss")
                (loss / 32).backward()
                metric = {"index": index, "task": row["task"], "episode_id": row["episode_id"],
                          "generation_index": row["generation_index"], "loss": float(loss.detach()),
                          "teacher_sampled_advantages": advantages,
                          "support": row["teacher_support_diagnostic"]}
                metrics.append(metric)
                write(args.output / mode / "microbatch-metrics.json", metrics)
                print(json.dumps({"stage": "backward", "arm": mode, "microbatch": index + 1}), flush=True)
            projections = {}
            for case in cases:
                name = case["row"]["task"]
                proposal = next(p for p in proposals if p["task"] == name and p["mode"] == "combined_teacher")
                if not proposal["complete"]:
                    continue
                model.train()  # non-reentrant activation checkpointing, zero dropout
                targets = proposal["action_ids"]
                values = logits(model, case["row"]["student_prefix_ids"], targets).float()
                selected = values.log_softmax(-1).gather(-1, torch.tensor(targets, device="cuda")[:, None]).mean()
                reference = torch.autograd.grad(selected, params, allow_unused=True)
                direction = -sum(float((p.grad.float() * g.float()).sum()) for p, g in zip(params, reference, strict=True)
                                 if p.grad is not None and g is not None)
                projections[name] = {"full_batch_gradient_direction_toward_proposal_logprob": direction,
                                     "proposal_correctness": "requires native/source review; positive is not a correctness proof"}
                del values, selected, reference
            norm = torch.nn.utils.clip_grad_norm_(params, 1.)
            if not torch.isfinite(norm) or norm <= 0:
                raise ValueError("nonfinite/zero batch gradient")
            optimizer.step()
            moments = sorted({int(v["step"].item()) for v in optimizer.state.values()})
            if moments != [1]:
                raise ValueError("one-update optimizer moments disagree")
            optimizer.zero_grad(set_to_none=True)
            model.eval()
            for case in cases:
                name = case["row"]["task"]
                proposal = next(p for p in proposals if p["task"] == name and p["mode"] == "combined_teacher")
                if name in before:
                    row = {**case["row"], "action_ids": proposal["action_ids"]}
                    projections[name]["actual_adam_mean_proposal_logprob_delta"] = float(score_sampled(model, row).mean()) - before[name]
            model.save_pretrained(args.output / mode / "student", selected_adapters=["default"])
            torch.save(optimizer.state_dict(), args.output / mode / "optimizer.pt")
            write(args.output / mode / "update.json", {"optimizer_updates": 1, "batch": 32,
                  "changed_teacher_rows": changed, "loss": sum(m["loss"] for m in metrics) / 32,
                  "grad_norm": float(norm), "optimizer_moment_steps": moments, "directions": projections,
                  "algorithm": "top20+tail reverseKL alpha1, clipped detached rollout correction cap2; no feedback CE or gold-call supervision",
                  "scope": "one isolated replay-batch diagnostic; not fresh on-policy training or an EMA campaign"})
            gc.collect()
            torch.cuda.empty_cache()
            await rollout.refresh()
            for cohort, names in [("failures", FAILURES), ("retention", RETENTION)]:
                agent.baselines[mode + "_" + cohort] = json.loads((args.output / "step-0000" / ("baseline_" + cohort) / "outcomes.json").read_text())
                await agent.collect(arm_index, mode + "_" + cohort, names, temperature=0.)
            outcome_paths[mode] = {name: str(args.output / f"step-{arm_index:04d}" / (mode + "_" + name) / "outcomes.json")
                                   for name in ["failures", "retention"]}
            await rollout.suspend_for_update()
            del optimizer
            torch.cuda.empty_cache()
        write(args.output / "receipt.json", {"runtime_completed": True, "semantic_qualified": None,
              "optimizer_updates_per_arm": 1, "total_diagnostic_updates": 2, "native_episodes_expected": 48,
              "native_episodes_referenced": (16 if referenced else 0) + (8 if args.control_reference else 0),
              "native_episodes_generated_this_attempt": (24 if args.control_reference else 32) if referenced else 48,
              "optimizer_updates_referenced": 1 if args.control_reference else 0,
              "optimizer_updates_applied_this_attempt": 1 if args.control_reference else 2,
              "same_decision_generations": 12, "outcomes": outcome_paths, "training_restarted": False,
              "qualification": "Independent native/source and retention audit required; no pass inferred from loss or proposal likelihood."})
    finally:
        await runner.cleanup()
        await rollout.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--control-reference", type=Path)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("retained attempt exists; do not overwrite")
    args.output.mkdir(parents=True)
    write(args.output / "start.json", {"started_unix": time.time(), "cpu_only": args.cpu_only})
    if args.cpu_only:
        source, _ = preflight(args)
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        prefix_inputs(args, source["cases"], tokenizer)
    else:
        asyncio.run(run(args))
