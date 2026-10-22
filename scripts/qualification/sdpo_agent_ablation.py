"""Bounded on-policy SDPO agent prototype; not the public frozen-teacher API."""

import argparse
import asyncio
import copy
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from aiohttp import web
from live import environment_config
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from sdpo_agent_config import (
    MODEL_REVISION,
    admission,
    aligned_messages,
    balanced_rows,
    feedback_text,
    observed_tool_error,
    population_rollouts,
    successful_reference,
    update_learning_rate,
)
from self_campaign import Campaign, write
from self_campaign_config import comparisons, summarize
from self_failure_diagnosis import logits
from self_feedback_native import prompt_ids as qwen_prompt_ids
from self_feedback_vllm import LocalVLLMRollout
from transformers import AutoModelForImageTextToText, AutoTokenizer, set_seed
from trl.experimental.sdpo.loss_utils import compute_topk_self_distillation_loss
from verifiers.v1.cli.eval.runner import run_eval
from verifiers.v1.configs.cli.eval import EvalConfig
from verifiers_preinstalled import configure_preinstalled_runtime
from vllm import SamplingParams


def prompt_ids(tokenizer, messages, tools):
    if getattr(tokenizer, "_sdpo_model_family", "qwen3.5") == "lfm2.5":
        from sdpo_lfm_runtime import prompt_ids as lfm_prompt_ids

        return lfm_prompt_ids(tokenizer, messages, tools)
    return qwen_prompt_ids(tokenizer, messages, tools)


class SamplingRollout(LocalVLLMRollout):
    """Same optimized engine; explicitly stochastic training, greedy diagnostics."""

    temperature = 0.

    async def generate(self, prefix, max_tokens=2048):
        if not 1 <= max_tokens <= getattr(self, "reply_limit", 2048):
            raise ValueError("unqualified reply budget")
        result = await self._generate(prefix, SamplingParams(
            temperature=self.temperature, top_p=1., max_tokens=max_tokens,
            stop_token_ids=[self.tokenizer.eos_token_id], logprobs=1))
        completion = result.outputs[0]
        ids = list(completion.token_ids)
        if list(result.prompt_token_ids) != prefix or not ids or not completion.logprobs:
            raise ValueError("native prompt/action/logprob mismatch")
        logps = [float(v[t].logprob) for v, t in zip(completion.logprobs, ids, strict=True)]
        if not all(math.isfinite(x) for x in logps):
            raise ValueError("nonfinite sampled log probabilities")
        return ids, completion.finish_reason == "stop", {
            "policy_version": self.version, "adapter_sha256": self.refreshes[-1]["weights_sha256"],
            "sampled_logprobs": logps, "finish_reason": completion.finish_reason,
            "temperature": self.temperature, "num_cached_tokens": result.num_cached_tokens}


class Agent(Campaign):
    def __init__(self, args):
        self.args, self.calls, self.curve, self.baselines = args, [], [], {}

    async def completions(self, request):
        body = await request.json()
        body["messages"] = aligned_messages(body["messages"])
        if getattr(self, "collection_turn_limit", 8) == 20:
            for message in body["messages"]:
                if message["role"] == "system":
                    message["content"] = message["content"].replace(
                        "a budget of at most 8 assistant turns", "a budget of at most 20 assistant turns")
        if getattr(self.args, "model_family", "qwen3.5") == "lfm2.5":
            from sdpo_lfm_runtime import completions

            return await completions(self, body)
        if getattr(self.args, "tool_parser_mode", "native") == "schema-bound-v1":
            from sdpo_schema_parser import schema_completions

            return await schema_completions(self, body)

        class Request:
            async def json(self):
                return body

        return await super().completions(Request())

    async def collect(self, step, cohort, names, temperature=0.):
        mtp_before = None
        if getattr(self.args, "capture_mtp_metrics", False):
            from sdpo_mtp_metrics import snapshot

            mtp_before = snapshot()
        guidance_profile = getattr(self.args, "run_profile", "ablation") == "guidance-five"
        pool_profile = getattr(self.args, "run_profile", "ablation") in (
            "pool124-qualify", "pool124-50", "pool124-bounded", "lfm-pool124-20")
        if pool_profile:
            from sdpo_pool124_config import collection_budget

            budget = collection_budget(cohort, self.args.run_profile)
            self.args.context_limit, self.args.reply_limit = budget["context"], budget["reply"]
            self.rollout.reply_limit = budget["reply"]
            self.collection_turn_limit = budget["turns"]
        if guidance_profile:
            evaluation = cohort == "heldout62"
            self.args.context_limit = 65536 if evaluation else 32768
            self.args.reply_limit = 8192 if evaluation else 4096
            self.rollout.reply_limit = self.args.reply_limit
            self.collection_turn_limit = 20 if evaluation else 8
        self.folder = self.args.output / f"step-{step:04d}" / cohort
        self.folder.mkdir(parents=True, exist_ok=False)
        self.calls = []
        self.rollout.temperature = temperature
        env = environment_config(getattr(self, "collection_turn_limit", 8), "limited_zapier")
        replicas = population_rollouts(cohort, getattr(self.args, "online_rollouts", 2))
        if pool_profile:
            replicas = budget["replicas"]
            env["agent"]["max_output_tokens"] = budget["episode"]
        if guidance_profile:
            replicas = 5 if cohort == "heldout62" else 2
            env["agent"]["max_output_tokens"] = 32768 if cohort == "heldout62" else 12288
        env["taskset"]["task"].update(capture_actions=True, deterministic_world=True)
        profile = getattr(self.args, "environment_profile", "frozen-a0c")
        if profile != "frozen-a0c":
            from progress_environment_profile import task_options, taskset_options

            env["taskset"]["task"].update(task_options(profile))
            env["taskset"].update(taskset_options(profile, names))
        write(self.folder / "environment-selection.json", {"profile": profile, "config": env})
        config = EvalConfig.model_validate({"env": env, "model": "local-self",
            "client": {"type": "eval", "base_url": "http://127.0.0.1:18774/v1", "api_key_var": "SELF_LOCAL_KEY"},
            "sampling": {"temperature": temperature, "max_tokens": self.args.reply_limit}, "num_tasks": len(names),
            "task_keys": names, "num_rollouts": replicas,
            "max_concurrent": getattr(self.args, "concurrency", 32), "shuffle": False,
            "output_dir": self.folder / "native", "run": {"name": f"{self.args.run_name}-{step}-{cohort}", "dir": "."},
            "push": False, "rich": None, "serve": None})
        print(json.dumps({"stage": "native_collection", "step": step, "cohort": cohort,
                          "temperature": temperature}), flush=True)
        episodes = await run_eval(config)
        summary = summarize(episodes, names, replicas)
        write(self.folder / "outcomes.json", summary)
        if mtp_before is not None:
            from sdpo_mtp_metrics import delta, snapshot

            mtp_after = snapshot()
            write(self.folder / "mtp-counters.json", {
                "before": mtp_before, "after": mtp_after, "delta": delta(mtp_before, mtp_after),
                "policy_version": self.rollout.version, "cohort": cohort, "step": step})
        expected_scoring_errors = False
        if (pool_profile or guidance_profile) and cohort == "heldout62" and summary["errors"]:
            from sdpo_pool124_config import known_frozen_scoring_errors

            expected_scoring_errors = known_frozen_scoring_errors(summary["errors"])
        if ((summary["errors"] and not expected_scoring_errors)
                or any(c["policy_version"] != self.rollout.version for c in self.calls)):
            raise ValueError("native errors or stale policy; preserve and stop")
        values = [e.model_dump(mode="json") for e in episodes]
        if temperature == 0. and getattr(self.args, "native_repeatability", False):
            from sdpo_greedy_repeatability import audit

            repeatability = audit(self.calls)
            write(self.folder / "native-greedy-repeatability.json", repeatability)
            if not repeatability["qualified"]:
                raise ValueError("complete native greedy calls differ at identical inputs; preserve attempt")
        if getattr(self.args, "teacher_feedback", None) in ("luna-or-solution", "prepared-guidance"):
            from sdpo_luna_context import native_task_bindings

            retained = [json.loads(line) for line in (self.folder / "native/traces.jsonl").read_text().splitlines() if line]
            self.luna_native_tasks = native_task_bindings(values, retained)
            if self.args.teacher_feedback == "prepared-guidance":
                # WireTaskData drops initial-state extensions. The retained
                # native episodes also preserve assessment archive lifecycle;
                # use them only after exact episode/user/task ID reconciliation.
                values = retained
        if cohort != "online_train":
            if step == 0:
                self.baselines[cohort] = summary
            self.curve.append({"step": step, "cohort": cohort, "summary": summary,
                "compared_to_base": comparisons(self.baselines[cohort], summary)})
            write(self.args.output / "outcome-curve.json", self.curve)
        return values, copy.deepcopy(self.calls)


def associate(calls, episodes, tokenizer, feedback_mode, luna_references=None, luna_native_tasks=None,
              context_limit=16384, thinking=False, prepared_guidance=None, guidance_context="taskwide",
              action_risk_sampling=False):
    """Join actual sampled actions by unique native tool IDs/history; retain final text."""
    by_tool, by_history, by_prompt = {}, {}, {}
    for episode in episodes:
        trace = episode["traces"][0]
        messages = [n.get("message") for n in trace["nodes"] if n.get("message")]
        user = next(m["content"] for m in messages if m["role"] == "user")
        by_prompt.setdefault(user, []).append(episode)
        for message in messages:
            for tool in message.get("tool_calls") or []:
                by_tool[tool["id"]] = episode
            if message["role"] == "tool":
                by_history[message["tool_call_id"]] = episode
    rows, excluded = [], []
    native_findings_cache = {}
    for index, call in enumerate(calls):
        if not call["complete"]:
            excluded.append({"generation_index": index,
                             "reason": call.get("limit_reason", "incomplete_generation")})
            continue
        tools = call["message"].get("tool_calls", [])
        history = [m for m in call["messages"] if m["role"] == "tool"]
        user = next(m["content"] for m in call["messages"] if m["role"] == "user")
        if tools and tools[0]["id"] in by_tool:
            episode = by_tool[tools[0]["id"]]
        elif history:
            episode = by_history[history[-1]["tool_call_id"]]
        else:
            # A first-turn final response may be identical in two deterministic
            # replicas. Reject ambiguous association rather than invent credit.
            candidates = by_prompt.get(user, [])
            if len(candidates) != 1:
                excluded.append({"generation_index": index, "reason": "ambiguous_first_turn_terminal"})
                continue
            episode = candidates[0]
        trace = episode["traces"][0]
        messages = [n.get("message") for n in trace["nodes"] if n.get("message")]
        assertions = trace.get("info", {}).get("automationbench", {}).get("assertions")
        if assertions is None:
            raise ValueError("missing native factual check results")
        current_ids = {t["id"] for t in tools}
        responses = [m for m in messages if m["role"] == "tool" and m.get("tool_call_id") in current_ids]
        if not tools and history:
            responses = history[-1:]
        observations = [{"tool": m.get("name"), "response": m.get("content")} for m in responses]
        solution = None
        reference = successful_reference(episode, by_prompt[user])
        if feedback_mode in ("solution-or-feedback", "luna-or-solution") and reference is not None:
            # Retain complete native calls and terminal behavior, no Luna/gold reconstruction.
            solution = json.dumps([n["message"] for n in reference["traces"][0]["nodes"]
                                   if (n.get("message") or {}).get("role") in ("assistant", "tool")], sort_keys=True)
        text = feedback_text(assertions, observations, solution)
        luna, fallback_reason, unavailable_tools = None, None, []
        if feedback_mode == "luna-or-solution":
            from sdpo_luna_context import reference_for_task, teacher_text, unavailable_reference_tools

            if luna_native_tasks is None or episode["id"] not in luna_native_tasks:
                raise ValueError("missing ID-joined native task authority")
            luna = reference_for_task(luna_native_tasks[episode["id"]], luna_references or {})
            fallback_reason = "missing_offline_source" if luna is None else None
            if luna is not None:
                unavailable_tools = unavailable_reference_tools(luna, call["tools"])
                if unavailable_tools:
                    luna = None
                    fallback_reason = "offline_source_uses_unavailable_tools"
                else:
                    text = teacher_text(luna)
        teacher_messages = call["messages"] + [{"role": "user", "content": text}]
        guidance_audit = None
        manifest_findings = None
        if feedback_mode == "prepared-guidance":
            from sdpo_guidance_context import native_findings
            from sdpo_guidance_context import teacher_messages as guidance_messages

            if luna_native_tasks is None or episode["id"] not in luna_native_tasks:
                raise ValueError("missing native retained task authority for prepared guidance")
            native_task = luna_native_tasks[episode["id"]]
            if prepared_guidance is None or native_task["name"] not in prepared_guidance:
                raise ValueError("missing prepared teacher guidance; no raw-context fallback")
            bundle = prepared_guidance[native_task["name"]]
            if episode["id"] not in native_findings_cache:
                native_findings_cache[episode["id"]] = native_findings(episode, bundle)
            manifest_findings = native_findings_cache[episode["id"]]
            teacher_messages, guidance_audit = guidance_messages(
                call, native_task, bundle, assertions, observations, manifest_findings,
                context_mode=guidance_context)
            text = teacher_messages[-1]["content"]
        if thinking:
            from sdpo_prompting import prompt_ids as thinking_prompt_ids

            teacher_prefix = thinking_prompt_ids(tokenizer, teacher_messages, call["tools"], thinking=True)
        else:
            teacher_prefix = prompt_ids(tokenizer, teacher_messages, call["tools"])
        row = {"task": episode["task"]["data"]["name"], "episode_id": episode["id"],
               "generation_index": index, "student_prefix_ids": call["prefix_ids"],
               "prepared_guidance_audit": guidance_audit,
               "native_manifest_findings": manifest_findings,
               "observed_obligation_failure": bool(manifest_findings and
                                                     manifest_findings["observed_obligation_failure"]),
               "teacher_prefix_ids": teacher_prefix,
               "action_ids": call["action_ids"], "complete": call["complete"],
               "terminal_text": not bool(tools), "teacher_feedback": text,
               "solution_used": solution is not None or luna is not None, "policy_version": call["policy_version"],
               "reference_episode_id": (luna["episode_id"] if luna is not None
                                        else reference["id"] if solution is not None else None),
               "offline_luna_reference_used": luna is not None,
               "offline_luna_missing_fallback": feedback_mode == "luna-or-solution" and luna is None,
               "offline_luna_fallback_reason": fallback_reason,
               "offline_luna_unavailable_tools": unavailable_tools,
               "offline_luna_source_sha256": luna["source_episode_sha256"] if luna is not None else None,
               "sampled_logprobs": call["sampled_logprobs"],
               "native_success": trace.get("metrics", {}).get("task_completed_correctly"),
               "executed_tool_responses": len(responses),
               "observed_tool_error": bool(tools) and observed_tool_error(responses),
               "full_native_feedback_checks": len(assertions)}
        if action_risk_sampling:
            from sdpo_action_risks import action_risks

            # Sampling metadata only: do not add guessed penalties or alter
            # original student/teacher tokens, tool execution, or the SDPO loss.
            row["observed_action_risks"] = action_risks(call, responses) if tools else []
        reason = admission(row, context=context_limit)
        if reason:
            excluded.append({"generation_index": index, "reason": reason, "task": row["task"],
                             "student_context_plus_target": len(row["student_prefix_ids"]) + len(row["action_ids"]),
                             "teacher_context_plus_target": len(row["teacher_prefix_ids"]) + len(row["action_ids"])})
        else:
            rows.append(row)
    return rows, excluded


def teacher_logits(model, row):
    model.set_adapter("sdpo_teacher")
    model.eval()
    with torch.no_grad():
        result = logits(model, row["teacher_prefix_ids"], row["action_ids"]).detach()
    model.set_adapter("default")
    model.train()
    return result


def loss_for(model, row, args, teacher=None):
    teacher = teacher_logits(model, row) if teacher is None else teacher
    scores = logits(model, row["student_prefix_ids"], row["action_ids"])
    total = scores.new_zeros((), dtype=torch.float32)
    sampled_advantages = []
    support = {"teacher_top1_ids": [], "teacher_top1_student_ranks": [],
               "teacher_mass_on_student_support": [], "teacher_top1_student_logps_before": [],
               "sampled_token_student_ranks": [], "actor_sampled_logps_before": [],
               "actor_rollout_logp_differences": [], "importance_weights": []}
    for start in range(0, len(row["action_ids"]), 32):
        s, t = scores[start:start + 32].float(), teacher[start:start + 32].float()
        per = compute_topk_self_distillation_loss(s, t, distillation_topk=args.topk,
                distillation_alpha=args.alpha, distillation_add_tail=True)
        target = torch.tensor(row["action_ids"][start:start + 32], device="cuda")[:, None]
        selected = s.log_softmax(-1).gather(-1, target).flatten()
        with torch.no_grad():
            preferred = t.argmax(-1, keepdim=True)
            student_support = s.detach().topk(args.topk, dim=-1).indices
            support["teacher_top1_ids"] += preferred.flatten().tolist()
            support["teacher_top1_student_ranks"] += (
                (s.detach() > s.detach().gather(-1, preferred)).sum(-1) + 1).tolist()
            support["sampled_token_student_ranks"] += (
                (s.detach() > s.detach().gather(-1, target)).sum(-1) + 1).tolist()
            support["teacher_mass_on_student_support"] += (
                t.log_softmax(-1).gather(-1, student_support).exp().sum(-1)).tolist()
            support["teacher_top1_student_logps_before"] += (
                s.detach().log_softmax(-1).gather(-1, preferred).flatten()).tolist()
        old = torch.tensor(row["sampled_logprobs"][start:start + 32], device="cuda")
        # Fresh collection per optimizer step; correct residual vLLM/actor
        # mismatch using the authors' bounded detached importance weight.
        ratio = (selected.detach() - old).clamp(-20, 20).exp().clamp(max=2.)
        support["actor_sampled_logps_before"] += selected.detach().tolist()
        support["actor_rollout_logp_differences"] += (selected.detach() - old).tolist()
        support["importance_weights"] += ratio.tolist()
        per = per * ratio
        total = total + per.sum() / len(row["action_ids"])
        sampled_advantages += (t.log_softmax(-1).gather(-1, target) -
                               s.detach().log_softmax(-1).gather(-1, target)).flatten().tolist()
    row["teacher_support_diagnostic"] = support
    return total, sampled_advantages


@torch.no_grad()
def score_sampled(model, row, preferred_ids=None):
    model.eval()
    value = logits(model, row["student_prefix_ids"], row["action_ids"]).float()
    target = torch.tensor(row["action_ids"], device="cuda")[:, None]
    if preferred_ids is not None:
        preferred = torch.tensor(preferred_ids, device="cuda")[:, None]
        logps = value.log_softmax(-1)
        return logps.gather(-1, target).flatten().cpu(), logps.gather(-1, preferred).flatten().cpu()
    if "teacher_support_diagnostic" in row:
        diagnostic = row["teacher_support_diagnostic"]
        preferred = torch.tensor(diagnostic["teacher_top1_ids"], device="cuda")[:, None]
        after = value.log_softmax(-1).gather(-1, preferred).flatten().cpu()
        before = torch.tensor(diagnostic["teacher_top1_student_logps_before"])
        diagnostic["actual_adam_teacher_top1_logprob_deltas"] = (after - before).tolist()
        diagnostic["teacher_top1_delta_modes"] = "legacy_train_before/eval_after; not independently same-mode"
    return value.log_softmax(-1).gather(-1, target).flatten().cpu()


async def main(args):
    started = time.monotonic()
    from sdpo_long_campaign_config import campaign_batch, campaign_learning_rate, execution_profile

    execution = execution_profile(args)
    batch_size = execution["batch"]
    guidance_profile = args.run_profile == "guidance-five"
    if not guidance_profile and (getattr(args, "concurrency", 32) != 32
            or getattr(args, "speculation", "none") != "none"
            or getattr(args, "teacher_guidance", None) is not None
            or getattr(args, "guidance_context", "taskwide") != "taskwide"
            or getattr(args, "disable_prefix_caching", False)
            or getattr(args, "disable_chunked_prefill", False)
            or getattr(args, "native_prefix_fixture", None) is not None
            or getattr(args, "native_repeatability", False)
            or getattr(args, "capture_mtp_metrics", False)
            or getattr(args, "heldout_base_reference", None) is not None
            or getattr(args, "selection_mode", "") == "first-risk-frontier"):
        raise ValueError("new speculation/guidance controls require the explicit new profile")
    if args.shared_budget is not None:
        from sdpo_execution_budget import install_deadline_timer

        install_deadline_timer(json.loads(args.shared_budget.read_text()))
    args.output.mkdir(parents=True, exist_ok=False)
    expected_revision = MODEL_REVISION
    if getattr(args, "model_family", "qwen3.5") == "lfm2.5":
        from sdpo_lfm_config import MODEL_REVISION as expected_revision
    if (Path(args.model).name != expected_revision
            or not 0 < args.lr <= 3e-5 or not 0 <= args.teacher_rate <= 1):
        raise ValueError("model pin/update cap mismatch")
    if args.thinking and args.tool_parser_mode != "schema-bound-v1" and args.run_profile != "lfm-pool124-20":
        raise ValueError("thinking requires the separate qualified reasoning/tool bridge")
    cohort = json.loads(args.cohorts.read_text())
    pool_profile = args.run_profile in ("pool124-qualify", "pool124-50", "pool124-bounded", "lfm-pool124-20")
    if pool_profile:
        from sdpo_pool124_config import diagnostic_groups, step_tasks, task_batch, validate_pool

        if args.training_pool is None:
            raise ValueError("exact training pool source required")
        validate_pool(cohort, json.loads(args.training_pool.read_text()), json.loads(args.heldout.read_text()))
    groups = [set(cohort[key]) for key in ("train", "development", "reserve")]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("training/development/reserve overlap")
    heldout = set(json.loads(args.heldout.read_text())["task_names"])
    if (set(cohort["train"]) | set(cohort["development"]) | set(cohort["reserve"])) & heldout:
        raise ValueError("heldout overlap")
    prepared_guidance = None
    if guidance_profile:
        prepared_guidance = json.loads(args.teacher_guidance.read_text())["guidance"]
        if set(prepared_guidance) != set(cohort["train"]) or len(cohort["train"]) != 20:
            raise ValueError("prepared guidance must cover all twenty frozen training tasks")
    write(args.output / "inputs.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    luna_references = None
    if args.teacher_feedback == "luna-or-solution":
        if args.luna_references is None:
            raise ValueError("offline Luna source required for this arm")
        reviewed = json.loads(args.luna_references.read_text())
        luna_references = reviewed["references"]
        if not reviewed["current_rubric_state_checks_complete"] or not luna_references:
            raise ValueError("offline source review missing")
        if not set(luna_references) <= set(cohort["train"]):
            raise ValueError("Luna context must be restricted to training tasks")
        write(args.output / "offline-luna-reference-receipt.json", {
            "sha256": hashlib.sha256(args.luna_references.read_bytes()).hexdigest(),
            "reference_tasks": sorted(luna_references),
            "fallback_tasks": sorted(set(cohort["train"]) - set(luna_references)),
            "external_model_calls": 0, "student_receives_reference": False})
    elif args.luna_references is not None:
        raise ValueError("unexpected Luna context in no-Luna control")
    os.environ.update(POSTTRAIN_VERIFIERS_PREINSTALLED="1", SELF_LOCAL_KEY="EMPTY")
    configure_preinstalled_runtime(interpreter=sys.executable)
    set_seed(18)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer._sdpo_model_family = getattr(args, "model_family", "qwen3.5")
    model_class = AutoModelForImageTextToText
    if tokenizer._sdpo_model_family == "lfm2.5":
        from transformers import AutoModelForCausalLM

        model_class = AutoModelForCausalLM
    base = model_class.from_pretrained(args.model, dtype=torch.bfloat16,
        device_map={"": "cuda:0"}, attn_implementation="sdpa", local_files_only=True)
    model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.,
        target_modules="all-linear", exclude_modules=r".*visual.*", task_type="CAUSAL_LM"))
    initial = args.output / "initial-student"
    model.save_pretrained(initial, selected_adapters=["default"])
    model.load_adapter(initial, adapter_name="sdpo_teacher", is_trainable=False)
    model.set_adapter("default")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    probe = tokenizer.encode("Local readiness", add_special_tokens=False)
    target = tokenizer.encode(" verified.", add_special_tokens=False)
    with torch.no_grad():
        active = logits(model, probe, target)
        with model.disable_adapter():
            bare = logits(model, probe, target)
        torch.testing.assert_close(active, bare, atol=1e-5, rtol=1e-5)
    write(args.output / "fresh-base-gate.json", {"max_abs_delta": float((active - bare).abs().max()),
          "B_matrices_zero": all(not p.any() for n, p in model.named_parameters() if "lora_B.default" in n)})
    args.feedback_mode = "minimal"
    agent = Agent(args)
    agent.model, agent.tokenizer = model, tokenizer
    if args.tool_parser_mode == "schema-bound-v1":
        from sdpo_schema_parser import parser_class

        agent.schema_parser, parser_receipt = parser_class()
        write(args.output / "tool-parser-receipt.json", parser_receipt)
    from sdpo_speculation import speculative_config

    rollout = await SamplingRollout.create(args.model, model, tokenizer, args.output,
        65536 if pool_profile or guidance_profile else args.context_limit,
        getattr(args, "concurrency", 32), 16,
        speculative_config=speculative_config(getattr(args, "speculation", "none"),
                                              getattr(args, "mtp_tokens", 2)),
        prefix_caching=not getattr(args, "disable_prefix_caching", False),
        chunked_prefill=not getattr(args, "disable_chunked_prefill", False),
        enforce_eager=getattr(args, "enforce_eager", False))
    rollout.reply_limit = args.reply_limit
    agent.rollout = rollout
    app = web.Application(client_max_size=16 * 1024 * 1024)
    app.router.add_post("/v1/chat/completions", agent.completions)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 18774).start()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=.01)
    updates = []
    try:
        await rollout.score_probe(model, probe, target)
        await rollout.batch_probe(probe, maximum=getattr(args, "concurrency", 32))
        if getattr(args, "native_prefix_fixture", None):
            from sdpo_native_prefix_gate import qualify as qualify_prefixes

            write(args.output / "native-prefix-qualification.json",
                  await qualify_prefixes(rollout, json.loads(args.native_prefix_fixture.read_text())))
        if args.qualify_capacity or args.thinking or args.run_profile == "campaign-qualify" or pool_profile:
            from sdpo_native_tool_gate import qualify

            await qualify(agent)
        if pool_profile:
            from sdpo_pool124_runtime import qualify_long_rollout

            await qualify_long_rollout(agent)
            diagnostics = diagnostic_groups(cohort["train"],
                json.loads(args.heldout.read_text())["task_names"], args.run_profile,
                cohort.get("reference_diagnostic"))
        else:
            diagnostics = [(group, cohort[group]) for group in ("train", "development")]
        if guidance_profile:
            from sdpo_pool124_runtime import qualify_long_rollout

            await qualify_long_rollout(agent)
            diagnostics = [("train", cohort["train"]),
                           ("heldout62", json.loads(args.heldout.read_text())["task_names"])]
        shared_control = None
        for group, names in diagnostics:
            if group == "heldout62" and getattr(args, "heldout_base_reference", None):
                from sdpo_shared_base import load

                shared_control = load(args.heldout_base_reference, Path(__file__).parent,
                                      vars(args), rollout.refreshes[0]["weights_sha256"], args.heldout)
                folder = args.output / "step-0000/heldout62"
                folder.mkdir(parents=True, exist_ok=False)
                write(folder / "shared-base.json", {
                    **shared_control, "source_reference": str(args.heldout_base_reference)})
                summary = shared_control["summary"]
                write(folder / "outcomes.json", summary)
                agent.baselines[group] = summary
                agent.curve.append({"step": 0, "cohort": group, "summary": summary,
                    "reused_control": shared_control["origin_run"],
                    "compared_to_base": comparisons(summary, summary)})
                write(args.output / "outcome-curve.json", agent.curve)
            else:
                await agent.collect(0, group, names)
        capacity = None
        if args.qualify_capacity:
            from sdpo_capacity import qualify_rollout

            if pool_profile:
                from sdpo_pool124_config import collection_budget

                budget = collection_budget("online_train", args.run_profile)
                args.context_limit, args.reply_limit = budget["context"], budget["reply"]
                rollout.reply_limit = budget["reply"]
            if guidance_profile:
                args.context_limit, args.reply_limit, rollout.reply_limit = 32768, 4096, 4096
            capacity = await qualify_rollout(args, model, tokenizer, rollout)
        for step in range(1, args.steps + 1):
            if time.monotonic() - started > execution["time_limit_s"]:
                raise TimeoutError("execution profile time limit; preserve evidence")
            tasks = step_tasks(cohort["train"], step) if pool_profile else cohort["train"]
            episodes, calls = await agent.collect(step, "online_train", tasks, temperature=1.)
            rows, excluded = associate(calls, episodes, tokenizer, args.teacher_feedback, luna_references,
                                       getattr(agent, "luna_native_tasks", None), args.context_limit, args.thinking,
                                       prepared_guidance, getattr(args, "guidance_context", "taskwide"),
                                       action_risk_sampling=args.selection_mode == "first-risk-frontier")
            if guidance_profile:
                from sdpo_guidance_sampling import batch as guidance_batch

                batch = guidance_batch(rows, cohort["train"], step, mode=args.selection_mode)
            elif pool_profile:
                batch = task_batch(rows, tasks, offset=step - 1)
            elif batch_size == 32:
                batch = campaign_batch(rows, cohort["train"], offset=step - 1)
            else:
                batch = balanced_rows(rows, batch_size, offset=step - 1, tasks=cohort["train"], mode=args.selection_mode)
            folder = args.output / f"step-{step:04d}" / "online_train"
            write(folder / "training-admission.json", {"admitted": len(rows), "exclusions": excluded,
                  "truncated_training_sequences": 0, "selected": [{k: r[k] for k in
                    ("task", "episode_id", "generation_index", "terminal_text", "solution_used")} for r in batch],
                  "eligible_terminal_actions": sum(r["terminal_text"] for r in rows),
                  "selection_mode": args.selection_mode,
                  "selected_task_population": tasks,
                  "distinct_selected_tasks": len({r["task"] for r in batch}),
                  "unique_selected_actions": len({(r["episode_id"], r["generation_index"]) for r in batch}),
                  "selected_observed_tool_errors": sum(r["observed_tool_error"] for r in batch),
                  "selected_observed_action_risks": sum(bool(r.get("observed_action_risks")) for r in batch),
                  "selected_terminal_actions": sum(r["terminal_text"] for r in batch),
                  "offline_luna_reference_actions": sum(r["offline_luna_reference_used"] for r in batch),
                  "offline_luna_fallback_actions": sum(r["offline_luna_missing_fallback"] for r in batch)})
            write(folder / "selected-transitions.json", batch)
            await rollout.suspend_for_update()
            if step == 1 and capacity is not None:
                from sdpo_capacity import qualify_actor

                if optimizer.state:
                    raise ValueError("capacity gate must precede all optimizer updates")
                qualify_actor(args, model, tokenizer, capacity, teacher_logits, loss_for, score_sampled)
                torch.cuda.empty_cache()  # release the synthetic4K probe's unused workspace before fitting
            optimizer.zero_grad(set_to_none=True)
            losses, before, advantages, preferred_before, preferred_ids_by_row = [], [], [], [], []
            for row in batch:
                teacher = teacher_logits(model, row)
                preferred_ids = teacher.argmax(-1).tolist()
                old, preferred_old = score_sampled(model, row, preferred_ids=preferred_ids)
                before.append(old)
                preferred_before.append(preferred_old)
                preferred_ids_by_row.append(preferred_ids)
                model.train()  # restore the original student loss mode after no-grad scoring
                loss, advantage = loss_for(model, row, args, teacher=teacher)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite SDPO loss")
                (loss / batch_size).backward()
                losses.append(float(loss.detach()))
                advantages.append(advantage)
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
            if not torch.isfinite(norm) or norm <= 0:
                raise ValueError("nonfinite/zero SDPO gradient")
            if args.lr_schedule == "hold10-decay20-flat":
                applied_lr = campaign_learning_rate(args.lr, step, args.lr_end_ratio)
            else:
                applied_lr = update_learning_rate(args.lr, step, args.steps, args.lr_schedule, args.lr_end_ratio)
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = applied_lr
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            direction = []
            for row, old, advantage, preferred_old, preferred_ids in zip(
                    batch, before, advantages, preferred_before, preferred_ids_by_row, strict=True):
                new, preferred_new = score_sampled(model, row, preferred_ids=preferred_ids)
                delta = new - old
                diagnostic = row["teacher_support_diagnostic"]
                if diagnostic["teacher_top1_ids"] != preferred_ids:
                    raise ValueError("preferred diagnostic IDs changed within an optimizer update")
                diagnostic["teacher_top1_student_eval_logps_before"] = preferred_old.tolist()
                diagnostic["actual_adam_teacher_top1_logprob_deltas"] = (preferred_new - preferred_old).tolist()
                diagnostic["teacher_top1_delta_modes"] = "eval_before/eval_after"
                a = torch.tensor(advantage)
                direction.append({"task": row["task"], "generation_index": row["generation_index"],
                    "teacher_support_diagnostic": row["teacher_support_diagnostic"],
                    "teacher_sampled_advantages": advantage, "actual_adam_logprob_deltas": delta.tolist(),
                    "sign_agreement_fraction": float(((a * delta) > 0).float().mean()),
                    "scope": "actual generated prefix and full batch Adam step, not a correctness proof"})
            write(folder / "teacher-adam-directions.json", direction)
            student = get_peft_model_state_dict(model, adapter_name="default")
            teacher = get_peft_model_state_dict(model, adapter_name="sdpo_teacher")
            teacher = {k: v.lerp(student[k].to(v.device), args.teacher_rate) for k, v in teacher.items()}
            set_peft_model_state_dict(model, teacher, adapter_name="sdpo_teacher")
            model.set_adapter("default")
            if step == args.steps or step % execution["save_every"] == 0:
                checkpoint = args.output / f"checkpoint-{step}"
                model.save_pretrained(checkpoint)
                torch.save(optimizer.state_dict(), checkpoint / "optimizer.pt")
                if batch_size == 32:
                    import random

                    import numpy as np

                    torch.save({"python": random.getstate(), "numpy": np.random.get_state(),
                                "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()},
                               checkpoint / "rng_state.pth")
                    write(checkpoint / "campaign-checkpoint.json", {
                        "step": step, "ema_updates": step, "batch": batch_size,
                        "learning_rate": applied_lr, "schedule": args.lr_schedule,
                        "resume_qualified": False, "run_profile": args.run_profile,
                        "files_sha256": {str(p.relative_to(checkpoint)): hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in checkpoint.rglob("*") if p.is_file()}})
            moment_steps = sorted({int(state["step"]) for state in optimizer.state.values() if "step" in state})
            if moment_steps != [step]:
                raise ValueError("optimizer moment steps disagree with update count")
            record = {"step": step, "batch": batch_size, "loss": sum(losses) / batch_size, "grad_norm": float(norm),
                      "learning_rate": applied_lr, "learning_rate_schedule": args.lr_schedule,
                      "optimizer_moment_steps": moment_steps,
                      "ema_updates": step, "elapsed_s": time.monotonic() - started}
            updates.append(record)
            write(args.output / "updates.json", updates)
            print(json.dumps({"stage": "optimizer_update", **record}), flush=True)
            await rollout.refresh()
            await rollout.score_probe(model, probe, target)
            if guidance_profile and step == 1:
                # Test the requested MTP path after a real adapter change, not
                # just on the zero-effect initialization. Preserve this step
                # even if readiness subsequently fails; never restart silently.
                from sdpo_native_tool_gate import qualify

                await rollout.batch_probe(probe, maximum=40)
                original_output = args.output
                args.output = original_output / "changed-adapter-qualification"
                args.output.mkdir(exist_ok=False)
                online_temperature = rollout.temperature
                rollout.temperature = 0.
                try:
                    await qualify(agent)
                    if getattr(args, "native_prefix_fixture", None):
                        from sdpo_native_prefix_gate import qualify as qualify_prefixes

                        write(args.output / "native-prefix-qualification.json",
                              await qualify_prefixes(rollout, json.loads(args.native_prefix_fixture.read_text())))
                finally:
                    args.output = original_output
                    rollout.temperature = online_temperature
            if step == args.steps or step % execution["evaluate_every"] == 0:
                for group, names in diagnostics:
                    await agent.collect(step, group, names)
        write(args.output / "receipt.json", {"completed": True, "optimizer_updates": len(updates),
            "elapsed_s": time.monotonic() - started, "external_calls": 0,
            "scope": "multi-turn SDPO adaptation on actual fresh student actions; no raw-feedback CE or gold action CE",
            "reserve_evaluated": False,
            "shared_base_control_origin": shared_control["origin_run"] if shared_control else None,
            "heldout62_evaluated": guidance_profile or pool_profile and args.run_profile != "pool124-qualify",
            "rollout_versions": rollout.refreshes})
    finally:
        await runner.cleanup()
        await rollout.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-family", choices=("qwen3.5", "lfm2.5"), default="qwen3.5")
    parser.add_argument("--cohorts", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--context-limit", type=int, choices=(16384, 32768), default=16384)
    parser.add_argument("--reply-limit", type=int, choices=(2048, 4096), default=2048)
    parser.add_argument("--qualify-capacity", action="store_true")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--shared-budget", type=Path)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--run-profile", choices=("ablation", "campaign-qualify", "campaign250", "pool124-qualify",
                                                "pool124-50", "pool124-bounded", "lfm-pool124-20", "guidance-five"), default="ablation")
    parser.add_argument("--concurrency", type=int, choices=(32, 40), default=32)
    parser.add_argument("--speculation", choices=("none", "mtp"), default="none")
    parser.add_argument("--mtp-tokens", type=int, choices=(1, 2), default=2)
    parser.add_argument("--plain-decoding-authorized", action="store_true",
                        help="Explicit user amendment for guidance-five standard decoding; no automatic fallback.")
    parser.add_argument("--disable-prefix-caching", action="store_true")
    parser.add_argument("--disable-chunked-prefill", action="store_true")
    parser.add_argument("--native-prefix-fixture", type=Path)
    parser.add_argument("--native-repeatability", action="store_true")
    parser.add_argument("--capture-mtp-metrics", action="store_true")
    parser.add_argument("--heldout-base-reference", type=Path)
    parser.add_argument("--teacher-guidance", type=Path)
    parser.add_argument("--guidance-context", choices=("taskwide", "feedback-only", "feedback-status"), default="taskwide")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--training-pool", type=Path)
    parser.add_argument("--batch-size", type=int, choices=(16, 32), default=16)
    parser.add_argument("--online-rollouts", type=int, choices=(2, 4, 8), default=2)
    parser.add_argument("--tool-parser-mode", choices=("native", "schema-bound-v1", "lfm2-native"), default="native")
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--lr-schedule", choices=("constant", "cosine-decay", "hold10-decay20-flat"), default="constant")
    parser.add_argument("--lr-end-ratio", type=float, default=.1)
    parser.add_argument("--alpha", type=float, choices=(.5, 1.), default=1.)
    parser.add_argument("--topk", type=int, choices=(20, 100), default=20)
    parser.add_argument("--teacher-rate", type=float, default=.01)
    parser.add_argument("--teacher-feedback", choices=("native-feedback", "solution-or-feedback", "luna-or-solution",
                                                     "prepared-guidance"), default="native-feedback")
    parser.add_argument("--luna-references", type=Path)
    parser.add_argument("--selection-mode", choices=("legacy-balanced", "distinct-balanced", "native-error-priority", "failure-frontier", "first-error-frontier", "first-risk-frontier"),
                        default="legacy-balanced")
    asyncio.run(main(parser.parse_args()))
