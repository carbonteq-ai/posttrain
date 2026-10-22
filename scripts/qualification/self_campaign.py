"""Local SELF campaign: one optimizer/scheduler, fresh round collections, native curves.

Research driver only. A plain single-device loader intentionally avoids Accelerate's
one-batch lookahead: future-round data must not be fetched before the preceding
optimizer/EMA update and policy refresh have completed.
"""

import argparse
import asyncio
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from aiohttp import web
from datasets import Dataset
from live import environment_config
from peft import LoraConfig, get_peft_model
from self_campaign_config import comparisons, schedule, summarize
from self_feedback_native import collect_transitions, prompt_ids
from self_feedback_probe import UpdateAudit
from self_feedback_signals import load_goal_context, requested_budget
from self_feedback_vllm import LocalVLLMRollout
from self_pilot_config import admit_sequences, resolve_pilot_tasks, sample_transitions
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoModelForImageTextToText, AutoTokenizer, TrainerCallback, set_seed
from trl import SFTConfig
from trl.experimental.self_feedback import SELFTrainer
from verifiers.v1.cli.eval.runner import run_eval
from verifiers.v1.configs.cli.eval import EvalConfig
from verifiers_preinstalled import configure_preinstalled_runtime
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tool_parsers.qwen3_engine_tool_parser import Qwen3EngineToolParser


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


class RoundDataset(IterableDataset):
    def __init__(self, campaign):
        self.campaign = campaign

    def __iter__(self):
        owner = self.campaign
        for start, end, count in owner.schedule:
            rows = owner.samples.get(start)
            if rows is None or len(rows) != count or owner.trainer.state.global_step != start:
                raise ValueError("future round requested before optimizer boundary and fresh collection")
            yield from rows
            if owner.trainer.state.global_step != end:
                raise ValueError("round exposures did not produce the expected number of updates")


class CampaignTrainer(SELFTrainer):
    def __init__(self, *args, **kwargs):
        self.micro_metrics = []
        self.diagnostic = False
        super().__init__(*args, **kwargs)

    def get_train_dataloader(self):
        if self.accelerator.num_processes != 1 or self.args.dataloader_num_workers:
            raise ValueError("campaign requires one device and no loader workers/prefetch")
        return DataLoader(self.train_dataset, batch_size=1, collate_fn=self.data_collator, num_workers=0)

    def compute_loss(self, *args, **kwargs):
        loss = super().compute_loss(*args, **kwargs)
        if not self.diagnostic:
            self.micro_metrics.append({"total": float(loss.detach()), **self.last_self_metrics})
        return loss


class CampaignCallback(TrainerCallback):
    def __init__(self, owner, loop):
        self.owner, self.loop = owner, loop
        self.optimizer_id = None

    def on_step_end(self, args, state, control, **kwargs):
        owner, trainer = self.owner, self.owner.trainer
        if trainer.accelerator.optimizer_step_was_skipped:
            raise ValueError("skipped optimizer step: retain evidence and stop campaign")
        if len(trainer.micro_metrics) != owner.args.batch_size:
            raise ValueError(f"wrong effective batch: {len(trainer.micro_metrics)}")
        if self.optimizer_id is None:
            self.optimizer_id = id(trainer.optimizer)
        if id(trainer.optimizer) != self.optimizer_id:
            raise ValueError("optimizer was replaced between rounds")
        metrics = {key: sum(row[key] for row in trainer.micro_metrics) / len(trainer.micro_metrics)
                   for key in trainer.micro_metrics[0]}
        trainer.micro_metrics.clear()
        row = {"step": state.global_step, "batch_size": owner.args.batch_size,
               "lr_after_step": trainer.lr_scheduler.get_last_lr(), **metrics}
        if not all(math.isfinite(v) for v in metrics.values()):
            raise ValueError("nonfinite loss component")
        with (owner.args.output / "updates.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps({"stage": "optimizer_update", **row}), flush=True)
        owner.updates.append(row)
        if state.global_step % owner.args.updates_per_round == 0:
            control.should_save = True
        return control

    def on_save(self, args, state, control, **kwargs):
        future = asyncio.run_coroutine_threadsafe(self.owner.boundary(state.global_step), self.loop)
        future.result()


class Campaign:
    def __init__(self, args):
        self.args = args
        self.schedule = schedule(args.steps, args.updates_per_round, args.batch_size)
        self.samples, self.updates, self.baselines, self.curve = {}, [], {}, []
        self.calls, self.folder = [], None
        self.fixed_rows = None
        self.trainer = None

    async def completions(self, request):
        body = await request.json()
        messages, tools = body["messages"], body.get("tools", [])
        prefix = prompt_ids(self.tokenizer, messages, tools)
        budget, fits = requested_budget(body, self.args.reply_limit, self.args.context_limit, len(prefix))
        if not fits:
            ids, complete = [], False
            evidence = {"limit_reason": "context_limit", "policy_version": self.rollout.version}
        else:
            ids, complete, evidence = await self.rollout.generate(prefix, max_tokens=budget)
            if not complete:
                evidence["limit_reason"] = "reply_limit" if budget == self.args.reply_limit else "requested_completion_limit"
        raw = self.tokenizer.decode(ids, skip_special_tokens=True)
        native = ChatCompletionRequest(model="local-self", messages=messages, tools=tools)
        parser = Qwen3EngineToolParser(self.tokenizer, tools=native.tools)
        parsed = parser.extract_tool_calls(raw, native) if complete else None
        tools_out = [call.model_dump() for call in parsed.tool_calls] if parsed else []
        message = {"role": "assistant", "content": parsed.content if parsed else raw}
        if tools_out:
            message["tool_calls"] = tools_out
        record = {"messages": messages, "tools": tools, "prefix_ids": prefix, "action_ids": ids,
                  "raw": raw, "message": message, "complete": complete,
                  "requested_completion_budget": budget, **evidence}
        if self.args.feedback_mode == "goals-observations-v4":
            user = next(m["content"] for m in messages if m["role"] == "user")
            record["teacher_goals"] = self.goal_by_prompt[user]
        self.calls.append(record)
        with (self.folder / "generation-calls.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        return web.json_response({"id": f"self-{len(self.calls)}", "object": "chat.completion",
            "created": int(time.time()), "model": "local-self",
            "choices": [{"index": 0, "message": message,
                         "finish_reason": "length" if not complete else "tool_calls" if tools_out else "stop"}],
            "usage": {"prompt_tokens": len(prefix), "completion_tokens": len(ids),
                      "total_tokens": len(prefix) + len(ids)}})

    async def collect(self, step, cohort, names):
        self.folder = self.args.output / f"step-{step:04d}" / cohort
        self.folder.mkdir(parents=True, exist_ok=False)
        self.calls = []
        print(json.dumps({"stage": "collection_start", "step": step, "cohort": cohort}), flush=True)
        env = environment_config(8, "limited_zapier")
        env["agent"]["max_output_tokens"] = 12288
        env["taskset"]["task"].update(capture_actions=True, deterministic_world=True)
        config = EvalConfig.model_validate({"env": env, "model": "local-self",
            "client": {"type": "eval", "base_url": "http://127.0.0.1:18774/v1", "api_key_var": "SELF_LOCAL_KEY"},
            "sampling": {"temperature": 0.0, "max_tokens": self.args.reply_limit},
            "num_tasks": len(names), "task_keys": names, "num_rollouts": 2,
            "max_concurrent": 32, "shuffle": False, "output_dir": self.folder / "native",
            "run": {"name": f"{self.args.run_name}-step{step}-{cohort}", "dir": "."},
            "push": False, "rich": None, "serve": None})
        episodes = await run_eval(config)
        summary = summarize(episodes, names, 2)
        write(self.folder / "outcomes.json", summary)
        if summary["errors"]:
            raise ValueError("native/scoring errors: retained outcomes.json; stop without substituting scores")
        if any(call["policy_version"] != self.rollout.version for call in self.calls):
            raise ValueError("stale rollout adapter")
        excluded = []
        rows, unexecuted = collect_transitions(self.calls, episodes, self.tokenizer,
                                              f"step{step}", excluded)
        write(self.folder / "all-complete-transitions.json", rows)
        rows, overlong = admit_sequences(rows, self.args.context_limit)
        write(self.folder / "transitions.json", rows)
        write(self.folder / "training-admission.json", {"incomplete_generations": excluded,
              "overlong_transitions": overlong, "terminal_unexecuted_tool_ids": unexecuted,
              "admitted": len(rows), "eligible_trajectories": len({r["trajectory_id"] for r in rows}),
              "truncated_training_sequences": 0, "training_allowed": cohort == "train"})
        if not rows:
            raise ValueError("empty complete transition population")
        if step == 0:
            self.baselines[cohort] = summary
        result = {"step": step, "cohort": cohort, "summary": summary,
                  "compared_to_step0": comparisons(self.baselines[cohort], summary),
                  "admitted_transitions": len(rows), "incomplete_generations": len(excluded),
                  "overlong_transitions": len(overlong)}
        self.curve.append(result)
        write(self.args.output / "outcome-curve.json", self.curve)
        print(json.dumps({"stage": "collection_complete", "step": step, "cohort": cohort,
                          "episodes": len(episodes), "metrics": summary["metrics"]}), flush=True)
        return rows

    def prepare_samples(self, step, rows):
        count = self.args.updates_per_round * self.args.batch_size
        self.samples[step] = sample_transitions(rows, count, seed=17 + step)
        write(self.args.output / f"step-{step:04d}" / "sampled-training-transitions.json", self.samples[step])

    def fixed_diagnostic(self, step):
        trainer, model = self.trainer, self.model
        was_training = model.training
        model.eval()
        trainer.diagnostic = True
        values = []
        try:
            with torch.no_grad():
                for row in self.fixed_rows:
                    loss = trainer.compute_loss(model, {"transitions": [row]})
                    values.append({"total": float(loss), **trainer.last_self_metrics})
        finally:
            trainer.diagnostic = False
            model.set_adapter("default")
            model.train(was_training)
        result = {"step": step, "rows": len(values),
                  "means": {key: sum(v[key] for v in values) / len(values) for key in values[0]},
                  "scope": "fixed development transitions, current EMA teacher; not a fixed-teacher loss"}
        if not all(math.isfinite(v) for v in result["means"].values()):
            raise ValueError("nonfinite fixed diagnostic")
        write(self.args.output / f"step-{step:04d}" / "fixed-loss.json", result)

    async def boundary(self, step):
        checkpoint = self.args.output / "checkpoints" / f"checkpoint-{step}"
        required = ["adapter_model.safetensors", "self_teacher/adapter_model.safetensors",
                    "optimizer.pt", "scheduler.pt", "trainer_state.json", "rng_state.pth"]
        if any(not (checkpoint / name).is_file() for name in required):
            raise ValueError("incomplete checkpoint: preserve and stop")
        state = json.loads((checkpoint / "trainer_state.json").read_text())
        if state["global_step"] != step:
            raise ValueError("checkpoint step mismatch")
        optimizer = torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=True)
        moments = {int(value["step"].item()) for value in optimizer["state"].values()}
        if moments != {step}:
            raise ValueError(f"optimizer moment steps reset or disagree: {moments}")
        write(checkpoint / "campaign-checkpoint.json", {"step": step, "optimizer_moment_steps": sorted(moments),
              "file_sha256": {name: hashlib.sha256((checkpoint / name).read_bytes()).hexdigest()
                              for name in required}, "exact_resume_qualified": False})
        await self.rollout.refresh()
        self.model.eval()
        parity = await self.rollout.score_probe(self.model, self.probe_prefix, self.probe_targets)
        train_rows = await self.collect(step, "train", self.train_names)
        await self.collect(step, "development", self.dev_names)
        await self.rollout.suspend_for_update()
        self.fixed_diagnostic(step)
        if step < self.args.steps:
            self.prepare_samples(step, train_rows)
        write(self.args.output / "progress.json", {"stage": "checkpoint_evaluated", "step": step,
              "optimizer_updates": len(self.updates), "policy_version": self.rollout.version,
              "actor_rollout_delta": parity["max_abs_delta"], "checkpoint": str(checkpoint)})

    async def run(self):
        args = self.args
        args.output.mkdir(parents=True, exist_ok=False)
        write(args.output / "run-inputs.json", {k: str(v) if isinstance(v, Path) else v
                                               for k, v in vars(args).items()})
        self.train_names = resolve_pilot_tasks(args.protocol, args.heldout)
        protocol = json.loads(args.protocol.read_text())
        self.dev_names = protocol["cohorts"]["development"]["task_names"]
        heldout = set(json.loads(args.heldout.read_text())["task_names"])
        if set(self.dev_names) & (heldout | set(self.train_names)) or len(set(self.dev_names)) != 24:
            raise ValueError("development cohort identity/leakage failure")
        if args.qualify:
            self.train_names = ["simple.gmail_reply_meeting_request", "simple.slack_ceo_announcement",
                                "hr.offboarding_automation", "finance.credit_limit_review"]
            self.dev_names = [protocol["cohorts"]["development"]["task_names"][0]]
        self.goal_by_prompt = {}
        if args.feedback_mode == "goals-observations-v4":
            bindings = json.loads(args.task_prompt_bindings.read_text())
            goal_admission = {}
            for name in self.train_names + self.dev_names:
                binding = bindings[name]
                initial_digest = hashlib.sha256(json.dumps(binding['initial_state'], sort_keys=True,
                                                separators=(',', ':')).encode()).hexdigest()
                if initial_digest != binding['initial_state_sha256']:
                    raise ValueError('initial source authority digest mismatch')
                context = load_goal_context(name, binding['initial_state'], binding['full_prompt'])
                if hashlib.sha256(binding["prompt"].encode()).hexdigest() != binding["sha256"]:
                    raise ValueError("task prompt binding digest mismatch")
                context["actual_request_sha256"] = binding["sha256"]
                goal_admission[name] = {'authority_status': context['authority_status'],
                                        'errors': context['authority_errors'], 'manifest_sha256': context['sha256']}
                self.goal_by_prompt[binding["prompt"]] = context if context['authority_status'] == 'verified' else None
            write(args.output / "teacher-goal-contexts.json", self.goal_by_prompt)
            write(args.output / "goal-admission.json", goal_admission)
        os.environ["POSTTRAIN_VERIFIERS_PREINSTALLED"] = "1"
        os.environ["SELF_LOCAL_KEY"] = "EMPTY"
        configure_preinstalled_runtime(interpreter=sys.executable)
        set_seed(17)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        self.model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16,
            device_map={"": "cuda:0"}, attn_implementation="sdpa", local_files_only=True)
        self.model = get_peft_model(self.model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
            target_modules="all-linear", exclude_modules=r".*visual.*", task_type="CAUSAL_LM"))
        if os.environ.get("VLLM_BATCH_INVARIANT") != "1":
            raise ValueError("qualified invariant vLLM path required")
        self.rollout = await LocalVLLMRollout.create(args.model, self.model, self.tokenizer,
                                                   args.output, args.context_limit, 32, 16)
        if args.qualify_capacity:
            from self_context_qualification import qualify_context

            args.episode_concurrency, args.server_sequences, args.kv_cache_gib = 32, 32, 16
            await qualify_context(args, self.model, self.tokenizer, self.rollout)
            return
        app = web.Application(client_max_size=16 * 1024 * 1024)
        app.router.add_post("/v1/chat/completions", self.completions)
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", 18774).start()
        try:
            train_rows = await self.collect(0, "train", self.train_names)
            dev_rows = await self.collect(0, "development", self.dev_names)
            self.fixed_rows = sample_transitions(dev_rows, 4 if args.qualify else 16, seed=999)
            write(args.output / "fixed-development-transitions.json", self.fixed_rows)
            self.probe_prefix = train_rows[0]["student_prefix_ids"]
            self.probe_targets = train_rows[0]["action_ids"][:16]
            await self.rollout.score_probe(self.model, self.probe_prefix, self.probe_targets)
            await self.rollout.suspend_for_update()
            self.trainer = CampaignTrainer(model=self.model, processing_class=self.tokenizer,
                sequence_limit=args.context_limit, train_dataset=Dataset.from_list(train_rows[:1]),
                data_collator=lambda rows: {"transitions": rows},
                args=SFTConfig(output_dir=str(args.output / "checkpoints"), max_steps=args.steps,
                    per_device_train_batch_size=1, gradient_accumulation_steps=args.batch_size,
                    learning_rate=1e-5, lr_scheduler_type="linear", warmup_steps=0,
                    bf16=True, gradient_checkpointing=True,
                    gradient_checkpointing_kwargs={"use_reentrant": False},
                    remove_unused_columns=False, dataset_kwargs={"skip_prepare_dataset": True},
                    max_length=None, report_to="none", logging_steps=1,
                    save_strategy="no", seed=17, data_seed=17))
            self.trainer.train_dataset = RoundDataset(self)
            self.audit = UpdateAudit(self.trainer)
            self.trainer.add_callback(self.audit)
            self.trainer.add_callback(CampaignCallback(self, asyncio.get_running_loop()))
            self.trainer.save_model(str(args.output / "step-0000" / "student-and-teacher"))
            self.fixed_diagnostic(0)
            self.prepare_samples(0, train_rows)
            started = time.perf_counter()
            result = await asyncio.to_thread(self.trainer.train)
            if len(self.updates) != args.steps or len(self.audit.updates) != args.steps:
                raise ValueError("optimizer/EMA update coverage differs from planned steps")
            write(args.output / "campaign-receipt.json", {"completed": True, "steps": args.steps,
                  "batch_size": args.batch_size, "rounds": len(self.schedule),
                  "transition_exposures": args.steps * args.batch_size,
                  "train_metrics": result.metrics, "wall_s_after_initial_collection": time.perf_counter() - started,
                  "ema_verified": True, "optimizer_updates": self.updates,
                  "refreshes": self.rollout.refreshes, "qualified": args.qualify,
                  "development_used_for_updates": False, "heldout_used": False,
                  "external_model_calls": 0, "truncated_training_sequences": 0})
        finally:
            await runner.cleanup()
            await self.rollout.aclose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--updates-per-round", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--context-limit", type=int, choices=(16384,), default=16384)
    parser.add_argument("--qualify", action="store_true")
    parser.add_argument("--qualify-capacity", action="store_true")
    parser.add_argument("--reply-limit", type=int, choices=(512, 2048), default=512)
    parser.add_argument("--feedback-mode", choices=("minimal", "goals-observations-v4"), default="minimal")
    parser.add_argument("--task-prompt-bindings", type=Path, default=Path('/workspace/driver/task-prompt-bindings.json'))
    args = parser.parse_args()
    schedule(args.steps, args.updates_per_round, args.batch_size)
    if args.batch_size != 16 or (not args.qualify and (args.steps, args.updates_per_round) != (256, 32)):
        parser.error("authorized campaign is256 updates/32 per round/batch16")
    asyncio.run(Campaign(args).run())


if __name__ == "__main__":
    main()
