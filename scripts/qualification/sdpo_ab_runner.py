"""Lean SDPO A/B runner on the workstation (plan D10, D11, D12, D16, D20).

Adapted from `sdpo_agent_ablation.py`: the same vLLM rollout engine
(`LocalVLLMRollout`), the same OpenAI-compatible local endpoint feeding the
native Verifiers environment, and the same TRL top-k + tail reverse-KL helper
with a clamped importance weight and an EMA teacher adapter. Changes:

- rollouts are token-consistent: each turn's prompt is the previous prompt plus
  the sampled reply plus the rendered tool results, so one episode is one token
  sequence and every assistant token is trained (tool results masked);
- the teacher sequence is the same episode with the D5 teacher first message and
  optional per-turn notes in the tool-result messages;
- LoRA rank 64, alpha 32, all linear layers; constant LR; one task (8 episodes)
  per optimizer step; the system prompt is the unchanged native one.

Phases: smoke, base, controls (with Gate B), arm (training + evaluation).
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import math
import os
import random
import sys
import time
import traceback
import uuid
from collections import Counter, defaultdict
from pathlib import Path

ROOT_PORT = 18774
DUMMY = "§SDPO_AB_DUMMY§"


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, default=str))
    tmp.replace(path)


def log(**fields):
    if SF_FINAL:  # S-final F12: a closed stdout must not end the run (R0 lost its 4th round to EPIPE)
        from sdpo_final import safe_log

        return safe_log(**fields)
    print(json.dumps({"t": round(time.time(), 1), **fields}, default=str), flush=True)


# ------------------------------------------------------------------ token-consistent rendering

THINKING = False  # `--thinking on` (D97): the chat template's enable_thinking for every rendered prompt; off = as before
SF_FINAL = False  # `--sf-final` (S-final, Phase 1 item 9): set in main; read by log() (F12) and the re-render check (F11)
SF_TRAINING_SAMPLING = "training"  # `--sampling training` (S-final F7): the training rollouts' sampling, for scoring
THINK_CLOSE = "</think>"


def render_delta(tokenizer, new_messages, thinking=None):
    """Tokens that follow an assistant turn's `<|im_end|>`: the new messages and the next generation prompt."""
    import copy

    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
                {"role": "assistant", "content": DUMMY}] + copy.deepcopy(list(new_messages))
    for message in messages:
        for call in message.get("tool_calls") or []:
            function = call.get("function", call)
            if isinstance(function.get("arguments"), str):
                function["arguments"] = json.loads(function["arguments"])
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=THINKING if thinking is None else thinking)
    anchor = DUMMY + tokenizer.eos_token
    index = text.index(anchor) + len(anchor)
    return tokenizer.encode(text[index:], add_special_tokens=False)


def first_prompt(tokenizer, messages, tools, thinking=None):
    thinking = THINKING if thinking is None else thinking
    try:
        from self_feedback_native import prompt_ids
    except ImportError:  # CPU-only checks without torch: the same template call, returned as a list
        import copy

        messages = copy.deepcopy(messages)
        for message in messages:
            for call in message.get("tool_calls", []):
                function = call["function"]
                if isinstance(function["arguments"], str):
                    function["arguments"] = json.loads(function["arguments"])
        encoded = tokenizer.apply_chat_template(messages, tools=tools, tokenize=True, add_generation_prompt=True,
                                                enable_thinking=thinking, return_dict=True)
        ids = encoded["input_ids"]
        return list(ids[0] if ids and isinstance(ids[0], list) else ids)
    return prompt_ids(tokenizer, messages, tools, enable_thinking=thinking)


def visible_reply(raw):
    """`--thinking on`: split a decoded reply into (reasoning, visible text, closed). The generation prompt already
    opened `<think>`, so the model writes its reasoning, `</think>`, then the reply the environment should see. A
    reply without `</think>` (cut inside the reasoning, or no reasoning at all) is passed on whole, as before."""
    if THINK_CLOSE not in raw:
        return "", raw, False
    reasoning, visible = raw.split(THINK_CLOSE, 1)
    return reasoning, visible.lstrip("\n"), True


# ------------------------------------------------------------------ rollout engine

def sampling_options(temperature, top_p, max_tokens, eos_token_id, seed, extra=None):
    """Keyword arguments of the rollout's vLLM `SamplingParams`. `extra` adds the fields beyond temperature and top_p
    (correction b, `--sampling card`: top_k, min_p, presence_penalty, repetition_penalty). The rollout calls
    vLLM's engine directly (AsyncLLM.generate with SamplingParams), not an OpenAI-compatible server, so every field
    reaches the sampler as a SamplingParams argument; no request extra-body is involved. Without `extra` the
    arguments are exactly the ones every earlier run used."""
    options = {"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens, "stop_token_ids": [eos_token_id],
               "logprobs": 1, "seed": seed}
    for key, value in (extra or {}).items():
        if key in options:
            raise ValueError(f"extra sampling field {key!r} would replace a fixed one")
        options[key] = value
    return options


def make_rollout_class():
    import torch
    from self_feedback_vllm import LocalVLLMRollout
    from trl.generation.async_vllm_session import AsyncVllmSession
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    class Rollout(LocalVLLMRollout):
        temperature, top_p, reply_limit = 1.0, 1.0, 4096

        @classmethod
        async def build(cls, model_path, model, tokenizer, output, context_limit, sequences, kv_cache_gib):
            self = cls(tokenizer, output)
            from posttrain.common.cuda import activate_cuda_toolkit

            activate_cuda_toolkit(torch)
            engine_args = AsyncEngineArgs(
                model=model_path, dtype="bfloat16", tensor_parallel_size=1, gpu_memory_utilization=0.35,
                max_model_len=context_limit, max_num_seqs=sequences, max_num_batched_tokens=context_limit,
                kv_cache_memory_bytes=kv_cache_gib * 1024**3, block_size=128, enable_lora=True, max_loras=1,
                max_lora_rank=64, enable_prefix_caching=True, enable_chunked_prefill=True, enforce_eager=False,
                language_model_only=True, seed=17, generation_config="vllm",
                override_generation_config={"eos_token_id": tokenizer.eos_token_id}, disable_log_stats=True,
                enable_sleep_mode=True, worker_extension_cls="self_feedback_vllm.SelfQualificationWorkerExtension")
            self.engine = AsyncLLM.from_engine_args(engine_args)
            (output / "engine-config.txt").write_text(repr(self.engine.vllm_config))

            async def synchronize(_version):
                await self._load_student(model)

            self.session = AsyncVllmSession(self.engine, synchronize, sleep_level=1)
            await self.refresh()
            return self

        async def _load_student(self, model):
            from vllm.lora.request import LoRARequest

            version = self.version + 1
            directory = self.output / "rollout-policy" / f"v{version}"
            directory.mkdir(parents=True)
            if model.active_adapter != "default":
                raise ValueError("rollout export requires the active student adapter")
            model.save_pretrained(directory, selected_adapters=["default"], safe_serialization=True)
            weights = directory / "adapter_model.safetensors"
            digest = hashlib.sha256(weights.read_bytes()).hexdigest()
            if self.request is not None and not await self.engine.remove_lora(self.request.lora_int_id):
                raise ValueError("engine did not remove the previous policy adapter")
            request = LoRARequest(f"ab-policy-v{version}", version + 1, str(directory))
            if not await self.engine.add_lora(request):
                raise ValueError("engine did not acknowledge the student adapter")
            if not await self.engine.reset_prefix_cache():
                raise ValueError("engine did not acknowledge prefix-cache invalidation")
            self.request, self.version = request, version
            self.refreshes.append({"policy_version": version, "weights_sha256": digest, "path": str(directory)})

        seed_salt = None
        sample_counter = 0
        extra_sampling = {}  # set per collection by Runtime.collect; empty except for --sampling card

        async def sample(self, prefix, max_tokens, seed=None):
            # precheck: an explicit per-request seed; every other caller passes none and keeps the salted counter
            if seed is None and self.seed_salt is not None:
                self.sample_counter += 1
                seed = request_seed(self.seed_salt, self.sample_counter)
            result = await self._generate(prefix, SamplingParams(**sampling_options(
                self.temperature, self.top_p, max_tokens, self.tokenizer.eos_token_id, seed, self.extra_sampling)))
            completion = result.outputs[0]
            ids = list(completion.token_ids)
            if list(result.prompt_token_ids) != prefix:
                raise ValueError("engine changed the supplied prompt")
            logps = [float(v[t].logprob) for v, t in zip(completion.logprobs or [], ids, strict=True)]
            return ids, logps, completion.finish_reason

        teacher_loads = 0
        student_request = None

        async def load_teacher(self, model):
            """R1 (D74): serve the EMA teacher adapter ("sdpo_teacher") instead of the student while the teacher
            continues at mistake points; `restore_student` puts the same student request back."""
            from vllm.lora.request import LoRARequest

            self.teacher_loads += 1
            directory = self.output / "rollout-teacher" / f"v{self.version}-{self.teacher_loads}"
            directory.mkdir(parents=True)
            model.save_pretrained(directory, selected_adapters=["sdpo_teacher"], safe_serialization=True)
            path = directory / "sdpo_teacher" if (directory / "sdpo_teacher").is_dir() else directory
            if not (path / "adapter_model.safetensors").is_file():
                raise ValueError("teacher export has no adapter weights")
            if not await self.engine.remove_lora(self.request.lora_int_id):
                raise ValueError("engine did not remove the student adapter")
            request = LoRARequest(f"ab-teacher-{self.teacher_loads}", 1_000_000 + self.teacher_loads, str(path))
            if not await self.engine.add_lora(request):
                raise ValueError("engine did not acknowledge the teacher adapter")
            if not await self.engine.reset_prefix_cache():
                raise ValueError("engine did not acknowledge prefix-cache invalidation")
            self.student_request, self.request = self.request, request
            self.refreshes.append({"teacher_load": self.teacher_loads, "path": str(path),
                                   "weights_sha256": hashlib.sha256((path / "adapter_model.safetensors").read_bytes())
                                   .hexdigest()})

        async def restore_student(self):
            if self.student_request is None:
                return
            if not await self.engine.remove_lora(self.request.lora_int_id):
                raise ValueError("engine did not remove the teacher adapter")
            if not await self.engine.add_lora(self.student_request):
                raise ValueError("engine did not acknowledge the student adapter")
            if not await self.engine.reset_prefix_cache():
                raise ValueError("engine did not acknowledge prefix-cache invalidation")
            self.request, self.student_request = self.student_request, None

    return Rollout


# ------------------------------------------------------------------ agent endpoint

class Agent:
    """Serves the environment's chat requests from the rollout engine, one token chain per episode."""

    align_budget = False  # handover recipe: the system prompt promises at most 8 assistant turns
    prompt_suffix = None  # correction b, collect phase: text appended to every request's task prompt

    def __init__(self, tokenizer, rollout, context_limit, reply_limit):
        self.tokenizer, self.rollout = tokenizer, rollout
        self.context_limit, self.reply_limit = context_limit, reply_limit
        self.reset()

    def reset(self):
        self.prompt_queue = {}
        self.chains, self.by_tool_id, self.by_text = {}, {}, {}
        self.audit = {"calls": 0, "chain_breaks": 0, "rerender_checks": 0, "rerender_equal": 0}
        self.check_rerender = 0

    def _new_chain(self, messages, tools):
        chain_id = uuid.uuid4().hex
        prefix = first_prompt(self.tokenizer, messages, tools)
        chain = {"id": chain_id, "tools": tools, "messages0": messages, "segments": [
            {"kind": "context", "ids": prefix, "messages": None}], "tokens": list(prefix)}
        self.chains[chain_id] = chain
        return chain

    async def completions(self, request):
        from aiohttp import web

        body = await request.json()
        messages, tools = request_messages(body["messages"], self.align_budget), body.get("tools", [])
        messages = with_prompt_suffix(messages, self.prompt_suffix)
        assistants = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
        self.audit["calls"] += 1
        if not assistants:
            messages, context_id = override_prompt(messages, self.prompt_queue)
            chain = self._new_chain(messages, tools)
            chain["context_id"] = context_id
        else:
            last = assistants[-1]
            ids = [c.get("id") for c in messages[last].get("tool_calls") or []]
            chain_id = next((self.by_tool_id[i] for i in ids if i in self.by_tool_id), None)
            if chain_id is None:
                key = json.dumps([messages[1].get("content"), len(assistants), messages[last].get("content")])
                chain_id = self.by_text.get(key)
            if chain_id is None:
                self.audit["chain_breaks"] += 1
                chain = self._new_chain(messages, tools)
                chain["rerendered_at_turn"] = len(assistants) + 1
            else:
                chain = self.chains[chain_id]
                new = messages[last + 1:]
                delta = render_delta(self.tokenizer, new)
                chain["segments"].append({"kind": "context", "ids": delta, "messages": new})
                chain["tokens"] += delta
                if self.check_rerender > 0:
                    self.check_rerender -= 1
                    self.audit["rerender_checks"] += 1
                    self.audit["rerender_equal"] += int(first_prompt(self.tokenizer, messages, tools)
                                                        == chain["tokens"])
                    if SF_FINAL:  # S-final F11: a chain whose fresh render differs is excluded from training
                        chain["rerender_mismatch"] = first_prompt(self.tokenizer, messages, tools) != chain["tokens"]
        prefix = list(chain["tokens"])
        budget = min(self.reply_limit, self.context_limit - len(prefix))
        if budget < 1:
            action, logps, finish = [], [], "context_limit"
        else:
            action, logps, finish = await self.rollout.sample(prefix, budget)
        complete = finish == "stop"
        limit = None if complete else ("context_limit" if finish == "context_limit"
                                       else "reply_limit" if budget == self.reply_limit else "context_limit")
        raw = self.tokenizer.decode(action, skip_special_tokens=True)
        shown = raw
        if THINKING:  # the reasoning stays in the chain (the model's own tokens) but never reaches the environment
            reasoning, shown, closed = visible_reply(raw)
        native, parser = tool_call_parser(self.tokenizer, messages, tools)
        parsed = parser.extract_tool_calls(shown, native) if complete else None
        calls = parsed_calls(parsed)
        message = {"role": "assistant", "content": (parsed.content if parsed else shown) or ""}
        if calls:
            message["tool_calls"] = calls
        segment = {"kind": "action", "ids": action, "logprobs": logps, "complete": complete,
                   "limit_reason": limit, "policy_version": self.rollout.version, "raw": raw,
                   "tool_call_ids": [c["id"] for c in calls]}
        if THINKING:
            segment.update(thinking_closed=closed, reasoning_chars=len(reasoning), visible_chars=len(shown))
        chain["segments"].append(segment)
        chain["tokens"] += action
        if action and action[-1] != self.tokenizer.eos_token_id:
            # an unfinished reply is closed for the history but never trained
            chain["segments"].append({"kind": "context", "ids": [self.tokenizer.eos_token_id], "messages": None})
            chain["tokens"].append(self.tokenizer.eos_token_id)
        for call in calls:
            self.by_tool_id[call["id"]] = chain["id"]
        key = json.dumps([messages[1].get("content") if len(messages) > 1 else None,
                          len(assistants) + 1, message["content"]])
        self.by_text[key] = chain["id"]
        return web.json_response({
            "id": f"ab-{uuid.uuid4().hex}", "object": "chat.completion", "created": int(time.time()),
            "model": "local", "choices": [{"index": 0, "message": message,
                                           "finish_reason": "length" if not complete else
                                           "tool_calls" if calls else "stop"}],
            "usage": {"prompt_tokens": len(prefix), "completion_tokens": len(action),
                      "total_tokens": len(prefix) + len(action)}})


def tool_call_parser(tokenizer, messages, tools):
    """The rollout's tool-call parser for one request: (vLLM ChatCompletionRequest, Qwen3EngineToolParser)."""
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.tool_parsers.qwen3_engine_tool_parser import Qwen3EngineToolParser

    native = ChatCompletionRequest(model="local", messages=messages, tools=tools)
    return native, Qwen3EngineToolParser(tokenizer, tools=native.tools)


def parsed_calls(parsed):
    """The parsed tool calls in the OpenAI shape, each with an id."""
    calls = [c.model_dump() for c in parsed.tool_calls] if parsed and parsed.tools_called else []
    for call in calls:
        if not call.get("id"):
            call["id"] = f"call-{uuid.uuid4().hex[:16]}"
    return calls


def request_messages(messages, align):
    """The request's messages, with the handover's turn-budget sentence when `align` is on: True for its 8-turn
    training wording, 20 for its heldout62 evaluation wording (979c8fe0:sdpo_agent_ablation.py:80-85)."""
    if not align:
        return messages
    from sdpo_ab_handover import aligned_messages

    messages = aligned_messages(messages)
    if align == 20:
        for message in messages:
            if message["role"] == "system":
                message["content"] = message["content"].replace(
                    "a budget of at most 8 assistant turns", "a budget of at most 20 assistant turns")
    return messages


# The handover's heldout62 scoring (979c8fe0:sdpo_pool124_config.py collection_budget("heldout62") and
# sdpo_agent_ablation.py collect, temperature 0): greedy, 20 turns, 65,536-token context, 8,192-token replies,
# 32,768-token episodes, 5 replicas.
HANDOVER_EVALUATION = {"temperature": 0.0, "top_p": 1.0, "turns": 20, "context": 65536, "reply": 8192,
                       "episode": 32768, "replicas": 5}


def join_chains(agent, records):
    """Episode id -> its token chain, matched by tool-call ids, else by the replies' text."""
    claimed, joined = set(), {}
    for record in records:
        trace = record["traces"][0]
        assistants = [n["message"] for n in trace["nodes"]
                      if (n.get("message") or {}).get("role") == "assistant"]
        chain_id = None
        for message in assistants:
            for call in message.get("tool_calls") or []:
                chain_id = agent.by_tool_id.get(call.get("id"))
                if chain_id:
                    break
            if chain_id:
                break
        if chain_id is None:
            user = next((n["message"]["content"] for n in trace["nodes"]
                         if (n.get("message") or {}).get("role") == "user"), None)
            if user is None:  # D49: an infrastructure failure never started (no messages): nothing to join
                continue
            # correction b: the model saw the task prompt with the suffix; the native trace keeps the original
            user = suffixed(user, getattr(agent, "prompt_suffix", None)) if isinstance(user, str) else user
            for candidate in agent.chains.values():
                actions = [s for s in candidate["segments"] if s["kind"] == "action"]
                if (candidate["id"] not in claimed and candidate["messages0"][1].get("content") == user
                        and len(actions) == len(assistants)
                        and all((a["raw"] or "").strip()[:200] in json.dumps(m.get("content") or "")
                                or not (a["raw"] or "").strip() for a, m in zip(actions, assistants))):
                    chain_id = candidate["id"]
                    break
        if chain_id is None or chain_id in claimed:
            continue
        chain = agent.chains[chain_id]
        if sum(s["kind"] == "action" for s in chain["segments"]) != len(assistants) or chain.get("rerendered_at_turn"):
            chain["mismatch"] = True
        claimed.add(chain_id)
        joined[record["id"]] = chain
    return joined


# ------------------------------------------------------------------ sequences

def student_sequence(chain):
    ids, targets, positions, old, turns = [], [], [], [], []
    turn = 0
    for segment in chain["segments"]:
        if segment["kind"] == "action":
            turn += 1
            start = len(ids)
            for k, token in enumerate(segment["ids"]):
                positions.append(len(ids) + k - 1)
                targets.append(token)
                turns.append(turn)
            old += segment["logprobs"]
            ids += segment["ids"]
            segment["_span"] = (start, len(ids))
        else:
            ids += segment["ids"]
    return {"ids": ids, "positions": positions, "targets": targets, "old": old, "turns": turns}


def teacher_sequence(tokenizer, chain, context):
    """The student's episode with the teacher's first message and per-turn notes.

    A note of a later turn is appended to that turn's last tool result (the default), or, when the context says
    note_role "user" (SDPO closure round 6, design suffix U), given as its own user message after that turn's tool
    results: inside the tool result the 4B teacher read a note-only fact as the tool's own output (21 Nov, S1PR)."""
    import copy

    first = copy.deepcopy(chain["messages0"])
    user_index = next(i for i, m in enumerate(first) if m["role"] == "user")
    content = context["first_message"]
    if 1 in context["notes"]:
        content += "\n\n" + context["notes"][1]
    first[user_index] = {"role": "user", "content": content}
    ids = first_prompt(tokenizer, first[:user_index + 1], chain["tools"])
    positions, targets, turn = [], [], 0
    for index, segment in enumerate(chain["segments"]):
        if index == 0:
            continue
        if segment["kind"] == "action":
            turn += 1
            for k, token in enumerate(segment["ids"]):
                positions.append(len(ids) + k - 1)
                targets.append(token)
            ids += segment["ids"]
        elif segment["messages"] is not None and (turn + 1) in context["notes"]:
            messages = copy.deepcopy(segment["messages"])
            if context.get("note_role") == "user":
                messages.append({"role": "user", "content": context["notes"][turn + 1]})
            else:
                target = max(i for i, m in enumerate(messages) if m["role"] in ("tool", "user"))
                body = messages[target].get("content")
                body = body if isinstance(body, str) else json.dumps(body)
                messages[target]["content"] = body + "\n\n" + context["notes"][turn + 1]
            ids += render_delta(tokenizer, messages)
        else:
            ids += segment["ids"]
    return {"ids": ids, "positions": positions, "targets": targets}


def restrict_to_calls(tokenizer, student, teacher):
    """D36: keep only the tokens of tool-call bodies (after <tool_call>, through </tool_call>) in the loss.

    The facts are about actions; on free text the student, who cannot see them, can only copy the teacher's
    generic reconsidering (arm A r2: 0.13 -> 1.38 hesitation phrases per transfer reply, 31% reply-limit hits).
    The <tool_call> opener itself is excluded, so the act-now-or-keep-thinking choice is not trained either."""
    keep = call_body_indices(tokenizer, student)
    if not keep:
        return None, None
    return pick(student, keep, ("positions", "targets", "old", "turns")), pick(teacher, keep, ("positions", "targets"))


def text_anchor(tokenizer, student):
    """D38: the assistant tokens outside tool-call bodies (free text and the <tool_call> opener), held at the
    untrained model. Masked arm A still drifted on them (transfer hesitation 0.13 -> 0.43 per reply) through
    the shared weights; anchoring them is the standard KL-to-reference term, applied only where there is no
    feedback to distil."""
    calls = set(call_body_indices(tokenizer, student))
    keep = [i for i in range(len(student["targets"])) if i not in calls]
    return pick(student, keep, ("positions", "targets", "old", "turns")) if keep else None


def call_body_indices(tokenizer, student):
    """Indices into the student's assistant tokens that lie after <tool_call> through </tool_call>."""
    opener, closer = (tokenizer.convert_tokens_to_ids(t) for t in ("<tool_call>", "</tool_call>"))
    if not isinstance(opener, int) or not isinstance(closer, int) or opener == tokenizer.unk_token_id:
        raise ValueError("tool-call tags are not single tokens")
    keep, inside, turn = [], False, None
    for index, (token, t) in enumerate(zip(student["targets"], student["turns"])):
        if t != turn:
            inside, turn = False, t
        if token == opener:
            inside = True
        elif inside:
            keep.append(index)
            inside = token != closer
    return keep


def pick(sequence, keep, keys):
    return {**sequence, **{k: [sequence[k][i] for i in keep] for k in keys}}


def reasoning_and_call_indices(tokenizer, student):
    """Arms 2/3 (plan of 15 Nov, Ali's default 3: train reasoning and tool calls, not the visible text): indices into
    the student's assistant tokens that are reasoning (through `</think>`) or tool calls (`<tool_call>` through
    `</tool_call>`, both tags). Left out: visible free text, the newlines right after `</think>`, the end-of-turn
    token. A reply without `</think>` is all reasoning (cut inside the thinking). Same classes as sdpo_a3's
    token_classes ("decision" kept only when it opens a call)."""
    think_close, opener, closer = (tokenizer.convert_tokens_to_ids(t) for t in ("</think>", "<tool_call>", "</tool_call>"))
    if any(not isinstance(x, int) or x == tokenizer.unk_token_id for x in (think_close, opener, closer)):
        raise ValueError("</think> and the tool-call tags must be single tokens")
    newlines = set(tokenizer.encode("\n", add_special_tokens=False) + tokenizer.encode("\n\n", add_special_tokens=False))
    keep, turn, state = [], None, None
    for index, (token, t) in enumerate(zip(student["targets"], student["turns"])):
        if t != turn:
            turn, state = t, "reasoning"
        if state == "reasoning":
            keep.append(index)
            if token == think_close:
                state = "await_decision"
            continue
        if state == "await_decision":
            if token in newlines:
                continue
            state = "call" if token == opener else "text"
        elif token == opener:
            state = "call"
        if state == "call":
            keep.append(index)
            if token == closer:
                state = "text"
    return keep


# ------------------------------------------------------------------ recipe R: the decision to act

ANCHORED_LOSSES = ("calls+anchor", "calls+anchor+decision")
DECISION_HARM_KINDS = ("prohibited", "changed")  # located "do not make this call" facts
DECISION_MISSING_KIND, DECISION_MISSING_LOCATION = "missing", "end_of_attempt"  # the "Not done" lines
DECISION_MASK_LOGIT = -1.0e4  # finite, so the masked token has probability 0 without inf - inf in the KL


def decision_tokens(tokenizer):
    """Ids of <tool_call> and <|im_end|> (single tokens in Qwen3.5)."""
    opener, end = (tokenizer.convert_tokens_to_ids(t) for t in ("<tool_call>", "<|im_end|>"))
    if (not isinstance(opener, int) or not isinstance(end, int) or opener == tokenizer.unk_token_id
            or end == tokenizer.unk_token_id):
        raise ValueError("decision tokens are not single tokens")
    return opener, end


def ends_paragraph(tokenizer, token, cache):
    """Whether `token` closes a paragraph: its text ends in a blank line (the rule `signal_positions` uses, so a
    run of three or more newlines counts as well as the plain "\\n\\n" token)."""
    if token not in cache:
        cache[token] = tokenizer.decode([token]).endswith("\n\n")
    return cache[token]


def rendered_facts(context):
    """The facts the teacher actually saw: the context's facts minus those the engine withheld. `coverage.suppressed`
    lists facts the plain style did not render (a manifest twin of a native line at the same call, an unresolved
    condition, a native check that disagrees, a goal on a successful attempt: D46) and `coverage.dropped` the
    fact ids behind "Not listed: N more" (the 8-line budget). Only a rendered fact is evidence for a decision."""
    coverage = context.get("coverage") or {}
    withheld = set()
    for entry in list(coverage.get("suppressed") or []) + list(coverage.get("dropped") or []):
        withheld.add(entry.get("id") if isinstance(entry, dict) else entry)
    return [f for f in context.get("facts") or [] if isinstance(f, dict) and f.get("id") not in withheld]


def decision_positions(tokenizer, student, context):
    """Recipe R: the act-or-not decision points the engine has evidence about, as indices into the student's
    assistant tokens (before `restrict_to_calls`), each with the only direction the loss may push:

    - harm openers (sign -1: P(<tool_call>) may only fall): the <tool_call> opener of every call the engine located
      a prohibited or changed-record fact at (facts with location at_call, by turn and 0-based call index);
    - final-reply decision points (sign +1: P(<tool_call>) may only rise): when the context carries a "Not done"
      fact (kind missing, location end_of_attempt) and the last reply made no call, that reply's first token,
      each paragraph start (the token after a blank line, `ends_paragraph`) and its <|im_end|> (absent when the
      reply limit cut it).

    Only facts the teacher saw count (`rendered_facts`). Harm facts whose turn has no such call, or whose call
    the engine could not locate (call None), are counted in "unmatched". Returns None without decision points."""
    opener, end = decision_tokens(tokenizer)
    targets, turns, paragraphs = student["targets"], student["turns"], {}
    openers = defaultdict(list)
    for index, (token, turn) in enumerate(zip(targets, turns)):
        if token == opener:
            openers[turn].append(index)
    chosen, unmatched = {}, 0
    facts = rendered_facts(context)
    for fact in facts:
        if fact.get("kind") in DECISION_HARM_KINDS and fact.get("location") == "at_call" and fact.get("turn"):
            call = fact.get("call")
            if call is not None and call < len(openers.get(fact["turn"], [])):
                chosen.setdefault(openers[fact["turn"]][call], (-1, "harm_opener"))
            else:
                unmatched += 1
    missing = any(f.get("kind") == DECISION_MISSING_KIND and f.get("location") == DECISION_MISSING_LOCATION
                  for f in facts)
    final = max(turns) if turns else None
    if missing and final is not None and not openers.get(final):
        indices = [i for i, t in enumerate(turns) if t == final]
        first, last = indices[0], indices[-1]
        chosen.setdefault(first, (1, "final_start"))
        for i in indices[1:]:
            if ends_paragraph(tokenizer, targets[i - 1], paragraphs):
                chosen.setdefault(i, (1, "final_end" if i == last and targets[i] == end else "final_paragraph"))
        if targets[last] == end:
            chosen.setdefault(last, (1, "final_end"))
    if not chosen:
        return None
    order = sorted(chosen)
    return {"indices": order, "signs": [chosen[i][0] for i in order], "reasons": [chosen[i][1] for i in order],
            "unmatched": unmatched, "opener": opener}


def decision_sequence(student, teacher, found):
    """The decision points as a loss sequence: student and teacher positions, the sampled token and its rollout
    log-probability (for the same importance weight as the other terms), sign and reason per point."""
    keep = found["indices"]
    return {"positions": [student["positions"][i] for i in keep],
            "teacher_positions": [teacher["positions"][i] for i in keep],
            "targets": [student["targets"][i] for i in keep], "old": [student["old"][i] for i in keep],
            "signs": list(found["signs"]), "reasons": list(found["reasons"]), "opener": found["opener"],
            "unmatched": found["unmatched"]}


def prepare_sequences(tokenizer, chain, context, loss_tokens):
    """The per-episode loss sequences for `train_arm`: {"student", "teacher", "anchor", "decision",
    "assistant_tokens"}, or the admission reason ("no_assistant_tokens", "no_call_tokens") when the chosen tokens
    leave nothing to train. For every value but calls+anchor+decision, "decision" is None and the other three are
    exactly D36/D38's; under calls+anchor+decision an attempt without any call is kept when it has decision points
    (its "student" then has no targets)."""
    student = student_sequence(chain)
    if not student["targets"]:
        return "no_assistant_tokens"
    teacher = teacher_sequence(tokenizer, chain, context)
    assistant_tokens = len(student["targets"])
    anchor = text_anchor(tokenizer, student) if loss_tokens in ANCHORED_LOSSES else None
    decision = None
    if loss_tokens == "calls+anchor+decision":
        found = decision_positions(tokenizer, student, context)
        decision = decision_sequence(student, teacher, found) if found else None
    if loss_tokens == "reasoning+calls":
        keep = reasoning_and_call_indices(tokenizer, student)
        if not keep:
            return "no_reasoning_or_call_tokens"
        student, teacher = pick(student, keep, ("positions", "targets", "old", "turns")), pick(teacher, keep, ("positions", "targets"))
    if loss_tokens in ("calls", *ANCHORED_LOSSES):
        calls, teacher_calls = restrict_to_calls(tokenizer, student, teacher)
        if calls is None:
            if decision is None:
                return "no_call_tokens"
            # recipe R: an attempt that never called (T4) keeps its decision points; the distillation sequence is
            # empty and the anchor covers every assistant token
            calls = pick(student, [], ("positions", "targets", "old", "turns"))
            teacher_calls = pick(teacher, [], ("positions", "targets"))
        student, teacher = calls, teacher_calls
    return {"student": student, "teacher": teacher, "anchor": anchor, "decision": decision,
            "assistant_tokens": assistant_tokens}


def bernoulli_kl(log_p, log_q):
    """KL(Bernoulli(p) || Bernoulli(q)) from log p and log q, in nats; the float reference for `decision_term`."""
    import math

    p, q = math.exp(log_p), math.exp(log_q)
    one_p, one_q = -math.expm1(log_p), -math.expm1(log_q)
    kl = p * (log_p - log_q)
    if one_p > 0.0:
        kl += one_p * (math.log(one_p) - (math.log(one_q) if one_q > 0.0 else -math.inf))
    return kl


def decision_active(sign, log_p_student, log_p_teacher):
    """The one-sided rule: a point contributes only when the teacher sits on the side its sign allows."""
    if sign > 0:
        return log_p_teacher > log_p_student
    if sign < 0:
        return log_p_teacher < log_p_student
    return True


def decision_term(head, hs, ht, decision, device):
    """Mean over the decision points of the importance-weighted Bernoulli reverse KL between the student's and
    the teacher's P(<tool_call>) at that point, one-sided per point (`decision_active`). The student is never
    pushed toward any particular text token: the loss sees only P(act) against P(anything else)."""
    import torch

    s = head(hs).float()
    with torch.no_grad():
        t = head(ht).float()
    opener = decision["opener"]
    log_s, log_t = s.log_softmax(-1), t.log_softmax(-1)
    lp_s, lp_t = log_s[:, opener], log_t[:, opener]
    column = torch.zeros(s.shape[-1], dtype=torch.bool, device=s.device)
    column[opener] = True
    masked_s, masked_t = s.masked_fill(column, DECISION_MASK_LOGIT), t.masked_fill(column, DECISION_MASK_LOGIT)
    l1_s = torch.logsumexp(masked_s, -1) - torch.logsumexp(s, -1)  # log(1 - p), exact and stable
    l1_t = torch.logsumexp(masked_t, -1) - torch.logsumexp(t, -1)
    kl = lp_s.exp() * (lp_s - lp_t) + l1_s.exp() * (l1_s - l1_t)
    signs = torch.tensor(decision["signs"], device=device)
    with torch.no_grad():
        active = ((signs > 0) & (lp_t > lp_s)) | ((signs < 0) & (lp_t < lp_s)) | (signs == 0)
        target = torch.tensor(decision["targets"], device=device)[:, None]
        old = torch.tensor(decision["old"], device=device, dtype=torch.float32)
        ratio = (log_s.gather(-1, target).flatten() - old).clamp(-20, 20).exp().clamp(max=2.0)
    per = kl * ratio * active.float()
    stats = {"points": len(decision["signs"]), "active": int(active.sum()), "active_points": active.tolist(),
             "ratio": ratio.tolist(),
             "p_student": lp_s.exp().detach().tolist(), "p_teacher": lp_t.exp().detach().tolist(),
             "kl": kl.detach().tolist(), "reasons": decision["reasons"], "signs": decision["signs"],
             "unmatched": decision["unmatched"]}
    return per.sum() / len(decision["signs"]), stats


def decision_summary(records):
    """Per-reason counts and mean P(act) for a batch's decision records (the monitor in decision.json)."""
    by_reason = defaultdict(lambda: {"points": 0, "active": 0, "p_student": 0.0, "p_teacher": 0.0})
    for record in records:
        for reason, sign, ps, pt, active in zip(record["reasons"], record["signs"], record["p_student"],
                                                 record["p_teacher"], record["active_points"]):
            entry = by_reason[reason]
            entry["points"] += 1
            entry["active"] += int(active)
            entry["p_student"] += ps
            entry["p_teacher"] += pt
    for entry in by_reason.values():
        entry["p_student"], entry["p_teacher"] = entry["p_student"] / entry["points"], entry["p_teacher"] / entry["points"]
        entry["active_share"] = entry["active"] / entry["points"]
    return {"episodes": len(records), "episodes_with_points": sum(r["points"] > 0 for r in records),
            "decision_only_episodes": sum(r.get("call_tokens", 1) == 0 for r in records),
            "unmatched_harm_facts": sum(r.get("unmatched", 0) for r in records), "by_reason": dict(by_reason)}


def call_spans(tokenizer, action_ids):
    """Per tool call in one reply: token range of its argument text (between the function tag and its end)."""
    text = tokenizer.decode(action_ids, skip_special_tokens=False)
    ends = []
    for i in range(len(action_ids)):
        ends.append(len(tokenizer.decode(action_ids[:i + 1], skip_special_tokens=False)))
    spans, cursor = [], 0
    while True:
        start = text.find("<tool_call>", cursor)
        if start < 0:
            break
        function = text.find(">", text.find("<function=", start)) + 1
        stop = text.find("</function>", function)
        if function <= 0 or stop < 0:
            break
        first = next((i for i, e in enumerate(ends) if e > function), None)
        last = next((i for i, e in enumerate(ends) if e >= stop), len(ends) - 1)
        if first is not None and last >= first:
            spans.append((first, last + 1))
        cursor = stop
    return spans


def parameter_spans(tokenizer, action_ids):
    """Per tool call in one reply: {parameter name: token range of its value}."""
    import re

    text = tokenizer.decode(action_ids, skip_special_tokens=False)
    ends = [len(tokenizer.decode(action_ids[:i + 1], skip_special_tokens=False)) for i in range(len(action_ids))]

    def tokens(a, b):
        first = next((i for i, e in enumerate(ends) if e > a), None)
        last = next((i for i, e in enumerate(ends) if e >= b), len(ends) - 1)
        return (first, last + 1) if first is not None and last >= first else None

    calls, cursor = [], 0
    while True:
        start = text.find("<function=", cursor)
        if start < 0:
            break
        stop = text.find("</function>", start)
        if stop < 0:
            break
        found = {}
        for match in re.finditer(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", text[start:stop], re.S):
            span = tokens(start + match.start(2), start + match.end(2))
            if span:
                found[match.group(1)] = span
        calls.append(found)
        cursor = stop
    return calls


# ------------------------------------------------------------------ model scoring and loss

def inner_hidden(model, ids, positions):
    import torch

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    tensor = torch.tensor([ids], device="cuda")
    output = base.model(input_ids=tensor, attention_mask=torch.ones_like(tensor), use_cache=False)
    hidden = output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]
    index = torch.tensor(positions, device="cuda")
    return hidden[0].index_select(0, index)


def lm_head(model):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.lm_head if hasattr(base, "lm_head") else base.get_output_embeddings()


def token_logprobs(model, ids, positions, targets, adapter="default", chunk=2048):
    import torch

    model.set_adapter(adapter)
    model.eval()
    out = []
    with torch.no_grad():
        hidden = inner_hidden(model, ids, positions)
        head = lm_head(model)
        for start in range(0, len(positions), chunk):
            logits = head(hidden[start:start + chunk]).float()
            target = torch.tensor(targets[start:start + chunk], device="cuda")[:, None]
            out += logits.log_softmax(-1).gather(-1, target).flatten().tolist()
    model.set_adapter("default")
    return out


def chunk_loss(head, hs, ht, target, old, topk, exclude=None, exclude_id=None):
    """Per-chunk top-k + tail loss. With `exclude` (recipe R), the rows it marks have token `exclude_id` masked out
    of both distributions before the loss, so the term holds the distribution over the other tokens only; the
    importance weight still comes from the unmasked student."""
    import torch
    from trl.experimental.sdpo.loss_utils import compute_topk_self_distillation_loss

    s = head(hs).float()
    with torch.no_grad():
        t = head(ht).float()
    s_kl, t_kl = s, t
    if exclude is not None and bool(exclude.any()):
        mask = torch.zeros_like(s, dtype=torch.bool)
        mask[:, exclude_id] = exclude
        s_kl, t_kl = s.masked_fill(mask, DECISION_MASK_LOGIT), t.masked_fill(mask, DECISION_MASK_LOGIT)
    per = compute_topk_self_distillation_loss(s_kl, t_kl, distillation_topk=topk, distillation_alpha=1.0,
                                              distillation_add_tail=True)
    selected = s.log_softmax(-1).gather(-1, target[:, None]).flatten()
    ratio = (selected.detach() - old).clamp(-20, 20).exp().clamp(max=2.0)
    return (per * ratio).sum(), ratio.detach()


def episode_loss(model, student, teacher, topk=20, chunk=1024, anchor=None, parts=None, decision=None,
                 decision_weight=1.0):
    """Mean over assistant tokens of the importance-weighted top-k + tail reverse KL.

    With `anchor` (D38), the mean over the anchored tokens of the same loss against the frozen untrained
    adapter on the student's own context is added with weight 1; `parts` receives both terms.

    With `decision` (recipe R, `decision_sequence`), a third term is added with weight `decision_weight`: the
    mean over the decision points of the one-sided Bernoulli KL between the student's and the teacher's
    P(<tool_call>) (`decision_term`). At those points, which are anchored tokens, the anchor holds only the
    distribution over the other tokens (`chunk_loss` with `exclude`), so the two terms do not fight: the
    decision goes to the teacher, the words stay at the untrained model. Without `decision` nothing changes."""
    import torch
    from torch.utils.checkpoint import checkpoint

    with torch.no_grad():
        model.set_adapter("sdpo_teacher")
        model.eval()
        if decision:
            both = inner_hidden(model, teacher["ids"], teacher["positions"] + decision["teacher_positions"]).detach()
            ht, ht_decision = both[:len(teacher["positions"])], both[len(teacher["positions"]):]
        else:
            ht = inner_hidden(model, teacher["ids"], teacher["positions"]).detach()
        if anchor:
            model.set_adapter("sdpo_anchor")
            ha = inner_hidden(model, student["ids"], anchor["positions"]).detach()
    model.set_adapter("default")
    model.train()
    count = len(student["targets"])
    hs = inner_hidden(model, student["ids"], student["positions"] + (anchor["positions"] if anchor else []))
    head = lm_head(model)

    def mean_loss(hidden, reference, sequence, exclude=None, exclude_id=None):
        if not sequence["targets"]:
            return hidden.new_zeros((), dtype=torch.float32), []
        target = torch.tensor(sequence["targets"], device="cuda")
        old = torch.tensor(sequence["old"], device="cuda", dtype=torch.float32)
        total, ratios = hidden.new_zeros((), dtype=torch.float32), []
        for start in range(0, len(sequence["targets"]), chunk):
            if exclude is None:
                part, ratio = checkpoint(chunk_loss, head, hidden[start:start + chunk], reference[start:start + chunk],
                                         target[start:start + chunk], old[start:start + chunk], topk,
                                         use_reentrant=False)
            else:
                part, ratio = checkpoint(chunk_loss, head, hidden[start:start + chunk], reference[start:start + chunk],
                                         target[start:start + chunk], old[start:start + chunk], topk,
                                         exclude[start:start + chunk], exclude_id, use_reentrant=False)
            total = total + part
            ratios += ratio.tolist()
        return total / len(sequence["targets"]), ratios

    if count:
        loss, ratios = mean_loss(hs[:count], ht, student)
    else:  # recipe R: an attempt that never called; only the anchor and the decision term remain
        loss, ratios = hs.new_zeros((), dtype=torch.float32), []
    if parts is not None:
        parts["distill"] = float(loss.detach())
    if decision and not anchor:
        raise ValueError("the decision term needs the anchor (calls+anchor+decision)")
    if anchor:
        exclude, rows = None, None
        if decision:
            index = {p: i for i, p in enumerate(anchor["positions"])}
            if any(p not in index for p in decision["positions"]):
                raise ValueError("a decision point is not an anchored token")
            rows = [index[p] for p in decision["positions"]]
            exclude = torch.zeros(len(anchor["positions"]), dtype=torch.bool, device=hs.device)
            exclude[rows] = True
        held, _ = mean_loss(hs[count:], ha, anchor, exclude, decision["opener"] if decision else None)
        loss = loss + held
        if parts is not None:
            parts["anchor"] = float(held.detach())
        if decision:
            term, stats = decision_term(head, hs[count:][rows], ht_decision, decision, hs.device)
            loss = loss + decision_weight * term
            if parts is not None:
                parts["decision"], parts["decision_stats"] = float(term.detach()), stats
            if not count:
                ratios = stats["ratio"]
    return loss, ratios


def decision_parity(linear, generator):
    """Recipe R: `decision_term` against the float reference `bernoulli_kl`, on random logits with every sign
    (two-sided, act-more, act-less) and every importance-weight regime (the rollout log-probability equal to the
    current one: weight 1; 0.5 nats below it: weight e^0.5; 2 nats below it: clamped to 2); returns the largest
    absolute difference."""
    import math
    import torch

    hs, ht = torch.randn(9, 16, generator=generator), torch.randn(9, 16, generator=generator)
    targets = torch.randint(0, 50, (9,), generator=generator).tolist()
    shifts = [0.0, 0.5, 2.0] * 3
    with torch.no_grad():
        current = linear(hs).log_softmax(-1).gather(-1, torch.tensor(targets)[:, None]).flatten().tolist()
        log_s, log_t = linear(hs).log_softmax(-1)[:, 7].tolist(), linear(ht).log_softmax(-1)[:, 7].tolist()
    old = [c - shift for c, shift in zip(current, shifts)]
    signs = [0, 0, 0, 1, 1, 1, -1, -1, -1]
    decision = {"opener": 7, "targets": targets, "old": old, "signs": signs, "reasons": ["check"] * 9, "unmatched": 0}
    term, stats = decision_term(linear, hs, ht, decision, hs.device)
    reference = sum(bernoulli_kl(ps, pt) * min(math.exp(shift), 2.0)
                    for sign, ps, pt, shift in zip(signs, log_s, log_t, shifts) if decision_active(sign, ps, pt)) / 9
    return abs(float(term) - reference)


def parity_checks(model, tokenizer, output, decision=False):
    """Loss parity with the TRL helper on CPU, and hidden-state head parity on the GPU model."""
    import torch
    from trl.experimental.sdpo.loss_utils import compute_topk_self_distillation_loss

    generator = torch.Generator().manual_seed(20261027)
    s = torch.randn(37, 50, generator=generator)
    t = torch.randn(37, 50, generator=generator)
    whole = compute_topk_self_distillation_loss(s, t, distillation_topk=20, distillation_alpha=1.0,
                                                distillation_add_tail=True).sum()
    pieces = sum(compute_topk_self_distillation_loss(s[i:i + 8], t[i:i + 8], distillation_topk=20,
                                                     distillation_alpha=1.0, distillation_add_tail=True).sum()
                 for i in range(0, 37, 8))
    linear = torch.nn.Linear(16, 50, bias=False)
    hs, ht = torch.randn(37, 16, generator=generator), torch.randn(37, 16, generator=generator)
    target = torch.randint(0, 50, (37,), generator=generator)
    old = torch.full((37,), -1.0)
    mine, _ = chunk_loss(linear, hs, ht, target, old, 20)
    reference = compute_topk_self_distillation_loss(linear(hs), linear(ht).detach(), distillation_topk=20,
                                                    distillation_alpha=1.0, distillation_add_tail=True)
    selected = linear(hs).log_softmax(-1).gather(-1, target[:, None]).flatten()
    reference = (reference * (selected.detach() - old).exp().clamp(max=2.0)).sum()
    probe = tokenizer.encode("Local readiness check for the SDPO runner.", add_special_tokens=False)
    with torch.no_grad():
        model.eval()
        full = model(input_ids=torch.tensor([probe], device="cuda"), use_cache=False).logits[0, :-1].float()
        mine_logits = lm_head(model)(inner_hidden(model, probe, list(range(len(probe) - 1)))).float()
    result = {"chunked_vs_whole": float((whole - pieces).abs()), "loss_vs_trl_reference": float((mine - reference).abs()),
              "hidden_head_vs_model_logits_max_abs": float((full - mine_logits).abs().max())}
    result["passed"] = (result["chunked_vs_whole"] < 1e-5 and result["loss_vs_trl_reference"] < 1e-5
                        and result["hidden_head_vs_model_logits_max_abs"] < 0.05)
    if decision:  # recipe R only: the Bernoulli term against its float reference
        result["decision_vs_reference"] = decision_parity(linear, generator)
        result["passed"] = result["passed"] and result["decision_vs_reference"] < 1e-5
    write(output / "parity.json", result)
    if not result["passed"]:
        raise ValueError(f"parity failed: {result}")
    return result


# ------------------------------------------------------------------ runtime

class Runtime:
    def __init__(self, args):
        self.args = args
        self.out = Path(args.output)

    async def start(self):
        import torch
        from aiohttp import web
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForImageTextToText, AutoTokenizer, set_seed
        from verifiers_preinstalled import configure_preinstalled_runtime

        os.environ.update(POSTTRAIN_VERIFIERS_PREINSTALLED="1", AB_LOCAL_KEY="EMPTY")
        configure_preinstalled_runtime(interpreter=sys.executable)
        set_seed(20261027)
        torch.set_num_threads(4)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model, local_files_only=True)
        base = AutoModelForImageTextToText.from_pretrained(self.args.model, dtype=torch.bfloat16,
            device_map={"": "cuda:0"}, attn_implementation="sdpa", local_files_only=True)
        self.model = get_peft_model(base, LoraConfig(r=self.args.lora_rank, lora_alpha=self.args.lora_alpha,
            lora_dropout=0.0, target_modules="all-linear", exclude_modules=r".*visual.*", task_type="CAUSAL_LM"))
        initial = self.out / "initial-student"
        self.model.save_pretrained(initial, selected_adapters=["default"])
        write(self.out / "initial-adapter-check.json",
              verify_adapter_config(initial / "adapter_config.json", self.args.lora_rank, self.args.lora_alpha))
        self.initial = initial
        self.model.load_adapter(initial, adapter_name="sdpo_teacher", is_trainable=False)
        if getattr(self.args, "loss_tokens", "all") in ANCHORED_LOSSES:
            self.model.load_adapter(initial, adapter_name="sdpo_anchor", is_trainable=False)
        self.model.set_adapter("default")
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model.enable_input_require_grads()
        if getattr(self.args, "loss_tokens", "all") == "calls+anchor+decision":
            parity_checks(self.model, self.tokenizer, self.out, decision=True)
        else:
            parity_checks(self.model, self.tokenizer, self.out)
        if getattr(self.args, "r1_branches", False):
            from sdpo_ab_branches import branch_parity_check

            branch_parity_check(self.out, self.args.r1_opener_cap)
        if getattr(self.args, "sf_final", False):  # S-final: the two-sided decision term against its float reference
            from sdpo_final import parity_check as final_parity_check

            final_parity_check(self.out, self.args.sf_cap, self.args.sf_floor)
        Rollout = make_rollout_class()
        context_limit, reply_limit = episode_limits(self.args)[1:3]
        self.rollout = await Rollout.build(self.args.model, self.model, self.tokenizer, self.out,
                                           context_limit, self.args.concurrency, self.args.kv_cache_gib)
        self.rollout.seed_salt = self.args.seed_salt
        self.agent = Agent(self.tokenizer, self.rollout, context_limit, reply_limit)
        app = web.Application(client_max_size=64 * 1024 * 1024)
        app.router.add_post("/v1/chat/completions", self.agent.completions)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        await web.TCPSite(self.runner, "127.0.0.1", ROOT_PORT).start()
        probe = self.tokenizer.encode("Local readiness", add_special_tokens=False)
        target = self.tokenizer.encode(" verified.", add_special_tokens=False)
        self.probe = (probe, target)
        self.startup_probe = await self.rollout.score_probe(self.model, probe, target)
        self.model.set_adapter("default")
        log(stage="ready")

    async def close(self):
        await self.runner.cleanup()
        await self.rollout.aclose()

    async def collect(self, label, names, attempts, temperature, top_p, prompt_queue=None, max_turns=None,
                      episode_tokens=None, sampling=None):
        """Run the native environment for names x attempts; save traces, chains and outcomes.

        `sampling` (correction b, collect phase only): the full sampling dict of this collection; its fields beyond
        temperature and top_p reach vLLM through `sampling_options`, and the dict is recorded in summary.json. Every
        other caller leaves it None: the rollout then samples with temperature and top_p alone, as before."""
        from sdpo_ab_metrics import episode_outcome, summary
        from sdpo_ab_replay import environment
        from verifiers.v1.cli.eval.runner import run_eval
        from verifiers.v1.configs.cli.eval import EvalConfig

        folder = self.out / label
        folder.mkdir(parents=True, exist_ok=False)
        self.agent.reset()
        self.agent.check_rerender = 0 if prompt_queue else 6  # a replaced prompt never equals the re-render
        self.agent.prompt_queue = prompt_queue or {}
        self.rollout.temperature, self.rollout.top_p = temperature, top_p
        if sampling is not None and (sampling["temperature"], sampling["top_p"]) != (temperature, top_p):
            raise ValueError("sampling dict disagrees with temperature/top_p")
        self.rollout.extra_sampling = {k: v for k, v in (sampling or {}).items() if k not in ("temperature", "top_p")}
        started = time.monotonic()
        turns, context_limit, reply_limit, rollout_timeout = episode_limits(self.args)
        env = environment(names, turns if max_turns is None else max_turns, max_output_tokens=context_limit,
                          rollout_timeout=rollout_timeout)
        if episode_tokens is not None:
            env["agent"]["max_output_tokens"] = episode_tokens
        config = EvalConfig.model_validate({
            "env": env, "model": "local",
            "client": {"type": "eval", "base_url": f"http://127.0.0.1:{ROOT_PORT}/v1", "api_key_var": "AB_LOCAL_KEY"},
            "sampling": {"temperature": temperature, "top_p": top_p, "max_tokens": reply_limit},
            "num_tasks": len(names), "task_keys": list(names), "num_rollouts": attempts,
            "max_concurrent": self.args.concurrency, "shuffle": False, "output_dir": folder / "native",
            "run": {"name": label, "dir": "."}, "push": False, "rich": None, "serve": None})
        log(stage="collect", label=label, tasks=len(names), attempts=attempts, temperature=temperature)
        episodes = await run_eval(config)
        records = [e.model_dump(mode="json") for e in episodes]
        joined = join_chains(self.agent, records)
        outcomes = []
        for record in records:
            chain = joined.get(record["id"])
            actions = [s for s in chain["segments"] if s["kind"] == "action"] if chain else None
            outcome = episode_outcome(record, actions)
            outcome["chain_joined"] = chain is not None and not chain.get("mismatch")
            outcome["tokens"] = len(chain["tokens"]) if chain else None
            outcome["context_id"] = chain.get("context_id") if chain else None
            outcome["runaway"] = runaway(actions) if actions else None
            outcome["reply_limit_cut"] = reply_limit_cut(actions) if actions else None
            outcome["runaway_definition"] = "visible>8000 or thinking unclosed (measures-phase2)"
            outcome["required_passed"], outcome["required_total"] = required_checks(record)
            outcomes.append(outcome)
        with gzip.open(folder / "chains.jsonl.gz", "wt") as stream:
            for record in records:
                chain = joined.get(record["id"])
                if chain:
                    stream.write(json.dumps({"episode_id": record["id"], "tools": chain["tools"],
                                             "messages0": chain["messages0"], "segments": chain["segments"],
                                             "context_id": chain.get("context_id"),
                                             "mismatch": bool(chain.get("mismatch"))}) + "\n")
        with gzip.open(folder / "episodes.jsonl.gz", "wt") as stream:
            for record in records:
                stream.write(json.dumps(record) + "\n")
        result = summary(outcomes, list(names), attempts)
        if prompt_queue is not None:
            left = sum(len(v) for v in self.agent.prompt_queue.values())
            used = sum(1 for o in outcomes if o["context_id"])
            if left:
                raise ValueError(f"prompt override incomplete: {left} queued contexts were never used")
            self.agent.audit["episodes_without_context"] = len(outcomes) - used
        result.update(label=label, elapsed_s=time.monotonic() - started, temperature=temperature, top_p=top_p,
                      endpoint_audit=self.agent.audit, chains_joined=sum(o["chain_joined"] for o in outcomes),
                      policy_version=self.rollout.version)
        if sampling is not None:
            result["sampling"] = dict(sampling)
        if THINKING:
            result["thinking"] = "on"
        if self.agent.prompt_suffix:
            result["prompt_suffix"] = self.agent.prompt_suffix
        write(folder / "outcomes.json", outcomes)
        write(folder / "summary.json", result)
        log(stage="collected", label=label, successes=result["successes"], mean=result["mean_task_success"],
            elapsed=round(result["elapsed_s"]), joined=result["chains_joined"], audit=self.agent.audit)
        return records, joined, outcomes, result


def load_collection(folder):
    from sdpo_ab_metrics import read_episodes  # noqa: F401 - keeps the dependency explicit

    records = [json.loads(line) for line in gzip.open(Path(folder) / "episodes.jsonl.gz", "rt")]
    chains = {}
    for line in gzip.open(Path(folder) / "chains.jsonl.gz", "rt"):
        value = json.loads(line)
        chains[value["episode_id"]] = value
    return records, chains


def load_pool_source(folder, fit):
    """(records of tasks in `fit`, number of episodes skipped) from a --solution-pool-from folder's episodes, read one
    line at a time: an episode of any other task (test60b, dev60, guard10, ...) is dropped before it is kept or shown
    to the engine, so it never enters the pool."""
    records, skipped = [], 0
    with gzip.open(Path(folder) / "episodes.jsonl.gz", "rt") as handle:
        for line in handle:
            record = json.loads(line)
            if task_of(record) in fit:
                records.append(record)
            else:
                skipped += 1
    return records, skipped


def task_of(record):
    return record["traces"][0]["task"]["data"]["name"]


def domain_of(name):
    return name.split(".", 1)[0]


# ------------------------------------------------------------------ gates

def span_records(rt, records, chains, contexts):
    """Bad spans: argument tokens of the calls behind prohibited / changed / tool-error facts."""
    spans, seen = [], set()
    for record in records:
        context = contexts.get(record["id"])
        chain = chains.get(record["id"])
        if not context or not chain or chain.get("mismatch"):
            continue
        student = student_sequence(chain)
        actions = [s for s in chain["segments"] if s["kind"] == "action"]
        for bad in context["bad_spans"]:
            if (record["id"], bad["turn"], bad["call_index"]) in seen:
                continue  # one call behind several facts is one span
            seen.add((record["id"], bad["turn"], bad["call_index"]))
            action = actions[bad["turn"] - 1]
            calls = call_spans(rt.tokenizer, action["ids"])
            if bad["call_index"] >= len(calls):
                continue
            a, b = calls[bad["call_index"]]
            offset = sum(len(x["ids"]) for x in actions[:bad["turn"] - 1])
            spans.append({"episode_id": record["id"], "task": task_of(record), "kind": bad["kind"],
                          "turn": bad["turn"], "call_index": bad["call_index"],
                          "index": list(range(offset + a, offset + b))})
    return spans


def mean_over(values, index):
    return sum(values[i] for i in index) / len(index)


def gate_b(rt, records, chains, siblings_by_task, contracts):
    from sdpo_ab_context import build_context

    placements = ("first", "notes", "both")
    scores, all_spans, contexts_by_placement = {}, None, {}
    student_lp = {}
    for placement in placements:
        contexts = {}
        for record in records:
            if record["traces"][0]["metrics"].get("task_completed_correctly") == 1.0:
                continue
            context = build_context(record, siblings_by_task[task_of(record)], "A", placement,
                                    contracts.get(task_of(record)))
            if context:
                contexts[record["id"]] = context
        contexts_by_placement[placement] = contexts
        spans = span_records(rt, records, chains, contexts)
        if all_spans is None:
            all_spans = spans
        teacher_lp = {}
        for episode_id in sorted({s["episode_id"] for s in spans}):
            chain = chains[episode_id]
            student = student_sequence(chain)
            if episode_id not in student_lp:
                student_lp[episode_id] = token_logprobs(rt.model, student["ids"], student["positions"],
                                                        student["targets"], "sdpo_teacher")
            teacher = teacher_sequence(rt.tokenizer, chain, contexts[episode_id])
            assert teacher["targets"] == student["targets"]
            teacher_lp[episode_id] = token_logprobs(rt.model, teacher["ids"], teacher["positions"],
                                                    teacher["targets"], "sdpo_teacher")
        rows = []
        for span in spans:
            s = mean_over(student_lp[span["episode_id"]], span["index"])
            t = mean_over(teacher_lp[span["episode_id"]], span["index"])
            rows.append({**{k: span[k] for k in ("episode_id", "task", "kind", "turn", "call_index")},
                         "tokens": len(span["index"]), "student": s, "teacher": t, "right_way": t < s})
        by_domain = defaultdict(list)
        for row in rows:
            by_domain[domain_of(row["task"])].append(row["right_way"])
        share = sum(r["right_way"] for r in rows) / len(rows) if rows else 0.0
        domains = {d: sum(v) / len(v) for d, v in by_domain.items()}
        scores[placement] = {"spans": len(rows), "right_way_share": share, "domains": domains,
                             "passed": bool(rows) and share >= 0.8 and min(domains.values()) >= 0.6,
                             "mean_margin": (sum(r["student"] - r["teacher"] for r in rows) / len(rows)) if rows else None,
                             "rows": rows}
        log(stage="gate_b", placement=placement, spans=len(rows), share=share, domains=domains)
    passing = [p for p in placements if scores[p]["passed"]]
    order = {"both": 0, "first": 1, "notes": 2}
    best = max(passing, key=lambda p: (scores[p]["right_way_share"], -order[p])) if passing else None
    return {"placements": scores, "chosen": best, "student_span_logprobs": {
        f"{s['episode_id']}|{s['turn']}|{s['call_index']}": mean_over(student_lp[s["episode_id"]], s["index"])
        for s in all_spans}, "spans": all_spans}


# ------------------------------------------------------------------ phases

def adapter_effect(before, after, minimum=1e-3):
    """Plan v4: the rollout engine's selected-token log-probs on the startup probe must move after an adapter load
    (max |delta| >= `minimum` nats); identical log-probs mean the engine is still sampling the base model."""
    change = max(abs(a - b) for a, b in zip(before["rollout"], after["rollout"], strict=True))
    if not change >= minimum:
        raise ValueError(f"adapter load left the rollout engine's log-probs unchanged (max |delta| {change:.2e} nats)")
    return {"rollout_logprob_max_abs_change": change, "parity_max_abs_delta": after["max_abs_delta"],
            "minimum": minimum}


def episode_limits(args):
    """Plan v4 (12 Nov 2026): the episode limits as settings, (max_turns, context, reply, rollout_timeout_s).
    Defaults are the D10 values (50 turns, 65,536 context, 4,096 reply, 3,600 s per rollout), so every earlier
    command line behaves as before; the reply limit can never exceed the context."""
    turns = getattr(args, "max_turns", 50)
    context = getattr(args, "context_limit", 65536)
    reply = getattr(args, "reply_limit", 4096)
    timeout = getattr(args, "rollout_timeout", 3600)
    if not (0 < turns and 0 < reply <= context and timeout > 0):
        raise ValueError(f"bad episode limits: turns {turns}, reply {reply}, context {context}, timeout {timeout}")
    return turns, context, reply, timeout


def collect_sampling(name):
    """Correction b: the evaluation sampling of `--phase collect --sampling NAME` ("protocol", the default: the
    frozen EVALUATION_SAMPLING; "card": the Qwen3.5-4B model card's non-thinking general settings)."""
    from sdpo_frozen_environment import COLLECT_SAMPLINGS

    if name == SF_TRAINING_SAMPLING:  # S-final F7: the training rollouts' sampling (temperature 1.0, top-p 1.0)
        from sdpo_frozen_environment import ROLLOUT_SAMPLING

        return dict(ROLLOUT_SAMPLING)
    if name not in COLLECT_SAMPLINGS:
        raise ValueError(f"unknown collect sampling {name!r}")
    return dict(COLLECT_SAMPLINGS[name])


def read_prompt_suffix(path):
    """Correction b: the text of `--prompt-suffix-file`, appended to the task's first user message. The file is used
    verbatim except for trailing newlines (a text file's final newline is not part of the instruction)."""
    text = Path(path).read_text(encoding="utf-8").rstrip("\n")
    if not text.strip():
        raise ValueError(f"prompt suffix file {path} is empty")
    return text


def with_prompt_suffix(messages, suffix):
    """The request's messages with `suffix` appended after one blank line to the first user message (the task
    prompt); every other message, and the caller's list, unchanged. No suffix: the messages themselves."""
    import copy

    if not suffix:
        return messages
    index = next((i for i, m in enumerate(messages) if m["role"] == "user"), None)
    if index is None:
        return messages
    if not isinstance(messages[index].get("content"), str):
        raise ValueError("prompt suffix needs a text task prompt")
    messages = copy.deepcopy(messages)
    messages[index]["content"] = suffixed(messages[index]["content"], suffix)
    return messages


def suffixed(text, suffix):
    return text + "\n\n" + suffix if suffix else text


def check_collect_options(args):
    """Correction b: --sampling card and --prompt-suffix-file apply to `--phase collect --protocol ours` only; anywhere
    else they would be silently ignored, so they are refused."""
    if args.phase in ("a3-continue", "a4-signal", "classic-train", "signal-check", "design-audit",
                      "online-train"):  # sdpo-weekend and SDPO v6: thinking on
        if getattr(args, "thinking", "off") != "on" or getattr(args, "sampling", "protocol") != "card-thinking":
            raise ValueError("--phase a3-continue/a4-signal need --thinking on --sampling card-thinking")
        return
    if getattr(args, "sf_final", False) and args.phase == "engine-train" and getattr(
            args, "sampling", "protocol") == "protocol" and not getattr(args, "prompt_suffix_file", None):
        return  # S-final: --thinking on|off follows Track G's result in training too (Phase 1 item 4)
    changed = (getattr(args, "sampling", "protocol") != "protocol" or getattr(args, "prompt_suffix_file", None)
               or getattr(args, "thinking", "off") != "off")
    if changed and (args.phase != "collect" or args.protocol != "ours"):
        raise ValueError("--sampling card, --prompt-suffix-file and --thinking on need --phase collect --protocol ours")


async def phase_collect_only(rt, args):
    names = json.loads(Path(args.tasks).read_text())["task_names"]
    from sdpo_frozen_environment import EVALUATION_SAMPLING

    if args.adapter:  # a trained checkpoint, loaded as run_arm_d24 does before its transfer evaluation
        from peft import load_peft_weights, set_peft_model_state_dict

        set_peft_model_state_dict(rt.model, load_peft_weights(args.adapter), adapter_name="default")
        rt.model.set_adapter("default")
        await rt.rollout.suspend_for_update()
        await rt.rollout.refresh()
        # plan v4: the rollout engine must actually apply the adapter (a vLLM adapter whose keys match no module is
        # ignored silently); score_probe also re-checks HF-vs-vLLM parity with the adapter on (<= 0.15 nats)
        after = await rt.rollout.score_probe(rt.model, *rt.probe)
        check = adapter_effect(rt.startup_probe, after)
        (rt.out / "adapter-check.json").write_text(json.dumps({"adapter": args.adapter, **check}, indent=1))
        log(stage="adapter-check", **check)
    if args.protocol == "handover":
        budget = HANDOVER_EVALUATION
        rt.agent.context_limit, rt.agent.reply_limit, rt.agent.align_budget = budget["context"], budget["reply"], 20
        await rt.collect(args.label, names, args.attempts, budget["temperature"], budget["top_p"],
                         max_turns=budget["turns"], episode_tokens=budget["episode"])
        return
    if getattr(args, "sampling", "protocol") == "protocol" and not getattr(args, "prompt_suffix_file", None):
        await rt.collect(args.label, names, args.attempts, EVALUATION_SAMPLING["temperature"],
                         EVALUATION_SAMPLING["top_p"])
        return
    # correction b: a named sampling and/or a task-prompt suffix, both recorded (inputs and summary.json)
    sampling = collect_sampling(args.sampling)
    rt.agent.prompt_suffix = read_prompt_suffix(args.prompt_suffix_file) if args.prompt_suffix_file else None
    try:
        await rt.collect(args.label, names, args.attempts, sampling["temperature"], sampling["top_p"],
                         sampling=sampling)
    finally:
        rt.agent.prompt_suffix = None


async def phase_smoke(rt, args):
    """Collection, contexts, Gate B scoring, one loss/backward per episode (no update) and capacity."""
    import torch
    from sdpo_ab_context import build_context, contract_for
    from sdpo_frozen_environment import EVALUATION_SAMPLING

    names = json.loads(Path(args.tasks).read_text())["task_names"]
    records, joined, outcomes, summary = await rt.collect("smoke", names, args.attempts,
                                                          EVALUATION_SAMPLING["temperature"],
                                                          EVALUATION_SAMPLING["top_p"])
    await rt.rollout.suspend_for_update()
    checks = []
    by_task = defaultdict(list)
    for record in records:
        by_task[task_of(record)].append(record)
    for record in records:
        chain = joined.get(record["id"])
        row = {"episode_id": record["id"], "task": task_of(record), "joined": chain is not None}
        try:
            contract = contract_for(task_of(record))
            row["contract_found"] = contract is not None
            context = build_context(record, by_task[task_of(record)], "A", "both", contract)
            row["context"] = context
            if context and chain:
                student, teacher = student_sequence(chain), teacher_sequence(rt.tokenizer, chain, context)
                row["targets_equal"] = student["targets"] == teacher["targets"]
                row["student_tokens"], row["teacher_tokens"] = len(student["ids"]), len(teacher["ids"])
                spans = span_records(rt, [record], {record["id"]: chain}, {record["id"]: context})
                row["spans"] = [{k: s[k] for k in ("kind", "turn", "call_index")} | {"tokens": len(s["index"])}
                                for s in spans]
                if spans:
                    slp = token_logprobs(rt.model, student["ids"], student["positions"], student["targets"],
                                         "sdpo_teacher")
                    tlp = token_logprobs(rt.model, teacher["ids"], teacher["positions"], teacher["targets"],
                                         "sdpo_teacher")
                    row["span_scores"] = [{"student": mean_over(slp, s["index"]), "teacher": mean_over(tlp, s["index"]),
                                           "text": rt.tokenizer.decode([student["targets"][i] for i in s["index"]])}
                                          for s in spans]
                    row["rollout_vs_actor_max_abs"] = max(abs(a - b) for a, b in zip(slp, student["old"]))
                torch.cuda.reset_peak_memory_stats()
                loss, ratios = episode_loss(rt.model, student, teacher)
                loss.backward()
                row["loss"] = float(loss.detach())
                row["mean_importance_weight"] = sum(ratios) / len(ratios)
                row["peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
                rt.model.zero_grad(set_to_none=True)
        except Exception:  # noqa: BLE001 - smoke evidence records every failure
            row["error"] = traceback.format_exc()
        checks.append(row)
        write(rt.out / "smoke-checks.json", checks)
    capacity = {}
    for length, actions in ((65536, 16384),):
        ids = [random.Random(1).randrange(1000, 100000) for _ in range(length)]
        positions = list(range(length - actions - 1, length - 1))
        student = {"ids": ids, "positions": positions, "targets": ids[length - actions:], "old": [-1.0] * actions}
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            loss, _ = episode_loss(rt.model, student, student)
            loss.backward()
            capacity[str(length)] = {"ok": True, "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
                                     "seconds": time.monotonic() - started}
        except torch.OutOfMemoryError:
            capacity[str(length)] = {"ok": False, "error": "out_of_memory"}
        rt.model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    write(rt.out / "capacity.json", capacity)
    log(stage="smoke_done", capacity=capacity)


async def phase_controls(rt, args):
    from sdpo_ab_context import build_context, contract_for
    from sdpo_frozen_environment import EVALUATION_SAMPLING

    fit = json.loads(Path(args.fit).read_text())["task_names"]
    transfer = json.loads(Path(args.transfer).read_text())["task_names"]
    if args.phase == "controls" and not (rt.out / "control" / "summary.json").exists():
        await rt.collect("control", fit + transfer, 4, EVALUATION_SAMPLING["temperature"], EVALUATION_SAMPLING["top_p"])
    records, chains = load_collection(args.base)
    records = [r for r in records if task_of(r) in fit]
    siblings = defaultdict(list)
    for record in records:
        siblings[task_of(record)].append(record)
    contracts = {name: contract_for(name) for name in fit}
    await rt.rollout.suspend_for_update()
    result = gate_b(rt, records, chains, siblings, contracts)
    good = good_spans(rt, records, chains)
    result["good_spans"] = good
    result["student_good_logprobs"] = {f"{g['episode_id']}|{g['turn']}|{g['call_index']}":
                                       g["student"] for g in good}
    write(rt.out / "gate-b.json", result)
    log(stage="gate_b_done", chosen=result["chosen"])


async def phase_gateb_diagnostic(rt, args):
    """Re-scoring only: Gate B for arm A and arm B contexts (placement both), full call spans and
    the identifying-parameter value spans (recipient, channel, record ids)."""
    from sdpo_ab_context import SALIENT_KEYS, build_context, contract_for

    fit = json.loads(Path(args.fit).read_text())["task_names"]
    records, chains = load_collection(args.base)
    records = [r for r in records if task_of(r) in fit]
    siblings = defaultdict(list)
    for record in records:
        siblings[task_of(record)].append(record)
    await rt.rollout.suspend_for_update()
    rows = []
    for record in records:
        if record["traces"][0]["metrics"].get("task_completed_correctly") == 1.0:
            continue
        chain = chains.get(record["id"])
        if not chain or chain.get("mismatch"):
            continue
        name = task_of(record)
        student = student_sequence(chain)
        actions = [s for s in chain["segments"] if s["kind"] == "action"]
        offsets = [sum(len(a["ids"]) for a in actions[:k]) for k in range(len(actions))]
        student_lp = None
        for arm in ("A", "B"):
            context = build_context(record, siblings[name], arm, "both", contract_for(name))
            if not context or not context["bad_spans"]:
                continue
            if arm == "B" and not context["solution_episode"]:
                continue
            if student_lp is None:
                student_lp = token_logprobs(rt.model, student["ids"], student["positions"], student["targets"],
                                            "sdpo_teacher")
            teacher = teacher_sequence(rt.tokenizer, chain, context)
            teacher_lp = token_logprobs(rt.model, teacher["ids"], teacher["positions"], teacher["targets"],
                                        "sdpo_teacher")
            seen = set()
            for bad in context["bad_spans"]:
                key = (bad["turn"], bad["call_index"])
                if key in seen:
                    continue
                seen.add(key)
                action = actions[bad["turn"] - 1]
                full = call_spans(rt.tokenizer, action["ids"])
                params = parameter_spans(rt.tokenizer, action["ids"])
                if bad["call_index"] >= len(full):
                    continue
                base = offsets[bad["turn"] - 1]
                a, b = full[bad["call_index"]]
                index = list(range(base + a, base + b))
                row = {"arm": arm, "task": name, "episode_id": record["id"], "kind": bad["kind"],
                       "turn": bad["turn"], "call_index": bad["call_index"], "solution": context["solution_episode"],
                       "student": mean_over(student_lp, index), "teacher": mean_over(teacher_lp, index)}
                ident = []
                if bad["call_index"] < len(params):
                    for key_name, (x, y) in params[bad["call_index"]].items():
                        if key_name in SALIENT_KEYS:
                            ident += list(range(base + x, base + y))
                if ident:
                    row["ident_student"] = mean_over(student_lp, ident)
                    row["ident_teacher"] = mean_over(teacher_lp, ident)
                rows.append(row)
        write(rt.out / "gate-b-diagnostic.json", rows)
    summary = {}
    for arm in ("A", "B"):
        mine = [r for r in rows if r["arm"] == arm]
        ident = [r for r in mine if "ident_student" in r]
        summary[arm] = {"spans": len(mine), "right_way_share": sum(r["teacher"] < r["student"] for r in mine) / max(1, len(mine)),
                        "ident_spans": len(ident),
                        "ident_right_way_share": sum(r["ident_teacher"] < r["ident_student"] for r in ident) / max(1, len(ident))}
    write(rt.out / "gate-b-diagnostic-summary.json", summary)
    log(stage="gate_b_diagnostic", **summary)


def good_spans(rt, records, chains):
    """Argument tokens of write calls in successful base episodes (clean-solution calls)."""
    from sdpo_ab_context import clean_sibling

    rows = []
    for record in records:
        if record["traces"][0]["metrics"].get("task_completed_correctly") != 1.0:
            continue
        cleaned = clean_sibling(record)
        chain = chains.get(record["id"])
        if not cleaned or not chain or chain.get("mismatch"):
            continue
        keep = {(c["turn"], c["call_index"]) for c in cleaned["calls"]}
        student = student_sequence(chain)
        actions = [s for s in chain["segments"] if s["kind"] == "action"]
        lp = token_logprobs(rt.model, student["ids"], student["positions"], student["targets"], "sdpo_teacher")
        offset = 0
        for turn, action in enumerate(actions, 1):
            for index, (a, b) in enumerate(call_spans(rt.tokenizer, action["ids"])):
                if (turn, index) in keep and b > a:
                    rows.append({"episode_id": record["id"], "task": task_of(record), "turn": turn,
                                 "call_index": index, "index": list(range(offset + a, offset + b)),
                                 "student": mean_over(lp, range(offset + a, offset + b))})
            offset += len(action["ids"])
    return rows


def rescore(rt, chains, spans, adapter="default"):
    out = {}
    for episode_id in sorted({s["episode_id"] for s in spans}):
        chain = chains[episode_id]
        student = student_sequence(chain)
        lp = token_logprobs(rt.model, student["ids"], student["positions"], student["targets"], adapter)
        for s in spans:
            if s["episode_id"] == episode_id:
                out[f"{episode_id}|{s['turn']}|{s['call_index']}"] = (mean_over(lp, s["index"]), len(s["index"]))
    return out


def mean_entropy(rt, chains, episode_ids, adapter="default", chunk=1024):
    """Mean student entropy (nats) over its own assistant tokens in a fixed set of base episodes."""
    import torch

    total, count = 0.0, 0
    rt.model.set_adapter(adapter)
    rt.model.eval()
    with torch.no_grad():
        head = lm_head(rt.model)
        for episode_id in episode_ids:
            student = student_sequence(chains[episode_id])
            hidden = inner_hidden(rt.model, student["ids"], student["positions"])
            for start in range(0, len(student["positions"]), chunk):
                logp = head(hidden[start:start + chunk]).float().log_softmax(-1)
                total += float(-(logp.exp() * logp).sum(-1).sum())
                count += logp.shape[0]
    rt.model.set_adapter("default")
    return total / max(1, count)


def reference_collection(args):
    """Episodes behind the Gate C references: the D25 remeasure collection in the d24 phase, else the base."""
    if getattr(args, "phase", None) == "d24":
        return fit_collection(args)
    return load_collection(args.base)


def gate_c(rt, args, gate):
    records, chains = reference_collection(args)
    bad = rescore(rt, chains, gate["spans"])
    fell = [bad[k][0] < v for k, v in gate["student_span_logprobs"].items() if k in bad]
    rt.gate_c_records = {"bad": [{"span": k, "before": v, "after": bad[k][0], "tokens": bad[k][1]}
                                 for k, v in gate["student_span_logprobs"].items() if k in bad]}
    good = rescore(rt, chains, gate["good_spans"])
    rt.gate_c_records["good"] = [{"span": k, "before": v, "after": good[k][0], "tokens": good[k][1]}
                                 for k, v in gate["student_good_logprobs"].items() if k in good]
    tokens = sum(n for _, n in good.values())
    change = (sum((good[k][0] - v) * good[k][1] for k, v in gate["student_good_logprobs"].items() if k in good)
              / tokens) if tokens else 0.0
    result = {"bad_spans": len(fell), "bad_fell_share": sum(fell) / len(fell) if fell else 0.0,
              "good_span_tokens": tokens, "good_mean_change_nats_per_token": change}
    result["passed"] = bool(fell) and result["bad_fell_share"] >= 0.7 and change >= -0.05
    return result


VARIANTS = (("F", "first"), ("C", "first"), ("C", "both"))


def select_context(rt, args):
    """D21 Gate B as selector: score F, C and C+notes on the failed base episodes of fit-20.

    Identical failing calls in an episode (same tool, canonical arguments) count once. Primary score:
    task-balanced right-way share on identifying-parameter tokens; tie-break: whole call."""
    from sdpo_ab_context import SALIENT_KEYS, build_context, contract_for, episode_calls

    fit = json.loads(Path(args.fit).read_text())["task_names"]
    records, chains = load_collection(args.base)
    records = [r for r in records if task_of(r) in fit]
    siblings = defaultdict(list)
    for record in records:
        siblings[task_of(record)].append(record)
    rows, spans = [], []
    for record in records:
        if record["traces"][0]["metrics"].get("task_completed_correctly") == 1.0:
            continue
        chain = chains.get(record["id"])
        if not chain or chain.get("mismatch"):
            continue
        name = task_of(record)
        contract = contract_for(name)
        reference = build_context(record, siblings[name], "A", "first", contract, variant="F")
        if not reference or not reference["bad_spans"]:
            continue
        calls = {(c["turn"], c["call_index"]): c for c in episode_calls(record)[0]}
        actions = [s for s in chain["segments"] if s["kind"] == "action"]
        offsets = [sum(len(a["ids"]) for a in actions[:k]) for k in range(len(actions))]
        mine, seen_calls = [], set()
        for bad in reference["bad_spans"]:
            call = calls.get((bad["turn"], bad["call_index"]))
            key = (call["name"], json.dumps(call["arguments"], sort_keys=True)) if call else (bad["turn"], bad["call_index"])
            if key in seen_calls:
                continue  # an identical failing call counts once
            seen_calls.add(key)
            action = actions[bad["turn"] - 1]
            full = call_spans(rt.tokenizer, action["ids"])
            if bad["call_index"] >= len(full):
                continue
            params = parameter_spans(rt.tokenizer, action["ids"])
            base = offsets[bad["turn"] - 1]
            a, b = full[bad["call_index"]]
            ident = []
            if bad["call_index"] < len(params):
                for key_name, (x, y) in params[bad["call_index"]].items():
                    if key_name in SALIENT_KEYS:
                        ident += list(range(base + x, base + y))
            mine.append({"episode_id": record["id"], "task": name, "kind": bad["kind"], "turn": bad["turn"],
                         "call_index": bad["call_index"], "index": list(range(base + a, base + b)), "ident": ident})
        if not mine:
            continue
        student = student_sequence(chain)
        student_lp = token_logprobs(rt.model, student["ids"], student["positions"], student["targets"], "sdpo_teacher")
        for span in mine:
            span["student"] = mean_over(student_lp, span["index"])
            span["ident_student"] = mean_over(student_lp, span["ident"]) if span["ident"] else None
        for variant, placement in VARIANTS:
            context = build_context(record, siblings[name], "A", placement, contract, variant=variant)
            teacher = teacher_sequence(rt.tokenizer, chain, context)
            if teacher["targets"] != student["targets"]:
                raise ValueError("teacher and student targets differ")
            teacher_lp = token_logprobs(rt.model, teacher["ids"], teacher["positions"], teacher["targets"],
                                        "sdpo_teacher")
            for span in mine:
                row = {k: span[k] for k in ("episode_id", "task", "kind", "turn", "call_index")}
                row.update(variant=f"{variant}+notes" if placement == "both" else variant,
                           student=span["student"], teacher=mean_over(teacher_lp, span["index"]),
                           ident_student=span["ident_student"],
                           ident_teacher=mean_over(teacher_lp, span["ident"]) if span["ident"] else None)
                rows.append(row)
        spans += mine
        write(rt.out / "gate-b-select-rows.json", rows)
    scores = {}
    for variant, placement in VARIANTS:
        label = f"{variant}+notes" if placement == "both" else variant
        mine = [r for r in rows if r["variant"] == label]
        whole, ident, domains = defaultdict(list), defaultdict(list), defaultdict(list)
        for r in mine:
            whole[r["task"]].append(r["teacher"] < r["student"])
            domains[domain_of(r["task"])].append(r["teacher"] < r["student"])
            if r["ident_student"] is not None:
                ident[r["task"]].append(r["ident_teacher"] < r["ident_student"])
        balanced = lambda groups: sum(sum(v) / len(v) for v in groups.values()) / len(groups) if groups else 0.0
        scores[label] = {"variant": variant, "placement": placement, "spans": len(mine), "tasks": len(whole),
                         "ident_spans": sum(len(v) for v in ident.values()),
                         "task_balanced_ident": balanced(ident), "task_balanced_whole": balanced(whole),
                         "span_share_whole": sum(r["teacher"] < r["student"] for r in mine) / max(1, len(mine)),
                         "domains_whole": {d: sum(v) / len(v) for d, v in domains.items()}}
        log(stage="gate_b_select", label=label, **{k: v for k, v in scores[label].items() if k != "domains_whole"})
    chosen = max(scores, key=lambda k: (scores[k]["task_balanced_ident"], scores[k]["task_balanced_whole"]))
    good = good_spans(rt, records, chains)
    result = {"chosen": chosen, "variant": scores[chosen]["variant"], "placement": scores[chosen]["placement"],
              "scores": scores,
              "spans": [{k: s[k] for k in ("episode_id", "task", "kind", "turn", "call_index", "index")} for s in spans],
              "student_span_logprobs": {f"{s['episode_id']}|{s['turn']}|{s['call_index']}": s["student"] for s in spans},
              "good_spans": good,
              "student_good_logprobs": {f"{g['episode_id']}|{g['turn']}|{g['call_index']}": g["student"] for g in good}}
    write(rt.out / "gate-b-select.json", result)
    log(stage="gate_b_selected", chosen=chosen)
    return result


async def train_arm(rt, args, lr, folder):
    import torch
    from peft import get_peft_model_state_dict, set_peft_model_state_dict, load_peft_weights
    from sdpo_frozen_environment import EVALUATION_SAMPLING, ROLLOUT_SAMPLING

    fit = json.loads(Path(args.fit).read_text())["task_names"]
    gate = rt.selection
    placement, variant = gate["placement"], gate["variant"]
    if getattr(rt, "engine_context", None) is None:
        from sdpo_ab_context import build_context, contract_for

        contracts = {name: contract_for(name) for name in fit}
    else:  # D44: the context engine (plain, placement both), truth-checked per context
        build_context, contracts = rt.engine_context, {name: None for name in fit}
    # fresh student and teacher from the zero-initialized adapter
    initial = load_peft_weights(str(rt.initial))
    set_peft_model_state_dict(rt.model, initial, adapter_name="default")
    set_peft_model_state_dict(rt.model, initial, adapter_name="sdpo_teacher")
    rt.model.set_adapter("default")
    parameters = [p for n, p in rt.model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=0.01)
    await rt.rollout.suspend_for_update()
    await rt.rollout.refresh()
    history, rng = [], random.Random(20261027)
    gate_result = None
    branches = getattr(args, "r1_branches", False)  # R1 (D74): teacher takeover branches; off = R0 exactly
    sf_on = getattr(args, "sf_final", False)  # S-final (Phase 1 item 9, sdpo_final): off = R0/R1 exactly
    if sf_on:
        import sdpo_final as sf
    safety = args.gate_mode == "safety"
    if safety:
        _, base_chains = reference_collection(args)
        entropy_episodes = gate.get("entropy_episodes") or sorted({s["episode_id"] for s in gate["spans"]})
        entropy0 = mean_entropy(rt, base_chains, entropy_episodes)
        write(rt.out / folder / "entropy-start.json", {"episodes": len(entropy_episodes), "mean_entropy": entropy0})
        log(stage="entropy_start", value=entropy0)
    for batch in range(1, args.batches + 1):
        label = f"{folder}/batch-{batch}"
        records, joined, outcomes, summary = await rt.collect(label, fit, getattr(args, "rollouts_per_task", 8),
                                                              ROLLOUT_SAMPLING["temperature"],
                                                              ROLLOUT_SAMPLING["top_p"])
        if branches:  # the teacher's continuations need the rollout engine awake (before it sleeps for training)
            branch_build = await r1_build_branches(rt, args, records, joined, label)
            branch_steps = []
        if sf_on:  # S-final stop branches (flip-required keep rule), before the engine sleeps for training
            sf_branch_build = await sf.build_round_branches(rt, args, records, joined, label, outcomes)
        await rt.rollout.suspend_for_update()
        by_task = defaultdict(list)
        for record in records:
            by_task[task_of(record)].append(record)
        order = list(fit)
        rng.shuffle(order)
        admission, steps = [], []
        guard = getattr(rt, "guard", None)
        references = guard.references(list(records) + list(getattr(rt, "guard_records", []))) if guard else {}
        guard_results, saved, prepared, timings = [], [], {}, []
        for name in order:
            if sf_on:  # S-final admission (F4, F5, F11) and the attempt's own decision points
                prepared[name] = sf.prepare_task(rt, args, name, by_task[name], joined, build_context, contracts,
                                                 placement, variant, saved, timings, admission,
                                                 sf_branch_build["redundant"])
                continue
            rows = []
            for record in by_task[name]:
                chain = joined.get(record["id"])
                entry = {"episode_id": record["id"], "task": name}
                if chain is None or chain.get("mismatch"):
                    admission.append({**entry, "admitted": False, "reason": "chain_not_joined"})
                    continue
                started = time.monotonic()
                try:
                    context = build_context(record, by_task[name], args.arm, placement, contracts[name],
                                            variant=variant)
                    timings.append({"episode_id": record["id"], "task": name, "arm": args.arm, "error": False,
                                    "seconds": time.monotonic() - started,
                                    "solution": bool(context and context.get("solution_episode"))})
                except Exception as error:  # noqa: BLE001 - with the guard on, any build error is a C5 failure
                    engine = getattr(rt, "engine_context", None) is not None
                    if not guard and not engine and not isinstance(error, ValueError):
                        raise
                    if guard:
                        failures = [{"check": "C5", "kind": "build_error",
                                     "line": f"build error: {type(error).__name__}: {error}"}]
                        guard_results.append((record["id"], failures))
                        admission.append({**entry, "admitted": False, "reason": "guard:C5", "lines": failures})
                    elif engine and not isinstance(error, ValueError):  # D45: an engine crash excludes, never stops
                        rt.engine_context.stats["builder_error"] += 1
                        admission.append({**entry, "admitted": False,
                                          "reason": f"builder_error: {type(error).__name__}: {error}"})
                    else:
                        admission.append({**entry, "admitted": False, "reason": f"builder: {error}"})
                    continue
                if not guard and context is not None and getattr(rt, "engine_context", None) is not None:
                    saved.append(context_record(context, args.arm))
                if guard:
                    failures = guard.check(record, context, references.get(name, []))
                    if context is not None or failures:
                        guard_results.append((record["id"], failures))
                    if context is not None:
                        saved.append(context_record(context, args.arm, failures))
                    if failures:
                        admission.append({**entry, "admitted": False,
                                          "reason": "guard:" + ",".join(sorted({f["check"] for f in failures})),
                                          "lines": failures})
                        continue
                if context is None:
                    admission.append({**entry, "admitted": False, "reason": "success_without_feedback_or_solution"})
                    continue
                sequences = prepare_sequences(rt.tokenizer, chain, context, args.loss_tokens)
                if isinstance(sequences, str):
                    admission.append({**entry, "admitted": False, "reason": sequences})
                    continue
                student, teacher = sequences["student"], sequences["teacher"]
                anchor, decision = sequences["anchor"], sequences["decision"]
                rows.append((record, context, student, teacher, anchor, decision))
                admission.append({**entry, "admitted": True, "tokens": len(student["ids"]),
                                  "assistant_tokens": sequences["assistant_tokens"], "loss_tokens": len(student["targets"]),
                                  "anchor_tokens": len(anchor["targets"]) if anchor else 0,
                                  "teacher_tokens": len(teacher["ids"]),
                                  "solution": context["solution_episode"], "kinds": sorted({f["kind"] if isinstance(f, dict) else str(f) for f in context["facts"]}),
                                  **({"decision_points": dict(Counter(decision["reasons"])),
                                      "decision_unmatched": decision["unmatched"],
                                      "decision_only": not student["targets"]} if decision else
                                     {"decision_points": {}} if args.loss_tokens == "calls+anchor+decision" else {})})
            prepared[name] = rows
        if getattr(rt, "engine_context", None) is not None:  # D47: later batches may draw on this batch's successes
            rt.engine_context.add_to_pool(records)
        write(rt.out / label / "build-timings.json", build_timing_summary(timings))
        if getattr(rt, "engine_context", None) is not None:
            write_jsonl_gz(rt.out / label / "contexts.jsonl.gz", saved)
            write(rt.out / label / "engine-stats.json", dict(rt.engine_context.stats))
            log(stage="engine_contexts", folder=folder, batch=batch, contexts=len(saved),
                manifest_items=rt.engine_context.stats["manifest_items"],
                no_manifest=rt.engine_context.stats["record_note:no manifest for task"])
        if guard:
            write_jsonl_gz(rt.out / label / "contexts.jsonl.gz", saved)
            verdict = guard_verdict(guard_results)
            write(rt.out / label / "guard.json", verdict)
            log(stage="guard", folder=folder, batch=batch, contexts=verdict["contexts"], failed=verdict["failed"])
            if verdict["stop"]:
                write(rt.out / label / "admission.json", admission)
                write(rt.out / folder / "guard-stop.json", {"batch": batch, **verdict})
                history.append({"batch": batch, "online_success": summary["mean_task_success"], "guard": verdict})
                write(rt.out / folder / "history.json", history)
                await rt.rollout.refresh()
                return {"gate_c": gate_result, "history": history, "stopped": "guard_stop", "last_passing": batch - 1,
                        "guard": {k: verdict[k] for k in ("contexts", "failed", "failed_share")}}
        if sf_on:  # S-final: mixed-task steps (F6); R0's per-task loop below then runs no step
            sf_round = sf.run_round(rt, args, prepared, sf_branch_build, parameters, optimizer, rng, folder, batch,
                                    label, outcomes, records, history)
            steps, order = sf_round["steps"], []
            if sf_round.get("stopped"):
                write(rt.out / label / "admission.json", admission)
                write(rt.out / label / "steps.json", steps)
                await rt.rollout.refresh()
                return sf_round["stopped"]
        decision_records = []
        for name in order:
            rows = prepared.get(name, [])
            if branches:
                points = branch_build["branches"].get(name, [])
                if not rows and points:  # R1: a task whose attempts give R0 nothing still trains its branches
                    if args.r1_branch_dry_run:
                        optimizer.zero_grad(set_to_none=True)
                        branch_steps.append(r1_dry_run_step(rt, args, name, 0, points, parameters, optimizer))
                        continue
                    steps.append(r1_branch_only_step(rt, args, name, points, parameters, optimizer))
                    log(stage="step", folder=folder, batch=batch, **steps[-1])
                    continue
            if not rows:
                steps.append({"task": name, "rows": 0, "skipped": True})
                continue
            optimizer.zero_grad(set_to_none=True)
            losses, ratios, excluded, anchored, decided = [], [], 0, [], []
            for record, context, student, teacher, anchor, decision in rows:
                try:
                    parts = {}
                    if decision is None:
                        loss, ratio = episode_loss(rt.model, student, teacher, anchor=anchor, parts=parts)
                    else:
                        loss, ratio = episode_loss(rt.model, student, teacher, anchor=anchor, parts=parts,
                                                   decision=decision, decision_weight=getattr(args, "decision_weight", 1.0))
                    (loss / len(rows)).backward()
                    losses.append(float(loss.detach()))
                    anchored += [parts["anchor"]] if "anchor" in parts else []
                    if "decision" in parts:
                        decided.append(parts["decision"])
                        decision_records.append({"episode_id": record["id"], "task": name, "loss": parts["decision"],
                                                 "call_tokens": len(student["targets"]), **parts["decision_stats"]})
                    ratios += ratio
                except torch.OutOfMemoryError:
                    excluded += 1
                    torch.cuda.empty_cache()
                    admission.append({"episode_id": record["id"], "task": name, "admitted": False,
                                      "reason": "out_of_memory_excluded", "tokens": len(student["ids"])})
            if branches:
                if args.r1_branch_dry_run:  # D74 item 5: each term's gradient norm at the untrained adapter, no step
                    branch_steps.append(r1_dry_run_step(rt, args, name, len(losses), points, parameters, optimizer))
                    continue
                branch_stats = r1_branch_backward(rt, args, points, args.r1_branch_weight)
            if not losses:
                steps.append({"task": name, "rows": len(rows), "skipped": True, "oom": excluded})
                continue
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            if not torch.isfinite(norm):
                raise ValueError("nonfinite gradient")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            student_state = get_peft_model_state_dict(rt.model, adapter_name="default")
            teacher_state = get_peft_model_state_dict(rt.model, adapter_name="sdpo_teacher")
            set_peft_model_state_dict(rt.model, {k: v.lerp(student_state[k].to(v.device), 0.01)
                                                 for k, v in teacher_state.items()}, adapter_name="sdpo_teacher")
            rt.model.set_adapter("default")
            steps.append({"task": name, "rows": len(rows), "oom": excluded, "loss": sum(losses) / len(losses),
                          **({"anchor_loss": sum(anchored) / len(anchored)} if anchored else {}),
                          **({"decision_loss": sum(decided) / len(decided), "decision_rows": len(decided)}
                             if decided else {}),
                          "grad_norm": float(norm), "mean_importance_weight": sum(ratios) / max(1, len(ratios)),
                          "clipped_share": sum(r >= 2.0 for r in ratios) / max(1, len(ratios))})
            if branches:
                steps[-1].update(branch_stats)
            log(stage="step", folder=folder, batch=batch, **steps[-1])
        write(rt.out / label / "admission.json", admission)
        write(rt.out / label / "steps.json", steps)
        if branches:
            write_jsonl_gz(rt.out / label / "branches.jsonl.gz", branch_build["rows"])  # with the per-point loss stats
            if args.r1_branch_dry_run:
                from sdpo_ab_branches import weight_suggestion

                reading = {"batch": batch, "steps": branch_steps, "suggestion": weight_suggestion(branch_steps)}
                write(rt.out / folder / "branch-gradnorms.json", reading)
                log(stage="r1_branch_dry_run", **reading["suggestion"])
                await rt.rollout.refresh()
                return {"gate_c": None, "history": history, "stopped": "r1_branch_dry_run", "dry_run": reading}
        if args.loss_tokens == "calls+anchor+decision":  # recipe R monitor: every decision point of the batch
            write(rt.out / label / "decision.json", {"summary": decision_summary(decision_records),
                                                     "episodes": decision_records})
            log(stage="decision", folder=folder, batch=batch, **decision_summary(decision_records))
        entry = {"batch": batch, "online_success": summary["mean_task_success"], "steps": len(
            [s for s in steps if not s.get("skipped")]), "mean_loss": (sum(s["loss"] for s in steps if "loss" in s)
                                                                       / max(1, sum("loss" in s for s in steps)))}
        if safety:
            reading = gate_c(rt, args, gate)
            reading["mean_entropy"] = mean_entropy(rt, base_chains, entropy_episodes)
            reading["entropy_start"] = entropy0
            reading["passed"] = (reading["good_mean_change_nats_per_token"] >= -0.05
                                 and reading["mean_entropy"] >= 0.5 * entropy0)
            reading["rule"] = "D22 safety: good spans >= -0.05 nats/token and entropy >= half the start"
            entry["gate_c"] = reading
            write(rt.out / folder / f"gate-c-batch-{batch}.json", {**reading, "records": rt.gate_c_records})
            log(stage="gate_c_safety", folder=folder, batch=batch, **reading)
            if not reading["passed"]:
                history.append(entry)
                write(rt.out / folder / "history.json", history)
                last = batch - 1
                if last >= 1 and args.eval_after and not any(h.get("fit_eval") and h["batch"] == last
                                                              for h in history):
                    weights = load_peft_weights(str(rt.out / folder / f"checkpoint-{last}"))
                    set_peft_model_state_dict(rt.model, weights, adapter_name="default")
                    rt.model.set_adapter("default")
                    await rt.rollout.refresh()
                    _, _, _, evaluation = await rt.collect(f"{folder}/eval-fit-{last}", fit, 4,
                                                           EVALUATION_SAMPLING["temperature"],
                                                           EVALUATION_SAMPLING["top_p"])
                    next(h for h in history if h["batch"] == last)["fit_eval"] = {
                        "mean_task_success": evaluation["mean_task_success"], "label": f"eval-fit-{last}"}
                    write(rt.out / folder / "history.json", history)
                else:
                    await rt.rollout.refresh()
                return {"gate_c": reading, "history": history, "stopped": "safety_stop", "last_passing": last}
            entry["last_passing"] = batch
        elif batch == 1 and args.gate_mode != "none":
            gate_result = gate_c(rt, args, gate)
            entry["gate_c"] = gate_result
            write(rt.out / folder / "gate-c.json", gate_result)
            log(stage="gate_c", folder=folder, **gate_result)
            if not gate_result["passed"]:
                history.append(entry)
                write(rt.out / folder / "history.json", history)
                await rt.rollout.refresh()
                return {"gate_c": gate_result, "history": history, "stopped": "gate_c_failed"}
        checkpoint = rt.out / folder / f"checkpoint-{batch}"
        rt.model.save_pretrained(checkpoint, selected_adapters=["default"])
        await rt.rollout.refresh()
        if sf_on:  # S-final: guard10 and the post-training round rules; a stop keeps checkpoint-(batch-1)
            sf_stops = await sf.after_checkpoint(rt, args, folder, batch, label)
            if sf_stops:
                history.append({**entry, "sf_stop": sf_stops})
                write(rt.out / folder / "history.json", history)
                return {"gate_c": gate_result, "history": history, "stopped": "sf_stop", "last_passing": batch - 1,
                        "sf_stop": sf_stops}
        if batch in args.eval_after:
            _, _, _, evaluation = await rt.collect(f"{folder}/eval-fit-{batch}", fit, 4,
                                                   EVALUATION_SAMPLING["temperature"], EVALUATION_SAMPLING["top_p"])
            entry["fit_eval"] = {"mean_task_success": evaluation["mean_task_success"], "label": f"eval-fit-{batch}"}
        history.append(entry)
        write(rt.out / folder / "history.json", history)
    return {"gate_c": gate_result, "history": history, "stopped": None}


# ------------------------------------------------------------------ R1 (plan D74): teacher takeover branches

R1_BRANCH_FLAGS = ("r1_branches", "r1_branch_weight", "r1_opener_cap", "r1_branch_dry_run", "r1_point_kinds",
                   "r1_branch_fix_hints")
WEEKEND_DEFAULTS = {"a3_items": None, "a3_episodes": None, "a4_limit": 200,  # sdpo-weekend (plan v4 A3): off by default
                    "classic_version": None, "classic_batch": 16, "classic_limit": 0,  # arms 2/3 (15 Nov)
                    "classic_longest_first": False,
                    # SDPO v6 (19 Nov): the design audit and the on-policy rounds, off unless their phase runs
                    "online_rounds": 3, "online_design": None, "online_ema": 0.01, "online_decision_token": False,
                    "online_call_text": "train",  # r3 (20 Nov): fix A, "skip" = free-text call values not trained
                    "audit_failures": 200,
                    "audit_successes": 100, "audit_designs": "V5,OUT,HINT", "audit_points": None}
SF_DEFAULTS = {"sf_final": False, "sf_branch_weight": None, "sf_decision_weight": None, "sf_cap": 0.5, "sf_floor": 0.2,
               "sf_dry_run": False, "sf_branch_share": 0.25, "sf_decision_share": 0.25, "sf_points_per_task": 2,
               "sf_branch_candidates": 3, "sf_keep": 1, "sf_max_repeat": 5, "sf_guard_tasks": None,
               "sf_guard_attempts": 8, "sf_guard_reference": None, "sf_guard_drop": 0.15, "sf_guard_loops_reference": None,
               "sf_norm_probe_steps": 3, "sf_mask_values": True, "sf_no_round_stops": False}


def sf_flags_touched(args):
    """S-final: whether any --sf-* flag differs from its default (then sdpo_final.check_final_options runs)."""
    return any(getattr(args, k, v) != v for k, v in SF_DEFAULTS.items())


def check_branch_options(args):
    """The R1 flags act only on engine-train with R0's loss (calls+anchor) and need their numbers stated."""
    if not getattr(args, "r1_branches", False):
        raise ValueError("--r1-branch-weight, --r1-opener-cap, --r1-branch-dry-run, --r1-point-kinds and "
                         "--r1-branch-fix-hints need --r1-branches")
    if args.phase != "engine-train" or args.loss_tokens != "calls+anchor" or args.arm is None:
        raise ValueError("--r1-branches needs --phase engine-train --loss-tokens calls+anchor --arm (R0's loss)")
    if args.r1_opener_cap is None or not 0.0 < args.r1_opener_cap <= 1.0:
        raise ValueError("--r1-branches needs --r1-opener-cap in (0, 1]")
    if args.r1_branch_dry_run:
        if args.r1_branch_weight is not None:
            raise ValueError("--r1-branch-dry-run measures at weight 1; it takes no --r1-branch-weight")
    elif args.r1_branch_weight is None or args.r1_branch_weight <= 0.0:
        raise ValueError("--r1-branches needs --r1-branch-weight > 0 (fixed before training from the dry run)")
    from sdpo_ab_execution import world_date

    if not world_date():
        raise ValueError("--r1-branches replays attempts: pin the world date (SDPO_AB_WORLD_DATE)")


async def r1_build_branches(rt, args, records, joined, label):
    """Every mistake point of the batch (sdpo_ab_branches.build_branches): the EMA teacher served by the rollout
    engine at temperature 1.0, top-p 1.0, the worlds rebuilt in one worker thread. Writes branch-build.json."""
    import asyncio
    import concurrent.futures
    import datetime as dt

    import sdpo_ab_world as world
    from sdpo_ab_branches import BRANCH_SAMPLING, HARM, STOP, build_branches, summarize
    from sdpo_ab_execution import world_date

    by_task = defaultdict(list)
    for record in records:
        by_task[task_of(record)].append(record)
    loop = asyncio.get_running_loop()
    worker = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="r1-world")
    started = time.monotonic()

    async def run_in_world(function, *arguments):
        return await loop.run_in_executor(worker, function, *arguments)

    async def generate(ids, max_tokens, seed):
        ids, _logps, finish = await rt.rollout.sample(list(ids), max_tokens, seed=seed)
        return ids, finish

    def parse_for(chain):
        def parse(text):
            native, parser = tool_call_parser(rt.tokenizer, chain["messages0"], chain["tools"])
            return parsed_calls(parser.extract_tool_calls(text, native))
        return parse

    saved = rt.rollout.temperature, rt.rollout.top_p, rt.rollout.extra_sampling
    rt.rollout.temperature, rt.rollout.top_p = BRANCH_SAMPLING["temperature"], BRANCH_SAMPLING["top_p"]
    rt.rollout.extra_sampling = {}
    await rt.rollout.load_teacher(rt.model)
    try:
        result = await build_branches(
            records, joined, tokenizer=rt.tokenizer, engine=rt.branch_engine, generate=generate, parse_for=parse_for,
            world=world, day=dt.date.fromisoformat(world_date()), run_in_world=run_in_world,
            siblings_of=lambda record: by_task[task_of(record)],
            seed_for=lambda key: request_seed(f"{args.seed_salt}|r1-branch", key), log=log,
            concurrency=args.concurrency, point_kinds=tuple(args.r1_point_kinds or (HARM, STOP)))
    finally:
        await rt.rollout.restore_student()
        rt.rollout.temperature, rt.rollout.top_p, rt.rollout.extra_sampling = saved
        worker.shutdown(wait=True)
    summary = summarize(result["rows"])
    write(rt.out / label / "branch-build.json", {"summary": summary, "counts": dict(result["counts"]),
                                                  "elapsed_s": time.monotonic() - started,
                                                  "engine_stats": dict(rt.branch_engine.stats)})
    log(stage="r1_branches", label=label, elapsed=round(time.monotonic() - started), **dict(result["counts"]))
    return result


def r1_branch_backward(rt, args, points, weight):
    """Backward of weight x the mean over the task's points of each point's loss (sdpo_ab_branches); each point's
    stats (loss, the student's probability on the inserted tokens, opener readings) go to its branches.jsonl row."""
    import torch
    from sdpo_ab_branches import branch_point_loss

    losses, oom = [], 0
    for branch in points:
        try:
            loss, stats = branch_point_loss(rt.model, branch, args.r1_opener_cap)
            (weight * loss / len(points)).backward()
            losses.append(float(loss.detach()))
            branch["row"].update(loss=losses[-1], **stats)
        except torch.OutOfMemoryError:
            oom += 1
            torch.cuda.empty_cache()
            branch["row"]["oom"] = True
    rt.model.set_adapter("default")
    return {"branch_points": len(points), "branch_oom": oom, "branch_weight": weight,
            **({"branch_loss": sum(losses) / len(losses)} if losses else {})}


def r1_grad_norm(parameters):
    import torch

    norms = [p.grad.detach().float().norm() for p in parameters if p.grad is not None]
    return float(torch.stack(norms).norm()) if norms else 0.0


def r1_dry_run_step(rt, args, name, rows, points, parameters, optimizer):
    """D74 item 5: R0's term's gradient norm (its backward already done), then the branch term's alone (weight 1);
    gradients cleared, no optimizer step."""
    r0 = r1_grad_norm(parameters) if rows else None
    optimizer.zero_grad(set_to_none=True)
    stats = r1_branch_backward(rt, args, points, 1.0) if points else {}
    branch = r1_grad_norm(parameters) if points else None
    optimizer.zero_grad(set_to_none=True)
    return {"task": name, "rows": rows, "r0_norm": r0, "branch_norm": branch, **stats}


def r1_branch_only_step(rt, args, name, points, parameters, optimizer):
    """A task whose attempts give R0's term nothing (no admitted row) but have branch points: R0's optimizer step
    (clip 1.0, AdamW, teacher EMA 0.01) on the branch term alone."""
    import torch
    from peft import get_peft_model_state_dict, set_peft_model_state_dict

    optimizer.zero_grad(set_to_none=True)
    stats = r1_branch_backward(rt, args, points, args.r1_branch_weight)
    if "branch_loss" not in stats:
        return {"task": name, "rows": 0, "skipped": True, **stats}
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
    if not torch.isfinite(norm):
        raise ValueError("nonfinite gradient")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    student_state = get_peft_model_state_dict(rt.model, adapter_name="default")
    teacher_state = get_peft_model_state_dict(rt.model, adapter_name="sdpo_teacher")
    set_peft_model_state_dict(rt.model, {k: v.lerp(student_state[k].to(v.device), 0.01)
                                         for k, v in teacher_state.items()}, adapter_name="sdpo_teacher")
    rt.model.set_adapter("default")
    return {"task": name, "rows": 0, "branch_only": True, "grad_norm": float(norm), **stats}


def publish_adapters(rt):
    """Make adapters readable by the inspecting account (0644 files, 0755 directories)."""
    for path in rt.out.rglob("*"):
        try:
            path.chmod(0o755 if path.is_dir() else 0o644)
        except OSError:
            pass
    rt.out.chmod(0o755)


async def phase_arm(rt, args):
    import shutil

    from peft import load_peft_weights, set_peft_model_state_dict
    from sdpo_frozen_environment import EVALUATION_SAMPLING

    if args.selection:
        rt.selection = json.loads(Path(args.selection).read_text())
        write(rt.out / "gate-b-select.json", rt.selection)
    else:
        await rt.rollout.suspend_for_update()
        rt.selection = select_context(rt, args)
        await rt.rollout.refresh()

    runs = []
    rates = (args.lr,) if args.gate_mode == "safety" else (args.lr, args.lr / 2)
    for attempt, lr in enumerate(rates):
        folder = f"arm{args.arm}-lr{lr:g}"
        result = await train_arm(rt, args, lr, folder)
        runs.append({"lr": lr, "folder": folder, **result})
        write(rt.out / "arm-result.json", runs)
        if result["stopped"] != "gate_c_failed":
            break
    if runs[-1]["stopped"] == "gate_b_failed":
        log(stage="arm_not_trained", reason="no Gate B placement passed")
        return
    final = runs[-1]
    evaluated = [h for h in final["history"] if "fit_eval" in h]
    if final["stopped"] and (final["stopped"] != "safety_stop" or not evaluated):
        log(stage="arm_stopped", arm=args.arm)
        return
    best = max(evaluated, key=lambda h: (h["fit_eval"]["mean_task_success"], -h["batch"]))
    shutil.copytree(rt.out / final["folder"] / f"checkpoint-{best['batch']}",
                    rt.out / f"carried-checkpoint-arm{args.arm}-batch{best['batch']}")
    weights = load_peft_weights(str(rt.out / final["folder"] / f"checkpoint-{best['batch']}"))
    set_peft_model_state_dict(rt.model, weights, adapter_name="default")
    rt.model.set_adapter("default")
    await rt.rollout.suspend_for_update()
    await rt.rollout.refresh()
    transfer = json.loads(Path(args.transfer).read_text())["task_names"]
    _, _, _, evaluation = await rt.collect(f"{final['folder']}/eval-transfer-{best['batch']}", transfer, 4,
                                           EVALUATION_SAMPLING["temperature"], EVALUATION_SAMPLING["top_p"])
    runs[-1]["best_batch"] = best["batch"]
    runs[-1]["transfer_mean_task_success"] = evaluation["mean_task_success"]
    write(rt.out / "arm-result.json", runs)


# ------------------------------------------------------------------ handover recipe (reconciliation 2026-10-30)

def action_position(chain, row):
    """Index in `chain["segments"]` of the action a handover row trains: the action segment whose preceding tokens
    are the row's student prefix."""
    prefix = []
    for position, segment in enumerate(chain["segments"]):
        if (segment["kind"] == "action" and len(prefix) == len(row["student_prefix_ids"])
                and list(segment["ids"]) == row["action_ids"]):
            if prefix != row["student_prefix_ids"]:
                raise ValueError(f"row prefix differs from its chain: {row['episode_id']}")
            return position
        prefix += segment["ids"]
    raise ValueError(f"row action not in its chain: {row['episode_id']}")


def engine_teacher_prefix(tokenizer, chain, position, context):
    """Correction b, `--style engine`: the teacher's tokens before the action at `position`, built exactly as
    `teacher_sequence` builds engine-train's teacher (the engine's first message, its turn-1 note, and its per-turn
    notes appended to the tool results before each later turn), cut where that action starts. Notes of later turns
    are therefore not seen; the action itself is appended by `loss_sequences`."""
    return teacher_sequence(tokenizer, {**chain, "segments": chain["segments"][:position]}, context)["ids"]


def engine_teacher_rows(tokenizer, batch, records, joined, teacher, context_limit):
    """Correction b, `--style engine`: the selected handover rows with the teacher prefix replaced by the engine's.

    Rows, admission and selection are `--style raw`'s (computed before this call), so the selected actions, their
    student sequences and their action tokens are unchanged; only `teacher_prefix_ids` differs. A row whose episode
    gets no usable engine context (none, untrue, builder error) or whose engine teacher would not fit the handover's
    context limit keeps the raw teacher, as the handover itself teaches that action, and the reason is counted in
    `teacher.stats` and recorded per row (`teacher_source`)."""
    from sdpo_ab_handover import admission

    by_id = {record["id"]: record for record in records}
    by_task = defaultdict(list)
    for record in records:
        by_task[task_of(record)].append(record)
    out = []
    for row in batch:
        chain = joined[row["episode_id"]]
        position = action_position(chain, row)
        context, reason = teacher.context(by_id[row["episode_id"]], by_task[row["task"]])
        if context is not None:
            ids = engine_teacher_prefix(tokenizer, chain, position, context)
            turn = sum(s["kind"] == "action" for s in chain["segments"][:position]) + 1
            notes = {t: text for t, text in sorted(context["notes"].items()) if t <= turn}
            candidate = {**row, "teacher_prefix_ids": ids, "teacher_feedback": context["first_message"],
                         "teacher_notes": notes, "teacher_source": "engine",
                         "teacher_solution_episode": context.get("solution_episode")}
            reason = admission(candidate, context=context_limit)
            if reason is None:
                teacher.stats["engine"] += 1
                out.append(candidate)
                continue
        teacher.stats["raw_kept:" + reason.split(":")[0]] += 1
        out.append({**row, "teacher_notes": {}, "teacher_source": "raw: " + reason, "teacher_solution_episode": None})
    return out


async def phase_handover(rt, args):
    """Train the handover pool124 recipe on --fit: per update, 2 rollouts per task (8 turns, 2,048-token replies,
    16,384-token context, the 8-turn system prompt), one failure-frontier action per task with raw native-check
    feedback or a successful rollout, one optimizer step, EMA 0.01. Evaluation is the shared collect --adapter path."""
    import shutil

    import torch
    from peft import get_peft_model_state_dict, load_peft_weights, set_peft_model_state_dict
    from sdpo_ab_handover import (SELECTION_MODE, TRAINING_BUDGET, handover_rows, loss_sequences, select, update,
                                   update_tasks)

    fit = json.loads(Path(args.fit).read_text())["task_names"]
    budget = TRAINING_BUDGET
    engine = teacher = None
    if args.style != "handover":  # D42/D43 style test: the frozen context engine words the same facts
        from sdpo_ab_engine import EngineFeedback, EngineTeacher, engine_version

        # correction b: --style engine builds and selects rows exactly as --style raw (with its raw-equality check)
        # and then gives the selected rows the engine's full context as the teacher (engine_teacher_rows)
        engine = EngineFeedback("raw" if args.style == "engine" else args.style)
        teacher = (EngineTeacher(fix_hints=args.engine_fix_hints, options=engine_options(args))
                   if args.style == "engine" else None)
    initial = load_peft_weights(str(rt.initial))
    set_peft_model_state_dict(rt.model, initial, adapter_name="default")
    set_peft_model_state_dict(rt.model, initial, adapter_name="sdpo_teacher")
    rt.model.set_adapter("default")
    parameters = [p for n, p in rt.model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.01)
    await rt.rollout.suspend_for_update()
    await rt.rollout.refresh()
    limits = (rt.agent.context_limit, rt.agent.reply_limit)
    rt.agent.context_limit, rt.agent.reply_limit, rt.agent.align_budget = budget["context"], budget["reply"], True
    updates = []
    try:
        for step in range(1, args.updates + 1):
            label = f"handover/update-{step:02d}"
            tasks = update_tasks(fit, step)
            records, joined, _, summary = await rt.collect(label, tasks, budget["replicas"], 1.0, 1.0,
                                                           max_turns=budget["turns"], episode_tokens=budget["episode"])
            await rt.rollout.suspend_for_update()
            rows, excluded = handover_rows(rt.tokenizer, records, joined, first_prompt, render_delta, budget["context"],
                                           feedback=engine)
            batch, shortfall = select(rows, tasks, step - 1)
            extra = {}
            if teacher is not None:
                batch = engine_teacher_rows(rt.tokenizer, batch, records, joined, teacher, budget["context"])
                extra = {"engine_teacher": dict(teacher.stats), "engine_context_stats": dict(teacher.engine.stats),
                         "teacher_sources": [r["teacher_source"] for r in batch]}
                log(stage="engine_teacher", update=step, **{k.replace(":", "_"): v for k, v in teacher.stats.items()})
            write(rt.out / label / "training-admission.json", {
                "style": args.style, "engine_version": engine_version() if engine else None,
                **({"engine_fix_hints": args.engine_fix_hints, "engine_options": engine_options(args)}
                   if teacher is not None else {}),
                "engine_rows": dict(engine.stats) if engine else None, **extra,
                "admitted": len(rows), "exclusions": excluded, "selection_mode": SELECTION_MODE, "shortfall": shortfall,
                "selected": [{k: r[k] for k in ("task", "episode_id", "generation_index", "terminal_text",
                                                "solution_used", "reference_episode_id")} for r in batch],
                "distinct_selected_tasks": len({r["task"] for r in batch}),
                "selected_terminal_actions": sum(r["terminal_text"] for r in batch),
                "selected_observed_tool_errors": sum(r["observed_tool_error"] for r in batch)})
            fields = ("task", "episode_id", "generation_index", "teacher_feedback") + (
                ("teacher_notes", "teacher_source", "teacher_solution_episode") if teacher is not None else ())
            write_jsonl_gz(rt.out / label / "teacher-feedback.jsonl.gz", [{k: r[k] for k in fields} for r in batch])
            if engine:
                engine.reset()
            if teacher is not None:
                teacher.reset()

            def loss_of(row):
                student, teacher = loss_sequences(row)
                return episode_loss(rt.model, student, teacher)[0]

            def ema():
                student_state = get_peft_model_state_dict(rt.model, adapter_name="default")
                teacher_state = get_peft_model_state_dict(rt.model, adapter_name="sdpo_teacher")
                set_peft_model_state_dict(rt.model, {k: v.lerp(student_state[k].to(v.device), 0.01)
                                                     for k, v in teacher_state.items()}, adapter_name="sdpo_teacher")
                rt.model.set_adapter("default")

            optimizer.zero_grad(set_to_none=True)
            losses, norm = update(batch, loss_of, lambda: torch.nn.utils.clip_grad_norm_(parameters, 1.0),
                                  optimizer.step, ema, lambda x: bool(torch.isfinite(x)))
            optimizer.zero_grad(set_to_none=True)
            updates.append({"update": step, "online_success": summary["mean_task_success"],
                            "selected": len(batch), "shortfall": shortfall, "admitted": len(rows),
                            "excluded": len(excluded),
                            "loss": sum(losses) / len(batch), "grad_norm": float(norm), "lr": args.lr})
            write(rt.out / "handover" / "updates.json", updates)
            log(stage="handover_update", **updates[-1])
            if step in args.save_at and step != args.updates:  # D43 length test: intermediate checkpoints
                rt.model.save_pretrained(rt.out / f"carried-checkpoint-handover-update{step}",
                                         selected_adapters=["default"])
            await rt.rollout.refresh()
    finally:
        rt.agent.context_limit, rt.agent.reply_limit = limits
        rt.agent.align_budget = False
    checkpoint = rt.out / "handover" / f"checkpoint-{args.updates}"
    rt.model.save_pretrained(checkpoint, selected_adapters=["default"])
    shutil.copytree(checkpoint, rt.out / f"carried-checkpoint-handover-update{args.updates}")


# ------------------------------------------------------------------ D27 in-training context guard

GUARD_STOP_SHARE = 0.10


class ContextGuard:
    """D27 guard: the checker's single gating implementation (`check_context_gating`, commit 34e9e267:
    C1D, C4-C9, C11, C12; no replay) applied to one built context."""

    def __init__(self, exclusions_path=None):
        import sdpo_ab_verify_contexts as checker

        self.v = checker
        self.checks = checker.GATING_CHECKS
        # loaded once; None is passed through when the registry does not exist (C9 is then skipped)
        self.exclusions = checker.load_exclusions(exclusions_path or checker.EXCLUSIONS)

    def references(self, records):
        """C1D references per task: distinct non-error calls of successful, harm-free episodes."""
        return self.v.successful_references(records)

    def check(self, record, context, references):
        """Failures as dicts {check, line, ...}; empty when the context passes every gating check.
        C11 placeholder lines count as failures (we want none)."""
        return self.v.check_context_gating(record, context, references, exclusions=self.exclusions)


def build_timing_summary(timings):
    """Context build times (seconds), overall and by task; the first build of a task group includes the
    builder's batched verification replays (corrections and solutions, verification "run")."""
    by_task = defaultdict(list)
    for t in timings:
        by_task[t["task"]].append(t["seconds"])
    return {"contexts": len(timings), "total_seconds": sum(t["seconds"] for t in timings),
            "with_solution": sum(bool(t.get("solution")) for t in timings),
            "errors": sum(bool(t.get("error")) for t in timings),
            "by_task": {k: {"builds": len(v), "seconds": sum(v), "first": v[0], "max": max(v)}
                        for k, v in sorted(by_task.items())}, "builds": timings}


def context_record(context, arm, failures=None):
    """What is saved per built context for review (D27)."""
    return {"episode_id": context.get("episode_id"), "arm": arm, "first_message": context.get("first_message"),
            "notes": context.get("notes"), "facts": context.get("facts"),
            "skipped_checks": context.get("skipped_checks"), "solution_episode": context.get("solution_episode"),
            "coverage": context.get("coverage"),
            "guard_failures": failures or []}


def write_jsonl_gz(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def guard_verdict(results, share=GUARD_STOP_SHARE):
    """results: [(episode_id, failures)] for every attempted build. Stop when more than `share` fail."""
    failed = [(e, f) for e, f in results if f]
    rate = len(failed) / len(results) if results else 0.0
    return {"contexts": len(results), "failed": len(failed), "failed_share": rate, "stop": rate > share,
            "failures": [{"episode_id": e, "checks": sorted({x["check"] for x in f}), "lines": f} for e, f in failed]}


# ------------------------------------------------------------------ D24 helpers (pure; tested on CPU)

RUNAWAY_CHARS = 8000


def verify_adapter_config(path, rank, alpha):
    """The saved initial adapter must carry the requested rank and alpha (D24: rank 16, alpha 32)."""
    config = json.loads(Path(path).read_text())
    result = {"path": str(path), "r": config.get("r"), "lora_alpha": config.get("lora_alpha"),
              "target_modules": config.get("target_modules"), "requested": {"r": rank, "lora_alpha": alpha}}
    if config.get("r") != rank or config.get("lora_alpha") != alpha:
        raise ValueError(f"initial adapter has r={config.get('r')} alpha={config.get('lora_alpha')}, "
                         f"requested r={rank} alpha={alpha}")
    return result


def request_seed(salt, counter):
    """A per-request sampling seed, so a run never replays another run's samples."""
    return int(hashlib.sha256(f"{salt}|{counter}".encode()).hexdigest()[:8], 16)


def override_prompt(messages, queue):
    """Replace the first user message by the next queued Gate A context for that request, if any."""
    import copy

    index = next((i for i, m in enumerate(messages) if m["role"] == "user"), None)
    if index is None:
        return messages, None
    pending = queue.get(messages[index].get("content"))
    if not pending:
        return messages, None
    item = pending.pop(0)
    messages = copy.deepcopy(messages)
    messages[index]["content"] = item["content"]
    return messages, item["context_id"]


def visible_chars(action):
    """Characters of the reply the environment saw: `visible_chars` when thinking was on (reasoning excluded), else
    the whole reply (thinking off: the raw reply is the visible reply)."""
    return action["visible_chars"] if action.get("visible_chars") is not None else len(action.get("raw") or "")


def runaway(actions):
    """measures-phase2.md (plan v4, thinking-aware): a reply whose visible text is longer than RUNAWAY_CHARS
    characters, or a reply that never closed its thinking. Long thinking alone is not a runaway; reply-limit cuts are
    recorded separately (reply_limit_cut). Runs made before this definition (eval-v2 c9f8bc37 and earlier) are
    re-scored with this same function from their chains (rescore_runaway.py)."""
    return any(visible_chars(a) > RUNAWAY_CHARS or a.get("thinking_closed") is False for a in actions)


def reply_limit_cut(actions):
    return any(a.get("limit_reason") == "reply_limit" for a in actions)


def legacy_runaway(actions):
    """The Phase 1 rule (raw text incl. thinking > RUNAWAY_CHARS, or a reply-limit cut); kept only to check
    re-scored runs against the flag they recorded."""
    return any(len(a.get("raw") or "") > RUNAWAY_CHARS or a.get("limit_reason") == "reply_limit" for a in actions)


def required_checks(record):
    """(passed, total) over scored positive native checks plus manifest goals (required effects)."""
    from sdpo_ab_metrics import manifest_findings, manifest_goals, negative_family, trace_of

    assertions = (trace_of(record).get("info") or {}).get("automationbench", {}).get("assertions") or []
    positive = [a for a in assertions if not a.get("excluded") and not negative_family(a["type"])]
    met, goals = manifest_goals(manifest_findings(record))
    return sum(bool(a.get("passed")) for a in positive) + met, len(positive) + goals


def episode_row(outcome):
    total = outcome.get("required_total") or 0
    return {"task": outcome["task"], "success": outcome["success"],
            "required_rate": (outcome["required_passed"] / total) if total else None,
            "harm": float(outcome["harm"]), "runaway": float(bool(outcome.get("runaway"))),
            "tool_errors": float(outcome["tool_errors"])}


def task_balanced(rows, tasks):
    """Per-task means, then the mean over the given tasks (each task weighs the same)."""
    out = {"tasks": len(tasks), "episodes": sum(r["task"] in tasks for r in rows), "per_task": {}}
    for key in ("success", "required_rate", "harm", "runaway", "tool_errors"):
        per = {}
        for task in tasks:
            values = [r[key] for r in rows if r["task"] == task and r[key] is not None]
            if values:
                per[task] = sum(values) / len(values)
        out[key] = sum(per.values()) / len(per) if per else None
        for task, value in per.items():
            out["per_task"].setdefault(task, {})[key] = value
    return out


def gate_a_verdict(variant, control):
    """D24 stop rule: required-check rate at most 5 points below control, harm share not above it,
    runaway share at most 2 points above it."""
    checks = {"required_rate": variant["required_rate"] >= control["required_rate"] - 0.05,
              "harm": variant["harm"] <= control["harm"] + 1e-12,
              "runaway": variant["runaway"] <= control["runaway"] + 0.02}
    return {"checks": checks, "passed": all(checks.values()),
            "deltas": {k: variant[k] - control[k] for k in ("success", "required_rate", "harm", "runaway",
                                                              "tool_errors")}}


def gate_a_queues(contexts_by_task, prompts, attempts):
    """Group tasks by their number of contexts; each group is one collection with attempts x contexts
    rollouts per task, and each rollout consumes one queued context."""
    groups = defaultdict(list)
    for task, contexts in contexts_by_task.items():
        if contexts:
            groups[len(contexts)].append(task)
    plans = []
    for count, tasks in sorted(groups.items()):
        queue = {}
        for task in tasks:
            items = [{"context_id": c["context_id"], "content": c["first_message"]}
                     for c in contexts_by_task[task] for _ in range(attempts)]
            if prompts[task] in queue:
                raise ValueError("two tasks share one request prompt")
            queue[prompts[task]] = items
        plans.append({"tasks": sorted(tasks), "attempts": count * attempts, "queue": queue})
    return plans


def recompute_outcomes(folder):
    """Episode outcomes of a stored collection with the current metric code (and the D24 fields)."""
    from sdpo_ab_metrics import episode_outcome

    records, chains = load_collection(folder)
    out = []
    for record in records:
        chain = chains.get(record["id"])
        actions = [s for s in chain["segments"] if s["kind"] == "action"] if chain else None
        outcome = episode_outcome(record, actions)
        outcome["runaway"] = runaway(actions) if actions else None
        outcome["required_passed"], outcome["required_total"] = required_checks(record)
        outcome["context_id"] = chain.get("context_id") if chain else None
        out.append(outcome)
    return out


# ------------------------------------------------------------------ D24 phases

def fit_collection(args):
    """fit-20 untrained-model episodes for D24/D25: the `remeasure` collection on the repaired
    environment, which covers every fit-20 task. The 2026-10-27 base episodes carry findings from the
    old manifests and are not used."""
    fit = json.loads(Path(args.fit).read_text())["task_names"]
    if not args.remeasure_dir:
        raise ValueError("D25: fit-20 inputs come from the remeasure collection (--remeasure-dir)")
    records, chains = load_collection(args.remeasure_dir)
    records = [r for r in records if task_of(r) in fit]
    missing = sorted(set(fit) - {task_of(r) for r in records})
    if missing:
        raise ValueError(f"remeasure collection lacks fit-20 tasks: {missing}")
    return records, chains


def fit_control_outcomes(args):
    """Gate A control: the untrained model on the repaired environment (the remeasure collection)."""
    fit = json.loads(Path(args.fit).read_text())["task_names"]
    return [o for o in recompute_outcomes(args.remeasure_dir) if o["task"] in fit]


def d24_contexts(args, arm, guard=None, collection=None, timings=None):
    """Variant-D teacher contexts (placement both) for every failed fit-20 untrained episode.

    With a guard (D27), contexts failing a gating check are excluded and reported; build errors count as C5.
    Returns (contexts by task, skipped, prompts by task, guard results [(episode id, failures)], saved)."""
    from sdpo_ab_context import build_context, contract_for, first_user

    records, _ = collection if collection is not None else fit_collection(args)
    siblings = defaultdict(list)
    for record in records:
        siblings[task_of(record)].append(record)
    references = guard.references(records) if guard else {}
    contexts, skipped, prompts, results, saved = defaultdict(list), [], {}, [], []
    for record in records:
        name = task_of(record)
        prompts[name] = first_user(record)
        if record["traces"][0]["metrics"].get("task_completed_correctly") == 1.0:
            continue
        started = time.monotonic()
        try:
            context = build_context(record, siblings[name], arm, "both", contract_for(name), variant=args.variant)
        except Exception as error:  # noqa: BLE001 - with the guard on, any build error is a C5 failure
            if timings is not None:
                timings.append({"episode_id": record["id"], "task": name, "arm": arm, "error": True,
                                "seconds": time.monotonic() - started})
            if not guard and not isinstance(error, ValueError):
                raise
            skipped.append({"episode_id": record["id"], "task": name, "reason": f"{type(error).__name__}: {error}"})
            if guard:
                results.append((record["id"], [{"check": "C5", "kind": "build_error",
                                                "line": f"build error: {type(error).__name__}: {error}"}]))
            continue
        if timings is not None:
            timings.append({"episode_id": record["id"], "task": name, "arm": arm, "error": False,
                            "seconds": time.monotonic() - started,
                            "solution": bool(context and context.get("solution_episode"))})
        failures = guard.check(record, context, references.get(name, [])) if guard else []
        if guard and (context is not None or failures):
            results.append((record["id"], failures))
        if context is not None:
            saved.append(context_record(context, arm, failures))
        if failures:
            skipped.append({"episode_id": record["id"], "task": name,
                            "reason": "guard:" + ",".join(sorted({f["check"] for f in failures})), "lines": failures})
            continue
        if context is None:
            skipped.append({"episode_id": record["id"], "task": name, "reason": "no context"})
            continue
        context["context_id"] = f"{arm}|{record['id']}"
        contexts[name].append(context)
    return contexts, skipped, prompts, results, saved


async def phase_gate_a(rt, args):
    """D24 Gate A: the untrained model acts from each variant's teacher first message."""
    from sdpo_frozen_environment import EVALUATION_SAMPLING

    control = [episode_row(o) for o in fit_control_outcomes(args) if o["success"] is not None]
    result = {"variant": args.variant, "attempts_per_context": args.gate_a_attempts, "arms": {}}
    for arm in ("A", "B"):
        timings = []
        contexts, skipped, prompts, guard_results, saved = d24_contexts(args, arm, getattr(rt, "guard", None),
                                                                        timings=timings)
        write(rt.out / "gate-a" / f"build-timings-arm{arm}.json", build_timing_summary(timings))
        write_jsonl_gz(rt.out / "gate-a" / f"contexts-arm{arm}.jsonl.gz", saved)
        guard_report = guard_verdict(guard_results) if getattr(rt, "guard", None) else None
        lookup = {c["context_id"]: c for cs in contexts.values() for c in cs}
        rows = []
        for plan in gate_a_queues(contexts, prompts, args.gate_a_attempts):
            label = f"gate-a/arm{arm}-n{plan['attempts'] // args.gate_a_attempts}"
            _, _, outcomes, _ = await rt.collect(label, plan["tasks"], plan["attempts"],
                                                 EVALUATION_SAMPLING["temperature"], EVALUATION_SAMPLING["top_p"],
                                                 prompt_queue=plan["queue"])
            for outcome in outcomes:
                if outcome["success"] is None or outcome["context_id"] is None:
                    continue
                row = episode_row(outcome)
                row["context_id"] = outcome["context_id"]
                row["solution"] = bool(lookup[outcome["context_id"]]["solution_episode"])
                rows.append(row)
        tasks = sorted({r["task"] for r in rows})
        variant = task_balanced(rows, tasks)
        matched = task_balanced(control, tasks)
        result["arms"][arm] = {
            "contexts": len(lookup), "contexts_with_solution": sum(bool(c["solution_episode"]) for c in lookup.values()),
            "skipped_contexts": skipped, "episodes": len(rows), "tasks": tasks,
            "variant": variant, "control_task_matched": matched, "verdict": gate_a_verdict(variant, matched),
            "guard": guard_report}
        if guard_report and guard_report["stop"]:
            result["arms"][arm]["verdict"]["passed"] = False
            result["arms"][arm]["verdict"]["reason"] = (
                f"guard: {guard_report['failed']} of {guard_report['contexts']} contexts failed gating checks "
                f"(> {GUARD_STOP_SHARE:.0%})")
        write(rt.out / "gate-a.json", result)
        log(stage="gate_a", arm=arm, contexts=len(lookup), episodes=len(rows),
            passed=result["arms"][arm]["verdict"]["passed"], deltas=result["arms"][arm]["verdict"]["deltas"])
    return result


def d24_reference(rt, args):
    """Gate C reference for D24 (good spans, entropy episodes, bad spans from variant D's located facts)
    and the Gate B diagnostic on offending-content spans when the builder exposes them."""
    from sdpo_ab_context import episode_calls

    contexts = d24_contexts(args, "A", getattr(rt, "guard", None))[0]
    records, chains = fit_collection(args)
    by_id = {r["id"]: r for r in records}
    spans, diagnostic, exposed = [], [], 0
    for task_contexts in contexts.values():
        for context in task_contexts:
            record = by_id[context["episode_id"]]
            chain = chains.get(record["id"])
            if not chain or chain.get("mismatch"):
                continue
            calls = {(c["turn"], c["call_index"]): c for c in episode_calls(record)[0]}
            actions = [s for s in chain["segments"] if s["kind"] == "action"]
            offsets = [sum(len(a["ids"]) for a in actions[:k]) for k in range(len(actions))]
            seen, mine = set(), []
            for bad in context.get("bad_spans") or []:
                call = calls.get((bad["turn"], bad["call_index"]))
                key = (call["name"], json.dumps(call["arguments"], sort_keys=True)) if call else (bad["turn"],)
                if key in seen or bad["turn"] > len(actions):
                    continue
                seen.add(key)
                action = actions[bad["turn"] - 1]
                full = call_spans(rt.tokenizer, action["ids"])
                if bad["call_index"] >= len(full):
                    continue
                base = offsets[bad["turn"] - 1]
                a, b = full[bad["call_index"]]
                span = {"episode_id": record["id"], "task": task_of(record), "kind": bad["kind"], "turn": bad["turn"],
                        "call_index": bad["call_index"], "index": list(range(base + a, base + b))}
                text = bad.get("offending_text")
                if text:
                    content = offending_span(rt.tokenizer, action["ids"], (a, b), text)
                    if content:
                        exposed += 1
                        span["offending"] = [base + i for i in content]
                mine.append(span)
            if not mine:
                continue
            student = student_sequence(chain)
            student_lp = token_logprobs(rt.model, student["ids"], student["positions"], student["targets"],
                                        "sdpo_teacher")
            teacher_lp = None
            for span in mine:
                span["student"] = mean_over(student_lp, span["index"])
                if "offending" in span:
                    if teacher_lp is None:
                        teacher = teacher_sequence(rt.tokenizer, chain, context)
                        teacher_lp = token_logprobs(rt.model, teacher["ids"], teacher["positions"],
                                                    teacher["targets"], "sdpo_teacher")
                    diagnostic.append({k: span[k] for k in ("episode_id", "task", "kind", "turn", "call_index")}
                                      | {"student": mean_over(student_lp, span["offending"]),
                                         "teacher": mean_over(teacher_lp, span["offending"])})
            spans += mine
    good = good_spans(rt, records, chains)
    gate_b = ({"skipped": True, "reason": "variant D exposes no offending-content spans (no bad_spans with "
               "offending_text)"} if not diagnostic else
              {"skipped": False, "spans": len(diagnostic),
               "right_way_share": sum(d["teacher"] < d["student"] for d in diagnostic) / len(diagnostic),
               "rows": diagnostic, "note": "diagnostic only; never blocks (D24)"})
    reference = {"chosen": f"{args.variant}+notes", "variant": args.variant, "placement": "both",
                 "spans": [{k: s[k] for k in ("episode_id", "task", "kind", "turn", "call_index", "index")}
                           for s in spans],
                 "student_span_logprobs": {f"{s['episode_id']}|{s['turn']}|{s['call_index']}": s["student"]
                                           for s in spans},
                 "good_spans": good,
                 "student_good_logprobs": {f"{g['episode_id']}|{g['turn']}|{g['call_index']}": g["student"]
                                           for g in good},
                 "entropy_episodes": sorted(c["episode_id"] for cs in contexts.values() for c in cs
                                            if c["episode_id"] in chains),
                 "gate_b_diagnostic": gate_b, "offending_spans_exposed": exposed}
    write(rt.out / "d24-reference.json", reference)
    log(stage="d24_reference", spans=len(spans), good=len(good), gate_b=gate_b.get("right_way_share", "skipped"))
    return reference


DOUBT_WORDS = ("Wait", "Actually", "Hmm")


def neutral_context(context):
    """S1 variant: the same facts under a requirements heading, without turn numbers or per-turn notes."""
    import re
    from sdpo_ab_context import FAILED_HEADING, SUCCESS_HEADING

    first = context["first_message"].replace(FAILED_HEADING, "Requirements for this task:")
    first = re.sub(r"Turn \d+: ", "", first.replace(SUCCESS_HEADING, "Requirements for this task:"))
    return {**context, "first_message": first, "notes": {}}


def signal_variants(context):
    return {"current": context, "first_only": {**context, "notes": {}}, "neutral": neutral_context(context)}


def doubt_ids(tokenizer):
    """First token of each doubt word, with and without a leading space."""
    return sorted({tokenizer(w, add_special_tokens=False)["input_ids"][0] for word in DOUBT_WORDS
                   for w in (word, " " + word)})


def signal_positions(tokenizer, student):
    """Reply starts, and free-text paragraph starts, as indices into the student's assistant tokens."""
    calls = set(call_body_indices(tokenizer, student))
    starts, paragraphs = [], []
    for i, (token, turn) in enumerate(zip(student["targets"], student["turns"])):
        if i == 0 or student["turns"][i - 1] != turn:
            starts.append(i)
        elif i not in calls and tokenizer.decode([student["targets"][i - 1]]).endswith("\n\n"):
            paragraphs.append(i)
    return starts, paragraphs, calls


def episode_signal(model, tokenizer, student, teachers, doubt, topk=20, chunk=1024):
    """S1: per teacher variant, on the student's own tokens (no training): log P(doubt word) at reply and
    paragraph starts for student and teacher, per-token top-k + tail reverse KL by token class, and per-token
    log-probs of the sampled tokens. The adapter is the untrained one."""
    import torch
    from trl.experimental.sdpo.loss_utils import compute_topk_self_distillation_loss

    starts, paragraphs, calls = signal_positions(tokenizer, student)
    opener = tokenizer.convert_tokens_to_ids("<tool_call>")
    head, ids = lm_head(model), torch.tensor(doubt, device="cuda")
    target = torch.tensor(student["targets"], device="cuda")[:, None]
    model.set_adapter("default")
    model.eval()
    out = {"starts": starts, "paragraphs": paragraphs, "teachers": {},
           "classes": ["opener" if t == opener else "call" if i in calls else "text"
                       for i, t in enumerate(student["targets"])]}
    with torch.no_grad():
        hs = inner_hidden(model, student["ids"], student["positions"])
        for name, teacher in teachers.items():
            ht = inner_hidden(model, teacher["ids"], teacher["positions"])
            rows = {"s_doubt": [], "s_token": [], "doubt": [], "token": [], "kl": []}
            for start in range(0, len(student["positions"]), chunk):
                s_logits = head(hs[start:start + chunk]).float()
                t_logits = head(ht[start:start + chunk]).float()
                for prefix, logits in (("s_", s_logits), ("", t_logits)):
                    logp = logits.log_softmax(-1)
                    rows[prefix + "doubt"] += logp[:, ids].logsumexp(-1).tolist()
                    rows[prefix + "token"] += logp.gather(-1, target[start:start + chunk]).flatten().tolist()
                rows["kl"] += compute_topk_self_distillation_loss(s_logits, t_logits, distillation_topk=topk,
                                                                  distillation_alpha=1.0,
                                                                  distillation_add_tail=True).tolist()
            out["student"] = {"doubt": rows.pop("s_doubt"), "token": rows.pop("s_token")}
            out["teachers"][name] = rows
            del ht
    return out


def signal_summary(rows, spans):
    """Aggregate episode_signal rows and the offending spans into the S1 numbers."""
    import statistics

    names = sorted({n for r in rows for n in r["teachers"]})
    result = {}
    for name in names:
        starts = [r["teachers"][name]["doubt"][i] - r["student"]["doubt"][i] for r in rows for i in r["starts"]]
        paras = [r["teachers"][name]["doubt"][i] - r["student"]["doubt"][i] for r in rows for i in r["paragraphs"]]
        mass = defaultdict(float)
        count = defaultdict(int)
        for r in rows:
            for kind, value in zip(r["classes"], r["teachers"][name]["kl"]):
                mass[kind] += value
                count[kind] += 1
        total = sum(mass.values()) or 1.0
        mine = [s for s in spans if s["variant"] == name]
        result[name] = {
            "reply_starts": len(starts),
            "doubt_push_reply_start_nats": statistics.fmean(starts) if starts else None,
            "student_doubt_prob_reply_start": statistics.fmean(
                math.exp(r["student"]["doubt"][i]) for r in rows for i in r["starts"]) if starts else None,
            "teacher_doubt_prob_reply_start": statistics.fmean(
                math.exp(r["teachers"][name]["doubt"][i]) for r in rows for i in r["starts"]) if starts else None,
            "paragraph_starts": len(paras),
            "doubt_push_paragraph_nats": statistics.fmean(paras) if paras else None,
            "kl_share": {k: mass[k] / total for k in mass},
            "kl_per_token": {k: mass[k] / count[k] for k in mass},
            "offending_spans": len(mine),
            "offending_right_way_share": sum(s["teacher"] < s["student"] for s in mine) / len(mine) if mine else None,
            "offending_mean_margin_nats": statistics.fmean(s["student"] - s["teacher"] for s in mine) if mine else None,
        }
    return result


def phase_signal(rt, args):
    """S1 (D41): teacher-signal diagnostic by re-scoring the fit-20 reference episodes under three teacher
    prompt variants, for arms A and B. Forward passes only; nothing is trained or generated."""
    from sdpo_ab_context import episode_calls

    records, chains = fit_collection(args)
    by_id = {r["id"]: r for r in records}
    doubt = doubt_ids(rt.tokenizer)
    report = {"doubt_token_ids": doubt, "doubt_tokens": [rt.tokenizer.decode([i]) for i in doubt], "arms": {}}
    for arm in args.arms:
        contexts = d24_contexts(args, arm, getattr(rt, "guard", None))[0]
        rows, spans = [], []
        for task_contexts in contexts.values():
            for context in task_contexts:
                record = by_id[context["episode_id"]]
                chain = chains.get(record["id"])
                if not chain or chain.get("mismatch"):
                    continue
                student = student_sequence(chain)
                if not student["targets"]:
                    continue
                variants = signal_variants(context)
                teachers = {n: teacher_sequence(rt.tokenizer, chain, c) for n, c in variants.items()}
                row = episode_signal(rt.model, rt.tokenizer, student, teachers, doubt)
                calls = {(c["turn"], c["call_index"]): c for c in episode_calls(record)[0]}
                actions = [s for s in chain["segments"] if s["kind"] == "action"]
                offsets = [sum(len(a["ids"]) for a in actions[:k]) for k in range(len(actions))]
                for bad in context.get("bad_spans") or []:
                    text, call = bad.get("offending_text"), calls.get((bad["turn"], bad["call_index"]))
                    if not text or not call or bad["turn"] > len(actions):
                        continue
                    action = actions[bad["turn"] - 1]
                    full = call_spans(rt.tokenizer, action["ids"])
                    if bad["call_index"] >= len(full):
                        continue
                    content = offending_span(rt.tokenizer, action["ids"], full[bad["call_index"]], text)
                    if not content:
                        continue
                    index = [offsets[bad["turn"] - 1] + i for i in content]
                    for name in variants:
                        spans.append({"episode_id": record["id"], "task": task_of(record), "kind": bad["kind"],
                                      "variant": name, "student": mean_over(row["student"]["token"], index),
                                      "teacher": mean_over(row["teachers"][name]["token"], index)})
                rows.append(row)
        summary = signal_summary(rows, spans)
        report["arms"][arm] = {"episodes": len(rows), "variants": summary,
                               "spans": [s for s in spans]}
        log(stage="signal", arm=arm, episodes=len(rows),
            **{n: {k: v for k, v in summary[n].items() if k in ("doubt_push_reply_start_nats",
                                                                   "offending_right_way_share")}
               for n in summary})
        write(rt.out / "signal.json", report)
    return report


def offending_span(tokenizer, action_ids, call_range, text):
    """Token indices (within the reply) of the first occurrence of `text` inside one call's argument span."""
    a, b = call_range
    target = normalize_space(text)
    if not target:
        return None
    pieces = [tokenizer.decode(action_ids[a:i + 1], skip_special_tokens=False) for i in range(a, b)]
    joined = normalize_space(pieces[-1]) if pieces else ""
    start = joined.find(target)
    if start < 0:
        return None
    end = start + len(target)
    first = next(i for i, p in enumerate(pieces) if len(normalize_space(p)) > start)
    last = next((i for i, p in enumerate(pieces) if len(normalize_space(p)) >= end), len(pieces) - 1)
    return list(range(a + first, a + last + 1))


def normalize_space(text):
    return " ".join(str(text).split()).casefold()


async def run_arm_d24(rt, args, arm, control_transfer):
    """Train one arm (D22 safety stop, no fit-20 evaluations), carry batch 4 or the last passing batch,
    evaluate transfer-20 at k = args.transfer_attempts and compare with the k = 8 control (D18)."""
    import shutil

    from peft import load_peft_weights, set_peft_model_state_dict
    from sdpo_ab_metrics import compare, summary
    from sdpo_frozen_environment import EVALUATION_SAMPLING

    args.arm = arm
    folder = f"arm{arm}-lr{args.lr:g}"
    result = await train_arm(rt, args, args.lr, folder)
    carried = result.get("last_passing") if result["stopped"] in ("safety_stop", "guard_stop") else args.batches
    record = {"arm": arm, "lr": args.lr, "folder": folder, **result, "carried_batch": carried}
    if not carried:
        record["transfer"] = None
        write(rt.out / f"arm{arm}-result.json", record)
        return record
    checkpoint = rt.out / folder / f"checkpoint-{carried}"
    shutil.copytree(checkpoint, rt.out / f"carried-checkpoint-arm{arm}-batch{carried}")
    set_peft_model_state_dict(rt.model, load_peft_weights(str(checkpoint)), adapter_name="default")
    rt.model.set_adapter("default")
    await rt.rollout.suspend_for_update()
    await rt.rollout.refresh()
    transfer = json.loads(Path(args.transfer).read_text())["task_names"]
    _, _, outcomes, _ = await rt.collect(f"{folder}/eval-transfer-{carried}", transfer, args.transfer_attempts,
                                         EVALUATION_SAMPLING["temperature"], EVALUATION_SAMPLING["top_p"])
    treated = summary(outcomes, transfer, args.transfer_attempts)
    comparison = compare(control_transfer, treated, transfer)
    record["transfer"] = {"summary": {k: treated[k] for k in ("mean_task_success", "successes", "harm_share",
                                                               "preservation_break_share", "truncation_share",
                                                               "truncation_by_limit", "tool_errors_per_episode")},
                          "runaway_share": sum(bool(o.get("runaway")) for o in outcomes) / max(1, len(outcomes)),
                          "vs_control_k8": comparison,
                          "d18_m8_to_m9": {"needs_delta": 0.05, "delta": comparison["delta"],
                                           "harm_change": comparison["harm_share_change"],
                                           "preservation_change": comparison["preservation_break_share_change"],
                                           "passed": comparison["delta"] >= 0.05
                                           and comparison["harm_share_change"] <= 0.05
                                           and comparison["preservation_break_share_change"] <= 0.05},
                          "d18_m7_to_m8": "not evaluable: D24 runs no fit-20 evaluation"}
    write(rt.out / f"arm{arm}-result.json", record)
    log(stage="arm_done", arm=arm, carried=carried, transfer=treated["mean_task_success"],
        delta=comparison["delta"], ci=comparison["delta_ci95"])
    return record


def subset_file(path, names, out):
    """A task-set file restricted to `names` (smoke runs); every name must be in the original set."""
    original = json.loads(Path(path).read_text())
    unknown = sorted(set(names) - set(original["task_names"]))
    if unknown:
        raise ValueError(f"tasks not in {path}: {unknown}")
    write(out, {**original, "task_names": [n for n in original["task_names"] if n in set(names)],
                "subset_of": str(path)})
    return str(out)


async def phase_contexts(rt, args):
    """Smoke hook (D27): build both arms' contexts (verification "run") for every failed episode of a
    collection (default: the remeasure set) and write them for a byte-for-byte comparison with laptop builds."""
    folder = Path(args.contexts_from or args.remeasure_dir)
    collection = load_collection(folder)
    report, timings = {}, []
    for arm in ("A", "B"):
        _, skipped, _, results, saved = d24_contexts(args, arm, getattr(rt, "guard", None), collection=collection,
                                                     timings=timings)
        write_jsonl_gz(rt.out / f"contexts-{arm}.jsonl.gz", saved)
        report[arm] = {"contexts": len(saved), "skipped": skipped,
                       "guard": guard_verdict(results) if getattr(rt, "guard", None) else None}
    write(rt.out / "contexts-report.json", report)
    write(rt.out / "contexts-timings.json", build_timing_summary(timings))
    log(stage="contexts", **{arm: r["contexts"] for arm, r in report.items()})


async def phase_engine_train(rt, args):
    """D44 recipe O: the per-tool-call scheme (one optimizer step per task per batch, every tool call, tool-call
    bodies distilled with the other assistant tokens held at the untrained model) on --fit, with the context
    engine's truth-checked plain contexts (placement both, expected values hidden; arm B: verified sibling solution).
    No gates (Ali, 2026-10-30); excluded contexts are counted in each batch's admission.json. Keeps every batch's
    checkpoint and carries the last."""
    import shutil

    from sdpo_ab_engine import EngineContext, engine_version

    if args.loss_tokens not in ANCHORED_LOSSES or args.arm is None:
        raise ValueError("engine-train needs --loss-tokens calls+anchor (or calls+anchor+decision) and --arm")
    args.eval_after, args.gate_mode = [], "none"
    rt.guard = None
    rt.selection = {"placement": "both", "variant": None}
    rt.engine_context = EngineContext(expected_values=args.engine_values == "shown", pool=args.solution_pool,
                                      fix_hints=args.engine_fix_hints, options=engine_options(args))
    log(stage="engine_options", values=args.engine_values, fix_hints=args.engine_fix_hints, engine=engine_version(),
        **engine_options(args))
    if getattr(args, "r1_branches", False):  # R1 (D74): R0's engine settings + the final note, no solution
        from sdpo_ab_branches import make_branch_engine

        branch_hints = args.engine_fix_hints if args.r1_branch_fix_hints is None else args.r1_branch_fix_hints
        rt.branch_engine = make_branch_engine(args.engine_values == "shown", branch_hints, engine_options(args))
        log(stage="r1_branch_engine", fix_hints=branch_hints, point_kinds=args.r1_point_kinds or ["harm", "stop"])
    if getattr(args, "sf_final", False):  # S-final: the branch teacher = R1's (R0's engine settings + the final note,
        from sdpo_ab_branches import make_branch_engine  # no solution; hints as --engine-fix-hints)

        rt.branch_engine = make_branch_engine(args.engine_values == "shown", args.engine_fix_hints, engine_options(args))
        log(stage="sf_branch_engine", fix_hints=args.engine_fix_hints, cap=args.sf_cap,
            points_per_task=args.sf_points_per_task, candidates=args.sf_branch_candidates, keep=args.sf_keep)
    fit = frozenset(json.loads(Path(args.fit).read_text())["task_names"])
    pool_sources = []
    for source in args.solution_pool_from or []:  # D47: successful attempts of an earlier run's batches
        if not args.solution_pool:
            raise ValueError("--solution-pool-from needs --solution-pool")
        records, skipped = load_pool_source(source, fit)
        rt.engine_context.add_to_pool(records)
        pool_sources.append({"source": str(source), "records": len(records), "skipped_not_fit": skipped,
                             "pool": rt.engine_context.stats["pool_records"]})
        log(stage="solution_pool", **pool_sources[-1])
    if pool_sources:
        write(rt.out / "solution-pool-sources.json", pool_sources)
    folder = f"engine-arm{args.arm}"
    result = await train_arm(rt, args, args.lr, folder)
    if getattr(args, "r1_branch_dry_run", False):  # nothing trained, no checkpoint to carry
        write(rt.out / "engine-train-result.json", {**result, "engine": engine_version(),
                                                    "engine_options": engine_options(args)})
        return
    if getattr(args, "sf_dry_run", False):  # S-final dry run: gradient norms only, nothing trained
        write(rt.out / "engine-train-result.json", {**result, "engine": engine_version(),
                                                    "engine_options": engine_options(args)})
        return
    last = max(h["batch"] for h in result["history"])
    shutil.copytree(rt.out / folder / f"checkpoint-{last}", rt.out / f"carried-checkpoint-engine-batch{last}")
    write(rt.out / "engine-train-result.json", {**result, "carried_batch": last, "engine": engine_version(),
                                                "engine_values": args.engine_values,
                                                "engine_fix_hints": args.engine_fix_hints,
                                                "engine_options": engine_options(args),
                                                "solution_pool_sources": pool_sources,
                                                "engine_stats": dict(rt.engine_context.stats)})
    log(stage="engine_train_done", carried=last, engine_stats=dict(rt.engine_context.stats))


async def phase_d24(rt, args):
    """D24 sequence in one task: Gate A, transfer-20 control top-up, then the arms that passed Gate A."""
    from sdpo_ab_metrics import summary
    from sdpo_frozen_environment import EVALUATION_SAMPLING

    if args.variant is None and ({"gate-a", "arms", "contexts"} & set(args.steps)):
        raise ValueError("D24 needs --variant")
    args.eval_after = []
    args.gate_mode = "safety"
    transfer = json.loads(Path(args.transfer).read_text())["task_names"] if args.transfer else None
    if transfer is None and ({"control", "arms"} & set(args.steps)):
        raise ValueError("control and arms need --transfer")
    rt.guard = ContextGuard(args.exclusions) if args.guard else None
    if "remeasure" in args.steps:
        fit = json.loads(Path(args.fit).read_text())["task_names"]
        await rt.collect("remeasure", fit, 4, EVALUATION_SAMPLING["temperature"], EVALUATION_SAMPLING["top_p"])
        args.remeasure_dir = str(rt.out / "remeasure")
    if rt.guard is not None and args.remeasure_dir:
        rt.guard_records = fit_collection(args)[0]
    if "signal" in args.steps:
        await rt.rollout.suspend_for_update()
        phase_signal(rt, args)
        return
    if "contexts" in args.steps:
        await phase_contexts(rt, args)
    if args.only_tasks:
        args.fit = subset_file(args.fit, args.only_tasks, rt.out / "fit-subset.json")
    if args.only_transfer_tasks:
        args.transfer = subset_file(args.transfer, args.only_transfer_tasks, rt.out / "transfer-subset.json")
        transfer = json.loads(Path(args.transfer).read_text())["task_names"]
    gate_a = None
    if "gate-a" in args.steps:
        gate_a = await phase_gate_a(rt, args)
    elif args.gate_a_result:
        gate_a = json.loads(Path(args.gate_a_result).read_text())
        write(rt.out / "gate-a.json", gate_a)
    control_dir = Path(args.control_transfer_dir) if args.control_transfer_dir else None
    if "control" in args.steps:
        # D25: a fresh untrained transfer-20 control at k = 8 in the repaired environment, with the
        # same env commit, code and sampling as the trained evaluations.
        await rt.collect("control-transfer", transfer, args.transfer_attempts, EVALUATION_SAMPLING["temperature"],
                         EVALUATION_SAMPLING["top_p"])
        control_dir = rt.out / "control-transfer"
    if "arms" not in args.steps:
        return
    if gate_a is None or control_dir is None:
        raise ValueError("arms need a Gate A result and the transfer-20 control")
    control = [o for o in recompute_outcomes(control_dir) if o["task"] in transfer]
    per_task = defaultdict(int)
    for o in control:
        per_task[o["task"]] += 1
    control_transfer = summary(control, transfer, max(per_task.values()) if per_task else args.transfer_attempts)
    write(rt.out / "control-transfer-summary.json", {k: control_transfer[k] for k in (
        "episodes", "successes", "mean_task_success", "harm_share", "preservation_break_share",
        "truncation_share", "truncation_by_limit", "tool_errors_per_episode", "task_success")})
    passing = [arm for arm in args.arms if gate_a["arms"][arm]["verdict"]["passed"]]
    forced = [arm for arm in args.arms if args.force_arms and arm not in passing]
    write(rt.out / "d24-plan.json", {"passing_arms": passing, "requested_arms": args.arms,
                                     "forced_arms_smoke_only": forced})
    passing = passing + forced
    if not passing:
        log(stage="d24_no_arm_passed_gate_a")
        return
    await rt.rollout.suspend_for_update()
    rt.selection = d24_reference(rt, args)
    await rt.rollout.refresh()
    for arm in passing:
        await run_arm_d24(rt, args, arm, control_transfer)


ENGINE_FLAG_PHASES = "--phase engine-train or --phase handover --style engine"


def budget_value(text):
    """--engine-line-budget / --engine-char-budget: a positive integer, or "none" for no limit."""
    if text == "none":
        return "none"
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("a budget is a positive integer or none")
    return value


def fix_hint_kinds(text):
    """--engine-fix-hints: comma-separated engine fix-hint kind names (abce.fix.FIX_HINT_KINDS), e.g. call_fix."""
    kinds = [k.strip() for k in text.split(",") if k.strip()]
    if not kinds:
        raise argparse.ArgumentTypeError("name at least one fix-hint kind, e.g. call_fix")
    return kinds


def engine_options(args):
    """Correction r0: the corrected engine's settings given on the command line, as EngineContext options. Only the
    flags that were set appear, so a run without them calls the engine exactly as before."""
    options = {}
    for name in ("line_budget", "char_budget"):
        value = getattr(args, f"engine_{name}", None)
        if value is not None:
            options[name] = None if value == "none" else value
    if getattr(args, "engine_not_done", None) is not None:
        options["not_done_source"] = args.engine_not_done
    if getattr(args, "engine_final_note", False):
        options["final_note"] = True
    return options


def check_engine_options(args):
    """The correction r0 engine flags act only where an EngineContext is built from the command line (engine-train,
    handover --style engine); anywhere else they would be silently ignored, so they are refused. The precheck's
    teacher options are registered and fixed, so it refuses them too."""
    if engine_options(args) and not (args.phase == "engine-train"
                                     or (args.phase == "handover" and args.style == "engine")):
        raise ValueError(f"--engine-line-budget, --engine-char-budget, --engine-not-done and --engine-final-note "
                         f"need {ENGINE_FLAG_PHASES}")


async def phase_precheck(rt, args):
    """The two checks before any training (sdpo_ab_precheck): the untrained model as the teacher (the zero-initialized
    adapter of `Runtime.start`, loaded as "sdpo_teacher"; sampling through the rollout engine's untrained policy),
    on the shipped sample of the recorded control run."""
    import torch
    from sdpo_ab_engine import engine_version
    from sdpo_ab_precheck import run_precheck
    from sdpo_frozen_environment import ROLLOUT_SAMPLING

    rt.rollout.temperature, rt.rollout.top_p = ROLLOUT_SAMPLING["temperature"], ROLLOUT_SAMPLING["top_p"]
    rt.rollout.extra_sampling = {}

    async def generate(ids, max_tokens, seed):
        ids, _logps, finish = await rt.rollout.sample(list(ids), max_tokens, seed=seed)
        return ids, finish

    def parse_for(chain):
        def parse(text):
            native, parser = tool_call_parser(rt.tokenizer, chain["messages0"], chain["tools"])
            return parsed_calls(parser.extract_tool_calls(text, native))
        return parse

    def teacher_logprobs(ids, positions):
        rt.model.set_adapter("sdpo_teacher")
        rt.model.eval()
        try:
            with torch.no_grad():
                hidden = inner_hidden(rt.model, ids, positions)
                head = lm_head(rt.model)
                return [head(hidden[s:s + 1024]).float().log_softmax(-1) for s in range(0, len(positions), 1024)]
        finally:
            rt.model.set_adapter("default")

    def score_next(ids, tokens):
        try:
            (logp,) = teacher_logprobs(list(ids), [len(ids) - 1])
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            return None
        values, indices = logp[0].topk(5)
        return ({t: float(logp[0][t]) for t in tokens},
                [[int(i), rt.tokenizer.decode([int(i)]), float(v)] for v, i in zip(values, indices)])

    def top1(ids, positions):
        try:
            return [int(t) for part in teacher_logprobs(list(ids), list(positions)) for t in part.argmax(-1)]
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            return None

    if getattr(args, "precheck_mode", "registered") == "branches":  # R1's check before training (D74)
        from sdpo_ab_branches import BRANCH_SAMPLING
        from sdpo_ab_r1check import run_branch_check

        rt.rollout.temperature, rt.rollout.top_p = BRANCH_SAMPLING["temperature"], BRANCH_SAMPLING["top_p"]

        def score_inserted(branch):
            """The student's (untrained adapter's) probability of every inserted token of a branch, on the
            student's branch sequence; None when out of memory."""
            student = branch["student"]
            try:
                with torch.no_grad():
                    rt.model.eval()
                    rt.model.set_adapter("default")
                    hidden = inner_hidden(rt.model, list(student["ids"]), list(student["positions"]))
                    head = lm_head(rt.model)
                    targets = torch.tensor(student["targets"], device=hidden.device)
                    logp = torch.cat([head(hidden[s:s + 1024]).float().log_softmax(-1).gather(
                        -1, targets[s:s + 1024, None]).flatten() for s in range(0, len(student["targets"]), 1024)])
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                return None
            openers = [float(logp[i].exp()) for i in branch["openers"]]
            return {"tokens": len(student["targets"]), "logp_mean": float(logp.mean()),
                    "p_mean": float(logp.exp().mean()), "logp_min": float(logp.min()), "opener_p": openers}

        await run_branch_check(rt.precheck, tokenizer=rt.tokenizer, generate=generate, parse_for=parse_for,
                               out=rt.out, suspend=rt.rollout.suspend_for_update, score_inserted=score_inserted,
                               concurrency=args.concurrency, engine_version=engine_version(), log=log)
        return
    if getattr(args, "precheck_mode", "registered") == "refine":
        from sdpo_ab_refine_check import TOPK, run_refine_check

        def distill(student_ids, student_positions, teacher_ids, teacher_positions, targets):
            """Per position: the loss's top-20 + tail term (alpha 1, no importance weight) between the student
            adapter on the student's sequence and the teacher adapter on the teacher's, the teacher's and student's
            log-probability of the target, and both top-1 tokens."""
            from trl.experimental.sdpo.loss_utils import compute_topk_self_distillation_loss

            try:
                with torch.no_grad():
                    rt.model.eval()
                    rt.model.set_adapter("sdpo_teacher")
                    ht = inner_hidden(rt.model, list(teacher_ids), list(teacher_positions))
                    rt.model.set_adapter("default")
                    hs = inner_hidden(rt.model, list(student_ids), list(student_positions))
                    head = lm_head(rt.model)
                    out = []
                    for s in range(0, len(targets), 1024):
                        sl, tl = head(hs[s:s + 1024]).float(), head(ht[s:s + 1024]).float()
                        per = compute_topk_self_distillation_loss(sl, tl, distillation_topk=TOPK,
                                                                  distillation_alpha=1.0, distillation_add_tail=True)
                        target = torch.tensor(targets[s:s + 1024], device=sl.device)[:, None]
                        ls, lt = sl.log_softmax(-1), tl.log_softmax(-1)
                        for d, ps, pt, s1, t1 in zip(per.tolist(), ls.gather(-1, target).flatten().tolist(),
                                                     lt.gather(-1, target).flatten().tolist(),
                                                     sl.argmax(-1).tolist(), tl.argmax(-1).tolist()):
                            out.append({"div": d, "logp_student": ps, "logp_teacher": pt, "top1_student": s1,
                                        "top1_teacher": t1})
                    return out
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                return None
            finally:
                rt.model.set_adapter("default")

        await run_refine_check(rt.precheck, tokenizer=rt.tokenizer, generate=generate, parse_for=parse_for,
                               distill=distill, suspend=rt.rollout.suspend_for_update, out=rt.out,
                               concurrency=args.concurrency, engine_version=engine_version(), log=log)
        return
    await run_precheck(rt.precheck, tokenizer=rt.tokenizer, generate=generate, parse_for=parse_for,
                       score_next=score_next, top1=top1, suspend=rt.rollout.suspend_for_update, out=rt.out,
                       concurrency=args.concurrency, engine_version=engine_version(), log=log)


def collect_inputs(args):
    """Correction b: what the collect phase's inputs file adds so the run describes itself: the full sampling dict
    and the prompt suffix exactly as appended (with its file's sha256). Nothing for other phases."""
    if args.phase != "collect" or args.protocol != "ours":
        return {}
    inputs = {"collect_sampling": collect_sampling(getattr(args, "sampling", "protocol"))}
    if getattr(args, "prompt_suffix_file", None):
        inputs["prompt_suffix"] = read_prompt_suffix(args.prompt_suffix_file)
        inputs["prompt_suffix_sha256"] = hashlib.sha256(Path(args.prompt_suffix_file).read_bytes()).hexdigest()
    return inputs


async def main(args):
    from sdpo_ab_execution import pin_world_clock

    # D31: one world clock for rollouts, contexts, replays and evaluation (SDPO_AB_WORLD_DATE, UTC); the
    # 2026-10-28 controls were recorded at UTC midnight of that day
    pinned = pin_world_clock()
    check_collect_options(args)
    check_engine_options(args)
    if any(getattr(args, k, None) not in (None, False) for k in R1_BRANCH_FLAGS):
        check_branch_options(args)
    if sf_flags_touched(args):  # S-final: any --sf-* flag given runs sdpo_final's option check
        from sdpo_final import check_final_options

        check_final_options(args)
        global SF_FINAL
        SF_FINAL = bool(args.sf_final)
    global THINKING
    THINKING = getattr(args, "thinking", "off") == "on"
    rt = Runtime(args)
    if args.phase == "precheck":  # before the model is loaded: inputs, recorded files, engine, replica imports
        if args.precheck_mode == "branches":
            from sdpo_ab_r1check import preflight as branch_preflight

            if not args.precheck_sample or args.precheck_wordings or args.precheck_source:
                raise ValueError("--precheck-mode branches needs --precheck-sample and takes no --precheck-wordings "
                                 "or --precheck-source (the sources are the sample file's container paths)")
            rt.precheck = branch_preflight(args.precheck_sample, None, pinned)
        elif args.precheck_mode == "refine":
            from sdpo_ab_refine_check import preflight as refine_preflight

            if not args.precheck_sample or args.precheck_wordings or args.precheck_source:
                raise ValueError("--precheck-mode refine needs --precheck-sample and takes no --precheck-wordings or "
                                 "--precheck-source (the sources are the sample file's container paths)")
            rt.precheck = refine_preflight(args.precheck_sample, None, pinned)
        else:
            from sdpo_ab_precheck import preflight

            if not (args.precheck_sample and args.precheck_wordings):
                raise ValueError("--phase precheck needs --precheck-sample and --precheck-wordings")
            rt.precheck = preflight(args.precheck_sample, args.precheck_wordings, args.precheck_source, pinned)
    rt.out.mkdir(parents=True, exist_ok=True)
    write(rt.out / f"inputs-{args.phase}.json", {**{k: v for k, v in vars(args).items()}, "world_date": pinned,
                                                 "tz": os.environ.get("TZ"), **collect_inputs(args)})
    started = time.monotonic()
    await rt.start()
    try:
        if args.phase == "smoke":
            await phase_smoke(rt, args)
        elif args.phase == "collect":
            await phase_collect_only(rt, args)
        elif args.phase == "gateb-diagnostic":
            await phase_gateb_diagnostic(rt, args)
        elif args.phase in ("controls", "gateb"):
            await phase_controls(rt, args)
        elif args.phase == "arm":
            await phase_arm(rt, args)
        elif args.phase == "d24":
            await phase_d24(rt, args)
        elif args.phase == "handover":
            await phase_handover(rt, args)
        elif args.phase == "engine-train":
            await phase_engine_train(rt, args)
        elif args.phase == "precheck":
            await phase_precheck(rt, args)
        elif args.phase == "a3-continue":  # sdpo-weekend (plan v4 A3): measurement only, nothing trains
            from sdpo_a3 import phase_a3

            await phase_a3(rt, args, sys.modules[__name__])  # this module: its THINKING is the one set above
        elif args.phase == "a4-signal":  # sdpo-weekend (plan v4 A4): forward passes only
            from sdpo_a3 import phase_a4

            await phase_a4(rt, args, sys.modules[__name__])
        elif args.phase == "signal-check":  # plan of 16 Nov, Step 2: forward passes only, nothing trains
            from sdpo_a3 import phase_signal

            await phase_signal(rt, args, sys.modules[__name__])
        elif args.phase == "classic-train":  # sdpo-weekend (arms 2/3, 15 Nov): classic SDPO, one round
            from sdpo_classic import phase_classic

            await phase_classic(rt, args, sys.modules[__name__])
        elif args.phase == "design-audit":  # SDPO v6 P1: forward passes and reflections only, nothing trains
            from sdpo_online import phase_design_audit

            await phase_design_audit(rt, args, sys.modules[__name__])
        elif args.phase == "online-train":  # SDPO v6 P2/P3: on-policy rounds, EMA teacher
            from sdpo_online import phase_online

            await phase_online(rt, args, sys.modules[__name__])
        else:
            raise ValueError("unknown phase")
        write(rt.out / f"receipt-{args.phase}.json", {"completed": True, "elapsed_s": time.monotonic() - started,
                                                      "code_sha": os.environ.get("SDPO_AB_CODE_SHA")})
    except Exception:
        write(rt.out / f"failure-{args.phase}.json", {"traceback": traceback.format_exc(),
                                                      "elapsed_s": time.monotonic() - started})
        raise
    finally:
        try:
            await rt.close()
        finally:
            publish_adapters(rt)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "collect", "controls", "gateb", "gateb-diagnostic", "arm", "d24",
                                                 "handover", "engine-train", "precheck", "a3-continue",
                                                 "a4-signal", "classic-train", "signal-check", "design-audit",
                                                 "online-train"),
                        required=True)
    parser.add_argument("--a3-items", help="a3-continue: the CPU-audited items (item_id, episode_id, arm, turn, digest)")
    parser.add_argument("--a3-episodes", help="a3-continue: the collect folder with episodes.jsonl.gz, chains.jsonl.gz")
    parser.add_argument("--a4-limit", type=int, default=200, help="a4-signal: how many of the a3 items to measure")
    parser.add_argument("--classic-version", choices=("N", "P", "V1", "V2", "V3-1", "V3-2", "V4-1", "V4-2", "V4-31", "V4-32", "V5"),
                        help="classic-train (arms 2/3; Step 3 versions 17 Nov): the feedback version")
    parser.add_argument("--classic-batch", type=int, default=16, help="classic-train: attempts per optimizer step")
    # SDPO v6 (19 Nov; weekend/sdpo-v6/PLAN.md): the design audit (P1) and the on-policy rounds (P2/P3)
    parser.add_argument("--online-rounds", type=int, default=3, help="online-train: on-policy rounds")
    import sdpo_online as _online  # 22 Nov (round 14): the design list lives with the online code, so every measured design
    # (the S1... family since round 5, S1FUHWNLGOJY in round 14) is accepted here too; it was never added to this list
    parser.add_argument("--online-design", choices=_online.DESIGNS,
                        help="online-train: the teacher design (sdpo_online.DESIGNS; W1-W4: r3 rewrites of V5)")
    parser.add_argument("--online-call-text", choices=("train", "skip"), default="train",
                        help="online-train (r3 fix A): skip = the values of free-text tool-call parameters are not trained")
    parser.add_argument("--online-ema", type=float, default=0.01, help="online-train: EMA teacher rate per step")
    parser.add_argument("--online-decision-token", action="store_true",
                        help="online-train: also train each turn's first token after </think> (call or reply)")
    parser.add_argument("--audit-failures", type=int, default=200, help="design-audit: failed attempts measured")
    parser.add_argument("--audit-successes", type=int, default=100, help="design-audit: successful attempts measured")
    parser.add_argument("--audit-designs", default="V5,OUT,HINT", help="design-audit: designs, comma-separated")
    parser.add_argument("--audit-points", help="design-audit (r3, report-only): divergence points (episode_id, turn, correct_source) for the fair decision test")
    parser.add_argument("--classic-limit", type=int, default=0, help="classic-train: memory proof on N attempts, nothing saved")
    parser.add_argument("--classic-longest-first", action="store_true", help="classic-train: the longest attempts first")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", default="collection")
    parser.add_argument("--tasks")
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--protocol", choices=("ours", "handover"), default="ours",
                        help="collect phase: our sampled scoring, or the handover's greedy 20-turn heldout62 scoring")
    parser.add_argument("--adapter", help="collect phase: a saved LoRA checkpoint to sample from (default: untrained)")
    parser.add_argument("--max-turns", type=int, default=50, help="collect: assistant turns per episode (D10: 50)")
    parser.add_argument("--reply-limit", type=int, default=4096, help="tokens per reply (D10: 4096; plan v4: 32768)")
    parser.add_argument("--context-limit", type=int, default=65536,
                        help="tokens of context per episode, also vLLM max_model_len (D10: 65536; plan v4: 131072)")
    parser.add_argument("--rollout-timeout", type=int, default=3600, help="seconds per rollout (D10: 3600)")
    parser.add_argument("--sampling", choices=("protocol", "card", "card-thinking", SF_TRAINING_SAMPLING), default="protocol",
                        help="collect phase, protocol ours: the frozen evaluation sampling (temperature 0.6, top_p "
                             "0.95), or the Qwen3.5-4B model card's non-thinking general settings (temperature 0.7, "
                             "top_p 0.8, top_k 20, min_p 0.0, presence_penalty 1.5, repetition_penalty 1.0)")
    parser.add_argument("--prompt-suffix-file",
                        help="collect phase, protocol ours: append this file's text (after one blank line) to the "
                             "task's user message for every attempt; recorded verbatim in the inputs")
    parser.add_argument("--thinking", choices=("on", "off"), default="off",
                        help="collect phase, protocol ours (D97): render every prompt with the chat template's "
                             "enable_thinking=True, so the model reasons inside <think>...</think> before each reply; "
                             "the reasoning stays in the token chain and is cut from what the environment sees. "
                             "off (default): enable_thinking=False, as every earlier run")
    parser.add_argument("--fit")
    parser.add_argument("--transfer")
    parser.add_argument("--base", help="base measurement collection folder (episodes + chains)")
    parser.add_argument("--controls", help="controls folder with gate-b.json")
    parser.add_argument("--arm", choices=("A", "B"))
    parser.add_argument("--gate-mode", choices=("effect", "safety", "none"), default="effect")
    parser.add_argument("--selection", help="gate-b-select.json from the arm-A task (arm B reuses it)")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--eval-after", type=int, nargs="*", default=[2, 4])
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--variant", choices=("F", "C", "D"), help="teacher-context variant (D24: D, placement both)")
    parser.add_argument("--seed-salt", help="per-request sampling seeds derive from this salt and a counter")
    parser.add_argument("--steps", nargs="*", default=["remeasure", "gate-a", "control", "arms"],
                        help="d24 steps, in order: remeasure (all fit-20, k = 4), contexts (build-only smoke hook), "
                             "gate-a, control (fresh transfer-20, k = --transfer-attempts), arms; or signal alone "
                             "(S1 teacher-signal re-scoring, D41)")
    parser.add_argument("--remeasure-dir", help="remeasure collection folder (when remeasure is not a step)")
    parser.add_argument("--contexts-from", help="collection folder for the contexts step (default: --remeasure-dir)")
    parser.add_argument("--guard", action=argparse.BooleanOptionalAction, default=True,
                        help="D27 in-training context guard (d24 phase)")
    parser.add_argument("--exclusions", help="exclusion registry for C9 (default: the checker's)")
    parser.add_argument("--control-transfer-dir", help="fresh transfer-20 control folder (when control is not a step)")
    parser.add_argument("--arms", nargs="*", default=["A", "B"], help="d24 arms to train when they pass Gate A")
    parser.add_argument("--gate-a-result", help="gate-a.json from an earlier d24 task (when gate-a is not a step)")
    parser.add_argument("--gate-a-attempts", type=int, default=2)
    parser.add_argument("--transfer-attempts", type=int, default=8)
    parser.add_argument("--rollouts-per-task", type=int, default=8)
    parser.add_argument("--solution-pool", action="store_true",
                        help="engine-train (D47): a failed attempt without a verified solution in its batch is offered "
                             "the successful attempts of earlier batches and of --solution-pool-from")
    parser.add_argument("--engine-values", choices=("hidden", "shown"), default="hidden",
                        help="engine-train (recipe V): show the expected values in the engine's contexts; every "
                             "context is still truth-checked (the leak test applies only with values hidden)")
    parser.add_argument("--engine-fix-hints", type=fix_hint_kinds, default=[], metavar="KIND[,KIND...]",
                        help="engine-train (recipe V): add the engine's fix hints of these kinds, comma-separated "
                             "(abce.fix.FIX_HINT_KINDS; the study approved only call_fix; needs --engine-values "
                             "shown); handover --style engine (correction r0): the engine teacher adds them (values "
                             "are always shown there)")
    parser.add_argument("--engine-line-budget", type=budget_value,
                        help="engine-train / handover --style engine (correction r0 engine): at most N statement "
                             "lines in the first message and per note, or none for no limit (unset: engine default)")
    parser.add_argument("--engine-char-budget", type=budget_value,
                        help="as --engine-line-budget, for the characters of the first message's statement lines")
    parser.add_argument("--engine-not-done", choices=("all", "benchmark"),
                        help="which 'Not done' lines the engine shows: all, or only benchmark-backed ones "
                             "(unset: engine default, all)")
    parser.add_argument("--engine-final-note", action="store_true",
                        help="the engine adds a note of the work still not done, placed right before the attempt's "
                             "final reply (an assistant message without a tool call)")
    parser.add_argument("--precheck-sample", help="precheck: the shipped sample file (/stage/precheck-sample.json)")
    parser.add_argument("--precheck-wordings",
                        help="precheck: the final-note wordings file (/stage/precheck-wordings.json)")
    parser.add_argument("--precheck-source",
                        help="precheck: the recorded collection folder (default: the sample file's container path)")
    parser.add_argument("--precheck-mode", choices=("registered", "refine", "branches"), default="registered",
                        help="precheck: the two registered checks (default), the refinement check of the engine "
                             "kinds content_at_call and value_location (sdpo_ab_refine_check), or R1's check before "
                             "training of its branches (sdpo_ab_r1check, D74); the last two need only "
                             "--precheck-sample")
    parser.add_argument("--solution-pool-from", nargs="*", default=[],
                        help="collection folders (episodes.jsonl.gz) whose successful attempts on --fit tasks seed the pool "
                             "(episodes of other tasks are skipped unread by the engine, counted per source)")
    parser.add_argument("--only-tasks", nargs="*", help="smoke: restrict Gate A and training to these fit-20 tasks")
    parser.add_argument("--only-transfer-tasks", nargs="*", help="smoke: restrict transfer evaluation to these tasks")
    parser.add_argument("--force-arms", action="store_true",
                        help="smoke only: train the requested arms even if they fail Gate A (recorded)")
    parser.add_argument("--loss-tokens", choices=("all", "calls", "calls+anchor", "calls+anchor+decision", "reasoning+calls"),
                        default="all",
                        help="d24 training: distil on every assistant token, only on tool-call bodies (D36), on "
                             "tool-call bodies with the other assistant tokens held at the untrained model (D38), "
                             "or D38 plus recipe R's one-sided Bernoulli KL on P(<tool_call>) at the decision "
                             "points the engine has evidence about (located harm openers may only lose P(call); "
                             "the final reply of an attempt with 'Not done' facts may only gain it)")
    parser.add_argument("--decision-weight", type=float, default=1.0,
                        help="calls+anchor+decision: weight of the decision term (fixed in advance, as the anchor's)")
    parser.add_argument("--style", choices=("handover", "raw", "plain_matched", "engine"), default="handover",
                        help="handover phase: the handover's own feedback text, or the frozen context engine's "
                             "raw (equal to it) or plain_matched wording of the same facts; engine: --style raw's "
                             "rows and selection, with the teacher seeing the engine's full context instead "
                             "(first message and per-turn notes up to the trained action; values shown, no fix "
                             "hints, same-batch verified sibling solution, truth-checked)")
    parser.add_argument("--updates", type=int, default=10,
                        help="handover phase: optimizer updates, one rollout collection each (the handover's step 10)")
    parser.add_argument("--save-at", type=int, nargs="*", default=[],
                        help="handover phase: also keep the student adapter after these updates")
    parser.add_argument("--kv-cache-gib", type=int, default=16)
    parser.add_argument("--r1-branches", action="store_true",
                        help="engine-train (R1, plan D74): add teacher takeover branches at mistake points (a stop "
                             "with benchmark work undone; a located harmful whole call) to R0's loss")
    parser.add_argument("--r1-branch-weight", type=float,
                        help="--r1-branches: the branch term's weight (fixed before training from the dry run)")
    parser.add_argument("--r1-opener-cap", type=float,
                        help="--r1-branches: the cap on the opener term's target P(<tool_call>) (0 < cap <= 1)")
    parser.add_argument("--r1-point-kinds", nargs="+", choices=("stop", "harm"),
                        help="--r1-branches: the mistake-point kinds that get branches (default both; R1 plan D95: stop)")
    parser.add_argument("--r1-branch-fix-hints", type=fix_hint_kinds, metavar="KIND[,KIND...]",
                        help="--r1-branches: the branch teacher's fix hints (default: --engine-fix-hints; R1 plan D95: "
                             "call_fix, as the D91 branch check measured)")
    parser.add_argument("--r1-branch-dry-run", action="store_true",
                        help="--r1-branches: round 1 only; per optimizer step, the gradient norm of R0's term and of "
                             "the branch term at the untrained adapter; nothing is trained or saved")
    parser.add_argument("--sf-final", action="store_true",
                        help="engine-train (S-final, Phase 1 item 9, sdpo_final): R0's loss plus the two-sided capped "
                             "decision term, flip-required stop branches with trained stops, success stops, arm-A "
                             "successes, runaway/incomplete/repeated-call drops, mixed-task steps, round monitors")
    parser.add_argument("--sf-branch-weight", type=float, help="--sf-final: the branch body term's weight (from the dry run)")
    parser.add_argument("--sf-decision-weight", type=float, help="--sf-final: the decision term's weight (from the dry run)")
    parser.add_argument("--sf-cap", type=float, default=0.5,
                        help="--sf-final: the constant target of an up push")
    parser.add_argument("--sf-floor", type=float, default=0.2,
                        help="--sf-final: the constant target of a down push")
    parser.add_argument("--sf-dry-run", action="store_true",
                        help="--sf-final: round 1 only; per step the gradient norm of R0's term, the branch body term "
                             "and the decision term at the untrained adapter (weight 1); nothing trained or saved")
    parser.add_argument("--sf-branch-share", type=float, default=0.25,
                        help="--sf-dry-run: the branch body term's gradient norm as a share of R0's (weight proposal)")
    parser.add_argument("--sf-decision-share", type=float, default=0.25,
                        help="--sf-dry-run: the decision term's gradient norm as a share of R0's (weight proposal)")
    parser.add_argument("--sf-points-per-task", type=int, default=2,
                        help="--sf-final: at most this many points of each kind per task per round")
    parser.add_argument("--sf-branch-candidates", type=int, default=3,
                        help="--sf-final: stop-branch candidates continued per task (the first points-per-task with a "
                             "kept call are trained)")
    parser.add_argument("--sf-keep", type=int, default=1, help="--sf-final: at most this many kept calls per branch")
    parser.add_argument("--sf-max-repeat", type=int, default=5,
                        help="--sf-final: an attempt with one identical call (same name and arguments) made this many "
                             "times is dropped (F5); the loop monitors use the same reading")
    parser.add_argument("--sf-guard-tasks", help="--sf-final: guard task file (guard10.json) scored after every round")
    parser.add_argument("--sf-guard-attempts", type=int, default=8, help="--sf-final: guard attempts per task")
    parser.add_argument("--sf-guard-reference", type=float,
                        help="--sf-final: the untrained guard task-mean success at the same settings (stop rule)")
    parser.add_argument("--sf-guard-drop", type=float, default=0.15,
                        help="--sf-final: stop when the guard mean falls below the reference by more than this")
    parser.add_argument("--sf-guard-loops-reference", type=int,
                        help="--sf-final: the untrained guard collection's loop attempts (turn limit or an identical call) "
                             "at the same settings; the guard_loops stop rule reads it")
    parser.add_argument("--sf-norm-probe-steps", type=int, default=3,
                        help="--sf-final: on the first N steps of every training round, measure each term's gradient "
                             "norm alone before the step (the mid-training distill/anchor/branch/decision split)")
    parser.add_argument("--sf-mask-values", action=argparse.BooleanOptionalAction, default=True,
                        help="--sf-final: leave the kept calls' parameter-value tokens out of the branch body term")
    parser.add_argument("--sf-no-round-stops", action="store_true",
                        help="--sf-final: record the round readings but never stop the run on them (smoke runs)")
    return parser


if __name__ == "__main__":
    asyncio.run(main(build_parser().parse_args()))
