#!/usr/bin/env python3
"""Replay recorded AutomationBench episodes turn by turn against vLLM.

Each recorded model call is re-sent with its exact prompt token ids and
generates exactly the recorded number of output tokens (``ignore_eos``), so the
prefill and decode work is identical across engine configurations. Turns of one
episode run in order; episodes run concurrently. Between turns the replay waits
the recorded gap between calls (tool execution plus harness), or no time.

This isolates server-side effects: prefix caching, prefill limits, KV capacity,
CUDA graphs, and speculative decoding over the recorded outputs' lengths.
It does not reproduce the model's own choices; the live benchmark does that.

``--mode collections`` (the default) mirrors RL rollout collection: the recorded
samples of ``--groups-per-collection`` tasks start together, the collection
ends when its slowest episode ends (the trainer's barrier), and the prefix
cache is reset before each collection as a policy synchronization does.
``--mode stream`` feeds episodes through a concurrency limit instead, which
resembles evaluation traffic rather than training.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Turn:
    # Tokens appended to the episode's running context before this call: the
    # whole prompt for the first call, then tool results plus the assistant
    # header. The replay sends running context + delta, where the running
    # context holds the tokens the replay actually generated, so prefix reuse
    # behaves as in a live episode.
    delta: list[int]
    output_tokens: int
    wait_before_s: float


@dataclass
class Episode:
    episode_id: str
    task: str
    turns: list[Turn]


@dataclass
class TurnResult:
    episode: int
    turn: int
    submitted_s: float
    first_token_s: float
    finished_s: float
    prompt_tokens: int
    cached_tokens: int
    output_tokens: int
    # Tokens of the episode's previous context (prompt plus generated output)
    # that this call could reuse; 0 for an episode's first call.
    reusable_tokens: int = 0
    task: str = ""


@dataclass
class RepeatResult:
    repeat: int
    wall_s: float
    turns: list[TurnResult] = field(default_factory=list)
    episode_wall_s: list[float] = field(default_factory=list)
    collection_wall_s: list[float] = field(default_factory=list)
    metrics_before: dict[str, float] = field(default_factory=dict)
    metrics_after: dict[str, float] = field(default_factory=dict)


def load_episodes(path: Path, copies: int) -> list[Episode]:
    """Rebuild every model call's exact prompt from a native Verifiers trace."""
    episodes = []
    with path.open() as handle:
        for line in handle:
            trace = json.loads(line)
            nodes = trace["nodes"]
            task = (
                trace.get("info", {}).get("automationbench", {}).get("task_name")
                or trace.get("task", {}).get("key")
                or ""
            )
            turns: list[Turn] = []
            previous_end = None
            previous_full: list[int] = []
            for call in trace["calls"]:
                node_index = call["node"]
                if node_index is None:  # the call failed before sampling
                    continue
                node = nodes[node_index]
                # Leading unmasked tokens of the sampled node are the assistant
                # header the renderer appended to the prompt.
                header = next(
                    (i for i, sampled in enumerate(node["mask"]) if sampled),
                    len(node["mask"]),
                )
                ancestors = []
                parent = node["parent"]
                while parent is not None:
                    ancestors.append(parent)
                    parent = nodes[parent]["parent"]
                prompt = [t for i in reversed(ancestors) for t in nodes[i]["token_ids"]]
                prompt += node["token_ids"][:header]
                output = len(node["token_ids"]) - header
                assert len(prompt) == call["usage"]["prompt_tokens"], trace["id"]
                assert output == call["usage"]["completion_tokens"], trace["id"]
                # Every recorded prompt extends the previous prompt plus output.
                assert prompt[: len(previous_full)] == previous_full, trace["id"]
                wait = 0.0 if previous_end is None else call["time"]["start"] - previous_end
                previous_end = call["time"]["end"]
                turns.append(Turn(prompt[len(previous_full) :], output, max(0.0, wait)))
                previous_full = prompt + node["token_ids"][header:]
            if turns:
                episodes.append(Episode(trace["id"], task, turns))
    return [e for _ in range(copies) for e in episodes]


def _metrics_snapshot() -> dict[str, float]:
    try:
        from vllm.v1.metrics.reader import Counter, get_metrics_snapshot
    except ImportError:
        return {}
    wanted = ("prefix_cache", "preemption", "prompt_tokens", "generation_tokens")
    values: dict[str, float] = {}
    for metric in get_metrics_snapshot():
        if isinstance(metric, Counter) and any(w in metric.name for w in wanted):
            values[metric.name] = values.get(metric.name, 0.0) + float(metric.value)
    return values


SAMPLING: dict[str, float] = {"temperature": 0.0, "top_p": 1.0}


async def _run_turn(engine, prompt, output_tokens, detokenize, lora, index):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.sampling_params import RequestOutputKind

    params = SamplingParams(
        max_tokens=output_tokens,
        min_tokens=output_tokens,
        ignore_eos=True,
        temperature=SAMPLING["temperature"],
        top_p=SAMPLING["top_p"],
        seed=(index[0] * 1000 + index[1]) % (2**31) if SAMPLING["temperature"] > 0 else None,
        detokenize=detokenize,
        output_kind=RequestOutputKind.DELTA,
    )
    submitted = time.perf_counter()
    first = None
    generated: list[int] = []
    cached = 0
    async for out in engine.generate(
        TokensPrompt(prompt_token_ids=prompt), params, uuid.uuid4().hex, lora_request=lora
    ):
        if out.outputs and out.outputs[0].token_ids:
            if first is None:
                first = time.perf_counter()
            generated.extend(out.outputs[0].token_ids)
        if out.num_cached_tokens:
            cached = out.num_cached_tokens
    finished = time.perf_counter()
    return TurnResult(
        episode=index[0],
        turn=index[1],
        submitted_s=submitted,
        first_token_s=first if first is not None else finished,
        finished_s=finished,
        prompt_tokens=len(prompt),
        cached_tokens=cached,
        output_tokens=len(generated),
    ), generated


async def _run_episode(engine, index, episode, tool_wait, detokenize, lora, result):
    start = time.perf_counter()
    context: list[int] = []
    for turn_index, turn in enumerate(episode.turns):
        if tool_wait == "recorded" and turn.wait_before_s > 0:
            await asyncio.sleep(turn.wait_before_s)
        prompt = context + turn.delta
        record, generated = await _run_turn(engine, prompt, turn.output_tokens, detokenize, lora, (index, turn_index))
        record.reusable_tokens = len(context)
        record.task = episode.task
        result.turns.append(record)
        context = prompt + generated
    result.episode_wall_s.append(time.perf_counter() - start)


async def _run_repeat(engine, episodes, concurrency, tool_wait, detokenize, lora, repeat):
    await engine.reset_prefix_cache()
    result = RepeatResult(repeat=repeat, wall_s=0.0, metrics_before=_metrics_snapshot())
    gate = asyncio.Semaphore(concurrency)

    async def one(index: int, episode: Episode):
        async with gate:
            await _run_episode(engine, index, episode, tool_wait, detokenize, lora, result)

    start = time.perf_counter()
    await asyncio.gather(*(one(i, e) for i, e in enumerate(episodes)))
    result.wall_s = time.perf_counter() - start
    result.metrics_after = _metrics_snapshot()
    return result


async def _run_collections(engine, episodes, groups_per_collection, tool_wait, detokenize, lora, repeat):
    """Run episodes as RL collections: each collection's episodes start together."""
    by_task: dict[str, list[int]] = {}
    for index, episode in enumerate(episodes):
        by_task.setdefault(episode.task, []).append(index)
    tasks = list(by_task)
    # Only full collections: a short final collection would shorten the barrier.
    collections = [
        [i for task in tasks[k : k + groups_per_collection] for i in by_task[task]]
        for k in range(0, len(tasks) - groups_per_collection + 1, groups_per_collection)
    ]
    result = RepeatResult(repeat=repeat, wall_s=0.0, metrics_before=_metrics_snapshot())

    start = time.perf_counter()
    for members in collections:
        await engine.reset_prefix_cache()  # a policy synchronization invalidates cached prefixes
        began = time.perf_counter()
        await asyncio.gather(
            *(_run_episode(engine, i, episodes[i], tool_wait, detokenize, lora, result) for i in members)
        )
        result.collection_wall_s.append(time.perf_counter() - began)
    result.wall_s = time.perf_counter() - start
    result.metrics_after = _metrics_snapshot()
    return result


async def _warm_up(engine, lora, vocab_size: int):
    # Unrelated tokens, so no recorded prefix is cached before measuring.
    prompts = [[(17 + 7919 * i + j) % (vocab_size - 1000) + 500 for j in range(2048)] for i in range(4)]
    # Same sampling as the measurement, so no path compiles during it.
    await asyncio.gather(*(_run_turn(engine, p, 64, True, lora, (-1, i)) for i, p in enumerate(prompts)))


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    k = (len(ordered) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def summarize_repeat(r: RepeatResult) -> dict[str, float]:
    ttft = [t.first_token_s - t.submitted_s for t in r.turns]
    latency = [t.finished_s - t.submitted_s for t in r.turns]
    prompt = sum(t.prompt_tokens for t in r.turns)
    cached = sum(t.cached_tokens for t in r.turns)
    output = sum(t.output_tokens for t in r.turns)
    episodes = len(r.episode_wall_s)
    collections = r.collection_wall_s
    later = [t for t in r.turns if t.reusable_tokens > 0]
    reusable = sum(t.reusable_tokens for t in later)
    reuse_lost = sum(max(0, t.reusable_tokens - t.cached_tokens) for t in later)
    first = [t for t in r.turns if t.reusable_tokens == 0]
    first_prefilled = sum(t.prompt_tokens - t.cached_tokens for t in first)
    unique_first: dict[str, int] = {}
    for t in first:
        unique_first[t.task] = max(unique_first.get(t.task, 0), t.prompt_tokens)
    return {
        "collections": len(collections),
        "collection_mean_s": statistics.fmean(collections) if collections else math.nan,
        "collection_p95_s": _percentile(collections, 0.95),
        "collection_total_s": sum(collections) if collections else math.nan,
        "episodes": episodes,
        "turns": len(r.turns),
        "wall_s": r.wall_s,
        "episodes_per_hour": episodes / r.wall_s * 3600.0,
        "episode_wall_mean_s": statistics.fmean(r.episode_wall_s),
        "episode_wall_p95_s": _percentile(r.episode_wall_s, 0.95),
        "ttft_mean_s": statistics.fmean(ttft),
        "ttft_p95_s": _percentile(ttft, 0.95),
        "turn_latency_mean_s": statistics.fmean(latency),
        "output_tokens_per_s": output / r.wall_s,
        "prompt_tokens": prompt,
        "cached_prompt_tokens": cached,
        "prefilled_tokens": prompt - cached,
        "prefix_hit_rate": cached / prompt if prompt else 0.0,
        "output_tokens": output,
        "reusable_tokens": reusable,
        "reuse_lost_tokens": reuse_lost,
        "reuse_lost_rate": reuse_lost / reusable if reusable else 0.0,
        "first_turn_prefilled_tokens": first_prefilled,
        "group_duplicate_prefill_tokens": max(0, first_prefilled - sum(unique_first.values())),
        "preemptions": sum(v - r.metrics_before.get(k, 0.0) for k, v in r.metrics_after.items() if "preemption" in k),
    }


def _environment_facts(model: str, traces: Path) -> dict[str, str]:
    facts = {"host": platform.node(), "model": model}
    facts["traces_sha256"] = hashlib.sha256(traces.read_bytes()).hexdigest()
    try:
        import torch
        import vllm

        facts["vllm_version"] = vllm.__version__
        facts["gpu"] = torch.cuda.get_device_name(0)
        root = Path(vllm.__file__).resolve().parents[1]
        facts["vllm_commit"] = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except Exception as error:  # facts are best-effort metadata
        facts["facts_error"] = repr(error)
    facts["VLLM_BATCH_INVARIANT"] = os.environ.get("VLLM_BATCH_INVARIANT", "0")
    return facts


async def main_async(args) -> None:
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.lora.request import LoRARequest
    from vllm.v1.engine.async_llm import AsyncLLM

    binding = json.loads(args.binding_config.read_text())
    engine_args = dict(binding["engine_args"])
    engine_args.update(json.loads(args.override) if args.override else {})
    episodes = load_episodes(args.traces, args.copies)
    if args.limit:
        episodes = episodes[: args.limit]
    lora = None
    if args.lora_dir:
        engine_args.setdefault("enable_lora", True)
        lora = LoRARequest("policy", 1, str(args.lora_dir))
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(model=args.model, **engine_args))
    cache_config = engine.vllm_config.cache_config
    # Prefix hits land on block boundaries; for hybrid models the block can be
    # large, so part of "reuse lost" is rounding rather than eviction.
    cache_facts = {
        "block_size": cache_config.block_size,
        "mamba_block_size": getattr(cache_config, "mamba_block_size", None),
        "mamba_cache_mode": getattr(cache_config, "mamba_cache_mode", None),
    }
    print(f"[{args.label}] cache {cache_facts}", flush=True)
    try:
        vocab = engine.model_config.get_vocab_size()
        await _warm_up(engine, lora, vocab)
        repeats = []
        for repeat in range(args.repeats):
            if args.mode == "collections":
                r = await _run_collections(
                    engine,
                    episodes,
                    args.groups_per_collection,
                    args.tool_wait,
                    not args.no_detokenize,
                    lora,
                    repeat,
                )
            else:
                r = await _run_repeat(
                    engine,
                    episodes,
                    args.concurrency,
                    args.tool_wait,
                    not args.no_detokenize,
                    lora,
                    repeat,
                )
            summary = summarize_repeat(r)
            print(
                f"[{args.label}] repeat {repeat}: collection {summary['collection_mean_s']:.1f} s "
                f"(p95 {summary['collection_p95_s']:.1f}, total {summary['collection_total_s']:.1f}), "
                f"{summary['episodes_per_hour']:.1f} episodes/h, "
                f"ttft {summary['ttft_mean_s'] * 1000:.0f} ms (p95 {summary['ttft_p95_s'] * 1000:.0f}), "
                f"prefix hit {summary['prefix_hit_rate']:.1%}, "
                f"prefilled {summary['prefilled_tokens']:,} tok, "
                f"reuse lost {summary['reuse_lost_rate']:.1%} ({summary['reuse_lost_tokens']:,} tok), "
                f"group duplicate prefill {summary['group_duplicate_prefill_tokens']:,} tok, "
                f"wall {summary['wall_s']:.1f} s",
                flush=True,
            )
            repeats.append({"summary": summary, "detail": asdict(r) if args.keep_detail else None})
    finally:
        engine.shutdown()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "label": args.label,
                "binding": binding,
                "engine_args": engine_args,
                "lora_dir": str(args.lora_dir) if args.lora_dir else None,
                "mode": args.mode,
                "concurrency": args.concurrency,
                "groups_per_collection": args.groups_per_collection,
                "sampling": dict(SAMPLING),
                "tool_wait": args.tool_wait,
                "detokenize": not args.no_detokenize,
                "copies": args.copies,
                "episodes": len(episodes),
                "turns_per_repeat": sum(len(e.turns) for e in episodes),
                "facts": _environment_facts(args.model, args.traces),
                "cache": cache_facts,
                "repeats": repeats,
            },
            indent=1,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--binding-config",
        type=Path,
        required=True,
        help="JSON from export_binding.py; its engine_args are used verbatim",
    )
    parser.add_argument(
        "--override", default=None, help="JSON object of engine arguments applied on top of the binding"
    )
    parser.add_argument("--lora-dir", type=Path, default=None)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("collections", "stream"), default="collections")
    parser.add_argument(
        "--groups-per-collection",
        type=int,
        default=16,
        help="tasks per RL collection; each contributes all its recorded samples",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--concurrency", type=int, default=32, help="stream mode only")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--copies", type=int, default=1, help="replay every recorded episode this many times")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--tool-wait", choices=("recorded", "zero"), default="recorded")
    parser.add_argument("--no-detokenize", action="store_true")
    parser.add_argument("--keep-detail", action="store_true")
    args = parser.parse_args()
    SAMPLING.update(temperature=args.temperature, top_p=args.top_p)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
