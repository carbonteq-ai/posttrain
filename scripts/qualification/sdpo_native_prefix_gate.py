"""Greedy native-prefix repeatability across cache boundaries and forty requests."""

import asyncio
import hashlib
import json


class NativePrefixMismatch(ValueError):
    def __init__(self, receipt):
        super().__init__("native prefix token/logprob invariance failed")
        self.receipt = receipt


async def qualify(rollout, fixture, max_tokens=256):
    temperature = getattr(rollout, "temperature", 0.)
    rollout.temperature = 0.
    try:
        return await _qualify(rollout, fixture, max_tokens)
    finally:
        rollout.temperature = temperature


async def _qualify(rollout, fixture, max_tokens):
    cases = fixture["cases"]
    if len(cases) != 4 or max_tokens != 256:
        raise ValueError("four retained native prefixes and256probe tokens required")
    for case in cases:
        ids = case["prefix_ids"]
        if (not ids or not all(type(token) is int for token in ids)
                or hashlib.sha256(json.dumps(ids).encode()).hexdigest() != case["prefix_sha256"]):
            raise ValueError("native prefix fixture identity differs")
    records, references = [], []

    async def reset():
        if not await rollout.engine.reset_prefix_cache():
            raise ValueError("native gate cache reset acknowledgement missing")

    async def generate(index, phase):
        ids, complete, info = await rollout.generate(cases[index]["prefix_ids"], max_tokens=max_tokens)
        return {"case": cases[index]["task"], "phase": phase, "ids": ids,
                "sampled_logprobs": info["sampled_logprobs"], "cached_tokens": info.get("num_cached_tokens"),
                "complete": complete}

    def compare(index, current):
        previous = references[index]
        equal = current["ids"] == previous["ids"]
        delta = (max((abs(a - b) for a, b in zip(current["sampled_logprobs"],
                    previous["sampled_logprobs"], strict=True)), default=0.) if equal else None)
        records.append({**current, "tokens_exact": equal, "max_abs_logp_delta": delta})
        if not equal or delta != 0.:
            raise NativePrefixMismatch({"qualified": False, "records": records,
                "references": references, "optimizer_updates": 0})

    for index in range(4):
        await reset()
        references.append(await generate(index, "cold_reference"))
    for tail in (640, 1280):
        for index, case in enumerate(cases):
            await reset()
            await rollout.generate(case["prefix_ids"][:-tail], max_tokens=1)
            compare(index, await generate(index, f"prewarm_tail_{tail}"))
    active, peak = 0, 0

    async def concurrent(index, phase):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            return await generate(index, phase)
        finally:
            active -= 1

    await reset()
    for phase in ("mixed40_cold", "mixed40_warm"):
        results = await asyncio.gather(*(concurrent(index % 4, phase) for index in range(40)))
        for index, result in enumerate(results):
            compare(index % 4, result)
    if peak != 40:
        raise ValueError("real native prefix gate did not dispatch40concurrent requests")
    return {"qualified": True, "records": records, "references": references,
            "peak_in_flight": peak, "optimizer_updates": 0, "max_probe_tokens": max_tokens,
            "scope": "Original native input tokens; synthetic256token runtime probes, never admitted to training. "
                     "Exact tokens/logprobs across selected cache states and forty concurrent requests; not universal trajectory determinism."}
