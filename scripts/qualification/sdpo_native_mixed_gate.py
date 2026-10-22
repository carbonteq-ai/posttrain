"""Full-budget numeric native-prefix controls in heterogeneous forty-request batches."""

import asyncio
import hashlib
import json
import time


class MixedPrefixMismatch(ValueError):
    def __init__(self, receipt):
        super().__init__("full-budget mixed native prefix invariance failed")
        self.receipt = receipt


async def qualify(rollout, fixture):
    original = rollout.temperature
    rollout.temperature = 0.
    try:
        return await _qualify(rollout, fixture)
    finally:
        rollout.temperature = original


async def _qualify(rollout, fixture):
    started = time.monotonic()
    cases, backgrounds = fixture["cases"], fixture["backgrounds"]
    if len(cases) != 4 or len(backgrounds) != 16:
        raise ValueError("four numeric cases and sixteen heterogeneous backgrounds required")
    for case in cases + backgrounds:
        ids = case["prefix_ids"]
        if (not ids or not all(type(i) is int for i in ids)
                or hashlib.sha256(json.dumps(ids).encode()).hexdigest() != case["prefix_sha256"]):
            raise ValueError("numeric prefix fixture identity differs")
    if any(c["budget"] != 8192 for c in cases):
        raise ValueError("critical native cases must retain original full8192reply budget")
    references, records = [], []
    active, peak = 0, 0

    async def generate(item):
        nonlocal active, peak
        role, index = item
        case = cases[index] if role == "case" else backgrounds[index]
        active += 1
        peak = max(active, peak)
        try:
            ids, complete, info = await rollout.generate(case["prefix_ids"],
                max_tokens=8192 if role == "case" else 64)
            return {"role": role, "index": index, "ids": ids, "complete": complete,
                    "sampled_logprobs": info["sampled_logprobs"],
                    "cached_tokens": info.get("num_cached_tokens")}
        finally:
            active -= 1

    for index in range(4):
        references.append(await generate(("case", index)))
    positions = {0, 7, 13, 20, 26, 30, 35, 39}
    items, ncase, nbg = [], 0, 0
    for position in range(40):
        if position in positions:
            items.append(("case", ncase % 4))
            ncase += 1
        else:
            items.append(("background", nbg % 16))
            nbg += 1
    for phase, order in (("mixed40_forward", items), ("mixed40_reverse", list(reversed(items)))):
        values = await asyncio.gather(*(generate(item) for item in order))
        for value in values:
            if value["role"] != "case":
                continue
            ref = references[value["index"]]
            exact = value["ids"] == ref["ids"]
            delta = (max((abs(a-b) for a, b in zip(value["sampled_logprobs"],
                     ref["sampled_logprobs"], strict=True)), default=0.) if exact else None)
            records.append({**value, "phase": phase, "tokens_exact": exact, "max_abs_logp_delta": delta})
            if not exact or delta != 0.:
                raise MixedPrefixMismatch({"qualified": False, "references": references, "records": records,
                                           "optimizer_updates": 0, "peak_in_flight": peak})
    if peak != 40:
        raise ValueError("mixed prefix test did not dispatch40requests")
    return {"qualified": True, "references": references, "records": records,
            "peak_in_flight": peak, "optimizer_updates": 0, "elapsed_s": time.monotonic()-started,
            "scope": "Numeric-only runtime controls, not training or task outcome evidence. Critical replies retain8192cap; background64token probes supply mixed prefill/decode pressure and are never training-admitted."}
