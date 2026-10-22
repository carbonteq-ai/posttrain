"""Read-only native speculation counters; acceptance is not a speedup estimate."""

import math
import time

NAMES = {"drafts": "vllm:spec_decode_num_drafts_total",
         "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
         "accepted_tokens": "vllm:spec_decode_num_accepted_tokens_total"}


def snapshot(registry=None):
    found = {name: [] for name in NAMES}
    try:
        if registry is None:
            from prometheus_client import REGISTRY

            registry = REGISTRY
        for metric in registry.collect():
            for sample in metric.samples:
                for name, expected in NAMES.items():
                    if sample.name == expected:
                        value = float(sample.value)
                        if not math.isfinite(value) or value < 0:
                            raise ValueError("invalid speculation counter")
                        found[name].append(value)
    except Exception as error:
        return {"available": False, "error_type": type(error).__name__, "monotonic_s": time.monotonic()}
    return {"available": all(found.values()), "monotonic_s": time.monotonic(),
            "counters": {name: sum(values) if values else None for name, values in found.items()}}


def delta(before, after):
    result = {"available": False, "elapsed_s": after["monotonic_s"]-before["monotonic_s"],
              "scope": "Native MTP counters during this collection, excluding readiness outside its interval. Acceptance is not a matched throughput or end-to-end speedup estimate."}
    if not before["available"] or not after["available"]:
        result["reason"] = "counter coverage unavailable; never interpreted as zero"
        return result
    values = {name: after["counters"][name]-before["counters"][name] for name in NAMES}
    if any(v < 0 for v in values.values()) or values["accepted_tokens"] > values["draft_tokens"]:
        result["reason"] = "counter reset or inconsistent speculative token counts"
        return result
    result.update(available=True, counters=values,
                  draft_token_acceptance=(values["accepted_tokens"]/values["draft_tokens"]
                                          if values["draft_tokens"] else None))
    return result
