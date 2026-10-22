"""Audit complete greedy native calls with identical actual policy and inputs."""

import hashlib
import itertools
import json


def audit(calls):
    groups = {}
    for index, call in enumerate(calls):
        if not call.get("complete") or call.get("temperature") != 0.:
            continue
        key = (tuple(call["prefix_ids"]), call["requested_completion_budget"],
               call["adapter_sha256"], call["temperature"])
        groups.setdefault(key, []).append((index, call))
    comparisons = []
    for key, values in groups.items():
        for (left, a), (right, b) in itertools.combinations(values, 2):
            common = 0
            for x, y in zip(a["action_ids"], b["action_ids"], strict=False):
                if x != y:
                    break
                common += 1
            comparisons.append({
                "indices": [left, right], "prefix_tokens": len(key[0]),
                "prefix_sha256": hashlib.sha256(json.dumps(list(key[0])).encode()).hexdigest(),
                "tokens_exact": a["action_ids"] == b["action_ids"],
                "common_output_tokens": common,
                "common_prefix_max_logp_delta": max((abs(x-y) for x, y in zip(
                    a["sampled_logprobs"][:common], b["sampled_logprobs"][:common], strict=True)), default=0.),
                "cached_tokens": [a.get("num_cached_tokens"), b.get("num_cached_tokens")]})
    differences = sum(not row["tokens_exact"] for row in comparisons)
    return {"qualified": bool(comparisons) and differences == 0,
            "complete_greedy_calls": sum(len(v) for v in groups.values()),
            "duplicate_input_groups": sum(len(v) > 1 for v in groups.values()),
            "compared_pairs": len(comparisons), "different_output_pairs": differences,
            "comparisons": comparisons,
            "scope": "Complete native greedy calls at identical input/budget/policy; logprob drift reported separately. Not universal determinism or reward evidence."}
