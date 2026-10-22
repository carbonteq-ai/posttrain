"""Bind prepared guidance to verified pinned native state normalization."""

import copy

from sdpo_guidance_context import digest


def bind(guidance, materials, normalization):
    if (not normalization["qualified"] or normalization["mismatch"]
            or normalization["native_episodes"] != 40
            or not normalization["normalizer_sha256"]):
        raise ValueError("normalization lacks complete native baseline evidence")
    result = copy.deepcopy(guidance)
    sources = {r["task"]: r["material"] for r in materials["rows"]}
    if set(result["guidance"]) != set(sources) or set(sources) != set(normalization["normalized_states"]):
        raise ValueError("normalization task population differs")
    for name, bundle in result["guidance"].items():
        original = digest(sources[name]["native_data"]["initial_state"])
        native = digest(normalization["normalized_states"][name])
        if (bundle["current_state_sha256"] != original
                or normalization["raw_source_hashes"][name] != original
                or normalization["canonical_hashes"][name] != native):
            raise ValueError("normalization source/hash binding differs")
        bundle["source_state_sha256"] = original
        bundle["current_state_sha256"] = native
        bundle["state_binding_normalization"] = {"receipt_sha256": digest(normalization),
            "normalizer_sha256": normalization["normalizer_sha256"],
            "scope": "Pinned WorldState constructor representation; original task values and instructions unchanged."}
    return result
