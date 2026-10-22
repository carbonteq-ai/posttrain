"""Pure cohort, candidate identity and zero-effect base adapter gates."""

import hashlib
import json

import torch
from prepare_self_transfer_reserve import select

MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
REFERENCE_ADAPTER_SHA = "0c5ba465dcd890c7c910da77f6a1d0a9956c18cafe21ac0371418c0b4341a2dc"
RESERVE_DIGEST = "6e24042a565ad2accb71294556a215771c86c3165399d66394b09c36944a3dd6"


def population(reserve, protocol, pool, heldout):
    names = reserve["task_names"]
    digest = hashlib.sha256(json.dumps(names, separators=(",", ":")).encode()).hexdigest()
    if digest != RESERVE_DIGEST or reserve["sorted_names_sha256"] != digest:
        raise ValueError("predeclared reserve identity changed")
    if names != select(protocol, pool, heldout):
        raise ValueError("reserve differs from independently selected cohort")
    return names


def candidates(selection):
    items = selection["candidates"]
    if selection["model_revision"] != MODEL_REVISION or not 3 <= len(items) <= 7:
        raise ValueError("pinned model or candidate count differs")
    names = [x["name"] for x in items]
    if len(set(names)) != len(names) or names[:2] != ["fresh_base", "checkpoint64"]:
        raise ValueError("fresh base and damaged checkpoint must be distinct named controls")
    if items[0]["kind"] != "zero_effect_base" or any(x["kind"] != "adapter" for x in items[1:]):
        raise ValueError("fresh base must be structurally verified, not a renamed trained adapter")
    if selection["primary"] not in names[2:] or not isinstance(selection["primary_qualified"], bool):
        raise ValueError("preselected primary and qualification status required")
    for item in items[1:]:
        if not item.get("path") or len(item.get("adapter_sha256", "")) != 64:
            raise ValueError("exact adapter path/hash required before opening reserve outcomes")
    if items[1]["adapter_sha256"] != REFERENCE_ADAPTER_SHA:
        raise ValueError("checkpoint64 control is not the retained reference")
    if len({x["adapter_sha256"] for x in items[1:]}) != len(items) - 1:
        raise ValueError("duplicate adapters require an explicit revised preselection")
    for item in items[2:]:
        if not item.get("receipt_path") or item.get("step") not in [5, 10]:
            raise ValueError("completed ablation receipt and selected saved step required")
    if not selection.get("decision_basis"):
        raise ValueError("record selection basis before opening reserve outcomes")
    return items


def check_adapter_config(config):
    if (config.get("bias") != "none" or config.get("lora_bias") or config.get("use_dora")
            or config.get("modules_to_save") or config.get("r") != 16 or config.get("lora_alpha") != 32
            or config.get("lora_dropout") != 0):
        raise ValueError("zero-effect base gate requires unchanged additive LoRA-only configuration")


def zero_effect_state(state):
    if (not state or not any(".lora_A." in k for k in state) or not any(".lora_B." in k for k in state)
            or any(not (".lora_A." in k or ".lora_B." in k) for k in state)):
        raise ValueError("non-LoRA saved state could modify the base")
    if not all(torch.isfinite(v).all() for v in state.values()):
        raise ValueError("nonfinite adapter cannot represent a verified base")
    return {k: torch.zeros_like(v) if ".lora_B." in k else v.clone() for k, v in state.items()}


def verify_loaded_state(expected, loaded):
    if set(expected) != set(loaded) or any(
        not torch.equal(expected[k].cpu(), loaded[k].cpu()) for k in expected
    ):
        raise ValueError("loaded adapter differs from selected state")
