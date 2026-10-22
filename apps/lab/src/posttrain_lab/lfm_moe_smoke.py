"""Bounded native tool-call and continuation probe for the LFM MoE screen."""

from __future__ import annotations

import json
from typing import Any

from posttrain.common import RunContext
from posttrain.serve import GenerationRequest, GenerationResult, ServeLaunchRequest, generate, launch, probe


def _tool_call(deltas: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    calls: dict[int, dict[str, Any]] = {}
    for delta in deltas:
        call = calls.setdefault(
            int(delta.get("index", 0)), {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
        )
        call["id"] += delta.get("id") or ""
        function = delta.get("function") or {}
        call["function"]["name"] += function.get("name") or ""
        call["function"]["arguments"] += function.get("arguments") or ""
    if len(calls) != 1:
        raise RuntimeError("native tool smoke requires exactly one parsed tool call")
    call = next(iter(calls.values()))
    if not call["id"] or call["function"]["name"] != "lookup_code":
        raise RuntimeError("native tool smoke returned the wrong tool or no call id")
    if json.loads(call["function"]["arguments"]) != {"key": "readiness"}:
        raise RuntimeError("native tool smoke returned incorrect function arguments")
    return call


def native_tool_smoke(context: RunContext, request: ServeLaunchRequest) -> GenerationResult:
    """Keep the server alive across a parsed call, synthetic result and final answer."""
    if request.inference.engine.get("reasoning_parser") == "lfm2_optional_thinking":
        from transformers import AutoTokenizer

        from .lfm_moe_reasoning import qualify_optional_reasoning

        base = request.inference.model.base
        tokenizer = AutoTokenizer.from_pretrained(base.repo_id, revision=base.revision)
        qualify_optional_reasoning(tokenizer)
        context.metric("serve/smoke/parser_preflight_passed", 1)
    tools = (
        {
            "type": "function",
            "function": {
                "name": "lookup_code",
                "description": "Return the secret readiness code for the supplied key.",
                "parameters": {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]},
            },
        },
    )
    messages: tuple[dict[str, Any], ...] = (
        {"role": "system", "content": "Use lookup_code to retrieve the requested code. Never invent a code."},
        {"role": "user", "content": 'Call lookup_code with exactly {"key":"readiness"}, then report the returned code.'},
    )
    with launch(context, request) as endpoint:
        if not probe(context, endpoint).model_available:
            raise RuntimeError("native tool smoke model is unavailable")
        first = generate(
            context,
            GenerationRequest(endpoint, messages, 8192, tools=tools, timeout_seconds=600),
            request.inference.model,
        )
        if "<think>" in first.content or "</think>" in first.content:
            raise RuntimeError("native tool smoke leaked reasoning markup into content")
        call = _tool_call(first.tool_call_deltas)
        continuation = (
            *messages,
            {"role": "assistant", "content": first.content, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": call["id"], "content": '{"code":"READY-2047"}'},
        )
        final = generate(
            context,
            GenerationRequest(endpoint, continuation, 8192, tools=tools, timeout_seconds=600),
            request.inference.model,
        )
        if final.tool_call_deltas or "READY-2047" not in final.content:
            raise RuntimeError("native tool smoke failed to consume the tool result and answer")
        if "<think>" in final.content or "</think>" in final.content:
            raise RuntimeError("native tool smoke leaked final reasoning markup into content")
        context.metric("serve/smoke/tool_continuation_passed", 1)
        return final
