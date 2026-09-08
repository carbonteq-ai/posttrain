"""Lossless message projection for renderer-parsed policy turns."""

from __future__ import annotations

import ast
import json
from collections.abc import Sequence
from typing import Any


def parsed_policy_message(
    parsed: Any,
    token_ids: Sequence[int],
    tokenizer: Any,
    *,
    tool_call_protocol: Any | None = None,
    tools: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Keep rejected tool syntax visible without making it executable.

    Token IDs remain replay authority. Decode rejected spans directly from that
    sequence; never substitute an empty action or repair arguments implicitly.
    Reasoning stays in its native channel and accepted calls stay structured.
    """
    content = [parsed.content] if parsed.content else []
    calls = []
    rejected = []
    for index, item in enumerate(parsed.tool_calls):
        if item.name is not None and item.status.value == "ok":
            calls.append(
                {
                    "id": item.id or f"call_{index}",
                    "name": item.name,
                    "arguments": item.arguments
                    if isinstance(item.arguments, str)
                    else json.dumps(item.arguments or {}, separators=(",", ":")),
                }
            )
        else:
            span = item.token_span
            if span is None or len(span) != 2 or not 0 <= span[0] < span[1] <= len(token_ids):
                raise ValueError("rejected policy tool call requires an exact sampled-token span")
            raw = tokenizer.decode(list(token_ids[span[0] : span[1]]), skip_special_tokens=False)
            if not raw:
                raise ValueError("rejected policy tool call decoded to empty evidence")
            content.append(raw)
            rejected.append(
                {
                    "type": "posttrain.rejected_tool_call",
                    "status": item.status.value,
                    "token_span": [span[0], span[1]],
                    "raw": raw,
                }
            )
    message: dict[str, Any] = {"role": "assistant", "content": "\n".join(content) or None}
    if parsed.reasoning_content is not None:
        message["reasoning_content"] = parsed.reasoning_content
    if calls:
        message["tool_calls"] = calls
    if rejected:
        message["provider_state"] = rejected
    if not calls and tool_call_protocol is not None and tool_call_protocol.id == "lfm2_pythonic":
        recovered = _lfm_pythonic_calls(message.get("content"), tool_call_protocol, tools)
        if recovered is not None:
            message["content"] = recovered[0]
            message["tool_calls"] = recovered[1]
    return message


def _lfm_pythonic_calls(
    content: object,
    protocol: Any,
    tools: Sequence[dict[str, Any]],
) -> tuple[str | None, list[dict[str, str]]] | None:
    """Recover the model's declared Python-call protocol without executing it."""

    if not isinstance(content, str):
        return None
    stripped = content.strip()
    start = str(protocol.start_token)
    end = str(protocol.end_token)
    if not stripped.startswith(start) or not stripped.endswith(end):
        return None
    expression = stripped[len(start) : -len(end)].strip()
    try:
        root = ast.parse(expression, mode="eval").body
    except SyntaxError:
        return None
    if not isinstance(root, ast.List) or not root.elts:
        return None
    allowed: set[str] = set()
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            allowed.add(str(function["name"]))
        elif isinstance(tool.get("name"), str):
            # Verifiers normalizes OpenAI tool definitions to its flat Tool
            # model before handing the request to an in-process policy client.
            allowed.add(str(tool["name"]))
    recovered: list[dict[str, str]] = []
    for index, item in enumerate(root.elts):
        if not isinstance(item, ast.Call) or not isinstance(item.func, ast.Name) or item.args:
            return None
        name = item.func.id
        if name not in allowed or any(keyword.arg is None for keyword in item.keywords):
            return None
        keys = [str(keyword.arg) for keyword in item.keywords]
        if len(keys) != len(set(keys)):
            return None
        try:
            arguments = {str(keyword.arg): ast.literal_eval(keyword.value) for keyword in item.keywords}
            encoded = json.dumps(arguments, separators=(",", ":"), ensure_ascii=False)
        except (TypeError, ValueError):
            return None
        recovered.append({"id": f"call_{index}", "name": name, "arguments": encoded})
    return None, recovered
