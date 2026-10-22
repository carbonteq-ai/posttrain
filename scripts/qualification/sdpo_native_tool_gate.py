"""Bounded native call/result/continuation witness; zero optimizer updates."""

import json
import uuid


async def qualify(agent):
    folder = agent.args.output / "native-tool-qualification"
    folder.mkdir(exist_ok=False)
    agent.folder, agent.calls = folder, []
    tools = [{"type": "function", "function": {"name": "lookup_code",
        "description": "Read the current readiness verification code.",
        "parameters": {"type": "object", "properties": {"key": {"type": "string", "enum": ["readiness"]}},
                       "required": ["key"], "additionalProperties": False}}}]
    messages = [{"role": "system", "content": "Use the available tool to answer accurately."},
                {"role": "user", "content": 'Call lookup_code with {"key":"readiness"}. '
                 "Then reply with exactly the code from its result, with no other text."}]

    class Request:
        async def json(self):
            return {"messages": messages, "tools": tools, "max_tokens": agent.args.reply_limit}

    response = await agent.completions(Request())
    first = json.loads(response.text)["choices"][0]["message"]
    calls = first.get("tool_calls") or []
    if (len(calls) != 1 or calls[0]["function"]["name"] != "lookup_code"
            or json.loads(calls[0]["function"]["arguments"]) != {"key": "readiness"}):
        raise ValueError("native tool qualification: incorrect tool or arguments")
    nonce = "READY-" + uuid.uuid4().hex[:8]
    messages += [first, {"role": "tool", "tool_call_id": calls[0]["id"],
                         "content": json.dumps({"code": nonce})}]
    response = await agent.completions(Request())
    final = json.loads(response.text)["choices"][0]["message"]
    if (final.get("content") or "").strip() != nonce or final.get("tool_calls"):
        raise ValueError("native continuation did not return the observed code in final content")
    if (len(agent.calls) != 2 or not all(c["complete"] for c in agent.calls)
            or any("<think>" in (m.get("content") or "") or "</think>" in (m.get("content") or "")
                   for m in (first, final))):
        raise ValueError("native continuation channels incomplete or leaked thinking markers")
    receipt = {"qualified": True, "optimizer_updates": 0, "thinking": agent.args.thinking,
               "context_limit": agent.args.context_limit, "reply_limit": agent.args.reply_limit,
               "native_calls": 2, "result_consumed": True, "correct_arguments": True,
               "reasoning_separate": True, "no_think_markup_in_content": True,
               "first_reasoning_characters": len(first.get("reasoning_content") or ""),
               "continuation_reasoning_characters": len(final.get("reasoning_content") or ""),
               "scope": "Actual local native transport witness, not task reward or learning evidence"}
    (folder / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
