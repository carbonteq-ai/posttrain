"""Native LFM template, reasoning and Python-call transport; no Qwen coercion."""

import copy
import json
import time


def prompt_ids(tokenizer, messages, tools):
    messages = copy.deepcopy(messages)
    for message in messages:
        for call in message.get("tool_calls") or []:
            function = call["function"]
            if isinstance(function["arguments"], str):
                function["arguments"] = json.loads(function["arguments"])
    return tokenizer.apply_chat_template(messages, tools=tools, tokenize=True,
        add_generation_prompt=True, preserve_thinking=False, return_dict=True,
        return_tensors="pt")["input_ids"][0].tolist()


def split_native(tokenizer, ids, complete):
    # LFM delimiters are special tokens: skip_special_tokens=True destroys them.
    if ids and ids[-1] == tokenizer.eos_token_id:
        ids = ids[:-1]
    raw = tokenizer.decode(ids, skip_special_tokens=False)
    if "</think>" not in raw:
        return raw, None, "", False
    reasoning, content = raw.split("</think>", 1)
    reasoning = reasoning.removeprefix("<think>")
    valid = complete and "<think>" not in content and "</think>" not in content
    return raw, reasoning, content.strip(), valid


async def completions(owner, body):
    from aiohttp import web
    from self_feedback_signals import requested_budget
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.tool_parsers.lfm2_tool_parser import Lfm2ToolParser

    messages, tools = copy.deepcopy(body["messages"]), body.get("tools", [])
    prefix = prompt_ids(owner.tokenizer, messages, tools)
    budget, fits = requested_budget(body, owner.args.reply_limit, owner.args.context_limit, len(prefix))
    if fits:
        ids, complete, evidence = await owner.rollout.generate(prefix, max_tokens=budget)
        if not complete:
            evidence["limit_reason"] = "reply_limit" if budget == owner.args.reply_limit else "requested_completion_limit"
    else:
        ids, complete = [], False
        evidence = {"limit_reason": "context_limit", "policy_version": owner.rollout.version}
    raw, reasoning, visible, channels_complete = split_native(owner.tokenizer, ids, complete)
    if complete and not channels_complete:
        complete = False
        evidence["limit_reason"] = "reasoning_channel_incomplete"
    # Protocol construction is deliberately after immutable native prefix rendering.
    native = ChatCompletionRequest(model="local-self", messages=copy.deepcopy(messages), tools=tools)
    parsed = Lfm2ToolParser(owner.tokenizer).extract_tool_calls(visible, native) if complete else None
    tools_out = [call.model_dump() for call in parsed.tool_calls] if parsed else []
    message = {"role": "assistant", "content": parsed.content if parsed else None}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    if tools_out:
        message["tool_calls"] = tools_out
    record = {"messages": messages, "tools": tools, "prefix_ids": prefix, "action_ids": ids,
              "raw": raw, "message": message, "complete": complete,
              "requested_completion_budget": budget, "tool_parser_mode": "lfm2-native",
              "thinking_enabled": True, "reasoning_content_present": bool(reasoning and reasoning.strip()),
              **evidence}
    closing = owner.tokenizer.convert_tokens_to_ids("</think>")
    record["reasoning_prefix_token_count"] = ids.index(closing) if closing in ids else len(ids)
    owner.calls.append(record)
    with (owner.folder / "generation-calls.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    return web.json_response({"id": f"lfm-local-{len(owner.calls)}", "object": "chat.completion",
        "created": int(time.time()), "model": "local-self",
        "choices": [{"index": 0, "message": message,
                     "finish_reason": "length" if not complete else "tool_calls" if tools_out else "stop"}],
        "usage": {"prompt_tokens": len(prefix), "completion_tokens": len(ids),
                  "total_tokens": len(prefix) + len(ids)}})
