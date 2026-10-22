"""Opt-in research adapter for a reproduced pinned Qwen parser type bug.

This is a transport treatment, not corrected-call supervision. It never supplies
an identifier or changes model tokens. The pinned native parser and its schema
discovery remain authoritative; this guard rejects coercions to undeclared JSON
types. The running capsules and installed vLLM source are never modified.
"""

import hashlib
import inspect
import json

PINNED_UTILS_SHA256 = "fd5153158235fbb5844a85f3081cab8757fdb75684c4416fd51da6d5bdff7d34"


def json_type(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, str):
        return "string"
    raise TypeError(f"not JSON: {type(value)}")


def guarded_coerce(raw, types, upstream):
    """Preserve raw text when native best-effort coercion violates declared types.

    Object-only scalar input still fails downstream validation. An allowed string
    alternative is not converted through json.loads into an undeclared scalar.
    Numeric/boolean declarations, valid containers and null keep native behavior.
    """
    value = upstream(raw, types)
    actual = json_type(value)
    if actual in types or (actual == "integer" and "number" in types):
        return value
    return raw


def parser_class():
    from vllm.parser.qwen3 import Qwen3Parser
    from vllm.tool_parsers.qwen3_engine_tool_parser import Qwen3EngineToolParser
    from vllm.tool_parsers.utils import coerce_to_schema_type, extract_types_from_schema

    path = inspect.getfile(coerce_to_schema_type)
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    if digest != PINNED_UTILS_SHA256:
        raise ValueError("schema guard requires the exact probed vLLM utils source")

    def coerce_value(value, schema):
        if isinstance(value, dict) and isinstance(schema.get("properties"), dict):
            return coerce_dict(value, schema["properties"])
        if isinstance(value, list) and isinstance(schema.get("items"), dict):
            changed = False
            for i, item in enumerate(value):
                value[i], item_changed = coerce_value(item, schema["items"])
                changed |= item_changed
            return value, changed
        raw = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        coerced = guarded_coerce(raw, extract_types_from_schema(schema), coerce_to_schema_type)
        return coerced, type(value) is not type(coerced) or value != coerced

    def coerce_dict(args, properties):
        changed = False
        for key, value in args.items():
            if isinstance(properties.get(key), dict):
                args[key], item_changed = coerce_value(value, properties[key])
                changed |= item_changed
        return args, changed

    class SchemaBoundEngine(Qwen3Parser):
        _coerce_dict = staticmethod(coerce_dict)

    class SchemaBoundToolParser(Qwen3EngineToolParser):
        _parser_engine_cls = SchemaBoundEngine

    return SchemaBoundToolParser, {"mode": "schema-bound-v1", "upstream_utils": path,
                                   "upstream_utils_sha256": digest,
                                   "installed_source_modified": False}


async def schema_completions(owner, body):
    """Campaign bridge with only the parser class changed; retain original IDs.

    Minimal-feedback SDPO only. Duplicate the small native bridge rather than
    replacing an installed parser symbol shared with other consumers.
    """
    import time

    from aiohttp import web
    from self_feedback_native import prompt_ids
    from self_feedback_signals import requested_budget
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    if owner.args.feedback_mode != "minimal":
        raise ValueError("schema bridge only qualified for minimal-feedback SDPO")
    messages, tools = body["messages"], body.get("tools", [])
    thinking = getattr(owner.args, "thinking", False)
    if thinking:
        from sdpo_prompting import prompt_ids as thinking_prompt_ids

        prefix = thinking_prompt_ids(owner.tokenizer, messages, tools, thinking=True)
    else:
        prefix = prompt_ids(owner.tokenizer, messages, tools)
    budget, fits = requested_budget(body, owner.args.reply_limit, owner.args.context_limit, len(prefix))
    if not fits:
        ids, complete = [], False
        evidence = {"limit_reason": "context_limit", "policy_version": owner.rollout.version}
    else:
        ids, complete, evidence = await owner.rollout.generate(prefix, max_tokens=budget)
        if not complete:
            evidence["limit_reason"] = ("reply_limit" if budget == owner.args.reply_limit
                                        else "requested_completion_limit")
    raw = owner.tokenizer.decode(ids, skip_special_tokens=True)
    native = ChatCompletionRequest(model="local-self", messages=messages, tools=tools,
                                   chat_template_kwargs={"enable_thinking": thinking})
    reasoning, visible = None, raw
    if thinking:
        from sdpo_prompting import split_reasoning

        reasoning, visible, channels_complete = split_reasoning(owner.tokenizer, raw, native)
        if complete and not channels_complete:
            complete = False
            evidence["limit_reason"] = "reasoning_channel_incomplete"
    parsed = owner.schema_parser(owner.tokenizer, tools=native.tools).extract_tool_calls(visible, native) if complete else None
    tools_out = [call.model_dump() for call in parsed.tool_calls] if parsed else []
    message = {"role": "assistant", "content": parsed.content if parsed else None if thinking else raw}
    if thinking and reasoning is not None:
        message["reasoning_content"] = reasoning
    if tools_out:
        message["tool_calls"] = tools_out
    record = {"messages": messages, "tools": tools, "prefix_ids": prefix, "action_ids": ids,
              "raw": raw, "message": message, "complete": complete,
              "requested_completion_budget": budget, "tool_parser_mode": "schema-bound-v1",
              "thinking_enabled": thinking, **evidence}
    if thinking:
        closing = owner.tokenizer.convert_tokens_to_ids("</think>")
        record["reasoning_prefix_token_count"] = ids.index(closing) if closing in ids else len(ids)
        record["reasoning_content_present"] = bool(reasoning and reasoning.strip())
    owner.calls.append(record)
    with (owner.folder / "generation-calls.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    return web.json_response({"id": f"self-{len(owner.calls)}", "object": "chat.completion",
        "created": int(time.time()), "model": "local-self",
        "choices": [{"index": 0, "message": message,
                     "finish_reason": "length" if not complete else "tool_calls" if tools_out else "stop"}],
        "usage": {"prompt_tokens": len(prefix), "completion_tokens": len(ids),
                  "total_tokens": len(prefix) + len(ids)}})
