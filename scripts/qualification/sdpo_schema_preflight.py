"""CPU-only exact pinned native parser and bridge qualification from a saved call."""

import argparse
import asyncio
import copy
import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from sdpo_schema_parser import parser_class
from self_campaign import Campaign
from transformers import AutoTokenizer
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tool_parsers.qwen3_engine_tool_parser import Qwen3EngineToolParser


def normalized(parsed):
    parsed = copy.deepcopy(parsed)
    for call in parsed.get("tool_calls", []):
        call.pop("id", None)
    return parsed


async def run(args):
    fixture = json.loads(args.fixture.read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    guarded, source = parser_class()
    request = ChatCompletionRequest(model="local-self", messages=fixture["messages"], tools=fixture["tools"])

    def parse(cls, raw):
        result = cls(tokenizer, tools=request.tools).extract_tool_calls(raw, request)
        return normalized({"content": result.content, "tool_calls": [c.model_dump() for c in result.tool_calls]})

    old, new = parse(Qwen3EngineToolParser, fixture["raw"]), parse(guarded, fixture["raw"])
    assert old["content"] == new["content"] and len(old["tool_calls"]) == len(new["tool_calls"]) == 1
    old_args = json.loads(old["tool_calls"][0]["function"]["arguments"])
    new_args = json.loads(new["tool_calls"][0]["function"]["arguments"])
    assert type(old_args["row"]) is int and type(new_args["row"]) is str
    assert new_args["row"] == str(old_args["row"])
    assert {k: v for k, v in old_args.items() if k != "row"} == {k: v for k, v in new_args.items() if k != "row"}
    assert parse(Qwen3EngineToolParser, fixture["raw"]) == old  # no global/installed mutation

    # Actual bridge input IDs, output IDs, budgets and content remain identical;
    # only the declared numeric-looking string type changes.
    class Rollout:
        version = 0

        async def generate(self, prefix, max_tokens):
            assert 0 < len(prefix) + max_tokens <= 16384
            return fixture["action_ids"], True, {"policy_version": 0}

    class Request:
        async def json(self):
            return {"messages": fixture["messages"], "tools": fixture["tools"]}

    from sdpo_schema_parser import schema_completions

    with tempfile.TemporaryDirectory() as tmp:
        owner = SimpleNamespace(args=SimpleNamespace(reply_limit=2048, context_limit=16384, feedback_mode="minimal"),
            tokenizer=tokenizer, rollout=Rollout(), calls=[], folder=Path(tmp), schema_parser=guarded)
        body = await Request().json()
        native_response = json.loads((await Campaign.completions(owner, Request())).body)
        native_row = owner.calls[-1]
        guarded_response = json.loads((await schema_completions(owner, body)).body)
        guarded_row = owner.calls[-1]
        for key in ("prefix_ids", "action_ids", "raw", "complete", "requested_completion_budget"):
            assert native_row[key] == guarded_row[key]
        assert native_response["usage"] == guarded_response["usage"]
        assert normalized(native_row["message"]) == normalized(native_response["choices"][0]["message"])
        assert normalized(guarded_row["message"]) == normalized(guarded_response["choices"][0]["message"])

    # Numeric declared fields must remain numeric; object/string/array cases
    # establish the fix does not turn every tool argument into text.
    controls = []
    for types, raw, expected_type in [(["integer"], "2", int), (["boolean"], "true", bool),
        (["object", "string"], '{"id":"2"}', dict), (["array", "string"], '["2"]', list),
        (["string", "object", "array", "null"], "500001", str), (["null", "string"], "null", type(None))]:
        tools = [{"type": "function", "function": {"name": "probe", "parameters": {"type": "object",
            "properties": {"value": {"type": types}}}}}]
        req = ChatCompletionRequest(model="local-self", messages=[{"role": "user", "content": "probe"}], tools=tools)
        text = '<tool_call>\n<function=probe>\n<parameter=value>\n' + raw + '\n</parameter>\n</function>\n</tool_call>'
        result = guarded(tokenizer, tools=req.tools).extract_tool_calls(text, req)
        value = json.loads(result.tool_calls[0].function.arguments)["value"]
        assert type(value) is expected_type, (types, raw, value)
        controls.append({"types": types, "raw": raw, "value": value, "python_type": type(value).__name__})
    output = {"qualified": True, "gpu_used": False, "source": source,
        "fixture_sha256": hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
        "native": old, "guarded": new, "controls": controls,
        "bridge_original_tokens_and_budgets_preserved": True,
        "native_parser_unchanged": True,
        "scope": "transport type repair, not task success or learning qualification"}
    args.output.write_text(json.dumps(output, indent=2))
    print(json.dumps(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(run(parser.parse_args()))
