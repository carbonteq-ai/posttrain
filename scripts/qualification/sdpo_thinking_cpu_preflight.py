"""Exact pinned tokenizer/reasoning/tool bridge replay; no model generation or GPU."""

import argparse
import asyncio
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from sdpo_prompting import prompt_ids, split_reasoning
from sdpo_schema_parser import parser_class, schema_completions
from self_feedback_native import prompt_ids as legacy_prompt_ids
from transformers import AutoTokenizer
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest


async def run(args):
    fixture = json.loads(args.fixture.read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    messages, tools = fixture["messages"], fixture["tools"]
    assert prompt_ids(tokenizer, messages, tools, False) == legacy_prompt_ids(tokenizer, messages, tools)
    thought_prefix = prompt_ids(tokenizer, messages, tools, True)
    assert thought_prefix != prompt_ids(tokenizer, messages, tools, False)
    native = ChatCompletionRequest(model="local-self", messages=messages, tools=tools,
                                   chat_template_kwargs={"enable_thinking": True})
    reasoning = "Inspect the observed result before selecting the known row."
    raw = reasoning + "</think>\n" + fixture["raw"]
    ids = tokenizer.encode(raw, add_special_tokens=False)
    decoded = tokenizer.decode(ids, skip_special_tokens=True)
    assert "</think>" in decoded
    hidden, visible, complete = split_reasoning(tokenizer, decoded, native)
    assert complete and hidden and reasoning in hidden and fixture["raw"].strip() in visible
    assert not split_reasoning(tokenizer, reasoning, native)[2]
    guarded, _ = parser_class()

    class Rollout:
        version = 0

        async def generate(self, prefix, max_tokens):
            assert prefix == thought_prefix
            assert 0 < max_tokens <= 4096 and len(prefix) + max_tokens <= 32768
            return ids, True, {"policy_version": 0}

    class Request:
        async def json(self):
            return {"messages": messages, "tools": tools}

    with tempfile.TemporaryDirectory() as tmp:
        owner = SimpleNamespace(args=SimpleNamespace(feedback_mode="minimal", context_limit=32768,
            reply_limit=4096, thinking=True), tokenizer=tokenizer, rollout=Rollout(), schema_parser=guarded,
            calls=[], folder=Path(tmp))
        response = await schema_completions(owner, await Request().json())
        message = json.loads(response.text)["choices"][0]["message"]
        assert message["reasoning_content"] and "</think>" not in (message["content"] or "")
        assert len(message["tool_calls"]) == 1
        assert type(json.loads(message["tool_calls"][0]["function"]["arguments"])["row"]) is str
        assert owner.calls[0]["action_ids"] == ids and owner.calls[0]["complete"]
    result = {"passed": True, "gpu_used": False, "model_generations": 0, "optimizer_updates": 0,
              "nonthinking_prefix_exactly_matches_legacy": True, "native_thinking_prefix_selected": True,
              "reasoning_and_tool_channels_separate": True, "unclosed_reasoning_rejected": True,
              "raw_thinking_plus_action_target_ids_unchanged": True,
              "scope": "CPU structural replay only; actual model/GPU native qualification still required"}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(run(parser.parse_args()))
