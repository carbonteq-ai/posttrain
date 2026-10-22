"""Native Qwen thinking prompts and separate reasoning/tool channels, project opt-in."""

import copy
import json


def prompt_ids(tokenizer, messages, tools, thinking=False):
    if getattr(tokenizer, "_sdpo_model_family", "qwen3.5") == "lfm2.5":
        from sdpo_lfm_runtime import prompt_ids as lfm_prompt_ids

        if not thinking:
            raise ValueError("pinned LFM native template prefills thinking")
        return lfm_prompt_ids(tokenizer, messages, tools)
    messages = copy.deepcopy(messages)
    for message in messages:
        for call in message.get("tool_calls") or []:
            function = call["function"]
            if isinstance(function["arguments"], str):
                function["arguments"] = json.loads(function["arguments"])
    return tokenizer.apply_chat_template(messages, tools=tools, tokenize=True,
        add_generation_prompt=True, enable_thinking=thinking, return_dict=True,
        return_tensors="pt")["input_ids"][0].tolist()


def split_reasoning(tokenizer, raw, request):
    # This selected vLLM uses the new adapter API, not extract_reasoning_content.
    from vllm.reasoning import ReasoningParserManager

    parser = ReasoningParserManager.get_reasoning_parser("qwen3")(tokenizer)
    reasoning, content = parser.extract_reasoning(raw, request)
    complete = ("</think>" in raw and bool(content and content.strip())
                and "<think>" not in content and "</think>" not in content)
    return reasoning, content, complete
