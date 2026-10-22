"""CPU-check the exact native LFM tokenizer/parser and immutable pool/config."""

import argparse
import hashlib
import json
from pathlib import Path


def preflight(root, model):
    from sdpo_lfm_config import MODEL_REVISION
    from sdpo_lfm_runtime import prompt_ids, split_native
    from sdpo_pool124_config import validate_pool
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.tool_parsers.lfm2_tool_parser import Lfm2ToolParser

    if model.name != MODEL_REVISION:
        raise ValueError("wrong LFM pin")
    tasks = validate_pool(json.loads((root / "cohorts.json").read_text()),
        json.loads((root / "train-pool.json").read_text()), json.loads((root / "heldout62.json").read_text()))
    config = AutoConfig.from_pretrained(model, local_files_only=True)
    if config.model_type != "lfm2" or AutoModelForCausalLM._model_mapping[type(config)].__name__ != "Lfm2ForCausalLM":
        raise ValueError("wrong native model class")
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    tools = [{"type": "function", "function": {"name": "lookup_code", "description": "Lookup",
        "parameters": {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]}}}]
    messages = [{"role": "user", "content": "Read readiness"}]
    prefix = prompt_ids(tokenizer, messages, tools)
    assert tokenizer.decode(prefix, skip_special_tokens=False).endswith("<think>")
    raw = "Read the exact key.</think><|tool_call_start|>[lookup_code(key='readiness')]<|tool_call_end|>"
    ids = tokenizer.encode(raw, add_special_tokens=False) + [tokenizer.eos_token_id]
    _, reasoning, visible, complete = split_native(tokenizer, ids, True)
    assert complete and reasoning == "Read the exact key."
    request = ChatCompletionRequest(model="local-self", messages=messages, tools=tools)
    result = Lfm2ToolParser(tokenizer).extract_tool_calls(visible, request)
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0].model_dump()
    assert call["function"]["name"] == "lookup_code" and json.loads(call["function"]["arguments"]) == {"key": "readiness"}
    history = messages + [{"role": "assistant", "tool_calls": [call], "reasoning_content": reasoning},
        {"role": "tool", "tool_call_id": call["id"], "content": '{"code":"READY-1234"}'}]
    rendered = tokenizer.decode(prompt_ids(tokenizer, history, tools), skip_special_tokens=False)
    assert "lookup_code(key='readiness')" in rendered and '"code":"READY-1234"' in rendered
    assert split_native(tokenizer, tokenizer.encode("unclosed", add_special_tokens=False), True)[3] is False
    receipt = {"qualified": True, "GPU_used": False, "model_revision": MODEL_REVISION,
        "source_manifest_sha256": hashlib.sha256((root / "source-sha256.txt").read_bytes()).hexdigest(),
        "model_class": "Lfm2ForCausalLM", "training_tasks": len(tasks), "heldout_tasks": 62,
        "native_thinking_prefill": True, "native_python_call_parse": True, "tool_result_rendered": True,
        "scope": "CPU transport/config proof only; real model/gradient/rollout gates still required"}
    (root / "evidence/cpu-preflight.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    print(json.dumps(preflight(args.root, args.model)))
