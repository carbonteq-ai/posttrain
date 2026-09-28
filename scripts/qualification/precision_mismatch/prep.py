"""Build the offline mismatch prompt set for Qwen3.5-0.8B (reasoning off, as the qwen3.5-off-v1 renderer).

Writes /work/prompts.json: 48 GSM8K train prompts in the gsm8k_v1 environment's
format (up to 1,024 generated tokens) and up to 16 AutomationBench agent
transcripts (their last assistant turn removed, 1,500 to 3,072 prompt tokens,
up to 512 generated tokens) from /work/transcripts/*.json when present.
"""

from __future__ import annotations

import glob
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import AutoTokenizer

MODEL, REVISION = "Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"
SYSTEM = (
    "Solve the grade-school math problem. Reason step by step, then give the final "
    "answer as a single number on the last line, prefixed with '#### ' (e.g. '#### 42')."
)


def _json_or(value: str, fallback: Any) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback


def _message(message: dict[str, Any]) -> dict[str, Any]:
    content = message["content"]
    entry: dict[str, Any] = {
        "role": message["role"],
        "content": content if isinstance(content, str) else json.dumps(content),
    }
    if message["role"] != "assistant":
        return entry
    entry["content"] = entry["content"] or ""
    calls = message.get("tool_calls")
    if isinstance(calls, str):
        calls = _json_or(calls, None)
    if calls:
        converted = []
        for call in calls:
            function = call.get("function") or call
            arguments = function.get("arguments") or "{}"
            if isinstance(arguments, str):
                arguments = _json_or(arguments, {"raw": arguments})
            converted.append({"type": "function", "function": {"name": function["name"], "arguments": arguments}})
        entry["tool_calls"] = converted
    return entry


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    parquet = glob.glob("/hf/hub/datasets--openai--gsm8k/snapshots/*/main/train-00000-of-00001.parquet")[0]
    rows = pd.read_parquet(parquet)
    prompts: list[dict[str, Any]] = []
    for index in random.Random(0).sample(range(len(rows)), 48):
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{SYSTEM}\n\n{rows.iloc[index]['question']}"}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
            enable_thinking=False,
        )
        prompts.append({"id": f"gsm8k-{index}", "set": "gsm8k", "prompt_ids": list(ids), "max_tokens": 1024})
    agent = 0
    for path in sorted(glob.glob("/work/transcripts/*.json")):
        if agent >= 16:
            break
        transcript = json.loads(Path(path).read_text())["transcript"]
        assistants = [index for index, message in enumerate(transcript) if message["role"] == "assistant"]
        if not assistants or assistants[-1] < 3:
            continue
        try:
            messages = [_message(message) for message in transcript[: assistants[-1]]]
            ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=False, enable_thinking=False
            )
        except Exception:  # noqa: BLE001 - skip transcripts the chat template cannot render
            continue
        if not 1500 <= len(ids) <= 3072:
            continue
        prompts.append({"id": "ab-" + Path(path).name[:12], "set": "agent", "prompt_ids": list(ids), "max_tokens": 512})
        agent += 1
    Path("/work/prompts.json").write_text(json.dumps(prompts))
    print({"gsm8k": 48, "agent": agent})


if __name__ == "__main__":
    main()
