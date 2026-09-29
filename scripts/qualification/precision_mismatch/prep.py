"""Build the prompt set for the offline precision-mismatch matrix.

Two sources, both rendered with the model's chat template (or ``--chat-template``):

* ``gsm8k``: GSM8K train questions in the gsm8k_v1 environment's single user-turn format.
* ``traces``: retained Verifiers trace JSON (Observatory ``/traces/<id>`` payloads, as in
  ``scratchpad/trace_audit/details``). Each trace becomes one agent context: the
  conversation up to (not including) a sampled assistant turn, with the trace's tool
  schemas, so the prompt is a real rollout state of the training environment.

usage (inside the runtime image, see run.sh):
  prep.py --model LiquidAI/LFM2.5-2.6B --revision <sha> --source traces --traces /work/traces \
      --num-prompts 64 --max-tokens 4096 --max-prompt-tokens 20480 --out /work/prompts.json
  prep.py --model Qwen/Qwen3.5-0.8B --revision <sha> --source gsm8k --num-prompts 48 \
      --max-tokens 1024 --template-kwargs '{"enable_thinking": false}' --out /work/prompts.json
"""

from __future__ import annotations

import argparse
import ast
import glob
import json
import random
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

GSM8K_SYSTEM = (
    "Solve the grade-school math problem. Reason step by step, then give the final "
    "answer as a single number on the last line, prefixed with '#### ' (e.g. '#### 42')."
)


def _parse(value: Any, fallback: Any) -> Any:
    """Trace fields are JSON or Python literals serialized as strings."""

    if not isinstance(value, str):
        return value
    if value == "None":
        return None
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(value)
        except (ValueError, SyntaxError):
            continue
    return fallback


def _message(message: dict[str, Any]) -> dict[str, Any]:
    content = _parse(message.get("content"), message.get("content"))
    entry: dict[str, Any] = {
        "role": message["role"],
        "content": "" if content is None else content if isinstance(content, str) else json.dumps(content),
    }
    if message["role"] == "tool" and message.get("tool_call_id"):
        entry["tool_call_id"] = message["tool_call_id"]
    if message["role"] != "assistant":
        return entry
    reasoning = _parse(message.get("reasoning_content"), None)
    if isinstance(reasoning, str) and reasoning:
        entry["reasoning_content"] = reasoning
    calls = _parse(message.get("tool_calls"), None)
    if calls:
        converted = []
        for call in calls:
            function = call.get("function") or call
            arguments = _parse(function.get("arguments") or "{}", {"raw": function.get("arguments")})
            converted.append(
                {
                    "id": call.get("id"),
                    "type": "function",
                    "function": {"name": function["name"], "arguments": arguments},
                }
            )
        entry["tool_calls"] = converted
    return entry


def _render(tokenizer: Any, messages: list[dict[str, Any]], tools: Any, kwargs: dict[str, Any]) -> list[int]:
    ids = tokenizer.apply_chat_template(
        messages, tools=tools or None, tokenize=True, add_generation_prompt=True, return_dict=False, **kwargs
    )
    return list(ids)


def _gsm8k(tokenizer: Any, args: argparse.Namespace, kwargs: dict[str, Any]) -> list[dict[str, Any]]:
    import pandas as pd

    parquet = glob.glob(f"{args.hf_home}/hub/datasets--openai--gsm8k/snapshots/*/main/train-00000-of-00001.parquet")[0]
    rows = pd.read_parquet(parquet)
    prompts = []
    for index in random.Random(args.seed).sample(range(len(rows)), args.num_prompts):
        messages = [{"role": "user", "content": f"{GSM8K_SYSTEM}\n\n{rows.iloc[index]['question']}"}]
        prompts.append(
            {
                "id": f"gsm8k-{index}",
                "set": "gsm8k",
                "prompt_ids": _render(tokenizer, messages, None, kwargs),
                "max_tokens": args.max_tokens,
            }
        )
    return prompts


def _traces(tokenizer: Any, args: argparse.Namespace, kwargs: dict[str, Any]) -> list[dict[str, Any]]:
    paths = sorted(glob.glob(str(Path(args.traces) / "*.json")))
    random.Random(args.seed).shuffle(paths)
    prompts: list[dict[str, Any]] = []
    for path in paths:
        if len(prompts) >= args.num_prompts:
            break
        payload = json.loads(Path(path).read_text())
        transcript = payload.get("transcript") or []
        raw = payload.get("raw") if isinstance(payload.get("raw"), dict) else {}
        tools = raw.get("tools") if raw else None
        tools = [{"type": "function", "function": tool} if "function" not in tool else tool for tool in tools or []]
        turns = [i for i, message in enumerate(transcript) if message.get("role") == "assistant" and i >= 2]
        # Prefer the deepest sampled turn whose context fits: long contexts stress
        # accumulated state (convolution, attention and KV) the most.
        for cut in reversed(turns):
            try:
                ids = _render(tokenizer, [_message(m) for m in transcript[:cut]], tools, kwargs)
            except Exception:  # noqa: BLE001 - a template that cannot render this trace is skipped
                break
            if args.min_prompt_tokens <= len(ids) <= args.max_prompt_tokens:
                prompts.append(
                    {
                        "id": f"trace-{Path(path).stem[:12]}-t{cut}",
                        "set": "agent",
                        "prompt_ids": ids,
                        "max_tokens": args.max_tokens,
                    }
                )
                break
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--source", choices=("gsm8k", "traces"), required=True)
    parser.add_argument("--traces", default="/work/traces")
    parser.add_argument("--chat-template", help="jinja file overriding the tokenizer's chat template")
    parser.add_argument("--template-kwargs", default="{}", help="JSON keyword arguments for the chat template")
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=1024, help="generation budget per prompt")
    parser.add_argument("--min-prompt-tokens", type=int, default=1)
    parser.add_argument("--max-prompt-tokens", type=int, default=20480)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hf-home", default="/hf")
    parser.add_argument("--out", default="/work/prompts.json")
    parser.add_argument("--append", action="store_true", help="add to an existing prompt file (mixed sets)")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if args.chat_template:
        tokenizer.chat_template = Path(args.chat_template).read_text()
    kwargs = json.loads(args.template_kwargs)
    prompts = _gsm8k(tokenizer, args, kwargs) if args.source == "gsm8k" else _traces(tokenizer, args, kwargs)
    if not prompts:
        raise SystemExit("no prompt fits the requested token bounds")
    if args.append and Path(args.out).exists():
        prompts = json.loads(Path(args.out).read_text()) + prompts
    Path(args.out).write_text(json.dumps(prompts))
    lengths = sorted(len(p["prompt_ids"]) for p in prompts)
    print(
        json.dumps(
            {
                "prompts": len(prompts),
                "prompt_tokens_min": lengths[0],
                "prompt_tokens_median": lengths[len(lengths) // 2],
                "prompt_tokens_max": lengths[-1],
            }
        )
    )


if __name__ == "__main__":
    main()
