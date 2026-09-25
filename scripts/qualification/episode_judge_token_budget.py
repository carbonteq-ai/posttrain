"""Measure retained episode-judge chat inputs with an immutable model tokenizer.

This is an offline preflight, not proof of vLLM serving-template parity. The
serving-side /tokenize guard remains authoritative for live requests.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from statistics import median
from typing import Any

from transformers import AutoTokenizer


def _count(tokenizer: Any, messages: list[dict[str, Any]]) -> int:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    token_ids = rendered["input_ids"] if isinstance(rendered, Mapping) else rendered
    if not isinstance(token_ids, list) or (token_ids and isinstance(token_ids[0], list)):
        raise TypeError("Gemma chat template did not return one flat token sequence")
    return len(token_ids)


def _summarize(values: list[int]) -> dict[str, int | float]:
    sorted_values = sorted(values)
    return {
        "count": len(values),
        "min": sorted_values[0],
        "median": median(sorted_values),
        "p95_nearest_rank": sorted_values[(95 * len(values) + 99) // 100 - 1],
        "max": sorted_values[-1],
        "above_12288": sum(value > 12_288 for value in values),
        "above_40000": sum(value > 40_000 for value in values),
        "above_56000": sum(value > 56_000 for value in values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, help="retained judge-inputs.jsonl")
    parser.add_argument("--model", default="google/gemma-4-12B-it")
    parser.add_argument("--revision", required=True, help="immutable tokenizer/model revision")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    original: list[int] = []
    compact: list[int] = []
    rows: list[dict[str, Any]] = []
    for line in args.inputs.read_text().splitlines():
        record = json.loads(line)
        messages = record["messages"]
        request = record["assessment_request"]
        if len(messages) != 2 or json.loads(messages[1]["content"]) != request:
            raise ValueError("judge input differs from its retained assessment request")
        compact_messages = [
            messages[0],
            {
                **messages[1],
                "content": json.dumps(request, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            },
        ]
        original_count = _count(tokenizer, messages)
        compact_count = _count(tokenizer, compact_messages)
        original.append(original_count)
        compact.append(compact_count)
        rows.append(
            {
                "trace_id": record["trace_id"],
                "task_name": record.get("task_name"),
                "original_input_tokens": original_count,
                "compact_input_tokens": compact_count,
            }
        )
    if not rows:
        raise ValueError("no retained judge inputs")
    result = {
        "tokenizer_model": args.model,
        "tokenizer_revision": args.revision,
        "source": str(args.inputs),
        "original": _summarize(original),
        "compact": _summarize(compact),
        "cases": rows,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
