"""Replay retained judge messages against a local OpenAI-compatible vLLM server."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import statistics
import time
import urllib.request
from pathlib import Path
from typing import Any

from automationbench_v1.episode_prompt import (
    EpisodeVerdict,
    WireEpisodeVerdict,
    normalize_wire_verdict,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8123")
    parser.add_argument("--model", default="google/gemma-4-12B-it")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument(
        "--retain-content",
        action="store_true",
        help="Retain raw model content in the report for qualitative prompt debugging",
    )
    parser.add_argument(
        "--wire-evidence-indexes",
        action="store_true",
        help="Use the compact judge wire schema and normalize evidence indexes before validation",
    )
    parser.add_argument(
        "--chat-template-kwargs-json",
        help="Model-specific chat-template kwargs as a JSON object",
    )
    parser.add_argument("--label", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--runtime-identity", required=True)
    args = parser.parse_args()
    if args.concurrency < 1 or args.repetitions < 1 or args.max_tokens < 1:
        parser.error("concurrency, repetitions, and max tokens must be positive")
    if args.temperature < 0:
        parser.error("temperature must be non-negative")
    if not 0 < args.top_p <= 1:
        parser.error("top-p must be in (0, 1]")
    if args.chat_template_kwargs_json is not None:
        try:
            args.chat_template_kwargs = json.loads(args.chat_template_kwargs_json)
        except json.JSONDecodeError as error:
            parser.error(f"invalid chat-template kwargs JSON: {error}")
        if not isinstance(args.chat_template_kwargs, dict):
            parser.error("chat-template kwargs must be a JSON object")
    else:
        args.chat_template_kwargs = {"enable_thinking": args.enable_thinking}
    return args


def _load_cases(path: Path) -> list[dict[str, Any]]:
    cases = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        messages = record.get("messages")
        input_digest = record.get("input_digest")
        if not isinstance(messages, list) or not messages:
            raise ValueError("every replay record must contain a non-empty messages list")
        if not isinstance(input_digest, str) or not input_digest:
            raise ValueError("every replay record must contain an input digest")
        cases.append(
            {
                "input_digest": input_digest,
                "source_input_digest": record.get("source_input_digest", input_digest),
                "case_id": record.get("case_id"),
                "messages": messages,
            }
        )
    if not cases:
        raise ValueError("replay corpus is empty")
    return cases


def _json_request(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _complete(
    *,
    base_url: str,
    model: str,
    case: dict[str, Any],
    max_tokens: int,
    timeout: float,
    chat_template_kwargs: dict[str, Any],
    temperature: float,
    top_p: float,
    top_k: int,
    wire_evidence_indexes: bool,
    retain_content: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    schema = WireEpisodeVerdict if wire_evidence_indexes else EpisodeVerdict
    response = _json_request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        {
            "model": model,
            "messages": case["messages"],
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_tokens": max_tokens,
            "chat_template_kwargs": chat_template_kwargs,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": schema.model_json_schema(),
                    "strict": True,
                },
            },
        },
        timeout,
    )
    elapsed = time.perf_counter() - started
    usage = response.get("usage") or {}
    choices = response.get("choices") or []
    finish_reason = choices[0].get("finish_reason") if choices else None
    message = choices[0].get("message") if choices else {}
    message = message if isinstance(message, dict) else {}
    content = message.get("content")
    content = content if isinstance(content, str) else ""
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    reasoning = reasoning if isinstance(reasoning, str) else ""
    verdict = None
    try:
        if wire_evidence_indexes:
            wire = WireEpisodeVerdict.model_validate_json(content)
            request = json.loads(case["messages"][1]["content"])
            verdict = normalize_wire_verdict(wire, request["valid_message_ids"])
        else:
            verdict = EpisodeVerdict.model_validate_json(content)
    except ValueError:
        pass
    result = {
        "case_id": case["case_id"],
        "input_digest": case["input_digest"],
        "source_input_digest": case["source_input_digest"],
        "elapsed_seconds": elapsed,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "finish_reason": finish_reason,
        "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "reasoning_sha256": hashlib.sha256(reasoning.encode()).hexdigest(),
        "content_characters": len(content),
        "reasoning_characters": len(reasoning),
        "structured_output_valid": verdict is not None,
        "verdict": verdict.model_dump(mode="json") if verdict is not None else None,
        "scores": (
            {name: assessment.score for name, assessment in verdict.assessments.items()}
            if verdict is not None
            else None
        ),
    }
    if retain_content:
        result["content"] = content
        result["reasoning"] = reasoning
    return result


def main() -> None:
    args = _arguments()
    corpus = _load_cases(args.inputs)
    requests = corpus * args.repetitions
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(
            pool.map(
                lambda case: _complete(
                    base_url=args.base_url,
                    model=args.model,
                    case=case,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout_seconds,
                    chat_template_kwargs=args.chat_template_kwargs,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    wire_evidence_indexes=args.wire_evidence_indexes,
                    retain_content=args.retain_content,
                ),
                requests,
            )
        )
    wall_time = time.perf_counter() - started
    completion_tokens = sum(result["completion_tokens"] for result in results)
    prompt_tokens = sum(result["prompt_tokens"] for result in results)
    latencies = [result["elapsed_seconds"] for result in results]
    report = {
        "schema_version": 1,
        "label": args.label,
        "model": args.model,
        "model_revision": args.model_revision,
        "runtime_identity": args.runtime_identity,
        "requests": len(results),
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "enable_thinking": args.chat_template_kwargs.get("enable_thinking"),
        "chat_template_kwargs": args.chat_template_kwargs,
        "wire_evidence_indexes": args.wire_evidence_indexes,
        "content_retained": args.retain_content,
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
        },
        "wall_time_seconds": wall_time,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "completion_tokens_per_second": completion_tokens / wall_time,
        "request_latency_seconds": {
            "mean": statistics.fmean(latencies),
            "median": statistics.median(latencies),
            "max": max(latencies),
        },
        "finish_reasons": {
            reason: sum(result["finish_reason"] == reason for result in results)
            for reason in sorted({str(result["finish_reason"]) for result in results})
        },
        "structured_output_valid": sum(result["structured_output_valid"] for result in results),
        "cases": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
