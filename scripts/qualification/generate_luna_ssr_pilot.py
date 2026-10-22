"""Bounded offline Qwen annotation pilot; outputs are candidates, never gold SFT."""

import argparse
import hashlib
import json
import time
from pathlib import Path

from luna_ssr_quality_stages import (
    WORKFLOW_PROMPT,
    critique_request,
    evidence_request,
    validate_ledger,
    validate_workflow,
)
from luna_ssr_sft import TAGS, validate_annotation


def parsed_json(text):
    # Native Qwen thinking is retained separately, never mistaken for final JSON.
    final = text.rsplit("</think>", 1)[-1].strip()
    if final.startswith("```json"):
        final = final[7:]
    elif final.startswith("```"):
        final = final[3:]
    if final.endswith("```"):
        final = final[:-3]
    return json.loads(final.strip())


def validate_skeleton(value):
    rows = value.get("structural_skeleton")
    if not isinstance(rows, list) or not rows:
        raise ValueError("missing structural skeleton")
    for row in rows:
        if row.get("tag") not in TAGS or not isinstance(row.get("summary"), str):
            raise ValueError("invalid skeleton tag or summary")
        if not 1 <= len(row["summary"].split()) <= 20:
            raise ValueError("skeleton summary outside1..20 words")
    return rows


def run(args):
    import torch
    from posttrain.common.cuda import activate_cuda_toolkit
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    started = time.time()
    if args.output.exists():
        raise ValueError("refusing existing model-output directory")
    args.output.mkdir(parents=True)
    activate_cuda_toolkit(torch)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    manifest = json.loads((args.cases / "manifest.json").read_text())
    cases = []
    for row in manifest["tasks"]:
        raw = (args.cases / row["case_file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["case_sha256"]:
            raise ValueError("case source hash mismatch")
        cases.append(json.loads(raw))
    engine = LLM(
        model=str(args.model),
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=32768,
        max_num_seqs=16,
        max_num_batched_tokens=8192,
        kv_cache_memory_bytes=16 * 1024**3,
        block_size=128,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        language_model_only=True,
        seed=17,
        generation_config="vllm",
        override_generation_config={"eos_token_id": tokenizer.eos_token_id},
    )
    (args.output / "engine-config.txt").write_text(repr(engine.llm_engine.vllm_config))
    sampling = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=4096, seed=17)
    summary = {"skeletons": [], "evidence": [], "rationales": [], "critiques": [], "excluded_requests": []}

    def generate(stage, entries, thinking=True, reply_limit=4096):
        admitted, prompts = [], []
        for task, request in entries:
            data = {key: value for key, value in request.items() if key != "system"}
            messages = [
                {"role": "system", "content": request["system"]},
                {"role": "user", "content": json.dumps(data, separators=(",", ":"))},
            ]
            ids = tokenizer.apply_chat_template(
                messages, tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=thinking
            )
            if len(ids) + reply_limit > 32768:
                summary["excluded_requests"].append(
                    {
                        "task": task,
                        "stage": stage,
                        "action_index": request.get("action_index"),
                        "reason": "overlong; not truncated",
                        "input_tokens": len(ids),
                    }
                )
                continue
            admitted.append((task, request, ids))
            prompts.append({"prompt_token_ids": ids})
        if not prompts:
            return []
        stage_sampling = (
            sampling
            if reply_limit == 4096
            else SamplingParams(
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                max_tokens=reply_limit,
                seed=17,
            )
        )

        def completed_rows():
            for offset in range(0, len(prompts), 16):
                results = engine.generate(prompts[offset : offset + 16], stage_sampling, use_tqdm=False)
                yield from zip(admitted[offset : offset + 16], results, strict=True)

        values = []
        with (args.output / (stage + ".jsonl")).open("x") as stream:
            for (task, request, ids), result in completed_rows():
                response = result.outputs[0]
                row = {
                    "task": task,
                    "stage": stage,
                    "request": request,
                    "input_tokens": len(ids),
                    "output_token_ids": response.token_ids,
                    "text": response.text,
                    "finish_reason": response.finish_reason,
                    "action_index": request.get("action_index"),
                    "thinking": thinking,
                    "sft_qualified": False,
                }
                try:
                    if response.finish_reason == "length":
                        raise ValueError("incomplete generation; not an SFT target")
                    value = parsed_json(response.text)
                    row["parsed"] = value
                    if stage == "skeletons":
                        if args.quality_stages:
                            validate_workflow(value)
                        else:
                            validate_skeleton(value)
                    elif stage == "evidence":
                        validate_ledger(request, value)
                    elif stage == "critiques":
                        if not isinstance(value.get("acceptable"), bool) or not isinstance(value.get("issues"), list):
                            raise ValueError("invalid critique structure")
                    elif stage == "rationales":
                        validate_annotation(request, value)
                    row["format_valid"] = True
                    row["parsed"] = value
                    values.append((task, value))
                except (ValueError, TypeError, KeyError, AttributeError) as exc:
                    row["format_valid"] = False
                    row["validation_error"] = f"{type(exc).__name__}: {exc}"
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                summary[stage].append(
                    {key: row[key] for key in ("task", "action_index", "input_tokens", "finish_reason", "format_valid")}
                )
                (args.output / "progress.json").write_text(json.dumps(summary, indent=2) + "\n")
        (args.output / "progress.json").write_text(json.dumps(summary, indent=2) + "\n")
        return values

    plans = []
    for case in cases:
        request = case["skeleton_request"]
        if args.quality_stages:
            request["system"] = WORKFLOW_PROMPT
            request["available_tool_names"] = [d["function"]["name"] for d in request.pop("available_tool_definitions")]
        plans.append((case["task"], request))
    skeletons = dict(
        generate(
            "skeletons", plans, thinking=not args.quality_stages, reply_limit=1536 if args.quality_stages else 4096
        )
    )
    requests = []
    for case in cases:
        if case["task"] not in skeletons:
            summary["excluded_requests"].append(
                {"task": case["task"], "stage": "rationales", "reason": "no format-valid skeleton"}
            )
            continue
        for request in case["rationale_request_templates"]:
            request["teacher_only_guidance"]["skeleton"] = skeletons[case["task"]]
            requests.append((case["task"], request))
    if args.quality_stages:
        evidence_entries = [(task, evidence_request(request)) for task, request in requests]
        ledgers = generate("evidence", evidence_entries, thinking=False, reply_limit=1536)
        # Match by exact input order/action identity; some evidence outputs may
        # fail. The persisted files, not a position-only zip, own this mapping.
        ledger_rows = (
            [json.loads(line) for line in (args.output / "evidence.jsonl").read_text().splitlines()] if ledgers else []
        )
        ledger_by_id = {
            (row["task"], row["action_index"]): row["parsed"] for row in ledger_rows if row.get("format_valid")
        }
        admitted = []
        for task, request in requests:
            ledger = ledger_by_id.get((task, request["action_index"]))
            if ledger is None:
                summary["excluded_requests"].append(
                    {
                        "task": task,
                        "stage": "rationales",
                        "action_index": request["action_index"],
                        "reason": "no verified-quote evidence ledger",
                    }
                )
                continue
            request["verified_quote_ledger"] = ledger
            request["system"] += (
                "\nUse the verified_quote_ledger to ground each factual claim. "
                "Explain only this next action in at most100 words. If unavailable, "
                "say supported false. Return the JSON once, without a checklist of these instructions."
            )
            admitted.append((task, request))
        requests = admitted
    generate("rationales", requests)
    if args.quality_stages and (args.output / "rationales.jsonl").exists():
        rows = [json.loads(line) for line in (args.output / "rationales.jsonl").read_text().splitlines()]
        reviews = [
            (row["task"], critique_request(row["request"], row["parsed"], row["request"]["verified_quote_ledger"]))
            for row in rows
            if row.get("format_valid")
        ]
        generate("critiques", reviews, thinking=False, reply_limit=1536)
    summary.update(
        {
            "completed": True,
            "model": str(args.model),
            "thinking": True,
            "context_limit": 32768,
            "reply_limit": 4096,
            "server_sequences": 16,
            "elapsed_seconds": time.time() - started,
            "optimizer_updates": 0,
            "quality_stages": args.quality_stages,
            "raw_generator_thinking_export_allowed": False,
            "native_replay_qualified": False,
            "sft_qualified": False,
            "quality_scope": "Format/reference-scope checks only. Semantic review, reference repair and execution still required.",
        }
    )
    (args.output / "receipt.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"completed": True, "skeletons": len(skeletons), "rationale_requests": len(requests)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quality-stages", action="store_true")
    run(parser.parse_args())
