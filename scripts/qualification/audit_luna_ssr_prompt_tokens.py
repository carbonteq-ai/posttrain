"""Measure exact native thinking-template input tokens without model/GPU calls."""

import argparse
import hashlib
import json
from pathlib import Path


def audit(root, tokenizer_path, output):
    from transformers import AutoTokenizer

    if output.exists():
        raise ValueError("preserve previous token audits")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    manifest = json.loads((root / "manifest.json").read_text())
    rows = []
    for source in manifest["tasks"]:
        raw = (root / source["case_file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != source["case_sha256"]:
            raise ValueError("case changed")
        case = json.loads(raw)
        counts = []
        for request in [case["skeleton_request"], *case["rationale_request_templates"]]:
            system = request["system"]
            data = {key: value for key, value in request.items() if key != "system"}
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(data, separators=(",", ":"))},
            ]
            counts.append(
                len(
                    tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        return_dict=False,
                        add_generation_prompt=True,
                        enable_thinking=True,
                    )
                )
            )
        rows.append(
            {
                "task": source["task"],
                "skeleton_input_tokens": counts[0],
                "rationale_input_tokens": counts[1:],
                "max_input_tokens": max(counts),
            }
        )
    result = {
        "tokenizer_path": str(tokenizer_path),
        "thinking": True,
        "tasks": rows,
        "maximum_input_tokens": max(row["max_input_tokens"] for row in rows),
        "measurement_scope": "Prepared inputs only. Rationale skeleton is a placeholder; generated skeleton and output tokens must be checked again before submission. No truncation.",
        "model_calls": 0,
        "generation_qualified": False,
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"tasks": len(rows), "maximum_input_tokens": result["maximum_input_tokens"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.root, args.tokenizer, args.output)
