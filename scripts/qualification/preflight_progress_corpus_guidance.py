"""CPU-only full-corpus teacher coverage and exact native-template size probe."""

import argparse
import json
from pathlib import Path

from progress_teacher_context import compose_corpus
from sdpo_agent_config import aligned_messages
from sdpo_prompting import prompt_ids


def inspect(materials, tokenizer):
    rows = []
    for bundle in materials:
        material, compiled = bundle["material"], bundle["compiled"]
        checks = [c for c in material["goal_manifest"]["checks"] if c["role"] in ("goal", "harm")]
        if compiled["audit"]["check_coverage"] != [c["check_id"] for c in checks]:
            raise ValueError("missing manifest requirement")
        messages = aligned_messages(material["native_data"]["prompt"])
        supplement = compose_corpus(material["task"], messages, compiled["sheet"], [], [])
        student = prompt_ids(tokenizer, messages, material["tool_definitions"], thinking=True)
        teacher = prompt_ids(tokenizer, messages + [{"role": "user", "content": supplement}],
                             material["tool_definitions"], thinking=True)
        rows.append({"task": material["task"], "checks": len(checks),
                     "initial_student_tokens": len(student), "initial_teacher_tokens": len(teacher),
                     "reply_reserve": 4096, "fits_initial_32k_with_reply": len(teacher) + 4096 <= 32768,
                     "semantic_qualified": False})
    names = [r["task"] for r in rows]
    if len(names) != 124 or len(set(names)) != 124:
        raise ValueError("full124 mandatory, no dropped tasks")
    return {"task_count": 124, "checks": sum(r["checks"] for r in rows), "rows": rows,
            "initial_fit_qualified": all(r["fits_initial_32k_with_reply"] for r in rows),
            "optimizer_updates": 0,
            "scope": "Exact initial native templates only. Growing histories require fresh whole-transition admission; no semantic/gradient/native-execution qualification."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materials", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    result = inspect(json.loads(args.materials.read_text()), tokenizer)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"tasks": result["task_count"], "initial_fit_qualified": result["initial_fit_qualified"],
                      "largest_prefix": max(r["initial_teacher_tokens"] for r in result["rows"])}))
    if not result["initial_fit_qualified"]:
        raise SystemExit(1)
