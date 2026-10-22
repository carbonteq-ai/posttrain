"""Missing-aware quality audit of synthetic annotations, without a model judge."""

import argparse
import json
from collections import Counter
from pathlib import Path

from generate_luna_ssr_pilot import parsed_json


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


def inspect_rationale(row):
    request = row["request"]
    try:
        value = parsed_json(row["text"])
    except (ValueError, TypeError):
        return {"task": row["task"], "action_index": row.get("action_index"), "json_parsed": False}
    if not isinstance(value, dict):
        return {
            "task": row["task"],
            "action_index": row.get("action_index"),
            "json_parsed": True,
            "annotation_object_valid": False,
        }
    prefix_text = json.dumps(request["prefix_evidence"], ensure_ascii=False)
    rationale = str(value.get("rationale", ""))
    # Review flags, not a semantic truth test: target arguments may contain
    # values known only to the privileged reference, never to the acting prefix.
    novel_target_literals = sorted(
        {
            text
            for text in strings(request["target_action"].get("arguments", {}))
            if len(text) >= 5 and text in rationale and text not in prefix_text
        }
    )
    allowed = {item["id"] for item in request["prefix_evidence"]}
    refs = value.get("evidence_refs", [])
    scoped = isinstance(refs, list) and bool(refs) and all(isinstance(ref, str) and ref in allowed for ref in refs)
    return {
        "task": row["task"],
        "action_index": row.get("action_index"),
        "json_parsed": True,
        "supported_claim": value.get("supported"),
        "rationale": rationale,
        "evidence_refs": refs,
        "reference_scope_valid": scoped,
        "review_flags": ["target_literal_absent_from_prefix"] if novel_target_literals else [],
        "target_literals_for_review": novel_target_literals,
        "semantic_grounding_verified": False,
        "sft_qualified": False,
    }


def analyze(cases, outputs):
    manifest = json.loads((cases / "manifest.json").read_text())
    expected = {(row["task"], index) for row in manifest["tasks"] for index in range(row["actions"])}
    stages = {}
    annotations = []
    for stage in ("skeletons", "evidence", "rationales", "critiques"):
        path = outputs / (stage + ".jsonl")
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
        stages[stage] = {
            "observed": len(rows),
            "format_and_admission_passed": sum(row.get("format_valid") is True for row in rows),
            "finish_reasons": dict(Counter(row["finish_reason"] for row in rows)),
            "validation_errors": dict(Counter(row["validation_error"] for row in rows if "validation_error" in row)),
            "generated_tokens": sum(len(row["output_token_ids"]) for row in rows),
            "thinking_delimiter_observed": sum("</think>" in row["text"] for row in rows),
        }
        if stage == "rationales":
            annotations = [inspect_rationale(row) for row in rows]
        if stage == "critiques":
            opinions = [row["parsed"] for row in rows if isinstance(row.get("parsed"), dict)]
            stages[stage]["critic_accepted"] = sum(value.get("acceptable") is True for value in opinions)
            stages[stage]["critic_rejected"] = sum(value.get("acceptable") is False for value in opinions)
            stages[stage]["scope"] = "Model opinion only, not proof of correctness or SFT admission"
    seen = [(row["task"], row["action_index"]) for row in annotations]
    if len(set(seen)) != len(seen) or set(seen) - expected:
        raise ValueError("duplicate or unexpected generated action identity")
    blocked = []
    for row in manifest["tasks"]:
        binding = row["binding"]
        if not binding["initial_state_matches"] or binding["unavailable_tools"] or binding["manifest_authority_errors"]:
            blocked.append({"task": row["task"], "binding": binding})
    prioritized = sorted(
        annotations,
        key=lambda row: (
            not bool(row.get("review_flags")),
            row.get("supported_claim") is not False,
            row.get("json_parsed") is not False,
            row["task"],
            row["action_index"],
        ),
    )
    examples, tasks_seen = [], set()
    for row in prioritized:
        if row["task"] not in tasks_seen:
            examples.append(row)
            tasks_seen.add(row["task"])
        if len(examples) == 4:
            break
    return {
        "model_output_directory": str(outputs),
        "receipt_present": (outputs / "receipt.json").exists(),
        "stages": stages,
        "expected_actions": len(expected),
        "observed_actions": len(seen),
        "missing_actions": [{"task": task, "action_index": index} for task, index in sorted(expected - set(seen))],
        "explicit_unsupported_annotations": sum(row.get("supported_claim") is False for row in annotations),
        "possible_literal_leakage_annotations": sum(bool(row.get("review_flags")) for row in annotations),
        "source_binding_blockers": blocked,
        "review_examples": examples,
        "annotations": annotations,
        "raw_generator_thinking_export_allowed": False,
        "reason": "Generator saw privileged reference/future context. Only separately reviewed prefix-grounded annotations may become student targets.",
        "native_replay_qualified": False,
        "sft_qualified": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("preserve previous audits")
    args.output.write_text(json.dumps(analyze(args.cases, args.outputs), indent=2) + "\n")
