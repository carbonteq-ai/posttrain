"""Export readable local generation evidence, preserving raw model output."""

import argparse
import hashlib
import json
from pathlib import Path


def export(cases, outputs, destination):
    manifest = json.loads((cases / "manifest.json").read_text())
    by_task = {}
    for row in manifest["tasks"]:
        raw = (cases / row["case_file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["case_sha256"]:
            raise ValueError("case digest mismatch")
        by_task[row["task"]] = json.loads(raw)
    stages = {}
    for stage in ("skeletons", "evidence", "rationales", "critiques"):
        path = outputs / (stage + ".jsonl")
        stages[stage] = (
            [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
        )
    lines = [
        "Qwen SFT data generation trace review",
        "",
        f"Output snapshot: {outputs.resolve()}",
        "These are generator diagnostics, not admitted SFT training examples.",
        "The generator saw Luna reference executions and privileged goals/harms.",
        "Its raw thinking below must not be exported as student reasoning targets.",
        "Luna tool calls below are supplied examples, not newly generated Qwen actions.",
        "A format pass does not establish correct reasoning or native task success.",
        "",
        "AVAILABLE OUTPUTS",
    ]
    for stage, rows in stages.items():
        lines.append(f"{stage}: {len(rows)} saved responses")
    if not stages["rationales"]:
        lines += [
            "",
            "Per-action rationales have not been saved in this snapshot.",
            "The baseline writes that stage after its whole batch completes.",
        ]
    names = sorted(
        {row["task"] for rows in stages.values() for row in rows},
        key=lambda name: (not name.startswith("simple."), name),
    )
    for number, name in enumerate(names, 1):
        case = by_task[name]
        lines += ["", "=" * 78, f"EXAMPLE {number}: {name}", "=" * 78, "", "NORMAL TASK REQUEST"]
        lines.extend(message["content"] for message in case["student_prompt"] if message["role"] == "user")
        lines += [
            "",
            f"LUNA SOURCE PARTIAL CREDIT: {case['source_partial_credit']}",
            "SOURCE COMPATIBILITY CHECKS",
            json.dumps(case["binding"], indent=2),
            "",
            "SUPPLIED LUNA TOOL CALLS (reference examples only)",
        ]
        for index, action in enumerate(case["teacher_only"]["reference_actions_and_observations"]):
            lines += [f"Action {index}", json.dumps(action["call"], indent=2)]
        lines += ["", f"Full inputs, observed tool results and goals: {(cases / (name + '.json')).resolve()}"]
        for stage, rows in stages.items():
            for row in rows:
                if row["task"] != name:
                    continue
                lines += [
                    "",
                    f"QWEN GENERATED {stage.upper()} — action {row.get('action_index')}",
                    f"Finish reason: {row['finish_reason']}; output tokens: {len(row['output_token_ids'])}",
                    f"Current format/admission check: {row.get('format_valid')}",
                ]
                if row.get("validation_error"):
                    lines.append("Check failure: " + row["validation_error"])
                lines += ["", "RAW GENERATED TEXT (unchanged)", row["text"], "END RAW GENERATED TEXT"]
    destination.write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "file": str(destination.resolve()),
                "examples": len(names),
                "responses": {stage: len(rows) for stage, rows in stages.items()},
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.cases, args.outputs, args.output)
