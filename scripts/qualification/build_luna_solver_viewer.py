"""Build a local comparison of actual control/guided solver calls and results."""

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from audit_luna_solver_pilot import audit
from luna_solver_prompts import text_content


def build(evidence, output):
    models = {}
    native_audit = audit(evidence)["models"]
    task_prompts = {}
    for size in ("4b", "9b", "27b"):
        root = evidence / size
        preparation_root = evidence / "critic27b" if size == "4b" and (evidence / "critic27b").exists() else root
        stages_path = preparation_root / "instruction-stages.jsonl"
        stages = (
            [json.loads(line) for line in stages_path.read_text().splitlines() if line.strip()]
            if stages_path.exists()
            else []
        )
        materials_path = root / "materials.json"
        if materials_path.exists():
            for material in json.loads(materials_path.read_text()).values():
                task_prompts[material["task"]] = material["user_request"]
        calls_path = root / "solver-calls.jsonl"
        calls = (
            [json.loads(line) for line in calls_path.read_text().splitlines() if line.strip()]
            if calls_path.exists()
            else []
        )
        results = {}
        for arm in ("control", "guided"):
            path = root / arm / "native/traces.jsonl"
            episodes = (
                [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
            )
            for episode in episodes:
                for trace in episode["traces"]:
                    for node in trace.get("nodes", []):
                        msg = node.get("message") or {}
                        if msg.get("role") == "tool":
                            results[(arm, msg.get("tool_call_id"))] = msg
        for row in calls:
            user = [text_content(m["content"]) for m in row["clean_student_request"]["messages"] if m["role"] == "user"]
            task_prompts[row["task"]] = "\n\n".join(user)
            choice = row["response"]["choices"][0]
            message = choice["message"]
            row["display"] = {
                "reasoning": message.get("reasoning_content") or message.get("reasoning") or "",
                "content": message.get("content") or "",
                "finish": choice["finish_reason"],
                "tool_calls": [
                    {"call": call, "result": results.get((row["arm"], call["id"]))}
                    for call in message.get("tool_calls") or []
                ],
            }
        models[size] = {
            "calls": calls,
            "stages": stages,
            "native_audit": native_audit[size]["episodes"],
            "preparation_errors": json.loads((preparation_root / "instruction-errors.json").read_text())
            if (preparation_root / "instruction-errors.json").exists()
            else {},
            "receipt": json.loads((root / "receipt.json").read_text()) if (root / "receipt.json").exists() else {},
            "execution_status": json.loads((root / "execution-status.json").read_text())
            if (root / "execution-status.json").exists()
            else {},
            "complete": (root / "receipt.json").exists(),
            "outcomes": json.loads((root / "outcomes.json").read_text()) if (root / "outcomes.json").exists() else {},
        }
    data = {
        "models": models,
        "tasks": task_prompts,
        "qualitative_review": json.loads((evidence / "qualitative-review.json").read_text())
        if (evidence / "qualitative-review.json").exists()
        else None,
        "snapshot": datetime.now(ZoneInfo("Asia/Karachi")).strftime("%d %b %Y, %H:%M PKT"),
    }
    template = Path(__file__).parent / "fixtures/luna_solver_viewer.html"
    page = template.read_text()
    page = page.replace("Instruction preparation · same local model", "Instruction preparation and review")
    page = page.replace('</select></div></div><div id="request"',
                        '<option value="27b">Qwen3.5-27B</option></select></div></div><div id="request"')
    count = len(task_prompts)
    page = page.replace("on the same two tasks", f"on {count} declared training tasks")
    page = page.replace("Two tasks are a diagnostic, not a benchmark", "These reused training tasks are diagnostics, not an unseen benchmark")
    available = [size for size in models if (evidence / size / "materials.json").exists()]
    if available:
        for size in models:
            if size not in available:
                page = page.replace(f'<option value="{size}">', f'<option value="{size}" disabled>')
        page = page.replace(f'<option value="{available[0]}">', f'<option value="{available[0]}" selected>')
    output.write_text(page.replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")))
    print(json.dumps({"file": str(output), "calls": {size: len(model["calls"]) for size, model in models.items()}}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.evidence, args.output)
