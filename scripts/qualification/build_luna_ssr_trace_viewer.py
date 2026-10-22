"""Build a private, self-contained trace comparison with no external requests."""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def read_run(root, pending):
    run = {"state": pending, "complete": (root / "receipt.json").exists()}
    for stage in ("skeletons", "evidence", "rationales", "critiques"):
        path = root / (stage + ".jsonl")
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
        run[stage] = [
            {
                **{key: value for key, value in row.items() if key != "output_token_ids"},
                "output_tokens": len(row["output_token_ids"]),
            }
            for row in rows
        ]
    audit_path = root / "full-audit-r1.json"
    run["audit_annotations"] = json.loads(audit_path.read_text())["annotations"] if audit_path.exists() else []
    rows = run["rationales"]
    if rows:
        run["summary"] = (
            f"{len(rows)} action annotations saved · "
            f"{sum(row['format_valid'] for row in rows)} passed automated checks · "
            f"{sum(row['finish_reason'] == 'length' for row in rows)} length-limited"
        )
    return run


def build(evidence):
    cases = evidence / "pilot14-r4"
    manifest_raw = (cases / "manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    tasks = []
    for row in manifest["tasks"]:
        raw = (cases / row["case_file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["case_sha256"]:
            raise ValueError("case source changed")
        case = json.loads(raw)
        domain, task = row["task"].split(".", 1)
        tasks.append(
            {
                "task": row["task"],
                "label": f"{domain.title()} / {task.replace('_', ' ')}",
                "prompt": "\n\n".join(m["content"] for m in case["student_prompt"] if m["role"] == "user"),
                "score": case["source_partial_credit"],
                "omitted": case["source_metrics"].get("assertions_excluded"),
                "actions": case["teacher_only"]["reference_actions_and_observations"],
                "goals": case["teacher_only"]["goals_and_constraints"],
                "binding": case["binding"],
            }
        )
    tasks.sort(key=lambda row: (not row["task"].startswith("simple."), row["task"]))
    data = {
        "snapshot": datetime.now(ZoneInfo("Asia/Karachi")).strftime("%d %b %Y, %H:%M PKT"),
        "case_digest": hashlib.sha256(manifest_raw).hexdigest(),
        "tasks": tasks,
        "runs": {
            "original": {
                size: read_run(
                    evidence / "generation-r1" / size, f"No saved {size.upper()} generation output in this snapshot."
                )
                for size in ("4b", "9b")
            },
            "revised": {
                "4b": read_run(
                    evidence / "quality-r3" / "4b",
                    "Revised4B pipeline queued after the9B baseline; no revised output saved yet.",
                ),
                "9b": {"state": "A revised9B arm has not been scheduled.", "complete": False},
            },
        },
    }
    template = (evidence / "trace-comparison-template.html").read_text()
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False).replace("<", "\\u003c")
    output = evidence / "trace-comparison.html"
    output.write_text(template.replace("__TRACE_DATA__", payload))
    print(json.dumps({"file": str(output.resolve()), "tasks": len(tasks), "bytes": output.stat().st_size}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    build(parser.parse_args().evidence)
