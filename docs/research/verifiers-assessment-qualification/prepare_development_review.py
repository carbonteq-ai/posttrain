"""Prepare inspectable development review inputs; never score or assign credit.

Native episodes remain authoritative. These derived files omit hidden assertions
and retain source digests and occurrence IDs for the human/Astra design review.
Initial simulator state is source material, not automatically an authored policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def encoded(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def prepare(index_path: Path, destination: Path) -> dict:
    index_bytes = index_path.read_bytes()
    index = json.loads(index_bytes)
    entries = index["entries"]
    if any(entry["split"] != "development" for entry in entries):
        raise ValueError("Review preparation accepts development entries only")
    names = [entry["task_name"] for entry in entries]
    if len(names) != len(set(names)):
        raise ValueError("Repeated task identities require an explicit review policy")
    prepared = []
    families: Counter = Counter()
    for entry in entries:
        source_path = Path(entry["episode_path"])
        raw = source_path.read_bytes()
        if digest(raw) != entry["episode_digest"]:
            raise ValueError(f"Episode bytes changed: {entry['task_name']}")
        episode = json.loads(raw)
        task = episode["task"]["data"]
        if task.get("task_name") != entry["task_name"]:
            raise ValueError("Index and retained task identity disagree")
        occurrences = []
        for trace in episode["traces"]:
            grouped: dict[str, list] = {}
            for event in trace.get("tool_execution_events", []):
                if event.get("source") == "tool_server":
                    grouped.setdefault(event["invocation_id"], []).append(event)
            for invocation_id, events in grouped.items():
                # No join to sampled calls, turns or tokens is inferred here.
                receipts = [json.loads(event["receipt_json"]) for event in events]
                occurrences.append(
                    {
                        "trace_id": trace["id"],
                        "origin": "tool_server",
                        "invocation_id": invocation_id,
                        "receipt_sequences": [event["receipt_seq"] for event in events],
                        "phases": [event["phase"] for event in events],
                        "receipts": receipts,
                    }
                )
        material = {
            "purpose": "Development reward-design review input, not a scored assessment",
            "task_name": entry["task_name"],
            "family": entry["family"],
            "attempt_id": entry["attempt_id"],
            "manifest_digest": entry["manifest_digest"],
            "source_episode_path": str(source_path),
            "source_episode_sha256": digest(raw),
            "source_generation": entry["source_generation"],
            "task_material": {
                key: task[key] for key in ("prompt", "system_prompt", "initial_state", "zapier_tools") if key in task
            },
            "execution": {key: episode[key] for key in ("ok", "errors") if key in episode},
            "observed_occurrences": occurrences,
            "limits": [
                "Initial state requires independent policy/visibility interpretation",
                "Receipt presence does not establish domain success or complete action coverage",
                "Occurrence IDs do not establish generated call, turn or token alignment",
                "No hidden assertion list or expected final state is used as policy authority",
                "Zero-child and startup failures are retained, not inferred task failures",
            ],
        }
        body = encoded(material)
        path = destination / "inputs" / f"{entry['attempt_id']}.json"
        if path.exists() and path.read_bytes() != body:
            raise ValueError("Refusing to overwrite a changed review input")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        prepared.append(
            {
                "task_name": entry["task_name"],
                "family": entry["family"],
                "attempt_id": entry["attempt_id"],
                "input_path": str(path.resolve()),
                "input_sha256": digest(body),
                "source_episode_sha256": digest(raw),
                "observed_occurrences": len(occurrences),
                "review_status": "not_substantively_reviewed",
            }
        )
        families[entry["family"]] += 1
    result = {
        "scope": "Development-only immutable review preparation; no reward or eligibility decisions",
        "index_path": str(index_path.resolve()),
        "index_sha256": digest(index_bytes),
        "entries": prepared,
        "task_count": len(prepared),
        "families": dict(families),
    }
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "manifest.json"
    body = encoded(result)
    if manifest_path.exists() and manifest_path.read_bytes() != body:
        raise ValueError("Use a new destination for a different review index")
    manifest_path.write_bytes(body)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    result = prepare(args.index, args.destination)
    print(
        json.dumps(
            {
                "task_count": result["task_count"],
                "families": len(result["families"]),
                "manifest": str(args.destination / "manifest.json"),
            }
        )
    )
