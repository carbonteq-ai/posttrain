"""Bind an explicit research-review ledger to development-only episode bytes.

This is offline review bookkeeping, not a scorer, reward implementation, or
substantive review of tasks absent from the separately authored case notes.
It never loads inventories, reserved splits, imported runtimes, or assertions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def build(index_path: Path, coverage_path: Path, cases_path: Path) -> dict:
    index_raw, coverage_raw, cases_raw = (path.read_bytes() for path in (index_path, coverage_path, cases_path))
    index, coverage, cases = (json.loads(raw) for raw in (index_raw, coverage_raw, cases_raw))
    indexed = {entry["task_name"]: entry for entry in index["entries"]}
    if len(indexed) != len(index["entries"]):
        raise ValueError("Duplicate development task")
    if any(entry["split"] != "development" for entry in index["entries"]):
        raise ValueError("Only explicitly indexed development episodes are permitted")
    notes = {entry["task_name"]: entry for entry in cases["cases"]}
    if len(notes) != len(cases["cases"]):
        raise ValueError("Duplicate case note")
    expected_names = [entry["task_name"] for entry in coverage["entries"]]
    if len(expected_names) != len(set(expected_names)):
        raise ValueError("Duplicate expected development task")
    rows = []
    for expected in coverage["entries"]:
        if expected["split"] != "development":
            raise ValueError("Reserved metadata is not permitted in this ledger")
        name = expected["task_name"]
        row = {
            "task_name": name,
            "family": expected["family"],
            "split": "development",
            "collection_status": expected["status"],
            "collection_unavailable": expected.get("unavailable", []),
            "review_status": "source_not_in_review_index",
            "policy_action_review": "not_substantively_reviewed",
            "reward_qualification": "not_granted_by_review",
            "token_projection": "unsupported",
        }
        row["collection_verification_scope"] = expected.get("verification_scope")
        if expected["status"] == "interrupted" and name not in indexed:
            row["review_status"] = "native_episode_unavailable_after_interruption"
            row["behavioral_evidence_status"] = "not_a_behavioral_failure_label"
        if name in indexed:
            entry = indexed[name]
            raw = Path(entry["episode_path"]).read_bytes()
            if sha(raw) != entry["episode_digest"]:
                raise ValueError(f"Changed source episode: {name}")
            episode = json.loads(raw)
            if episode["task"]["data"].get("task_name") != name:
                raise ValueError(f"Task/source identity mismatch: {name}")
            occurrences = {}
            errors = []
            scores = []
            trace_ids = []
            for trace in episode["traces"]:
                trace_ids.append(trace["id"])
                errors.extend(error.get("type", "unknown") for error in trace.get("errors", []))
                for reward_name, reward in trace.get("rewards", {}).items():
                    scores.append({"name": reward_name, **reward})
                for event in trace.get("tool_execution_events", []):
                    if event.get("source") == "tool_server":
                        key = (trace["id"], event["invocation_id"])
                        occurrences.setdefault(key, []).append(event["phase"])
            row.update(
                attempt_id=entry["attempt_id"],
                source_episode_path=entry["episode_path"],
                source_episode_sha256=entry["episode_digest"],
                source_generation=entry["source_generation"],
                trace_ids=trace_ids,
                execution_ok=episode["ok"],
                execution_error_types=errors,
                observed_tool_server_occurrences=len(occurrences),
                occurrence_phase_sets=dict(Counter("+".join(v) for v in occurrences.values())),
                official_rewards=scores,
                review_status="identity_outcome_and_receipt_structure_screened_only",
            )
            row["source_verification"] = entry.get("verification", {})
            row["source_verification_interpretation"] = (
                "Verification errors concern replay evidence; current_outcome_verified=False "
                "with assertions_not_fully_satisfied is an unsolved official outcome, "
                "not a source corruption claim."
            )
            official = [score["score"] for score in scores if score["name"] == "partial_credit"]
            row["official_outcome_class"] = (
                "execution_unavailable"
                if not episode["ok"]
                else "official_score_unavailable"
                if len(official) != 1
                else "official_full"
                if official[0] == 1
                else "official_zero"
                if official[0] == 0
                else "official_partial"
            )
            if not episode["ok"]:
                row["behavioral_evidence_status"] = (
                    "partial_execution_terminal_assessment_unavailable"
                    if occurrences
                    else "startup_execution_unavailable"
                )
            else:
                row["behavioral_evidence_status"] = "retained_not_automatically_complete"
            if name in notes:
                note = notes[name]
                if note.get("attempt_id") and note["attempt_id"] != entry["attempt_id"]:
                    raise ValueError(f"Review attempt mismatch: {name}")
                for occurrence_id in note.get("occurrence_ids", []):
                    matching = [key for key in occurrences if key[1] == occurrence_id]
                    if len(matching) != 1:
                        raise ValueError(f"Unresolved or ambiguous cited occurrence {name}: {occurrence_id}")
                row["policy_action_review"] = note["scope"]
                row["review_status"] = note["scope"]
                row["case_review"] = note
        rows.append(row)
    if set(indexed) - {row["task_name"] for row in rows}:
        raise ValueError("Index has tasks outside expected development coverage")
    unresolved_notes = set(notes) - set(indexed)
    if unresolved_notes:
        raise ValueError(f"Reviewed notes lack indexed source: {sorted(unresolved_notes)}")
    family_summary = {}
    for family in sorted({row["family"] for row in rows}):
        members = [row for row in rows if row["family"] == family]
        family_summary[family] = {
            "expected": len(members),
            "indexed": sum("source_episode_sha256" in row for row in members),
            "review_status_counts": dict(Counter(row["review_status"] for row in members)),
            "official_outcome_counts": dict(
                Counter(row.get("official_outcome_class", "source_not_in_review_index") for row in members)
            ),
        }
    return {
        "schema_version": 1,
        "purpose": "Research review coverage; not a scorer, eligibility decision, or all-task substantive acceptance",
        "input_bindings": {
            "index": {"path": str(index_path.resolve()), "sha256": sha(index_raw)},
            "expected_coverage": {"path": str(coverage_path.resolve()), "sha256": sha(coverage_raw)},
            "case_notes": {"path": str(cases_path.resolve()), "sha256": sha(cases_raw)},
        },
        "expected_development_tasks": len(rows),
        "indexed_and_hash_verified": len(indexed),
        "native_episode_unavailable": sum("source_episode_sha256" not in row for row in rows),
        "source_verification_error_count": sum(
            bool(row.get("source_verification", {}).get("verification_error")) for row in rows
        ),
        "review_status_counts": dict(Counter(row["review_status"] for row in rows)),
        "family_counts": dict(Counter(row["family"] for row in rows)),
        "family_summary": family_summary,
        "families_with_case_review": sorted({row["family"] for row in rows if "case_review" in row}),
        "reserved_task_payloads_used": 0,
        "rules_implemented_by_this_review": 0,
        "entries": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path)
    parser.add_argument("coverage", type=Path)
    parser.add_argument("cases", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = build(args.index, args.coverage, args.cases)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "expected_development_tasks",
                    "indexed_and_hash_verified",
                    "review_status_counts",
                    "families_with_case_review",
                )
            },
            indent=2,
        )
    )
