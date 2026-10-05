"""Replay every AutomationBench Simple manifest on GPT Luna's recorded episode.

Run from an ``environments/automationbench_v1`` checkout of carbonteq-ai/verifiers-environments
(its manifests and drafts are read relative to the working directory):

    uv run --frozen python <rl>/docs/research/verifiers-assessment-qualification/reward-candidate/simple-manifest-replay.py \
        <rl>/docs/research/verifiers-assessment-qualification/luna-development-review-coverage.json OUT.json

``--write-rebound COVERAGE TASK...`` writes the rebound drafts of the named tasks.

Goals and harms are read with ``manifest_outcomes``, the function manifest step credit uses in
training, so the selection counts exactly what training rewards. One process, sequential.

A draft that binds the whole prompt (system message included, which states the turn budget it
was authored at) is replayed rebound to the user message, ``prompt[1].content``, when that
message is identical in Luna's episode and in the 14- and 16-turn tasks and the draft never reads
the system message; the row records ``rebound`` and ``write_rebound`` writes those drafts back.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from pathlib import Path

import verifiers.v1 as vf
from automationbench_v1 import manifest_assessments
from automationbench_v1.capture import canonical_json
from automationbench_v1.contracts import load_task_contract as packaged_loader
from automationbench_v1.contracts.engine import binding_reason
from automationbench_v1.contracts.loader import load_contract
from automationbench_v1.contracts.models import CheckSpec
from automationbench_v1.manifest_assessments import OUTPUT_KIND as RECORD_OUTPUT
from automationbench_v1.manifest_assessments import ManifestAssessmentTask
from automationbench_v1.manifest_obligation_assessments import OBLIGATION_OUTPUT
from automationbench_v1.manifest_step_credit import _latest_payloads, manifest_outcomes
from automationbench_v1.public_state import public_initial_state
from automationbench_v1.taskset import (
    AutomationBenchConfig,
    AutomationBenchData,
    AutomationBenchTaskConfig,
    AutomationBenchTaskset,
)
from automationbench_v1.tools import AutomationBenchState

DOMAINS = ["simple", "sales", "marketing", "operations", "support", "finance", "hr"]


def tasks_at(turn_budget: int) -> dict[str, object]:
    config = AutomationBenchConfig(
        domains=DOMAINS, task={"toolset": "limited_zapier", "search_top_k": 20, "turn_budget": turn_budget}
    )
    return {task.data.task_name: task for task in AutomationBenchTaskset(config).load()}


def manifest_path(name: str, catalog: dict[str, str]) -> Path | None:
    draft = Path("manifest-drafts/tasks") / name / "draft.json"
    if draft.exists():
        return draft
    if name in catalog:
        return Path("src/automationbench_v1/contracts") / catalog[name]
    return None


PROMPT = ["task_evidence", "prompt"]
USER_MESSAGE = ["task_evidence", "prompt", 1, "content"]


def digest(value) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def rebound(raw: dict, prompts: list[list[dict]]) -> dict | None:
    """The draft bound to the user message instead of the whole prompt, or None when not admissible.

    ``prompts`` are Luna's episode prompt then the 14- and 16-turn task prompts."""

    paths = [binding["path"] for binding in raw["bindings"]]
    if PROMPT not in paths or any(path[:3] == [*PROMPT, 0] for path in paths):
        return None
    if '"system' in canonical_json({key: value for key, value in raw.items() if key != "bindings"}).lower():
        return None
    whole = next(binding for binding in raw["bindings"] if binding["path"] == PROMPT)
    if whole["canonical_sha256"] != digest(prompts[0]):
        return None  # not authored against Luna's task
    users = {
        canonical_json(prompt[1]["content"])
        for prompt in prompts
        if len(prompt) > 1 and prompt[1].get("role") == "user"
    }
    if len(users) != 1 or not all(len(prompt) > 1 for prompt in prompts):
        return None
    content = prompts[0][1]["content"]
    if (narrowed := narrowed_requests(raw, content)) is None:
        return None
    bindings = [binding for binding in raw["bindings"] if binding["path"] not in (PROMPT, USER_MESSAGE)]
    bindings.append({"path": USER_MESSAGE, "canonical_sha256": digest(content)})
    return {**narrowed, "bindings": sorted(bindings, key=lambda binding: canonical_json(binding["path"]))}


def _spellings(value) -> set[str]:
    if isinstance(value, bool):
        return set()  # a boolean is never stated verbatim
    if isinstance(value, (int, float)) and float(value).is_integer():
        return {str(int(value)), f"{int(value):,}"}
    return {str(value)}


def narrowed_requests(raw: dict, content) -> dict | None:
    """Request fields authorized by the whole prompt, re-authorized by the user message.

    A field moves only when the user message states it: its literal value appears in the message
    (case-insensitive), or it is copied from inside the message. Otherwise None, and the draft keeps
    its whole-prompt binding. Any other reference to the whole prompt also refuses."""

    text = (content if isinstance(content, str) else canonical_json(content)).casefold()
    sources = {}
    for key, source in raw["sources"].items():
        if not (isinstance(source, dict) and "member_key" in source and "fields" in source):
            sources[key] = source
            continue
        fields = {}
        for name, field in source["fields"].items():
            if PROMPT not in field["authority_paths"]:
                fields[name] = field
                continue
            if field.get("copy_from") is not None:
                if field["copy_from"][: len(USER_MESSAGE)] != USER_MESSAGE:
                    return None
            elif not any(spelling.casefold() in text for spelling in _spellings(field["value"])):
                return None
            paths = [USER_MESSAGE if path == PROMPT else path for path in field["authority_paths"]]
            fields[name] = {**field, "authority_paths": paths}
        sources[key] = {**source, "fields": fields}
    narrowed = {**raw, "sources": sources}
    if canonical_json(PROMPT) in canonical_json({k: v for k, v in narrowed.items() if k != "bindings"}):
        return None
    return narrowed


def binding(task, contract) -> str | None:
    evidence = {
        "task_evidence": {
            "prompt": task.data.model_dump(mode="json")["prompt"],
            "initial": public_initial_state(task.data.initial_state),
        }
    }
    return binding_reason(evidence, contract)


def goal_statuses(trace, contract) -> tuple[int, int]:
    """(passed, undecided) required goals: obligation findings marked required and record goals."""

    goal_checks = {check.check_id for check in contract.checks if getattr(check, "role", None) == "goal"}
    findings = [item for item in _latest_payloads(trace, OBLIGATION_OUTPUT) if item.get("kind") == "finding"]
    passed = sum(
        item.get("required") is True and item.get("status") == "valid" and item.get("value") == 1 for item in findings
    )
    # An abstained goal finding does not say whether it was required: count it as undecided.
    undecided = sum(item.get("status") != "valid" and item.get("check_id") in goal_checks for item in findings)
    goals = {check.check_id for check in contract.checks if isinstance(check, CheckSpec) and check.role == "goal"}
    latest = {}
    for payload in _latest_payloads(trace, RECORD_OUTPUT):
        for result in payload["results"]:
            if result["check_id"] in goals:
                latest[result["check_id"]] = result
    passed += sum(result["status"] == "valid" and result["value"] == 1 for result in latest.values())
    undecided += sum(result["status"] != "valid" for result in latest.values()) + len(goals - set(latest))
    return passed, undecided


def replay(name: str, contract, entry: dict) -> dict:
    raw = Path(entry["source_episode_path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != entry["source_episode_sha256"]:
        raise ValueError("luna_episode_digest_mismatch")
    trace = vf.WireEpisode.model_validate_json(raw).traces[0]
    data = AutomationBenchData.model_validate(trace.task.data.model_dump(mode="json"))
    trace.state = AutomationBenchState(
        world=trace.info["automationbench"]["end_state"],
        initial_state=data.initial_state,
        assertions=data.assertions,
        artifacts=dict(trace.state.artifacts),
    )
    manifest_assessments.load_task_contract = lambda _name: contract
    try:
        asyncio.run(ManifestAssessmentTask(data, AutomationBenchTaskConfig(capture_actions=True)).score(trace))
    finally:
        manifest_assessments.load_task_contract = packaged_loader
    occurrences, required, harms = manifest_outcomes(trace)
    passed, undecided = goal_statuses(trace, contract)
    return {
        "passed": passed,
        "undecided": undecided,
        "assessment_errors": len(trace.assessment_errors or ()),
        "credit_errors": len(trace.credit_errors or ()),
        "required": required,
        "witnessed": len(occurrences),
        "harms": len(harms),
    }


def main() -> None:
    coverage = {item["task_name"]: item for item in json.load(open(sys.argv[1]))["entries"]}
    catalog = json.loads(Path("src/automationbench_v1/contracts/catalog.json").read_text())["tasks"]
    names = sorted(
        {path.name for path in Path("manifest-drafts/tasks").iterdir() if path.name.startswith("simple.")}
        | {name for name in catalog if name.startswith("simple.")}
    )
    loaded = {budget: tasks_at(budget) for budget in (14, 16)}
    rows = []
    for name in names:
        entry = coverage.get(name) or {}
        rewards = entry.get("official_rewards") or [{}]
        row = {
            "task": name,
            "luna_class": entry.get("official_outcome_class"),
            "luna_score": rewards[0].get("score"),
        }
        try:
            path = manifest_path(name, catalog)
            row["source"] = str(path)
            row["installed"] = name in catalog
            raw = json.loads(path.read_bytes())
            if entry.get("source_episode_path") and not row["installed"]:
                episode = vf.WireEpisode.model_validate_json(Path(entry["source_episode_path"]).read_bytes())
                prompts = [episode.traces[0].task.data.model_dump(mode="json")["prompt"]] + [
                    loaded[budget][name].data.model_dump(mode="json")["prompt"] for budget in (14, 16)
                ]
                if (changed := rebound(raw, prompts)) is not None:
                    raw, row["rebound"] = changed, True
            contract = load_contract(json.dumps(raw).encode())
            row["binding_14"] = binding(loaded[14][name], contract)
            row["binding_16"] = binding(loaded[16][name], contract)
            if entry.get("source_episode_path"):
                row |= replay(name, contract, entry)
        except Exception as error:  # noqa: BLE001 - one task's failure is recorded, not fatal
            row["error"] = f"{type(error).__name__}: {str(error)[:200]}"
        rows.append(row)
        print(json.dumps(row), flush=True)
    Path(sys.argv[2]).write_text(json.dumps(rows, indent=1) + "\n")


def write_rebound(names: list[str]) -> None:
    """Write the rebound drafts of ``names`` (the tasks being installed) back to their draft files."""

    coverage = {item["task_name"]: item for item in json.load(open(sys.argv[2]))["entries"]}
    loaded = {budget: tasks_at(budget) for budget in (14, 16)}
    for name in names:
        path = Path("manifest-drafts/tasks") / name / "draft.json"
        raw = json.loads(path.read_bytes())
        episode = vf.WireEpisode.model_validate_json(Path(coverage[name]["source_episode_path"]).read_bytes())
        prompts = [episode.traces[0].task.data.model_dump(mode="json")["prompt"]] + [
            loaded[budget][name].data.model_dump(mode="json")["prompt"] for budget in (14, 16)
        ]
        if (changed := rebound(raw, prompts)) is not None:
            path.write_text(json.dumps(changed, indent=2, ensure_ascii=False) + "\n")
            print("rebound", name)


if __name__ == "__main__":
    if sys.argv[1] == "--write-rebound":
        write_rebound(sys.argv[3:])
    else:
        main()
