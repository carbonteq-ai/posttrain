# ruff: noqa: E701, E702, E741, B905
"""Offline manifest-guard (harm) rescoring of recorded AutomationBench episodes.

Phase 1 (replay_worker.py) re-executes each episode's recorded tool calls, in
order, against the task's initial world with the tools of the environment
revision the episode ran on, and rebuilds the tool-server receipts and state
writes a live episode would have produced. Phase 2 (here) runs the CURRENT
installed guard manifests over that inventory, exactly as
``manifest_guard_assessments.assess_guard`` evaluates them, and counts distinct
(check, invocation) harms, which is what ``manifest_step_credit`` sums into the
per-turn ``manifest_harms`` components.

Run with the environment repo's interpreter (the verifiers-environments checkout is
AUTOMATIONBENCH_ENV_REPO, default ../verifiers-environments; its automationbench_v1
manifests are the "current" ones):

  PY=$AUTOMATIONBENCH_ENV_REPO/environments/automationbench_v1/.venv/bin/python
  $PY rescore.py payloads.jsonl.gz out.jsonl [--workers 4] [--env-map env_revisions.json] [--current-tools]

Input lines: {"trace_id", "run", "payload": full Doris trace payload} or compact
harvested records ({"id","task_name","nodes",...}; task data then comes from the
installed taskset and replay uses current tools). ``--env-map`` maps
info.environment_id -> {"revision": <verifiers-environments sha>}; each
revision's src tree is extracted once with ``git archive`` into ./envs/.

Validation (2026-10-11, 5,658 episodes of 48 guarded tasks): 3,453 of 3,647 episodes with
recorded harms match exactly; every mismatch is a higher re-scored count, 186 of 194 from
the r9/r10 manifest-steps runs, whose environment revisions recorded call arguments in a
form some guards could not match (REPLAY_ERA_CAPTURE=1 reproduces those under-counts).
5,531 replays are exact (no tool-result or final-world mismatch). About 0.08 s per episode.

Inputs come from fetch_slim.py (read-only Doris, run inside the Trackio container);
env_revisions.json comes from resolve_envs.py.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENV_REPO = Path(os.environ.get("AUTOMATIONBENCH_ENV_REPO", "../verifiers-environments"))
CURRENT_SRC = ENV_REPO / "environments/automationbench_v1/src"


# ------------------------------------------------------------------ inputs
def recorded_harms(payload: dict) -> int | None:
    evidence = (payload.get("info") or {}).get("posttrain_turn_rewards") or {}
    total, seen = 0, False
    for assessment in evidence.get("assessments") or []:
        for component in assessment.get("components") or []:
            if component.get("name") == "manifest_harms":
                seen = True
                total += int(component.get("value") or 0)
    return total if seen else None


_TASKS: dict | None = None


def task_data(name: str) -> dict:
    global _TASKS
    if _TASKS is None:
        from automationbench_v1.taskset import AutomationBenchConfig, AutomationBenchTaskset, get_available_domains

        # Task data as the current taskset builds it (no turn budget / world-time prompt
        # decoration): compact records carry no task data of their own.
        taskset = AutomationBenchTaskset(
            AutomationBenchConfig.model_validate(
                {"domains": sorted(get_available_domains()), "task": {"toolset": "limited_zapier"}}
            )
        )
        _TASKS = {t.data.task_name: t.data.model_dump(mode="json") for t in taskset.load()}
    return _TASKS[name]


def episode(record: dict) -> dict:
    if "payload" in record:
        p = record["payload"]
        info = p.get("info") or {}
        return {
            "id": record["trace_id"],
            "run": record.get("run"),
            "data": p["task"]["data"],
            "nodes": p["nodes"],
            "end_state": (info.get("automationbench") or {}).get("end_state"),
            "environment_id": info.get("environment_id"),
            "recorded_harms": recorded_harms(p),
        }
    return {
        "id": record["id"],
        "run": record.get("run"),
        "data": task_data(record["task_name"]),
        "nodes": record["nodes"],
        "end_state": None,
        "environment_id": None,
        "recorded_harms": record.get("harms"),
    }


# ------------------------------------------------------------------ phase 1
def revision_src(revision: str | None) -> Path:
    if revision is None:
        return CURRENT_SRC
    target = HERE / "envs" / revision[:12]
    if not (target / "src/automationbench_v1").exists():
        target.mkdir(parents=True, exist_ok=True)
        archive = subprocess.run(
            ["git", "-C", str(ENV_REPO), "archive", revision, "environments/automationbench_v1/src"],
            check=True,
            capture_output=True,
        ).stdout
        subprocess.run(["tar", "-x", "--strip-components=2", "-C", str(target)], input=archive, check=True)
    return target / "src"


def run_replays(episodes: list[dict], src: Path, workers: int) -> dict[str, dict]:
    """Replay episodes with one revision's tools in up to ``workers`` subprocesses."""
    chunks = [episodes[i::workers] for i in range(min(workers, len(episodes)))]

    def run(chunk):
        payload = "".join(json.dumps({k: e[k] for k in ("id", "data", "nodes", "end_state")}) + "\n" for e in chunk)
        proc = subprocess.run(
            ["nice", sys.executable, "-W", "ignore", str(HERE / "replay_worker.py"), "--src", str(src)],
            input=payload,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONWARNINGS": "ignore"},
        )
        out = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
        if len(out) != len(chunk):
            sys.stderr.write(proc.stderr[-3000:])
        return out

    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=max(1, len(chunks))) as pool:
        for out in pool.map(run, chunks):
            results.update({item["id"]: item for item in out})
    return results


# ------------------------------------------------------------------ phase 2
def guard_harms(data: dict, replayed: dict) -> dict:
    from automationbench_v1.contracts.engine import binding_reason
    from automationbench_v1.contracts.guards import GuardCheck, evaluate_guard, plan_guard_instances
    from automationbench_v1.contracts.loader import load_task_contract
    from automationbench_v1.contracts.populations import InitialCollectionSource
    from automationbench_v1.contracts.requests import RequestSource
    from automationbench_v1.contracts.tables import TableSource
    from automationbench_v1.manifest_guard_assessments import (
        capture_guard_inputs,
        required_tables,
        restore_guard_inputs,
    )

    try:
        from automationbench_v1.manifest_source import manifest_source_material
    except ImportError:  # older engines (scoring with an episode's own-era manifests)

        def manifest_source_material(raw):
            return {
                "task_evidence": raw["task_evidence"],
                "tool_execution_events": raw["tool_execution_events"],
                "state_write_receipts": raw["state_write_receipts"],
            }

    from automationbench_v1.public_state import public_initial_state

    try:
        contract = load_task_contract(data["task_name"])
    except ValueError as error:
        return {"guarded": False, "reason": str(error)[:120]}
    checks = [c for c in contract.checks if isinstance(c, GuardCheck)]
    if not checks:
        return {"guarded": False, "reason": "no_guard_checks"}
    raw = {
        "task_evidence": {
            "task_name": data["task_name"],
            "initial": public_initial_state(data["initial_state"]),
            "final": replayed["final"],
            "prompt": data["prompt"],
            "complete": True,
            "authored_outputs": None,
        },
        "tool_execution_events": replayed["events"],
        "state_write_receipts": replayed["writes"],
    }
    safe = manifest_source_material(raw)
    material = {"source": safe, **capture_guard_inputs(safe, contract)}
    tables, effects = restore_guard_inputs(material, contract)
    authority = binding_reason(safe, contract)
    harms, abstained, unavailable, per_check = set(), 0, 0, {}
    for check in checks:
        keys = required_tables(check)
        cases, _ = plan_guard_instances(
            check, tables[check.population], effects[check.source], *(effects[i.source] for i in check.alternatives)
        )
        unavailable += sum(f.status != "qualified" for f in effects[check.source].effects)
        if authority is not None:
            abstained += len(cases)
            continue
        evaluation = evaluate_guard(
            safe,
            check,
            {n: tables[n] for n in keys},
            effects[check.source],
            effect_source=contract.sources[check.source],
            table_sources={
                n: s
                for n, s in contract.sources.items()
                if n in keys and isinstance(s, (TableSource, InitialCollectionSource, RequestSource))
            },
            join_effects={i.alias: effects[i.source] for i in check.effect_joins},
            join_sources={i.alias: contract.sources[i.source] for i in check.effect_joins},
            alternative_effects={i.alias: effects[i.source] for i in check.alternatives},
            alternative_sources={i.alias: contract.sources[i.source] for i in check.alternatives},
        )
        hits = {(f.check_id, f.occurrence) for f in evaluation.findings if f.value == 1 and f.occurrence}
        abstained += sum(f.value is None for f in evaluation.findings)
        harms |= hits
        if hits:
            per_check[check.check_id] = len(hits)
    return {
        "guarded": True,
        "harms": len(harms),
        "harm_checks": per_check,
        "abstained": abstained,
        "unavailable_effects": unavailable,
        "authority": authority,
        "contract_revision": contract.revision,
    }


def score(ep: dict, replayed: dict) -> dict:
    out = {
        "id": ep["id"],
        "run": ep["run"],
        "task": ep["data"]["task_name"],
        "environment_id": ep["environment_id"],
        "recorded_harms": ep["recorded_harms"],
        "replay_seconds": replayed.get("replay_seconds"),
    }
    if "error" in replayed:
        out["error"] = replayed["error"]
        return out
    out.update(
        stats=replayed["stats"],
        world_diff=replayed["world_diff"],
        fidelity=replayed["fidelity"],
        deterministic=replayed["deterministic"],
    )
    started = time.perf_counter()
    try:
        out.update(guard_harms(ep["data"], replayed))
    except Exception as error:  # noqa: BLE001
        out["error"] = f"guard {type(error).__name__}: {error}"[:400]
    out["score_seconds"] = round(time.perf_counter() - started, 4)
    return out


def records(path: str):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--workers", type=int, default=2, help="replay subprocesses (capped at 4)")
    parser.add_argument("--env-map", help="environment_id -> {revision} JSON; unmapped ids use current tools")
    parser.add_argument("--current-tools", action="store_true", help="replay every episode with current tools")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    started = time.perf_counter()
    env_map = json.loads(Path(args.env_map).read_text()) if args.env_map and not args.current_tools else {}
    episodes = [episode(r) for i, r in enumerate(records(args.input)) if not args.limit or i < args.limit]
    groups: dict[str | None, list[dict]] = defaultdict(list)
    for ep in episodes:
        groups[(env_map.get(ep["environment_id"]) or {}).get("revision")].append(ep)
    replays: dict[str, dict] = {}
    for revision, members in groups.items():
        src = revision_src(revision)
        t0 = time.perf_counter()
        replays.update(run_replays(members, src, min(args.workers, 4)))
        print(
            f"replayed {len(members)} with {revision or 'current'} in {time.perf_counter() - t0:.1f}s", file=sys.stderr
        )
    t0 = time.perf_counter()
    with open(args.output, "w") as out:
        for ep in episodes:
            replay = replays.get(ep["id"], {"error": "replay missing"})
            result = score(ep, replay)
            result["env_revision"] = (env_map.get(ep["environment_id"]) or {}).get("revision")
            out.write(json.dumps(result) + "\n")
    print(
        f"scored {len(episodes)} in {time.perf_counter() - t0:.1f}s; total {time.perf_counter() - started:.1f}s",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
