"""Replay each Luna-top20 draft manifest on its recorded Luna episode (one process, sequential)."""

import asyncio
import hashlib
import json
import sys
from pathlib import Path

import verifiers.v1 as vf
from automationbench_v1 import manifest_assessments
from automationbench_v1.contracts.engine import binding_reason
from automationbench_v1.contracts.loader import load_contract
from automationbench_v1.manifest_assessments import ManifestAssessmentTask
from automationbench_v1.manifest_guard_assessments import GUARD_OUTPUT
from automationbench_v1.manifest_obligation_assessments import OBLIGATION_OUTPUT
from automationbench_v1.manifest_step_credit import _latest_payloads
from automationbench_v1.public_state import public_initial_state
from automationbench_v1.taskset import (
    AutomationBenchConfig,
    AutomationBenchData,
    AutomationBenchTaskConfig,
    AutomationBenchTaskset,
)
from automationbench_v1.tools import AutomationBenchState

AUDIT = json.load(open(sys.argv[1]))
OUT = sys.argv[2]
COV = {
    e["task_name"]: e
    for e in json.load(
        open(
            "/home/hammad/projects/rl/docs/research/verifiers-assessment-qualification/luna-development-review-coverage.json"
        )
    )["entries"]
}
DRAFTS = Path("/nonexistent-drafts")
config = AutomationBenchConfig(
    domains=["simple", "sales", "marketing", "operations", "support", "finance", "hr"],
    task={"toolset": "limited_zapier", "search_top_k": 20, "turn_budget": 16},
)
TASKS = {t.data.task_name: t for t in AutomationBenchTaskset(config).load()}
from automationbench_v1.contracts import load_task_contract as real_loader  # noqa: E402

results = []
for entry in AUDIT["tasks"]:
    name = entry["task"]
    row = {"task": name, "domain": entry["domain"], "audit_status": entry["whole_task_status"]}
    try:
        dp = DRAFTS / name / "draft.json"
        if not dp.exists():
            cat = json.loads(Path("src/automationbench_v1/contracts/catalog.json").read_text())["tasks"]
            dp = Path("src/automationbench_v1/contracts") / cat[name]
        raw = dp.read_bytes()
        row["source"] = str(dp)
        row["draft_unchanged"] = hashlib.sha256(raw).hexdigest() == entry.get("draft_sha256")
        # Parse the draft exactly as the installed loader would.
        contract = load_contract(raw)
        task = TASKS[name]
        ev = {
            "task_evidence": {
                "prompt": task.data.model_dump(mode="json")["prompt"],
                "initial": public_initial_state(task.data.initial_state),
            }
        }
        row["binding"] = binding_reason(ev, contract)
        manifest_assessments.load_task_contract = lambda _n, c=contract: c
        e = COV.get(name)
        row["luna_class"] = e and e.get("official_outcome_class")
        ep = vf.WireEpisode.model_validate_json(Path(e["source_episode_path"]).read_bytes())
        tr = ep.traces[0]
        data = AutomationBenchData.model_validate(tr.task.data.model_dump(mode="json"))
        tr.state = AutomationBenchState(
            world=tr.info["automationbench"]["end_state"],
            initial_state=data.initial_state,
            assertions=data.assertions,
            artifacts=dict(tr.state.artifacts),
        )
        t = ManifestAssessmentTask(data, AutomationBenchTaskConfig(capture_actions=True))
        asyncio.run(t.score(tr))
        row["assessment_errors"] = len(tr.assessment_errors or ())
        row["credit_errors"] = len(tr.credit_errors or ())
        obl = [p for p in _latest_payloads(tr, OBLIGATION_OUTPUT) if p.get("kind") == "finding"]
        grd = [p for p in _latest_payloads(tr, GUARD_OUTPUT) if p.get("kind") == "finding"]
        req = [p for p in obl if p.get("required") is True]
        row["required"] = len(req)
        row["required_valid"] = sum(p.get("status") == "valid" for p in req)
        row["required_pass"] = sum(p.get("status") == "valid" and p.get("value") == 1 for p in req)
        row["guards"] = len(grd)
        row["harms"] = sum(p.get("status") == "valid" and p.get("value") == 1 for p in grd)
        row["abstained"] = sum(p.get("status") != "valid" for p in obl + grd)
    except Exception as err:  # noqa: BLE001
        row["error"] = f"{type(err).__name__}: {str(err)[:200]}"
    finally:
        manifest_assessments.load_task_contract = real_loader
    results.append(row)
    print(json.dumps(row), flush=True)
json.dump(results, open(OUT, "w"), indent=1)
