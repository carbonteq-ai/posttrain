"""Time episode scoring with and without manifests on real Luna episodes (single process)."""

import asyncio
import json
import os
import statistics
import sys
import time

import verifiers.v1 as vf
from automationbench_v1.contracts.loader import supported_tasks
from automationbench_v1.manifest_assessments import ManifestAssessmentTask
from automationbench_v1.taskset import AutomationBenchData, AutomationBenchTask, AutomationBenchTaskConfig
from automationbench_v1.tools import AutomationBenchState

mix = set(json.load(open(sys.argv[1])))
idx = {
    e["task_name"]: e
    for e in json.load(
        open(
            "/home/hammad/projects/rl/docs/research/verifiers-assessment-qualification/luna-development-review-coverage.json"
        )
    )["entries"]
}
rows = []
for name in sorted(set(supported_tasks()) & mix):
    e = idx.get(name)
    if not e or not os.path.exists(e.get("source_episode_path", "")):
        continue
    raw = open(e["source_episode_path"], "rb").read()
    timing = {}
    for label, cls in (("baseline", AutomationBenchTask), ("manifest", ManifestAssessmentTask)):
        episode = vf.WireEpisode.model_validate_json(raw)
        trace = episode.traces[0]
        data = AutomationBenchData.model_validate(trace.task.data.model_dump(mode="json"))
        trace.state = AutomationBenchState(
            world=trace.info["automationbench"]["end_state"],
            initial_state=data.initial_state,
            assertions=data.assertions,
            artifacts=dict(trace.state.artifacts),
        )
        task = cls(data, AutomationBenchTaskConfig(capture_actions=True))
        start = time.perf_counter()
        try:
            asyncio.run(task.score(trace))
            timing[label] = time.perf_counter() - start
        except Exception as error:  # noqa: BLE001
            timing[label] = None
            timing[label + "_error"] = str(error)[:120]
    rows.append({"task": name, "bytes": len(raw), **timing})
    print(json.dumps(rows[-1]), flush=True)
ok = [r for r in rows if r.get("manifest") is not None and r.get("baseline") is not None]
extra = [r["manifest"] - r["baseline"] for r in ok]
print(
    "SUMMARY",
    json.dumps(
        {
            "tasks": len(rows),
            "measured": len(ok),
            "baseline_mean_s": round(statistics.mean(r["baseline"] for r in ok), 3),
            "manifest_mean_s": round(statistics.mean(r["manifest"] for r in ok), 3),
            "manifest_p90_s": round(sorted(r["manifest"] for r in ok)[int(0.9 * (len(ok) - 1))], 3),
            "manifest_max_s": round(max(r["manifest"] for r in ok), 3),
            "extra_mean_s": round(statistics.mean(extra), 3),
        }
    ),
)
