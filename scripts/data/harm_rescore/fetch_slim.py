# ruff: noqa: E701, E702, E741, B905
"""Read-only: slim payloads (fields the harm re-scorer needs) for /tmp/hr_ids.txt -> /tmp/hr_slim.jsonl.gz."""

import gzip
import json
import os

import pymysql

c = pymysql.connect(
    host=os.environ["TRACKIO_DORIS_HOST"],
    port=int(os.environ["TRACKIO_DORIS_PORT"]),
    user=os.environ["TRACKIO_DORIS_USER"],
    password=os.environ["TRACKIO_DORIS_PASSWORD"],
    database=os.environ["TRACKIO_DORIS_DATABASE"],
    read_timeout=900,
    connect_timeout=10,
)
cur = c.cursor()
ids = [l.strip() for l in open("/tmp/hr_ids.txt") if l.strip()]
n = 0
with gzip.open("/tmp/hr_slim.jsonl.gz", "wt") as out:
    for i in range(0, len(ids), 20):
        chunk = ids[i : i + 20]
        cur.execute(
            "SELECT trace_id, run_name, fact_task_reward, payload FROM traces WHERE project_id='posttrain-lab'"
            " AND trace_type='verifiers' AND trace_id IN (" + ",".join(["%s"] * len(chunk)) + ")",
            chunk,
        )
        for tid, run, reward, payload in cur.fetchall():
            p = json.loads(payload)
            info = p.get("info") or {}
            slim = {
                "task": {"data": (p.get("task") or {}).get("data")},
                "nodes": p.get("nodes"),
                "tools": p.get("tools"),
                "metrics": p.get("metrics"),
                "is_truncated": p.get("is_truncated"),
                "ok": p.get("ok"),
                "info": {
                    "environment_id": info.get("environment_id"),
                    "automationbench": {"end_state": (info.get("automationbench") or {}).get("end_state")},
                    "posttrain_turn_rewards": info.get("posttrain_turn_rewards"),
                },
            }
            out.write(json.dumps({"trace_id": tid, "run": run, "reward": reward, "payload": slim}) + "\n")
            n += 1
print("done", n, "of", len(ids))
