#!/usr/bin/env python3
"""Write one row per LFM2.5 AutomationBench training episode with its failure signals.

Runs read-only SQL against the Trackio Doris database (connection from the
TRACKIO_DORIS_* environment variables, as the Trackio server is configured), one
training run at a time so each query stays small. Columns: trace id, run, task name,
partial credit, pass, assertion counts, tool-mistake counters, truncation and error
flags, model calls, episode ending, whether the episode carries manifest turn evidence,
whether that evidence records a harm, and a setup fingerprint (a hash of the system
prompt and tool list), so episodes are only compared within one setup.
``select_dpo_contrasts.py`` reads the result.

usage: episode_signals.py OUT.tsv [--project posttrain-lab] [--model %LFM2.5%]
"""

from __future__ import annotations

import argparse
import os

import pymysql

COLUMNS = [
    "trace_id",
    "run",
    "task",
    "reward",
    "pass",
    "scored",
    "passed",
    "mistakes",
    "invalid_args",
    "unknown_id",
    "unknown_tool",
    "empty_results",
    "truncated",
    "has_error",
    "model_calls",
    "ending",
    "has_manifest",
    "harmful",
    "setup",
]

EPISODES_SQL = """
SELECT trace_id, run_name, COALESCE(get_json_string(payload, '$.task.data.task_name'), fact_task_id), fact_task_reward,
  get_json_double(payload, '$.metrics.task_completed_correctly'), get_json_double(payload, '$.metrics.assertions_scored'),
  get_json_double(payload, '$.metrics.assertions_passed'), get_json_double(payload, '$.metrics.tool_mistakes'),
  get_json_double(payload, '$.metrics.tool_invalid_arguments'), get_json_double(payload, '$.metrics.tool_unknown_id'),
  get_json_double(payload, '$.metrics.tool_unknown_tool'), get_json_double(payload, '$.metrics.tool_empty_results'),
  COALESCE(fact_is_truncated, FALSE), COALESCE(fact_has_error, FALSE), fact_model_calls,
  get_json_string(metadata, '$.episode_ending'),
  locate('"manifest_harms"', payload) > 0,
  payload REGEXP '"manifest_harms","status":"valid","value":[1-9]',
  left(md5(concat(COALESCE(get_json_string(payload, '$.nodes[0].message.content'), ''), '|',
                  COALESCE(get_json_string(payload, '$.tools'), ''))), 12)
FROM traces WHERE project_id = %s AND trace_type = 'verifiers' AND run_name = %s
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out")
    parser.add_argument("--project", default="posttrain-lab")
    parser.add_argument("--model", default="%LFM2.5%")
    args = parser.parse_args()
    connection = pymysql.connect(
        host=os.environ["TRACKIO_DORIS_HOST"],
        port=int(os.environ.get("TRACKIO_DORIS_PORT", "9030")),
        user=os.environ["TRACKIO_DORIS_USER"],
        password=os.environ["TRACKIO_DORIS_PASSWORD"],
        database=os.environ.get("TRACKIO_DORIS_DATABASE", "trackio"),
        read_timeout=900,
        connect_timeout=10,
    )
    with connection.cursor() as cursor, open(args.out, "w", encoding="utf-8") as out:
        cursor.execute(
            "SELECT DISTINCT run_name FROM traces WHERE project_id = %s AND trace_type = 'verifiers'"
            " AND run_name LIKE 'train.%%' AND fact_model LIKE %s",
            (args.project, args.model),
        )
        runs = sorted(row[0] for row in cursor.fetchall())
        out.write("\t".join(COLUMNS) + "\n")
        for run in runs:
            cursor.execute(EPISODES_SQL, (args.project, run))
            rows = cursor.fetchall()
            for row in rows:
                out.write("\t".join("" if value is None else str(value) for value in row) + "\n")
            out.flush()
            print(run, len(rows), flush=True)


if __name__ == "__main__":
    main()
