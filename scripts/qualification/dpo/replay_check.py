"""Replay every recorded attempt on the pinned environment and compare.

For each attempt row (attempt_rows.py) the row's calls are replayed with sftgen/abtool2 (same path as the SFT gate
`replay_score`): the later-rollout world first, then the first-rollout world, keeping the world whose tool results are
byte-equal to the recorded ones (the runner's own episodes do not record which world they ran in; validate_row does
the same). Recorded per attempt:
  results_equal    every tool result byte-equal to the replay, same count
  success_equal    replayed success (task_completed_correctly 1.0 and no failed assertion row) = the recorded success
  checks_equal     the replayed failed checks (type + params, excluded rows included) = the recorded failed checks
  replay_failed    the replayed failed checks
Run with the pinned environment's interpreter (tools/pyfix11/python) and ABTOOL2_EXPECT_ENV=28146f96...
usage: replay_check.py ROWS.jsonl OUT.jsonl [--shard K/N]
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "sftgen"))

import abtool2  # noqa: E402
from rowlib import content_text, row_calls  # noqa: E402


def check_key(a: dict) -> str:
    return json.dumps([a.get("type"), a.get("params")], sort_keys=True, ensure_ascii=False)


def replay_row(row: dict) -> dict:
    md = row["metadata"]
    calls = row_calls(row)
    recorded = [content_text(m["content"]) for m in row["messages"] if m["role"] == "tool"]
    out = world = None
    equal = False
    for fresh in (False, True):
        out = abtool2.replay(md["task"], calls, None, fresh=fresh, cap=False)
        replayed = [content_text(c["result"]) for c in out["calls"]]
        world = "first-rollout" if fresh else "later-rollout"
        if len(replayed) == len(recorded) and replayed == recorded:
            equal = True
            break
    failed = sorted(check_key(a) for a in out["assertion_results"] if not a.get("passed"))
    recorded_failed = sorted(check_key(a) for a in md["recorded_failed_checks"])
    success = out["task_completed_correctly"] == 1.0 and not failed
    return {"id": row["id"], "task": md["task"], "episode_id": md["episode_id"], "world": world,
            "results_equal": equal, "success_equal": success == (md["outcome"]["success"] == 1.0),
            "checks_equal": failed == recorded_failed, "replay_success": success,
            "recorded_success": md["outcome"]["success"], "replay_failed": failed, "recorded_failed": recorded_failed,
            "statuses": [c["status"] for c in out["calls"]]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("rows")
    parser.add_argument("out")
    parser.add_argument("--shard", default="0/1")
    args = parser.parse_args(argv)
    k, n = map(int, args.shard.split("/"))
    abtool2.check_pin(abtool2.environment_pin())
    done = 0
    with open(args.rows) as rows, open(args.out, "w") as w:
        for i, line in enumerate(rows):
            if i % n != k:
                continue
            row = json.loads(line)
            try:
                r = replay_row(row)
            except Exception as error:  # noqa: BLE001 - a replay that raises is recorded, never skipped
                r = {"id": row["id"], "task": row["metadata"]["task"], "episode_id": row["metadata"]["episode_id"],
                     "error": f"{type(error).__name__}: {error}"[:300]}
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
            w.flush()
            done += 1
    print(json.dumps({"shard": args.shard, "replayed": done}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
