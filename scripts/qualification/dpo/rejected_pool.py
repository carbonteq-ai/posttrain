"""The rejected pool: the SFT model's failures that are real, reproduced failures.

An attempt is admitted only if
  - it failed (success < 1) with at least one recorded failed task check, and the replay on the pinned environment
    reproduced exactly the same failed checks (replay_check.py, checks_equal);
  - its token chain cross-checks are exact (attempt_rows.py: no unclosed tool call, every call answered, final reply);
  - it was not an infrastructure failure, was ended by the agent itself (not the 50-turn limit), was not cut by the
    reply or context limit, is not runaway, and its chain is joined;
  - its rendered length is within 32,768 tokens and its structure is valid (validate_row `length`, `structure`).
Text gates are not applied: the rejected side is what DPO should push down. Failures with harm or tool errors are kept.
Each failed check gets its class from the SDPO A1 table (a1-w6-rows: C/U wrong content, T/N missed action, N call
never made, F forbidden action); a check without a class is labelled "unclassified".
usage: rejected_pool.py ROWS_ALL.jsonl REPLAY_ALL.jsonl FGATES_DIR A1_ROWS.jsonl.gz REJECTED.jsonl MANIFEST.json"""

import gzip
import json
import sys
from collections import Counter
from pathlib import Path

CLASS_NAMES = {"C": "wrong content", "U": "wrong content", "T": "missed action", "N": "call never made",
               "F": "forbidden action"}


def check_key(type_, params):
    return json.dumps([type_, params], sort_keys=True, ensure_ascii=False)


def reason_out(row, replay, gates):
    md, o = row["metadata"], row["metadata"]["outcome"]
    if o["success"] == 1:
        return "not a failure"
    if not md["recorded_failed_checks"]:
        return "no recorded failed check"
    if not md["exact"]:
        return "chain cross-check: " + "; ".join(md["chain_problems"])[:120]
    if replay is None or "error" in replay:
        return "replay missing or raised"
    if not replay["checks_equal"] or not replay["success_equal"]:
        return "replay did not reproduce the failed checks"
    if o["infrastructure_failure"]:
        return "infrastructure failure"
    if o["stop_condition"] != "agent_completed":
        return "not ended by the agent: " + str(o["stop_condition"])
    if o["reply_limit_cut"]:
        return "reply limit cut"
    if o["runaway"]:
        return "runaway"
    if not o["chain_joined"]:
        return "chain not joined"
    if gates is None:
        return "length/structure not checked"
    if not gates["structure"]["pass"]:
        return "structure"
    if not gates["length"]["pass"]:
        return "over 32,768 tokens"
    return None


def main(rows_all, replay_all, fgates_dir, a1_rows, out, manifest_path):
    replays = {json.loads(line)["id"]: json.loads(line) for line in open(replay_all)}
    fgates = {}
    for k in range(4):
        for r in json.load(open(Path(fgates_dir) / f"fgates-{k}.json"))["rows"]:
            fgates[r["id"]] = r["gates"]
    classes = {}
    with gzip.open(a1_rows, "rt") as s:
        for line in s:
            a = json.loads(line)
            classes[(a["episode_id"], check_key(a["type"], a["params"]))] = a["class"]
    manifest, admitted = {}, 0
    with open(out, "w") as w:
        for line in open(rows_all):
            row = json.loads(line)
            md = row["metadata"]
            if md["outcome"]["success"] == 1:
                continue
            why = reason_out(row, replays.get(row["id"]), fgates.get(row["id"]))
            labels = [CLASS_NAMES.get(classes.get((md["episode_id"], check_key(c["type"], c["params"]))), "unclassified")
                      for c in md["recorded_failed_checks"]]
            entry = {"task": md["task"], "admitted": why is None, "reason": why or "real, reproduced failure",
                     "failure_classes": dict(Counter(labels)), "harm": bool(md["outcome"]["harm"]),
                     "tool_errors": md["outcome"]["tool_errors"], "failed_checks": len(labels)}
            manifest[row["id"]] = entry
            if why is None:
                row["metadata"]["failure_classes"] = entry["failure_classes"]
                w.write(json.dumps(row, ensure_ascii=False) + "\n")
                admitted += 1
    json.dump(manifest, open(manifest_path, "w"), indent=1, ensure_ascii=False)
    out_reasons = Counter(e["reason"] for e in manifest.values() if not e["admitted"])
    print(json.dumps({"failures": len(manifest), "admitted": admitted,
                      "tasks": len({e["task"] for e in manifest.values() if e["admitted"]}),
                      "excluded": dict(out_reasons.most_common()),
                      "admitted_with_harm": sum(e["harm"] for e in manifest.values() if e["admitted"]),
                      "unclassified_checks": sum(e["failure_classes"].get("unclassified", 0)
                                                 for e in manifest.values() if e["admitted"])}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:7]))
