"""Provenance and exclusion checks of the SFT model's recorded attempts.

Checks (each printed with its evidence; exit 0 only if all hold):
  run pins       env-commit.txt = the pinned environment 28146f96...; the run's adapter = P9 e3
                 (/evidence/sft-v4-20261113-r2/e3); attempts 8 per task; the task file's sha256 = the shipped train-d204
  frozen sets    the run's task names equal train-d204 (names hash as in sets-phase2-frozen.json, tried as sorted and
                 file order, joined by newline or as compact JSON); no name in test120 or guard10
  attempts       1,632 attempts, 8 per task, every chain joined
  setup          within each task every attempt's system message, user message and tool list are byte-identical
usage: python3 -I provenance_check.py RUN_ROOT SETS_DIR FROZEN.json ROWS.jsonl"""
import hashlib
import json
import sys
from collections import defaultdict

PIN = "28146f967f972f4bb4c037d81d4c041dd7464a57"
P9 = "/evidence/sft-v4-20261113-r2/e3"


def name_hashes(names):
    out = set()
    for ordered in (sorted(names), list(names)):
        out.add(hashlib.sha256("\n".join(ordered).encode()).hexdigest())
        out.add(hashlib.sha256(json.dumps(ordered, separators=(",", ":")).encode()).hexdigest())
    return out


def names_of(path):
    d = json.load(open(path))
    items = d if isinstance(d, list) else d.get("tasks") or d.get("task_names")
    return [x if isinstance(x, str) else x.get("name") or x.get("task") for x in items]


def main(root, sets_dir, frozen_path, rows_path):
    ok = True

    def check(name, cond, **evidence):
        nonlocal ok
        ok &= bool(cond)
        print(("PASS " if cond else "FAIL ") + name, json.dumps(evidence, default=str)[:400])

    env = open(f"{root}/env-commit.txt").read().strip()
    inputs = json.load(open(f"{root}/run/inputs-collect.json"))
    check("environment pin", env == PIN, env=env)
    check("adapter is P9 e3", inputs["adapter"] == P9, adapter=inputs["adapter"])
    check("8 attempts per task", inputs["attempts"] == 8, attempts=inputs["attempts"])
    shipped = dict(line.split()[::-1] for line in open(f"{root}/inputs-sha256.txt") if line.strip())
    local = hashlib.sha256(open(f"{sets_dir}/train-d204.json", "rb").read()).hexdigest()
    check("task file = shipped train-d204", shipped.get("train-d204.json") == local, shipped=shipped.get("train-d204.json"),
          local=local)
    frozen = json.load(open(frozen_path))["sets"]
    train = names_of(f"{sets_dir}/train-d204.json")
    test = set(names_of(f"{sets_dir}/test120.json"))
    guard = set(names_of(f"{sets_dir}/guard10.json"))
    check("train-d204 names hash = frozen record", frozen["train-d204"]["names_sha256"] in name_hashes(train),
          tasks=len(train))
    check("test120 names hash = frozen record",
          frozen["test120"]["names_sha256"] in name_hashes(names_of(f"{sets_dir}/test120.json")), tasks=len(test))
    check("guard10 names hash = frozen record",
          frozen["guard10"]["names_sha256"] in name_hashes(names_of(f"{sets_dir}/guard10.json")), tasks=len(guard))
    rows = [json.loads(line) for line in open(rows_path)]
    tasks = defaultdict(list)
    for r in rows:
        tasks[r["metadata"]["task"]].append(r)
    check("1,632 attempts, 8 per task", len(rows) == 1632 and all(len(v) == 8 for v in tasks.values()),
          rows=len(rows), tasks=len(tasks))
    check("tasks = train-d204", set(tasks) == set(train), extra=sorted(set(tasks) - set(train)),
          missing=sorted(set(train) - set(tasks)))
    check("no test120/guard10 task", not (set(tasks) & (test | guard)), overlap=sorted(set(tasks) & (test | guard)))
    check("every chain joined", all(r["metadata"]["outcome"]["chain_joined"] for r in rows),
          not_joined=sum(not r["metadata"]["outcome"]["chain_joined"] for r in rows))
    differing = [t for t, v in tasks.items()
                 if len({json.dumps([r["messages"][:2], r["tools"]], ensure_ascii=False) for r in v}) > 1]
    check("same system, user and tools within each task", not differing, differing=differing)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:5]))
