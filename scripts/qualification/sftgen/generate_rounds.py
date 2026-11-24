"""Plan v4 P7: teacher data generation in rounds, one admitted trajectory per task.

Round r runs one attempt for every task not yet admitted that has had fewer than --rounds real attempts (Ali, 12 Nov:
an attempt lost to infrastructure -- a provider rate limit, 5xx, transport failure or refused routing, or no episode
at all -- does not count, and the task gets another; at most --max-rounds rounds in all), in chunks of --chunk tasks (a crash or a stop
loses at most the chunk in flight), each chunk through teacher_runner.py -> episode_rows.py -> validate_row.py
--thinking. A task's first admitted row is kept; later rounds only retry tasks without one. Resumable: a chunk with
admitted.jsonl and report.json is reused; a chunk folder left without them is moved aside and rerun.
Stops before the next chunk when the proxy's STOP file (spend cap) exists.
--revalidate: no generation at all; re-runs validate_row on every stored chunk's rows.jsonl (e.g. after a gate fix)
and rebuilds data.jsonl and the coverage from the rounds exactly as they were run (each round's stored tasks.json).

Outputs in --out: rounds/r<r>/c<i>/ (runner output, rows.jsonl, report.json, admitted.jsonl, logs), data.jsonl (the
admitted rows, one per task, sorted by task), coverage.json and coverage.md (every task: admitted in which round, or
never admitted with each attempt's reason), render/ (render_check.py --thinking on data.jsonl).

usage: python3 generate_rounds.py --tasks-json FROZEN.json --set train-d204 --base-url http://127.0.0.1:PORT/v1 \
           --proxy-dir PROXY_DIR --out OUT [--rounds 3] [--chunk 34] [--concurrency 4]
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
from settings import CAPPED, ENV_PYTHON as ENV_PY, FRAMEWORK_PYTHON as SFT_PY, REPO  # noqa: E402
MODEL = "qwen3.5-397b-a17b"


def log(message: str) -> None:
    print(time.strftime("%m-%d %H:%M"), message, flush=True)


def run(cmd: list, *, log_path: Path, mem: str, env: dict | None = None, cwd: Path | None = None) -> int:
    with log_path.open("w") as stream:
        return subprocess.run([*CAPPED, *map(str, cmd)], stdout=stream, stderr=subprocess.STDOUT, cwd=cwd,
                              env={**os.environ, "CAPPED_MEM": mem, **(env or {})}).returncode


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def chunk_done(folder: Path) -> bool:
    return (folder / "report.json").exists() and (folder / "admitted.jsonl").exists()


def validate_chunk(folder: Path) -> None:
    # exit 1 only means some row failed a gate; a crash leaves no report
    for name in ("report.json", "admitted.jsonl"):
        (folder / name).unlink(missing_ok=True)
    run([ENV_PY, HERE / "validate_row.py", folder / "rows.jsonl", "--report", folder / "report.json", "--thinking",
         "--admitted-out", folder / "admitted.jsonl"], log_path=folder / "validate.log", mem="6G")
    if not chunk_done(folder):
        raise RuntimeError(f"validate_row left no report for {folder} (see validate.log)")


def run_chunk(folder: Path, names: list[str], args) -> None:
    run_done = (folder / "run" / "summary.json").exists() and (folder / "run" / "episodes.jsonl.gz").exists()
    if run_done and json.loads((folder / "run" / "summary.json").read_text()).get("tasks") == names:
        # the teacher run finished (paid for); only rows/validation are missing: rebuild them, never regenerate
        log(f"{folder.name}: reusing the completed teacher run; rebuilding rows")
        for name in ("rows.jsonl", "rows.log", "report.json", "admitted.jsonl"):
            (folder / name).unlink(missing_ok=True)
        build_rows(folder, args)
        validate_chunk(folder)
        return
    if folder.exists():
        aside = folder.with_name(f"{folder.name}.incomplete-{int(time.time())}")
        folder.rename(aside)
        log(f"{folder.name}: incomplete earlier run moved to {aside.name}")
    folder.mkdir(parents=True)
    code = run([ENV_PY, HERE / "teacher_runner.py", "--tasks", *names, "--attempts", 1, "--concurrency",
                args.concurrency, "--base-url", args.base_url, "--output", folder / "run",
                "--label", f"p7-{folder.parent.name}-{folder.name}"],
               log_path=folder / "runner.log", mem="10G", env={"POSTTRAIN_VERIFIERS_PREINSTALLED": "1"}, cwd=folder)
    if code != 0:
        raise RuntimeError(f"teacher_runner failed for {folder} (exit {code}, see runner.log)")
    build_rows(folder, args)
    validate_chunk(folder)


def build_rows(folder: Path, args) -> None:
    code = run([ENV_PY, HERE / "episode_rows.py", folder / "run", args.proxy_dir, folder / "rows.jsonl",
                "--model", MODEL, "--prefix", args.prefix], log_path=folder / "rows.log", mem="6G")
    if code != 0:
        raise RuntimeError(f"episode_rows failed for {folder} (exit {code}, see rows.log)")


INFRA = re.compile(r"upstream (409|429|5\d\d)\b|transport", re.I)


def infra_failures(folder: Path) -> dict:
    """task -> message for episodes that ended in a provider/infrastructure error (stop_condition error and every
    recorded error is a provider error matching INFRA): no teacher behaviour was judged, so the attempt is not counted."""
    out = {}
    path = folder / "run" / "episodes.jsonl.gz"
    if not path.exists():
        return out
    with gzip.open(path, "rt") as stream:
        for line in stream:
            record = json.loads(line)
            trace = (record.get("traces") or [{}])[0]
            errors = (trace.get("errors") or []) + (record.get("errors") or [])
            task = ((record.get("task") or {}).get("data") or {}).get("name")
            if trace.get("stop_condition") == "error" and errors and all(
                    e.get("type") == "ProviderError" and INFRA.search(e.get("message") or "") for e in errors):
                out[task] = errors[0]["message"][:160]
    return out


def chunk_outcomes(folder: Path) -> tuple[dict, dict]:
    """(task -> admitted row, task -> reason) for one chunk."""
    admitted = {r["metadata"]["task"]: r for r in read_jsonl(folder / "admitted.jsonl")}
    reasons = {}
    for row in json.loads((folder / "report.json").read_text())["rows"]:
        if not row["pass"]:
            failed = [k for k, g in row["gates"].items() if not g["pass"] and not g.get("advisory")]
            score = row["gates"].get("replay_score", {}).get("task_completed_correctly")
            reasons[row["task"]] = f"gates failed: {', '.join(failed)} (replay score {score})"
    skipped = json.loads((folder / "rows.log").read_text().strip().splitlines()[-1]).get("skipped") or []
    for s in skipped:
        reasons[s["task"]] = f"no row: {s['reason']}"
    return admitted, reasons


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tasks-json", type=Path, required=True)
    parser.add_argument("--set", default="train-d204")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--proxy-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3, help="real teacher attempts per task (infra losses excluded)")
    parser.add_argument("--max-rounds", type=int, default=6, help="rounds in all, including infrastructure top-ups")
    parser.add_argument("--chunk", type=int, default=34)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--prefix", default="sft-phase2a-v4")
    parser.add_argument("--revalidate", action="store_true", help="re-run the gates on stored rows; no generation")
    args = parser.parse_args(argv)
    tasks = sorted(json.loads(args.tasks_json.read_text())["sets"][args.set]["per_task"])
    args.out.mkdir(parents=True, exist_ok=True)
    admitted: dict[str, dict] = {}
    attempts: dict[str, list[dict]] = {t: [] for t in tasks}
    stopped = None
    real = lambda t: sum(1 for a in attempts[t] if not a.get("infra"))  # noqa: E731
    last_round = 0
    for r in range(1, args.max_rounds + 1):
        rdir = args.out / "rounds" / f"r{r}"
        plan = rdir / "tasks.json"
        if args.revalidate:  # the round as it was run
            if not plan.exists():
                break
            todo = json.loads(plan.read_text())
        else:
            todo = [t for t in tasks if t not in admitted and real(t) < args.rounds]
            if not todo or stopped:
                break
            rdir.mkdir(parents=True, exist_ok=True)
            if plan.exists() and json.loads(plan.read_text()) != todo:
                raise RuntimeError(f"{plan} lists other tasks than this resume would run; refusing to mix")
            plan.write_text(json.dumps(todo))
        last_round = r
        chunks = [todo[i:i + args.chunk] for i in range(0, len(todo), args.chunk)]
        log(f"round {r}: {len(todo)} tasks in {len(chunks)} chunks")
        for i, names in enumerate(chunks):
            folder = rdir / f"c{i}"
            if args.revalidate:
                if not (folder / "rows.jsonl").exists():
                    stopped = f"round {r} chunk {i} has no stored rows (generation stopped there)"
                    break
                log(f"round {r} chunk {i}: re-validating stored rows")
                validate_chunk(folder)
            elif not chunk_done(folder):
                if (args.proxy_dir / "STOP").exists():
                    stopped = f"spend cap reached before round {r} chunk {i}"
                    log(stopped)
                    break
                log(f"round {r} chunk {i}: {len(names)} tasks")
                run_chunk(folder, names, args)
            got, reasons = chunk_outcomes(folder)
            infra = infra_failures(folder)
            for t in names:
                if t in got:
                    attempts[t].append({"round": r, "admitted": True, "id": got[t]["id"]})
                    admitted.setdefault(t, got[t])
                elif t in infra or t not in reasons:
                    attempts[t].append({"round": r, "admitted": False, "infra": True,
                                        "reason": f"infrastructure, not counted: {infra.get(t, 'no episode recorded')}"})
                else:
                    attempts[t].append({"round": r, "admitted": False, "reason": reasons[t]})
            ledger = (args.proxy_dir / "ledger.jsonl").read_text().splitlines()
            spent = json.loads(ledger[-1]).get("spent_after") if ledger else 0.0
            log(f"round {r} chunk {i}: {len(got)}/{len(names)} admitted; total admitted {len(admitted)}/{len(tasks)};"
                f" spent ${spent}")
    rows = [admitted[t] for t in sorted(admitted)]
    with (args.out / "data.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")  # = rowlib.serialize_row (framework materialization format)
    never = {t: a for t, a in attempts.items() if t not in admitted}
    coverage = {"tasks": len(tasks), "admitted": len(rows), "never_admitted": len(never), "stopped": stopped,
                "admitted_by_round": {r: sum(1 for t in admitted if attempts[t][-1]["round"] == r)
                                      for r in range(1, last_round + 1)},
                "infra_attempts": sum(1 for a_list in attempts.values() for a in a_list if a.get("infra")),
                "attempts": attempts}
    (args.out / "coverage.json").write_text(json.dumps(coverage, indent=1))
    lines = [f"# P7 coverage: {len(rows)} of {len(tasks)} training tasks admitted", "",
             f"Admitted by round: {coverage['admitted_by_round']}. Stopped early: {stopped or 'no'}.", "",
             "## Never admitted", "", "| Task | Attempts (round: reason) |", "|---|---|"]
    lines += [f"| {t} | " + "<br>".join(f"r{a['round']}: {a['reason']}" for a in a_list) + " |"
              for t, a_list in sorted(never.items())]
    (args.out / "coverage.md").write_text("\n".join(lines) + "\n")
    log(f"data.jsonl: {len(rows)} rows; never admitted {len(never)}")
    if rows:
        if (args.out / "render").exists():
            shutil.rmtree(args.out / "render")
        code = run([SFT_PY, HERE / "render_check.py", args.out / "data.jsonl", args.out / "render", "--thinking"],
                   log_path=args.out / "render.log", mem="8G", cwd=REPO)
        log(f"render_check exit {code} (see render.log)")
        if code != 0:
            return 3
    return 0 if not stopped else 2


if __name__ == "__main__":
    raise SystemExit(main())
