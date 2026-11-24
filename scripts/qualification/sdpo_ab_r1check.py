"""R1's check before training (plan D74, "Check before training"; `--phase precheck --precheck-mode branches`; no
training, no bar in the code).

Sample (a shipped file, `select`): failed attempts that are not infrastructure failures, up to 2 per task by
`random.Random("sdpo-ab-r1check:" + task).sample(sorted(ids), min(2, n))`, ids pooled over the file's sources, tasks
of train-d62 outside test60b, dev60, guard10, dev17 and heldout62 (episode lines of other tasks are never parsed).
Sources: R0's round 1 (`--source <run>:engine-armB/batch-1`, D74's sample) or, for a dry run, the untrained screen
attempts (screen364; never run B1, as D74 says). The job recomputes the population and the draw and refuses a file
that differs.

Branches are built exactly as training builds them (`sdpo_ab_branches.build_branches`: R1's engine, points, replay,
harm confirmation, continuation, keep rule, sequences). For each point, on the world the replay rebuilt:
- stop point: the kept calls applied to the attempt's final world;
- harm point: the world just before the harmful call, the kept calls in its place, then the rest of the attempt's
  recorded calls replayed in order (a recorded call that now fails, harness status or the engine's tool-error rule,
  is counted as no longer applying; it changes nothing it could not change).
Measured against the attempt's recorded end (the replay reproduces it, else the attempt is excluded and counted),
flip-aware (the `passed` field, whatever the excluded flag): worse = partial credit down or any check that passed
before fails after; improved = partial credit up and nothing worse; else same. A point with no kept call applies
nothing (same, `applied` false).
Readings (no bar): missing required checks (outside the must-not family) covered, per check type and point kind;
calls before the first fix (the continuation's calls replayed one at a time; a fix = a required check failed at the
point passes); share of harm points whose teacher alternative is a call (its first reply makes a call; any call);
the student's probability on the inserted tokens (`score_inserted`, GPU only; None on CPU); everything split by
whether the task has a correct solution in the pool (R0's pool sources, or a success of the task in the sampled
source).
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import sdpo_ab_branches as br
import sdpo_ab_precheck as pc
import sdpo_ab_refine_check as rc

SAMPLE_SALT = "sdpo-ab-r1check:"
SEED_SALT = "sdpo-ab-r1check"
PER_TASK = 2
SCREEN_SOURCES = ("sdpo-ab-20261104-screen364-r1:screen364",)  # the dry run's sample; B1 is never sampled (D74)
POOL_SOURCES = ("sdpo-ab-20261104-screen364-r1:screen364", "sdpo-ab-20261104-t63card-control-r1:control-t63card")
OUTCOME_KEYS = ("partial_credit", "required_passed", "required_total", "stop_condition", "harm", "assistant_turns")


# ------------------------------------------------------------------ sample

def eligibility(outcome):
    """None for a failed attempt that is not an infrastructure failure, else the reason."""
    if outcome.get("infrastructure_failure") or outcome.get("success") is None:
        return "infrastructure_failure"
    return "succeeded" if outcome["success"] != 0.0 else None


def population(outcomes, tasks):
    out, reasons = defaultdict(list), Counter()
    for outcome in outcomes:
        if outcome["task"] in tasks:
            reason = eligibility(outcome)
            reasons[reason or "eligible"] += 1
            if reason is None:
                out[outcome["task"]].append(outcome["episode_id"])
    return {t: sorted(v) for t, v in sorted(out.items())}, reasons


def choose(by_task):
    return {t: random.Random(SAMPLE_SALT + t).sample(sorted(ids), min(PER_TASK, len(ids)))
            for t, ids in sorted(by_task.items()) if ids}


def parse_source(spec):
    """"<run>:<path under run/>" -> (run, path, label = the path)."""
    run, _, path = spec.partition(":")
    if not run or not path:
        raise ValueError(f"a source is <run>:<path under run/>, got {spec!r}")
    return run, path, path.strip("/")


def solved_tasks(folders, tasks):
    out = set()
    for folder in folders:
        out |= {o["task"] for o in json.loads((Path(folder) / "outcomes.json").read_text())
                if o["task"] in tasks and o.get("success") == 1.0}
    return out


def sample_file(evidence, sources, train_path, protected_paths, pool_sources=POOL_SOURCES, env_commit=None):
    """The shipped input (refine-check layout, so `sdpo_ab_refine_check.preflight` checks it): sources with hashes,
    environment commit and world date, the selection, and per task whether a correct solution is in the pool."""
    evidence = Path(evidence)
    entries, folders = {}, {}
    for spec in sources:
        run, path, label = parse_source(spec)
        root, folder = evidence / run, evidence / run / "run" / path
        inputs = sorted((root / "run").glob("inputs-*.json"))
        world = json.loads(inputs[0].read_text()) if inputs else {}
        commit = (root / "env-commit.txt").read_text().strip() if (root / "env-commit.txt").exists() else env_commit
        if not commit:
            raise ValueError(f"{root}: no env-commit.txt; pass --env-commit")
        folders[label] = folder
        entries[label] = {"run": run, "label": label, "container_dir": f"/evidence/{run}/run/{path}",
                          "sha256": {n: rc.sha256_of(folder / n) for n in rc.SOURCE_FILES}, "env_commit": commit,
                          "code_sha": (root / "code-sha.txt").read_text().strip() if (root / "code-sha.txt").exists()
                          else None, "world_date": world.get("world_date"), "tz": world.get("tz"),
                          "seed_salt": world.get("seed_salt")}
    tasks, protected_counts, dropped = rc.training_tasks(train_path, protected_paths)
    wanted = set(tasks)
    per_source, reasons, by_id = {}, {}, {}
    for label, folder in folders.items():
        outcomes = json.loads((folder / "outcomes.json").read_text())
        per_source[label], reasons[label] = population(outcomes, wanted)
        by_id.update({o["episode_id"]: {**o, "source": label} for o in outcomes if o["task"] in wanted})
    merged, _ = rc.merge_populations(per_source)
    chosen = choose(merged)
    pool_folders = [evidence / r / "run" / p for r, p, _ in map(parse_source, pool_sources)]
    in_pool = solved_tasks(pool_folders, set(chosen))
    in_source = solved_tasks(folders.values(), set(chosen))
    attempts = sorted(({"episode_id": i, "task": t, "source": by_id[i]["source"],
                        **{k: by_id[i].get(k) for k in OUTCOME_KEYS}} for t, ids in chosen.items() for i in ids),
                      key=lambda a: (a["task"], a["episode_id"]))
    return {
        "definition": {
            "check": "plan D74, check before training (sdpo_ab_r1check)",
            "population": "failed attempts, not infrastructure failures, of train-d62 tasks outside test60b, dev60, "
                          "guard10, dev17 and heldout62, in the sources",
            "draw": f"random.Random({SAMPLE_SALT!r} + task).sample(sorted(ids), min({PER_TASK}, n)), ids pooled over "
                    "the sources",
            "solution_in_pool": "the task has a success in the pool sources (R0's --solution-pool-from) or in the "
                                "sampled sources",
            "bars": "none in the code (registered in D74: improved >= 30%, worse <= 20%, per point kind)"},
        "sources": entries, "pool_sources": list(pool_sources),
        "train_set": {"path": str(train_path), "sha256": rc.sha256_of(train_path)},
        "protected_sets": {n: {"path": str(p), "sha256": rc.sha256_of(p)} for n, p in protected_paths.items()},
        "population_by_task": merged, "selected_by_task": chosen, "attempts": attempts,
        "solution_in_pool": {t: t in in_pool or t in in_source for t in sorted(chosen)},
        "counts": {"training_tasks": len(tasks) + len(dropped), "training_tasks_protected": dropped,
                   "protected_set_sizes": protected_counts,
                   "by_reason": {k: dict(sorted(v.items())) for k, v in reasons.items()},
                   "population": sum(map(len, merged.values())), "tasks_with_population": len(merged),
                   "selected": len(attempts), "selected_per_source": dict(Counter(a["source"] for a in attempts)),
                   "tasks_with_solution": sum(t in in_pool or t in in_source for t in chosen)}}


def load(plan, keep_tasks=None):
    """(attempts, {task: records of the sampled tasks}, {id: source label}, {id: chain}); the population and the draw
    are recomputed from the recorded files and must equal the shipped ones."""
    sample = plan["sample"]
    attempts = [a for a in sample["attempts"] if keep_tasks is None or a["task"] in keep_tasks]
    tasks = {a["task"] for a in attempts}
    wanted = {a["episode_id"] for a in attempts}
    records, labels, per_source, chains = defaultdict(list), {}, {}, {}
    for label, folder in plan["folders"].items():
        per_source[label], _ = population(json.loads((Path(folder) / "outcomes.json").read_text()), tasks)
        for record in rc.iter_task_records(Path(folder) / "episodes.jsonl.gz", tasks):
            records[record["traces"][0]["task"]["data"]["name"]].append(record)
            labels[record["id"]] = label
        chains.update({c["episode_id"]: c for c in pc.iter_jsonl_gz(Path(folder) / "chains.jsonl.gz", wanted)
                       if c["episode_id"] in wanted})
    merged, _ = rc.merge_populations(per_source)
    for task in sorted(tasks):
        if merged.get(task) != sample["population_by_task"].get(task):
            raise ValueError(f"the shipped population of {task} differs from the one the recorded runs give")
        if choose({task: merged[task]})[task] != sample["selected_by_task"].get(task):
            raise ValueError(f"the shipped sample of {task} is not the registered draw")
    if wanted - set(labels):
        raise ValueError(f"{len(wanted - set(labels))} sampled attempts are not in the recorded runs")
    return attempts, records, labels, chains


def preflight(sample_path, folders=None, pinned=None, check_engine=True):
    """The refine check's preflight (hashes, one environment and world date, the engine, manifests, replica)."""
    plan = rc.preflight(sample_path, folders, pinned, check_engine)
    return {**plan, "mode": "branches"}


# ------------------------------------------------------------------ measures

def arguments_text(arguments):
    return arguments if arguments is None or isinstance(arguments, str) else json.dumps(arguments)


def required_failed(assertions):
    from sdpo_ab_metrics import negative_family

    return [i for i, a in enumerate(assertions)
            if not negative_family(str(a.get("type"))) and a.get("passed") is not True]


def verdict(before_credit, after_credit, flipped):
    if flipped or after_credit < before_credit:
        return "worse"
    return "improved" if after_credit > before_credit else "same"


def measure_point(world, built, snapshots, recorded, point, executed, kept, succeeded_fn=None):
    """Runs in the world thread. The point's world after its kept calls (harm: in place of the harmful call, the rest
    of the attempt replayed), against the recorded end; and the continuation's calls replayed one at a time from the
    point (calls before the first fix)."""
    succeeded_fn = succeeded_fn or br.call_succeeded
    final = world.grade(built)
    n = point.get("recorded_index") if point["kind"] == br.HARM else None
    state = snapshots[n] if n is not None else None
    probe = world.fork(built, state)
    base = world.grade(probe)["assertion_results"]
    failed_at_point = required_failed(base)
    first_fix, results_differ, after_kept = None, 0, None
    for index, call in enumerate(executed):
        ((content, status),) = world.execute(probe, [(call["name"], arguments_text(call["arguments"]))])
        graded = world.grade(probe)
        results_differ += content != call.get("result")
        if first_fix is None and any(graded["assertion_results"][i].get("passed") is True for i in failed_at_point):
            first_fix = index
        if index == len(kept) - 1:
            after_kept = graded
    rest = {"calls": 0, "failed_now": 0, "result_changed": 0}
    if point["kind"] == br.HARM and kept:
        after_world = world.fork(built, state)
        for call in [executed[i] for i in kept]:
            world.execute(after_world, [(call["name"], arguments_text(call["arguments"]))])
        for call in recorded[n + 1:]:
            ((content, status),) = world.execute(after_world, [(call["tool"], arguments_text(call["arguments"]))])
            rest["calls"] += 1
            rest["failed_now"] += not succeeded_fn(content, status)
            rest["result_changed"] += content != call.get("recorded_result")
        after = world.grade(after_world)
    else:
        after = after_kept if kept else final
    before = final["assertion_results"]
    flipped = rc.flips(before, after["assertion_results"])
    missing = required_failed(before)
    covered = [i for i in missing if after["assertion_results"][i].get("passed") is True]
    return {"applied": bool(kept), "before_credit": final["partial_credit"], "after_credit": after["partial_credit"],
            "after_success": bool(after.get("task_completed_correctly")), "flips": flipped,
            "verdict": verdict(final["partial_credit"], after["partial_credit"], flipped),
            "missing_required": [str(before[i].get("type")) for i in missing],
            "covered_required": [str(before[i].get("type")) for i in covered],
            "failed_required_at_point": len(failed_at_point),
            "calls_before_first_fix": first_fix, "fix_within_kept": first_fix is not None and first_fix < len(kept),
            "replayed_results_differ": results_differ, "rest_of_attempt": rest if point["kind"] == br.HARM else None}


def share(part, whole):
    return part / whole if whole else None


def summarize_kind(rows, kind):
    mine = [r for r in rows if r.get("kind") == kind and "dropped" not in r and "check" in r]
    checks = [r["check"] for r in mine]
    verdicts = Counter(c["verdict"] for c in checks)
    by_type = defaultdict(lambda: {"missing": 0, "covered": 0})
    for c in checks:
        for t in c["missing_required"]:
            by_type[t]["missing"] += 1
        for t in c["covered_required"]:
            by_type[t]["covered"] += 1
    missing, covered = sum(v["missing"] for v in by_type.values()), sum(v["covered"] for v in by_type.values())
    fixes = [c["calls_before_first_fix"] for c in checks if c["calls_before_first_fix"] is not None]
    inserted = [r["inserted"]["p_mean"] for r in mine if r.get("inserted")]
    out = {"points": len(mine), "attempts": len({r["episode_id"] for r in mine}),
           "applied": sum(c["applied"] for c in checks), "verdicts": dict(verdicts),
           "improved_share": share(verdicts["improved"], len(checks)), "worse_share": share(verdicts["worse"],
                                                                                            len(checks)),
           "same_share": share(verdicts["same"], len(checks)),
           "after_success": sum(c["after_success"] for c in checks),
           "missing_required": missing, "covered_required": covered, "covered_share": share(covered, missing),
           "missing_required_by_type": {t: {**v, "share": share(v["covered"], v["missing"])}
                                        for t, v in sorted(by_type.items())},
           "calls_before_first_fix": {"points_with_a_fix": len(fixes), "points_without": len(checks) - len(fixes),
                                      "mean": statistics.mean(fixes) if fixes else None,
                                      "median": statistics.median(fixes) if fixes else None,
                                      "distribution": dict(sorted(Counter(fixes).items())),
                                      "fix_within_kept": sum(c["fix_within_kept"] for c in checks)},
           "inserted_p_mean": {"points": len(inserted), "mean": statistics.mean(inserted) if inserted else None},
           "replayed_results_differ": sum(c["replayed_results_differ"] for c in checks)}
    if kind == br.HARM:
        out["alternative_is_call"] = {"first_reply": share(sum(r["alternative_first_reply_call"] for r in mine),
                                                           len(mine)),
                                      "any_call": share(sum(r["executed_calls"] > 0 for r in mine), len(mine))}
        out["rest_of_attempt"] = {k: sum((c["rest_of_attempt"] or {}).get(k, 0) for c in checks)
                                  for k in ("calls", "failed_now", "result_changed")}
        per_attempt = defaultdict(list)
        for c, r in zip(checks, mine):
            per_attempt[r["episode_id"]].append(c["verdict"])
        attempt_verdicts = Counter("worse" if "worse" in v else "improved" if "improved" in v else "same"
                                   for v in per_attempt.values())
        out["per_attempt"] = {"verdicts": dict(attempt_verdicts),
                              "improved_share": share(attempt_verdicts["improved"], len(per_attempt)),
                              "worse_share": share(attempt_verdicts["worse"], len(per_attempt))}
    return out


def summarize(rows, counts):
    def block(selected):
        return {kind: summarize_kind(selected, kind) for kind in (br.STOP, br.HARM)}

    excluded = {r["episode_id"] for r in rows if str(r.get("dropped", "")).startswith(("replay_mismatch",
                                                                                       "replay_error"))}
    return {"all": block(rows),
            "with_solution": block([r for r in rows if r.get("with_solution")]),
            "without_solution": block([r for r in rows if not r.get("with_solution")]),
            "excluded_attempts_replay": len(excluded),
            "check_errors": dict(Counter(r["kind"] for r in rows if "check_error" in r)),
            "drops": dict(Counter(str(r["dropped"]).split(":")[0] for r in rows if "dropped" in r)),
            "counts": dict(counts)}


# ------------------------------------------------------------------ the job

async def run_branch_check(plan, *, tokenizer, generate, parse_for, out, suspend=None, score_inserted=None,
                           concurrency=32, engine=None, world=None, log=None, engine_version=None, keep_tasks=None,
                           seed_for=None, loaded=None):
    """generate(ids, max_tokens, seed) -> (ids, finish): the teacher (round 1: the untrained model) at temperature
    1.0, top-p 1.0; parse_for(chain) -> parse(text) -> calls; score_inserted(branch) -> {"p_mean", ...} or None (the
    student's probability on the inserted tokens; GPU); suspend(): awaited after all sampling, before scoring."""
    import asyncio
    import concurrent.futures
    import time

    from sdpo_ab_runner import request_seed, task_of, write

    if world is None:
        import sdpo_ab_world as world
    log = log or (lambda **fields: print(json.dumps(fields, default=str), flush=True))
    started = time.monotonic()
    attempts, records, labels, chains = loaded or load(plan, keep_tasks)
    engine = engine or br.make_branch_engine(True, ["call_fix"], rc.R0_OPTIONS)
    by_id = {r["id"]: r for rs in records.values() for r in rs}
    targets = [by_id[a["episode_id"]] for a in attempts]
    with_solution = plan["sample"].get("solution_in_pool") or {}
    loop = asyncio.get_running_loop()
    worker = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="r1check-world")
    scored = []

    async def run_in_world(function, *arguments):
        return await loop.run_in_executor(worker, function, *arguments)

    def siblings_of(record):
        key = rc.batch_key(record, labels[record["id"]])
        return [r for r in records[task_of(record)] if rc.batch_key(r, labels[r["id"]]) == key]

    async def on_point(*, record, point, built, snapshots, recorded, result, row, sequences, **_):
        try:
            row["check"] = await run_in_world(measure_point, world, built, snapshots, recorded, point,
                                              result["executed"], result["kept"])
        except Exception as error:  # noqa: BLE001 - recorded and counted; the point is left out of the readings
            row["check_error"] = f"{type(error).__name__}: {error}"
            log(stage="r1check_error", episode=record["id"], kind=point["kind"], error=row["check_error"])
            return
        row["alternative_first_reply_call"] = bool(result["turns"] and result["turns"][0]["calls"])
        if sequences:
            scored.append((row, sequences))
        log(stage="r1check_point", episode=record["id"], kind=point["kind"], verdict=row["check"]["verdict"])

    try:
        result = await br.build_branches(
            targets, {a["episode_id"]: chains.get(a["episode_id"]) for a in attempts}, tokenizer=tokenizer,
            engine=engine, generate=generate, parse_for=parse_for, world=world, day=plan["day"],
            run_in_world=run_in_world, siblings_of=siblings_of,
            seed_for=seed_for or (lambda key: request_seed(SEED_SALT, key)), log=log, concurrency=concurrency,
            on_point=on_point)
    finally:
        worker.shutdown(wait=True)
    log(stage="r1check_built", elapsed=round(time.monotonic() - started), counts=dict(result["counts"]))
    if suspend is not None:
        await suspend()
    for row, sequences in scored:
        row["inserted"] = score_inserted(sequences) if score_inserted else None
    rows = result["rows"]
    for row in rows:
        row["with_solution"] = bool(with_solution.get(row["task"]))
        row["source"] = labels.get(row["episode_id"])
    rows.sort(key=lambda r: (r["episode_id"], str(r.get("kind")), r.get("turn") or 0, r.get("call") or 0))
    report = {"definition": plan["sample"]["definition"], "sources": plan["sample"]["sources"],
              "sample_counts": plan["sample"]["counts"],
              "plan": {"sample_path": plan.get("sample_path"), "folders": {k: str(v) for k, v in plan["folders"].items()},
                       "world_date": str(plan["day"]), "env_commit": plan.get("env_commit"),
                       "engine_version": engine_version, "seed_salt": SEED_SALT,
                       "branch": {"max_calls": br.BRANCH_MAX_CALLS, "keep": br.BRANCH_KEEP, "arm": br.BRANCH_ARM,
                                  "sampling": br.BRANCH_SAMPLING},
                       "kept_tasks": sorted(keep_tasks) if keep_tasks else None,
                       "inserted_scored": score_inserted is not None},
              "attempts": len(attempts), "summary": summarize(rows, result["counts"]),
              "branch_summary": br.summarize(rows), "engine_stats": dict(getattr(engine, "stats", {})),
              "rows": rows, "elapsed_s": time.monotonic() - started}
    write(Path(out) / "branch-check.json", report)
    print(summary_text(report), flush=True)
    return report


def summary_text(report):
    s = report["summary"]
    lines = [f"r1 check: {report['attempts']} attempts; replay-excluded attempts {s['excluded_attempts_replay']}; "
             f"drops {s['drops']}"]
    for kind in (br.STOP, br.HARM):
        k = s["all"][kind]
        lines.append(f"{kind}: {k['points']} points ({k['attempts']} attempts, {k['applied']} applied): {k['verdicts']}"
                     f" improved {pc.fmt(k['improved_share'])} worse {pc.fmt(k['worse_share'])}; covered "
                     f"{k['covered_required']}/{k['missing_required']}")
    return "\n".join(lines)


# ------------------------------------------------------------------ laptop: select and the CPU dry run

def dry_run(sample_path, evidence, tokenizer_path, out, keep_tasks=None):
    """The whole data path on CPU with the real engine, tokenizer and replayed worlds; the teacher is mocked as in the
    branch dry run (`sdpo_ab_branches.dry_run`): it re-issues, one call per turn after a line of text, the calls of a
    recorded success of the task the student had not made, then a final reply."""
    import asyncio

    from transformers import AutoTokenizer

    from sdpo_ab_runner import task_of
    from sdpo_ab_world import recorded_calls

    sample = json.loads(Path(sample_path).read_text())
    folders = {label: Path(evidence) / s["run"] / "run" / s["container_dir"].split("/run/", 1)[1]
               for label, s in sample["sources"].items()}
    plan = preflight(sample_path, folders)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    loaded = load(plan, keep_tasks)
    attempts, records = loaded[0], loaded[1]
    by_id = {r["id"]: r for rs in records.values() for r in rs}
    successes = {t: [r for r in rs if br.succeeded(r)] for t, rs in records.items()}
    mocks = {}
    for a in attempts:
        record = by_id[a["episode_id"]]
        done = {(c["tool"], arguments_text(c["arguments"])) for c in recorded_calls(record)[0]}
        calls = [(c["tool"], arguments_text(c["arguments"])) for s in successes.get(task_of(record), [])[:1]
                 for c in recorded_calls(s)[0]]
        mocks[record["id"]] = [c for c in calls if c not in done][:br.BRANCH_MAX_CALLS]

    async def generate(ids, budget, key):
        episode, *_rest, index = key.split("|")
        calls, index = mocks[episode], int(index)
        text = (f"I still need to do this step ({calls[index][0]}).\n\n" + br.render_calls_text([calls[index]])
                if index < len(calls) else "All required work is now done.")
        return tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id], "stop"

    Path(out).mkdir(parents=True, exist_ok=True)
    return asyncio.run(run_branch_check(plan, tokenizer=tokenizer, generate=generate,
                                        parse_for=lambda chain: br.parse_calls_text, out=out, concurrency=4,
                                        seed_for=lambda key: key, loaded=loaded, log=lambda **fields: None))


def main(argv=None):
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="R1 check before training (D74): sample file and CPU dry run")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("select", help="write the sample file from recorded runs")
    p.add_argument("--evidence", default=rc.DEFAULT_EVIDENCE)
    p.add_argument("--source", action="append", help="<run>:<path under run/>, e.g. "
                   "sdpo-ab-20261105-r0-train-r1:engine-armB/batch-1 (R0's round 1); default: the screen dry run")
    p.add_argument("--env-commit", help="the environment commit when the run has no env-commit.txt")
    p.add_argument("--train", default=str(repo / "docs/research/sdpo-automationbench/train-d62.json"))
    p.add_argument("--out", required=True)
    d = sub.add_parser("dry-run", help="the whole check on CPU with a mocked teacher")
    d.add_argument("--sample", required=True)
    d.add_argument("--evidence", default=rc.DEFAULT_EVIDENCE)
    d.add_argument("--tokenizer", required=True)
    d.add_argument("--tasks", help="comma-separated subset")
    d.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "select":
        content = sample_file(args.evidence, args.source or list(SCREEN_SOURCES), args.train,
                              rc.default_protected(repo), env_commit=args.env_commit)
        Path(args.out).write_text(json.dumps(content, indent=1) + "\n")
        print(json.dumps(content["counts"], indent=1))
        return 0
    report = dry_run(args.sample, args.evidence, args.tokenizer, args.out,
                     set(args.tasks.split(",")) if args.tasks else None)
    print(json.dumps({k: report[k] for k in ("attempts", "summary", "elapsed_s")}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
