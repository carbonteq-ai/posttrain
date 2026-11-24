"""Refinement check before R1 (no training; `--phase precheck --precheck-mode refine`): two engine fix-hint kinds of
engine refine-r1 7c1382f (vendored), measured on the untrained model's failed screen attempts of the training tasks.

Sample (`select`, on the laptop; shipped as the sample file and recomputed by the job): the recorded screen attempts of
the train-d62 tasks in the runs screen364 (`sdpo-ab-20261104-screen364-r1`) and B1 (`sdpo-ab-20261104-t63card-control-
r1`), never a task of test60b, dev60, guard10, dev17 or heldout62 (episode lines of other tasks are never parsed);
failed attempts that are not infrastructure failures and left required work undone (outcomes required_passed <
required_total); up to 2 per task by `random.Random("sdpo-ab-r1refine:" + task).sample(sorted(ids), min(2, n))`.

Teacher: the untrained model (zero-initialized LoRA, as every train phase starts), contexts built as training builds
them (EngineContext, arm B, plain, placement both, expected values shown, no budgets, not_done_source benchmark,
siblings = the attempt's own batch of 8 of its run, solution pool = every recorded attempt of the task in both runs):
- r0     = R0's context: fix hints ["call_fix"];
- r0_cac = r0 + content_at_call: fix hints ["call_fix", "content_at_call"];
- r1     = R0 + the final note (R1's context): fix hints ["call_fix"], final_note on;
- r1_vl  = r1 + value_location: fix hints ["call_fix", "value_location"], final_note on.
Each added context must equal its base exactly once its added lines are removed (body lines and note text);
otherwise the attempt is excluded from that measure and counted.

(a) content_at_call, per line: on the call-body tokens of the student's own call the line is pinned to (the tokens
`restrict_to_calls` trains), the teacher-vs-student divergence (the loss's per-token top-20 + tail measure, alpha 1,
without the importance weight) with the r0 context and with the r0 context + that one line (the other content_at_call
lines of the attempt removed); and, at the first token of the value of the call's field that lacks the needed text, the
teacher's probability of the student's token and whether the teacher's top-1 differs from it, under both contexts.

(b) value_location, per attempt where it adds lines and the attempt ends with a final reply after a call: the teacher
continues from the stop point (`sdpo_ab_precheck.continue_attempt`: at most 10 calls, the rollouts' limits and parser,
in the attempt's own world rebuilt by replaying its recorded calls and verified to score its recorded result) once
with r1 and once with r1_vl (the same per-turn seeds), each in a fresh rebuild. Flip-aware: worse = any check that
passed before the continuation fails after it (the `passed` field, whatever the excluded flag); improved = partial
credit above the recorded one and nothing worse; else same. Missing required checks fixed: non-must-not checks failed
before and passed after.

Every per-line / per-attempt row and the summaries are written; nothing is judged against a bar here.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import sdpo_ab_precheck as pc

SAMPLE_SALT = "sdpo-ab-r1refine:"
SEED_SALT = "sdpo-ab-r1refine"
PER_TASK = 2
SOURCES = (("sdpo-ab-20261104-screen364-r1", "screen364"), ("sdpo-ab-20261104-t63card-control-r1", "control-t63card"))
CONTAINER_DIR = "/evidence/{run}/run/{label}"
SOURCE_FILES = pc.SOURCE_FILES
PROTECTED_SETS = ("test60b", "dev60", "guard10", "dev17", "heldout62")
BATCH_SIZE = pc.BATCH_SIZE
R0_OPTIONS = {"line_budget": None, "char_budget": None, "not_done_source": "benchmark"}
ENGINE_SETTINGS = {"r0": (("call_fix",), False), "r0_cac": (("call_fix", "content_at_call"), False),
                   "r1": (("call_fix",), True), "r1_vl": (("call_fix", "value_location"), True)}
ADDED = {"r0_cac": ("r0", "content_at_call"), "r1_vl": ("r1", "value_location")}
NOTE_PREFIX = "Note on your next action: "
TOPK = 20
HEAD = re.compile(r'^\{"id": "([0-9a-f]+)", "env": \{[^{}]*\}, "task": \{"type": "[^"]*", "data": \{"idx": \d+, '
                  r'"name": "([^"]+)"')


# ------------------------------------------------------------------ sample

def task_names(path):
    return list(json.loads(Path(path).read_text())["task_names"])


def iter_task_records(path, tasks):
    """Stream an episodes file: a line is parsed only when its head names a task in `tasks` (the task is read from
    the line head by a regex first, so lines of other tasks, the protected sets' included, are never parsed)."""
    with gzip.open(path, "rt") as stream:
        for line in stream:
            match = HEAD.match(line[:2000])
            if not match:
                raise ValueError(f"{path}: episode line head not recognised: {line[:120]!r}")
            if match.group(2) in tasks:
                record = json.loads(line)
                if record["id"] != match.group(1):
                    raise ValueError(f"{path}: line head id differs from the record's")
                yield record


def eligibility(outcome):
    """None when an attempt is in the population (not an infrastructure failure, failed, required work undone),
    else the first criterion it fails."""
    if outcome.get("infrastructure_failure") or outcome.get("success") is None:
        return "infrastructure_failure"
    if outcome["success"] != 0.0:
        return "succeeded"
    passed, total = outcome.get("required_passed"), outcome.get("required_total")
    if passed is None or total is None or not passed < total:
        return "no_required_work_undone"
    return None


def population(outcomes, tasks):
    """({task: sorted ids}, reason counts) over the outcomes of the given tasks."""
    out, reasons = defaultdict(list), Counter()
    for outcome in outcomes:
        if outcome["task"] not in tasks:
            continue
        reason = eligibility(outcome)
        reasons[reason or "eligible"] += 1
        if reason is None:
            out[outcome["task"]].append(outcome["episode_id"])
    return {t: sorted(v) for t, v in sorted(out.items())}, reasons


def choose(by_task):
    """The registered draw: up to 2 per task, seeded by the task name."""
    return {t: random.Random(SAMPLE_SALT + t).sample(sorted(ids), min(PER_TASK, len(ids)))
            for t, ids in sorted(by_task.items()) if ids}


def merge_populations(per_source):
    """{task: sorted ids over every source}; an id in two sources is refused."""
    merged, seen = defaultdict(list), {}
    for label, by_task in per_source.items():
        for task, ids in by_task.items():
            for i in ids:
                if i in seen:
                    raise ValueError(f"episode {i} is in {seen[i]} and {label}")
                seen[i] = label
            merged[task] += ids
    return {t: sorted(v) for t, v in sorted(merged.items())}, seen


def training_tasks(train_path, protected_paths):
    """(training tasks outside every protected set, {protected set: task count}, {training task: [sets]} dropped)."""
    train = task_names(train_path)
    protected = {name: set(task_names(path)) for name, path in protected_paths.items()}
    dropped = {t: sorted(n for n, s in protected.items() if t in s) for t in train}
    dropped = {t: v for t, v in dropped.items() if v}
    return [t for t in train if t not in dropped], {n: len(s) for n, s in protected.items()}, dropped


def select_sample(folders, train_path, protected_paths, fixes_path=None):
    """The selection part of the sample file. `folders`: {label: recorded collection folder}."""
    tasks, protected_counts, dropped = training_tasks(train_path, protected_paths)
    wanted = set(tasks)
    per_source, reasons, by_id = {}, {}, {}
    for label, folder in folders.items():
        outcomes = json.loads((Path(folder) / "outcomes.json").read_text())
        per_source[label], reasons[label] = population(outcomes, wanted)
        by_id.update({o["episode_id"]: o for o in outcomes if o["task"] in wanted})
    merged, source_of = merge_populations(per_source)
    chosen = choose(merged)
    picked = {i for ids in chosen.values() for i in ids}
    fixed = set()
    if fixes_path:
        fixed = {json.loads(line)["task"] for line in Path(fixes_path).read_text().splitlines() if line.strip()}
    attempts = []
    for label, folder in folders.items():
        mine = {i for i in picked if source_of[i] == label}
        if not mine:
            continue
        records = {r["id"]: r for r in iter_task_records(Path(folder) / "episodes.jsonl.gz", set(chosen))
                   if r["id"] in mine}
        chains = {c["episode_id"]: c for c in pc.iter_jsonl_gz(Path(folder) / "chains.jsonl.gz", mine)
                  if c["episode_id"] in mine}
        for i in sorted(mine):
            record, outcome = records[i], by_id[i]
            messages = pc.messages_of(record)
            attempts.append({
                "episode_id": i, "task": outcome["task"], "source": label,
                "repetition_index": (record.get("run") or {}).get("repetition_index"),
                "partial_credit": record["traces"][0]["rewards"]["partial_credit"]["score"],
                "required_passed": outcome["required_passed"], "required_total": outcome["required_total"],
                "stop_condition": outcome.get("stop_condition"),
                "assistant_turns": sum(m.get("role") == "assistant" for m in messages),
                "tool_calls": sum(len(m.get("tool_calls") or []) for m in messages if m.get("role") == "assistant"),
                "stop_point": pc.final_reply_shape(messages) is None,
                "final_reply_shape": pc.final_reply_shape(messages) or "final_reply_after_a_call",
                "task_fixed_after_recording": outcome["task"] in fixed,
                **pc.chain_summary(chains.get(i), record)})
    attempts.sort(key=lambda a: (a["task"], a["episode_id"]))
    return {"population_by_task": merged, "selected_by_task": chosen, "attempts": attempts,
            "counts": {"training_tasks": len(tasks) + len(dropped), "training_tasks_protected": dropped,
                       "protected_set_sizes": protected_counts,
                       "by_reason": {k: dict(sorted(v.items())) for k, v in reasons.items()},
                       "population": sum(map(len, merged.values())), "tasks_with_population": len(merged),
                       "tasks_without_population": sorted(set(tasks) - set(merged)),
                       "selected": len(picked), "with_stop_point": sum(a["stop_point"] for a in attempts),
                       "task_fixed_after_recording": sum(a["task_fixed_after_recording"] for a in attempts),
                       "selected_per_source": dict(Counter(a["source"] for a in attempts))}}


def sha256_of(path):
    return pc.sha256_file(path)


def sample_file(evidence, train_path, protected_paths, fixes_path=None):
    """The shipped input: definition, sources (hashes, environment commit, world date), the training and protected
    set files' hashes, and the selection."""
    evidence = Path(evidence)
    folders = {label: evidence / run / "run" / label for run, label in SOURCES}
    sources = {}
    for run, label in SOURCES:
        root = evidence / run
        inputs = json.loads((root / "run" / "inputs-collect.json").read_text())
        sources[label] = {"run": run, "label": label, "container_dir": CONTAINER_DIR.format(run=run, label=label),
                          "sha256": {n: sha256_of(folders[label] / n) for n in SOURCE_FILES},
                          "env_commit": (root / "env-commit.txt").read_text().strip(),
                          "code_sha": (root / "code-sha.txt").read_text().strip(),
                          "world_date": inputs.get("world_date"), "tz": inputs.get("tz"),
                          "seed_salt": inputs.get("seed_salt"), "attempts": inputs.get("attempts")}
    selection = select_sample(folders, train_path, protected_paths, fixes_path)
    return {
        "definition": {
            "population": "recorded screen attempts (screen364, B1 t63card control) of the train-d62 tasks outside "
                          "test60b, dev60, guard10, dev17 and heldout62 that are not infrastructure failures, failed, "
                          "and left required work undone (outcomes required_passed < required_total)",
            "draw": f"random.Random({SAMPLE_SALT!r} + task).sample(sorted(ids), min({PER_TASK}, n)), ids pooled "
                    "over both runs",
            "engine_settings": {k: {"fix_hints": list(v[0]), "final_note": v[1], **R0_OPTIONS}
                                for k, v in ENGINE_SETTINGS.items()},
            "teacher_arm": pc.TEACHER_ARM, "siblings": "the attempt's own batch of 8 of its run",
            "solution_pool": "every recorded attempt of the sampled tasks in both runs (successes kept)",
            "a": "content_at_call: per line, call-body tokens of the pinned call, divergence (top-20 + tail) r0 vs "
                 "r0 + the line; teacher probability and top-1 at the first token of the field lacking the text",
            "b": f"value_location: continuation from the stop point (max {pc.CHECK2_MAX_CALLS} calls, temperature "
                 "1.0, top-p 1.0) with r1 vs r1_vl; flip-aware worse = any check passed before and failed after",
            "bars": "none in the code (set by the owner before the run)"},
        "sources": sources,
        "train_set": {"path": str(train_path), "sha256": sha256_of(train_path)},
        "protected_sets": {n: {"path": str(p), "sha256": sha256_of(p)} for n, p in protected_paths.items()},
        "task_fixes": {"path": str(fixes_path), "sha256": sha256_of(fixes_path)} if fixes_path else None,
        **selection}


# ------------------------------------------------------------------ contexts

def view(context):
    return {"first_message": context["first_message"], "notes": dict(context["notes"]),
            "final_note": context.get("final_note")}


def strip_lines(context, lines):
    """The context's first message, notes and final note with the given hint lines removed: each body line equal to
    one, and " " + line inside a note (a note left with nothing but its prefix is dropped)."""
    drop = set(lines)
    first = "\n".join(part for part in context["first_message"].split("\n") if part not in drop)
    notes = {}
    for turn, text in context["notes"].items():
        for line in lines:
            text = text.replace(" " + line, "", 1)
        if text.strip() != NOTE_PREFIX.strip():
            notes[turn] = text
    return {"first_message": first, "notes": notes, "final_note": context.get("final_note")}


def added_lines(context, kind):
    return [m for m in ((context.get("fix_hints") or {}).get("lines") or []) if m.get("kind") == kind]


def isolate(context, keep, lines):
    """`context` with every line of `lines` removed except `keep` (a full context dict)."""
    return {**context, **strip_lines(context, [m["line"] for m in lines if m["line"] != keep["line"]])}


def teacher_engines(engine_class=None):
    if engine_class is None:
        from sdpo_ab_engine import EngineContext as engine_class
    return {name: engine_class(expected_values=True, pool=True, fix_hints=list(kinds),
                               options={**R0_OPTIONS, **({"final_note": True} if final else {})})
            for name, (kinds, final) in ENGINE_SETTINGS.items()}


def batch_key(record, label):
    index = (record.get("run") or {}).get("repetition_index")
    if not isinstance(index, int):
        raise ValueError(f"episode {record.get('id')} has no repetition index")
    return label, index // BATCH_SIZE


def field_token(tokenizer, action_ids, call_index, name):
    """(action-relative index of the first token of the value of field `name` in call `call_index` of one reply, how
    it was found) or (None, reason). The field is a parameter of that name (case and punctuation ignored), else a JSON
    key of that name inside a parameter's value."""
    key = lambda text: re.sub(r"[^a-z0-9]", "", str(text).lower())
    text = tokenizer.decode(action_ids, skip_special_tokens=False)
    ends = [len(tokenizer.decode(action_ids[:i + 1], skip_special_tokens=False)) for i in range(len(action_ids))]
    starts = [m.start() for m in re.finditer(r"<function=", text)]
    if call_index >= len(starts):
        return None, "call_not_in_reply"
    stop = text.find("</function>", starts[call_index])
    block_end = stop if stop >= 0 else len(text)
    params = list(re.finditer(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", text[starts[call_index]:block_end],
                              re.S))
    at, how = None, None
    for match in params:
        if key(match.group(1)) == key(name):
            at, how = starts[call_index] + match.start(2), "parameter"
            break
    if at is None:
        pattern = re.compile(r'"' + re.escape(str(name)) + r'"\s*:\s*"?', re.I)
        for match in params:
            found = pattern.search(match.group(2))
            if found:
                at, how = starts[call_index] + match.start(2) + found.end(), "json_key"
                break
    if at is None:
        return None, "field_not_in_call"
    index = next((i for i, e in enumerate(ends) if e > at), None)
    return (index, how) if index is not None else (None, "field_at_reply_end")


def field_name(meta):
    return meta.get("column") if meta.get("family") in ("sheets", "salesforce") and meta.get("column") else \
        meta.get("field")


def prepare_attempt(tokenizer, attempt, record, chain, siblings, engines, saved):
    """Contexts of one attempt and what each measure needs: (item, {measure: exclusion reason})."""
    from sdpo_ab_runner import context_record, engine_teacher_prefix, student_sequence, teacher_sequence

    built, reasons = {}, {}
    for name, engine in engines.items():
        try:
            context = engine(record, siblings, pc.TEACHER_ARM, "both")
        except ValueError as error:  # the engine's truth check
            reasons[name] = f"untrue: {error}"
            continue
        except Exception as error:  # noqa: BLE001 - an engine crash excludes the attempt from that measure
            reasons[name] = f"builder_error: {type(error).__name__}: {error}"
            continue
        if context is None:
            reasons[name] = "no_context"
            continue
        built[name] = context
        saved.append({**context_record(context, pc.TEACHER_ARM), "setting": name,
                      "final_note": context.get("final_note"), "fix_hints_lines": [
                          {k: m.get(k) for k in ("kind", "item", "line", "turn", "index", "at_turn")}
                          for m in (context.get("fix_hints") or {}).get("lines") or []]})
    out = {"attempt": attempt, "record": record, "chain": chain, "contexts": built, "cac": [], "vl": [],
           "reasons": {}, "solution_episode": (built.get("r0") or {}).get("solution_episode")}
    problem = pc.chain_problem(chain, record)
    if problem not in (None, "final_action_has_calls"):
        out["reasons"] = {"a": problem, "b": problem}
        return out
    for added, (base, kind) in ADDED.items():
        measure = "a" if kind == "content_at_call" else "b"
        missing = [n for n in (base, added) if n not in built]
        if missing:
            out["reasons"][measure] = "; ".join(f"{n}: {reasons[n]}" for n in missing)
            continue
        lines = added_lines(built[added], kind)
        if strip_lines(built[added], [m["line"] for m in lines]) != view(built[base]):
            out["reasons"][measure] = f"{added} differs from {base} beyond its {kind} lines"
            continue
        out["cac" if measure == "a" else "vl"] = lines
        if not lines:
            out["reasons"][measure] = f"no {kind} line"
    student = student_sequence(chain)
    out["student"] = student
    if out["cac"]:
        groups = pc.call_body_groups(tokenizer, student)
        r0 = teacher_sequence(tokenizer, chain, built["r0"])
        if r0["targets"] != student["targets"]:
            raise ValueError(f"teacher and student targets differ: {record['id']}")
        out["teacher_r0"] = r0
        actions = [s for s in chain["segments"] if s["kind"] == "action"]
        out["lines"] = []
        for meta in out["cac"]:
            line = {"meta": meta, "tokens": groups.get((meta["turn"], meta["index"]), [])}
            if not line["tokens"]:
                line["reason"] = "call_tokens_not_found"
            teacher = teacher_sequence(tokenizer, chain, isolate(built["r0_cac"], meta, out["cac"]))
            if teacher["targets"] != student["targets"]:
                raise ValueError(f"teacher and student targets differ: {record['id']}")
            line["teacher"] = teacher
            name = field_name(meta)
            if meta["turn"] - 1 < len(actions) and name:
                index, how = field_token(tokenizer, actions[meta["turn"] - 1]["ids"], meta["index"], name)
            else:
                index, how = None, "no_field_name" if not name else "turn_not_in_chain"
            if index is not None:
                start = student["turns"].index(meta["turn"])
                line["field_token"] = {"student_index": start + index, "how": how, "field": name,
                                       "token": student["targets"][start + index]}
                if start + index not in line["tokens"]:
                    line["field_token"]["outside_call_tokens"] = True
            else:
                line["field_token"] = {"student_index": None, "how": how, "field": name}
            out["lines"].append(line)
    if out["vl"]:
        if problem == "final_action_has_calls":
            out["reasons"]["b"] = "no_stop_point: the attempt does not end with a final reply after a call"
        else:
            position, turn = pc.final_position(chain)
            finals = {c: (built[c].get("final_note") or {}) for c in ("r1", "r1_vl")}
            if any(f.get("turn") not in (None, turn) for f in finals.values()):
                out["reasons"]["b"] = f"final_note_turn is not the final reply's turn {turn}"
            else:
                out["prefixes"] = {c: engine_teacher_prefix(tokenizer, chain, position, built[c]) for c in ("r1", "r1_vl")}
                if any(len(p) >= pc.ROLLOUT_LIMITS["context"] for p in out["prefixes"].values()):
                    out["reasons"]["b"] = "teacher_prefix_over_context_limit"
                actions = [s for s in chain["segments"] if s["kind"] == "action"]
                out.update(position=position, turn=turn, output_before=sum(len(a["ids"]) for a in actions[:-1]),
                           final_note_present=finals["r1"].get("turn") is not None)
    return out


def load_sources(plan, keep_tasks=None):
    """Every record of the sampled tasks (both runs, file order), the population recomputed and checked against the
    shipped one, and the sampled attempts' chains."""
    sample = plan["sample"]
    attempts = [a for a in sample["attempts"] if keep_tasks is None or a["task"] in keep_tasks]
    tasks = {a["task"] for a in attempts}
    records, labels, per_source, chains = defaultdict(list), {}, {}, {}
    wanted = {a["episode_id"] for a in attempts}
    for label, folder in plan["folders"].items():
        outcomes = json.loads((Path(folder) / "outcomes.json").read_text())
        per_source[label], _ = population(outcomes, tasks)
        for record in iter_task_records(Path(folder) / "episodes.jsonl.gz", tasks):
            records[record["traces"][0]["task"]["data"]["name"]].append(record)
            labels[record["id"]] = label
        mine = {a["episode_id"] for a in attempts if a["source"] == label}
        chains.update({c["episode_id"]: c for c in pc.iter_jsonl_gz(Path(folder) / "chains.jsonl.gz", mine)
                       if c["episode_id"] in mine})
    merged, _ = merge_populations(per_source)
    for task in sorted(tasks):
        if merged.get(task) != sample["population_by_task"].get(task):
            raise ValueError(f"the shipped population of {task} differs from the one the recorded runs give")
        if choose({task: merged[task]})[task] != sample["selected_by_task"].get(task):
            raise ValueError(f"the shipped sample of {task} is not the registered draw")
    missing = wanted - {r["id"] for rs in records.values() for r in rs}
    if missing:
        raise ValueError(f"{len(missing)} sampled attempts are not in the recorded runs")
    return attempts, records, labels, chains


def prepare_all(plan, tokenizer, engines=None, keep_tasks=None, log=None):
    """Load the sources and build every context of the sample: (items, exclusions, saved contexts, engines)."""
    log = log or (lambda **fields: None)
    attempts, records, labels, chains = load_sources(plan, keep_tasks)
    engines = engines or teacher_engines()
    for engine in engines.values():
        if hasattr(engine, "add_to_pool"):
            for task_records in records.values():
                engine.add_to_pool(task_records)
    by_id = {r["id"]: r for rs in records.values() for r in rs}
    items, exclusions, saved = [], [], []
    for attempt in attempts:
        record = by_id[attempt["episode_id"]]
        key = batch_key(record, labels[record["id"]])
        siblings = [r for r in records[attempt["task"]] if batch_key(r, labels[r["id"]]) == key]
        item = prepare_attempt(tokenizer, attempt, record, chains.get(attempt["episode_id"]), siblings, engines, saved)
        for measure, reason in sorted(item["reasons"].items()):
            exclusions.append({"episode_id": attempt["episode_id"], "task": attempt["task"], "measure": measure,
                               "reason": reason})
        items.append(item)
        log(stage="refine_context", episode=attempt["episode_id"], cac=len(item["cac"]), vl=len(item["vl"]))
    return items, exclusions, saved, engines


def context_counts(items, exclusions):
    """The dry run's and the job's context counts."""
    lengths = []
    for item in items:
        lengths += [len(item["teacher_r0"]["ids"])] if "teacher_r0" in item else []
        lengths += [len(line["teacher"]["ids"]) for line in item.get("lines") or []]
        lengths += [len(p) for p in (item.get("prefixes") or {}).values()]
    a = [i for i in items if i.get("lines")]
    b = [i for i in items if i.get("prefixes") and "b" not in i["reasons"]]
    return {"attempts": len(items), "tasks": len({i["attempt"]["task"] for i in items}),
            "content_at_call_lines": sum(len(i["cac"]) for i in items), "content_at_call_attempts": len(a),
            "content_at_call_lines_with_call_tokens": sum(bool(l["tokens"]) for i in a for l in i["lines"]),
            "content_at_call_field_token_located": sum(l["field_token"]["student_index"] is not None
                                                       for i in a for l in i["lines"]),
            "value_location_lines": sum(len(i["vl"]) for i in items),
            "value_location_attempts": sum(bool(i["vl"]) for i in items), "value_location_continuable": len(b),
            "content_at_call_by_type": dict(Counter(l["meta"].get("type") for i in a for l in i["lines"])),
            "exclusions": dict(Counter(f"{e['measure']}: {e['reason'].split(':')[0]}" for e in exclusions)),
            "prompt_tokens_max": max(lengths) if lengths else None,
            "engine_settings": {k: list(v[0]) + (["final_note"] if v[1] else []) for k, v in ENGINE_SETTINGS.items()}}


# ------------------------------------------------------------------ measures

def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def line_row(item, line, scores):
    """One content_at_call line's row from the scores under r0 and r0 + the line ({condition: [per-token dict]})."""
    meta, attempt = line["meta"], item["attempt"]
    row = {"episode_id": attempt["episode_id"], "task": attempt["task"], "source": attempt["source"],
           "task_fixed_after_recording": attempt.get("task_fixed_after_recording"),
           "with_solution": bool(item.get("solution_episode")), "line": meta["line"], "item": meta["item"],
           "check_type": meta.get("type"), "family": meta.get("family"), "field": field_name(meta),
           "missing": meta.get("missing"), "turn": meta["turn"], "call": meta["index"],
           "call_tokens": len(line["tokens"]), "field_token": {k: v for k, v in line["field_token"].items()},
           "reason": line.get("reason")}
    if not line["tokens"] or scores is None:
        return row
    field = line["field_token"]["student_index"]
    for condition in ("r0", "line"):
        per = scores[condition]
        divs = [p["div"] for p in per[:len(line["tokens"])]]
        row[condition] = {"div_mean": mean(divs), "div_sum": sum(divs),
                          "teacher_logp_mean": mean(p["logp_teacher"] for p in per[:len(line["tokens"])]),
                          "top1_differs": sum(p["top1_teacher"] != t for p, t in zip(per, line["targets"]))}
        if field is not None:
            at = per[len(line["tokens"])] if field not in line["tokens"] else per[line["tokens"].index(field)]
            row[condition]["field"] = {"p_teacher": pc.math_exp(at["logp_teacher"]), "logp_teacher": at["logp_teacher"],
                                       "top1_teacher": at["top1_teacher"],
                                       "top1_differs": at["top1_teacher"] != line["field_token"]["token"]}
    row["delta_div_mean"] = row["line"]["div_mean"] - row["r0"]["div_mean"]
    if field is not None:
        at = scores["r0"][len(line["tokens"])] if field not in line["tokens"] else \
            scores["r0"][line["tokens"].index(field)]
        row["field_token"]["p_student"] = pc.math_exp(at["logp_student"])
    return row


def summarize_lines(rows):
    scored = [r for r in rows if "r0" in r]
    located = [r for r in scored if "field" in r["r0"]]
    field = lambda c, k: [r[c]["field"][k] for r in located]
    flips = Counter((r["r0"]["field"]["top1_differs"], r["line"]["field"]["top1_differs"]) for r in located)
    out = {"lines": len(rows), "scored": len(scored), "attempts": len({r["episode_id"] for r in scored}),
           "tasks": len({r["task"] for r in scored}),
           "div_mean": {c: mean(r[c]["div_mean"] for r in scored) for c in ("r0", "line")},
           "div_pooled_per_token": {c: (sum(r[c]["div_sum"] for r in scored) / sum(r["call_tokens"] for r in scored)
                                        if scored else None) for c in ("r0", "line")},
           "delta_div_mean": mean(r["delta_div_mean"] for r in scored),
           "lines_div_up": sum(r["delta_div_mean"] > 0 for r in scored),
           "lines_div_down": sum(r["delta_div_mean"] < 0 for r in scored),
           "field_token": {"located": len(located),
                           "p_teacher_mean": {c: mean(field(c, "p_teacher")) for c in ("r0", "line")},
                           "p_teacher_down": sum(r["line"]["field"]["p_teacher"] < r["r0"]["field"]["p_teacher"]
                                                 for r in located),
                           "top1_differs": {c: sum(field(c, "top1_differs")) for c in ("r0", "line")},
                           "agree_to_differ": flips[(False, True)], "differ_to_agree": flips[(True, False)]},
           "by_family": dict(Counter(r["family"] for r in scored)),
           "unscored": dict(Counter(r.get("reason") or "no_scores" for r in rows if "r0" not in r))}
    return out


def flips(before, after):
    """Checks that passed before and do not pass after (the `passed` field, whatever the excluded flag)."""
    if len(before) != len(after):
        raise ValueError("assertion lists differ in length")
    out = []
    for index, (a, b) in enumerate(zip(before, after)):
        if a.get("type") != b.get("type") or a.get("params") != b.get("params"):
            raise ValueError(f"assertion {index} differs in type or parameters")
        if a.get("passed") is True and b.get("passed") is not True:
            out.append({"index": index, "type": b.get("type"), "before_excluded": a.get("excluded"),
                        "after_excluded": b.get("excluded")})
    return out


def required_fixed(before, after):
    """(failed required checks before, of those passing after): checks outside the must-not family."""
    from sdpo_ab_metrics import negative_family

    failed = [i for i, a in enumerate(before) if not negative_family(str(a.get("type"))) and a.get("passed") is not True]
    return len(failed), sum(after[i].get("passed") is True for i in failed)


def verdict(before_credit, after_credit, flipped):
    if flipped:
        return "worse"
    return "improved" if after_credit > before_credit else "same"


def summarize_continuations(rows):
    out = {"attempts": len(rows)}
    for condition in ("r1", "r1_vl"):
        done = [r[condition] for r in rows if r.get(condition)]
        counts = Counter(c["verdict"] for c in done)
        failed, fixed = sum(c["required_failed_before"] for c in done), sum(c["required_fixed"] for c in done)
        out[condition] = {"continued": len(done), "counts": dict(counts),
                          "shares": {k: (counts[k] / len(done) if done else None) for k in ("improved", "same", "worse")},
                          "required_failed_before": failed, "required_fixed": fixed,
                          "required_fixed_share": fixed / failed if failed else None,
                          "mean_required_fixed_share": mean(c["required_fixed"] / c["required_failed_before"]
                                                            for c in done if c["required_failed_before"]),
                          "credit_down_without_flip": sum(c["after"]["partial_credit"] < c["before_credit"]
                                                          and not c["flips"] for c in done),
                          "mean_credit_after": mean(c["after"]["partial_credit"] for c in done)}
    paired = [r for r in rows if r.get("r1") and r.get("r1_vl")]
    credit = lambda r, c: r[c]["after"]["partial_credit"]
    out["paired"] = {"attempts": len(paired), "vl_credit_higher": sum(credit(r, "r1_vl") > credit(r, "r1") for r in paired),
                     "vl_credit_lower": sum(credit(r, "r1_vl") < credit(r, "r1") for r in paired),
                     "vl_fixed_more": sum(r["r1_vl"]["required_fixed"] > r["r1"]["required_fixed"] for r in paired),
                     "vl_fixed_fewer": sum(r["r1_vl"]["required_fixed"] < r["r1"]["required_fixed"] for r in paired),
                     "verdict_pairs": dict(Counter(f"{r['r1']['verdict']}->{r['r1_vl']['verdict']}" for r in paired))}
    return out


# ------------------------------------------------------------------ the job

def preflight(sample_path, folders=None, pinned=None, check_engine=True):
    """Before the model is loaded: the sample, the recorded runs' files (sha256 as shipped), one environment commit
    and world date for both runs (and SDPO_AB_ENV_COMMIT, when set, equal to it), the engine's support for both kinds,
    the task manifests, and the replica's imports. `folders`: {label: folder} (default: the container paths)."""
    import datetime as dt
    import os

    sample = json.loads(Path(sample_path).read_text())
    folders = {label: Path((folders or {}).get(label) or source["container_dir"])
               for label, source in sample["sources"].items()}
    for label, source in sample["sources"].items():
        for name, digest in source["sha256"].items():
            got = sha256_of(folders[label] / name)
            if got != digest:
                raise ValueError(f"{folders[label] / name}: sha256 {got} is not the shipped {digest}")
    commits = {s["env_commit"] for s in sample["sources"].values()}
    days = {s["world_date"] for s in sample["sources"].values()}
    if len(commits) != 1 or len(days) != 1:
        raise ValueError(f"the recorded runs differ in environment {commits} or world date {days}")
    (commit,), (day,) = commits, days
    env_commit = os.environ.get("SDPO_AB_ENV_COMMIT")
    if env_commit and env_commit != commit:
        raise ValueError(f"environment {env_commit} is not the recorded runs' {commit} (launch with SDPO_AB_ENV_COMMIT "
                         "set to it)")
    if pinned and pinned != day:
        raise ValueError(f"world clock pinned to {pinned}, the recorded runs used {day}")
    from sdpo_ab_execution import pin_world_clock

    pin_world_clock(day)
    if check_engine:
        from sdpo_ab_engine import ENGINE_OPTIONS, engine_supports, engine_version, load_engine, missing_manifests

        if not engine_supports(ENGINE_OPTIONS):
            raise ValueError(f"the context engine ({engine_version()}) does not take {ENGINE_OPTIONS}")
        load_engine()
        from abce.fix import OPT_IN_KINDS, selected_kinds

        selected_kinds(sorted({k for kinds, _ in ENGINE_SETTINGS.values() for k in kinds}))
        if "content_at_call" not in OPT_IN_KINDS:
            raise ValueError("the vendored engine has no opt-in content_at_call kind (re-vendor refine-r1)")
        missing = missing_manifests(sorted(sample["selected_by_task"]))
        if missing:
            raise ValueError(f"task manifests do not load for {missing}")
    import sdpo_ab_world

    sdpo_ab_world.check_imports()
    return {"sample": sample, "folders": folders, "day": dt.date.fromisoformat(day), "sample_path": str(sample_path),
            "env_commit": env_commit or "unchecked (SDPO_AB_ENV_COMMIT not set)", "mode": "refine"}


async def run_refine_check(plan, *, tokenizer, generate, parse_for, distill, suspend, out, concurrency=32,
                           engines=None, keep_tasks=None, world=None, log=None, engine_version=None):
    """Both measures on the shipped sample. Model-facing parts are injected (CPU tests use stubs):

    generate(ids, max_tokens, seed) -> (ids, finish_reason): the untrained model at temperature 1.0, top-p 1.0;
    parse_for(chain) -> parse(text) -> calls; distill(student_ids, student_positions, teacher_ids, teacher_positions,
    targets) -> [{"div", "logp_teacher", "top1_teacher", "logp_student", "top1_student"}] per position (the student
    adapter on the student's own sequence, the teacher adapter on the teacher's; div = the loss's top-20 + tail term),
    or None (out of memory); suspend(): awaited after all sampling, before scoring."""
    import asyncio
    import concurrent.futures
    import time

    from sdpo_ab_runner import render_delta, request_seed, write, write_jsonl_gz

    if world is None:
        import sdpo_ab_world as world
    log = log or (lambda **fields: print(json.dumps(fields, default=str), flush=True))
    started = time.monotonic()
    out = Path(out)
    items, exclusions, saved, engines = prepare_all(plan, tokenizer, engines, keep_tasks, log)
    write_jsonl_gz(out / "refine-contexts.jsonl.gz", saved)
    counts = context_counts(items, exclusions)
    log(stage="refine_contexts", **{k: v for k, v in counts.items() if not isinstance(v, dict)})

    def exclude(item, measure, reason):
        exclusions.append({"episode_id": item["attempt"]["episode_id"], "task": item["attempt"]["task"],
                           "measure": measure, "reason": reason})

    # (b) continuations: r1 and r1_vl, each in its own rebuilt world, the same per-turn seeds
    day = plan["day"]
    loop = asyncio.get_running_loop()
    worker = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="refine-world")
    gate = asyncio.Semaphore(max(1, concurrency))
    continuable = [i for i in items if i.get("prefixes") and "b" not in i["reasons"]]
    rows_b = []

    async def in_world(function, *arguments):
        return await loop.run_in_executor(worker, function, *arguments)

    async def carry_on(item):
        eid, attempt = item["attempt"]["episode_id"], item["attempt"]
        parse = parse_for(item["chain"])
        row = {"episode_id": eid, "task": attempt["task"], "source": attempt["source"],
               "task_fixed_after_recording": attempt.get("task_fixed_after_recording"),
               "with_solution": bool(item.get("solution_episode")), "final_note_present": item["final_note_present"],
               "value_location_lines": [m["line"] for m in item["vl"]]}
        for condition in ("r1", "r1_vl"):
            try:
                built, report = await in_world(world.rebuild, item["record"], day)
            except Exception as error:  # noqa: BLE001 - recorded, counted and excluded
                exclude(item, "b", f"replay_error: {type(error).__name__}: {error}")
                return
            if report["variant"] == "unmatched":
                exclude(item, "b", "replay_variant_unmatched: the recorded initial world is neither setup variant")
                return
            if not report["verified"]:
                exclude(item, "b", f"replay_mismatch: replayed {report['replayed']['partial_credit']}, recorded "
                                   f"{report['recorded']['partial_credit']}")
                return

            async def turn_sample(ids, budget, index):
                async with gate:
                    return await generate(ids, budget, request_seed(SEED_SALT, f"{eid}|continue|{index}"))

            result = await pc.continue_attempt(
                item["prefixes"][condition], tokenizer=tokenizer, generate=turn_sample, parse=parse,
                execute=lambda pairs, built=built: in_world(world.execute, built, pairs),
                render=lambda messages: render_delta(tokenizer, messages), turns_before=item["turn"] - 1,
                output_before=item["output_before"])
            after = await in_world(world.grade, built)
            before = report["assertion_results"]
            flipped = flips(before, after["assertion_results"])
            failed, fixed = required_fixed(before, after["assertion_results"])
            credit = report["recorded"]["partial_credit"]
            row[condition] = {
                "teacher_tokens": len(item["prefixes"][condition]), "stop": result["stop"],
                "executed_calls": result["executed_calls"], "turns": result["turns"], "before_credit": credit,
                "after": {k: after[k] for k in ("partial_credit", "task_completed_correctly", "assertions_passed",
                                                "assertions_scored", "assertions_excluded")},
                "flips": flipped, "required_failed_before": failed, "required_fixed": fixed,
                "assertion_changes": pc.assertion_changes(before, after["assertion_results"]),
                "verdict": verdict(credit, after["partial_credit"], flipped),
                "replay": {k: report[k] for k in ("variant", "calls_replayed", "assertions_equal")}}
            log(stage="refine_continued", episode=eid, condition=condition, before=credit,
                after=after["partial_credit"], verdict=row[condition]["verdict"])
        rows_b.append(row)

    await asyncio.gather(*(carry_on(i) for i in continuable))
    worker.shutdown(wait=True)
    log(stage="refine_b_done", continued=len(rows_b), elapsed=round(time.monotonic() - started))

    # (a) after all sampling: teacher and student forward passes only
    await suspend()
    rows_a = []
    for item in items:
        for line in item.get("lines") or []:
            line["targets"] = [item["student"]["targets"][i] for i in line["tokens"]]
            scores = None
            if line["tokens"]:
                extra = [] if line["field_token"]["student_index"] in (None, *line["tokens"]) else \
                    [line["field_token"]["student_index"]]
                indices = line["tokens"] + extra
                targets = [item["student"]["targets"][i] for i in indices]
                student = item["student"]
                scores = {}
                for condition, teacher in (("r0", item["teacher_r0"]), ("line", line["teacher"])):
                    got = distill(student["ids"], [student["positions"][i] for i in indices], teacher["ids"],
                                  [teacher["positions"][i] for i in indices], targets)
                    if got is None:
                        exclude(item, "a", f"{condition}: out_of_memory")
                        scores = None
                        break
                    scores[condition] = got
            rows_a.append(line_row(item, line, scores))
    log(stage="refine_a_done", lines=len(rows_a), elapsed=round(time.monotonic() - started))

    result = {
        "definition": plan["sample"]["definition"], "sources": plan["sample"]["sources"],
        "sample_counts": plan["sample"]["counts"],
        "plan": {"sample_path": plan.get("sample_path"), "folders": {k: str(v) for k, v in plan["folders"].items()},
                 "world_date": str(day), "env_commit": plan.get("env_commit"), "engine_version": engine_version,
                 "seed_salt": SEED_SALT, "rollout_limits": pc.ROLLOUT_LIMITS, "topk": TOPK,
                 "kept_tasks": sorted(keep_tasks) if keep_tasks else None},
        "context_counts": counts, "exclusions": exclusions,
        "exclusion_counts": dict(Counter(f"{e['measure']}: {e['reason'].split(':')[0]}" for e in exclusions)),
        "engine_stats": {c: dict(getattr(e, "stats", {})) for c, e in engines.items()},
        "content_at_call": {"summary": summarize_lines(rows_a),
                            "by_solution": {str(s): summarize_lines([r for r in rows_a if r["with_solution"] == s])
                                            for s in (True, False)},
                            "rows": rows_a},
        "value_location": {"summary": summarize_continuations(rows_b),
                           "by_solution": {str(s): summarize_continuations([r for r in rows_b if r["with_solution"] == s])
                                           for s in (True, False)},
                           "rows": sorted(rows_b, key=lambda r: r["episode_id"])},
        "elapsed_s": time.monotonic() - started}
    write(out / "refine-check.json", result)
    print(summary_text(result), flush=True)
    return result


def summary_text(result):
    a, b = result["content_at_call"]["summary"], result["value_location"]["summary"]
    f = a["field_token"]
    lines = [f"refine check: {result['context_counts']['attempts']} attempts; exclusions {result['exclusion_counts']}",
             f"(a) content_at_call: {a['scored']} of {a['lines']} lines scored ({a['attempts']} attempts); divergence "
             f"per line r0 {pc.fmt(a['div_mean']['r0'])} -> +line {pc.fmt(a['div_mean']['line'])} (up {a['lines_div_up']}, "
             f"down {a['lines_div_down']}); field token located {f['located']}: P(student token) r0 "
             f"{pc.fmt(f['p_teacher_mean']['r0'])} -> {pc.fmt(f['p_teacher_mean']['line'])}, top-1 differs "
             f"{f['top1_differs']['r0']} -> {f['top1_differs']['line']}"]
    for condition in ("r1", "r1_vl"):
        c = b[condition]
        lines.append(f"(b) {condition}: {c['continued']} continued, {c['counts']}, required fixed "
                     f"{c['required_fixed']}/{c['required_failed_before']}")
    lines.append(f"(b) paired: {b['paired']}")
    return "\n".join(lines)


def dry_run(plan, tokenizer, engines=None, keep_tasks=None):
    """Every context of the sample built on CPU (no model): the counts, written nowhere."""
    items, exclusions, _saved, engines = prepare_all(plan, tokenizer, engines, keep_tasks)
    return {**context_counts(items, exclusions), "engine_stats": {c: dict(getattr(e, "stats", {}))
                                                                  for c, e in engines.items()}}


# ------------------------------------------------------------------ command line

import os  # noqa: E402

DEFAULT_EVIDENCE = os.environ.get("SDPO_AB_EVIDENCE", "evidence")
DEFAULT_SETS = os.environ.get("SDPO_AB_SETS", "scripts/qualification/fixtures/automationbench")


def default_protected(repo):
    return {"test60b": Path(DEFAULT_SETS) / "test60b.json", "dev60": Path(DEFAULT_SETS) / "dev60.json",
            "guard10": Path(DEFAULT_SETS) / "guard10.json", "dev17": Path(DEFAULT_SETS) / "dev17.json",
            "heldout62": Path(repo) / "scripts/qualification/fixtures/automationbench_heldout62_v1.json"}


def main(argv=None):
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="refinement check: sample (laptop) and CPU dry run")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("select", help="write the sample file from the recorded runs")
    p.add_argument("--evidence", default=DEFAULT_EVIDENCE)
    p.add_argument("--train", default=str(repo / "docs/research/sdpo-automationbench/train-d62.json"))
    p.add_argument("--task-fixes", default=f"{DEFAULT_EVIDENCE}/sdpo-ab-20261104-split/task-fixes.jsonl")
    p.add_argument("--out", required=True)
    d = sub.add_parser("dry-run", help="build every context of the sample on CPU and print the counts")
    d.add_argument("--sample", required=True)
    d.add_argument("--evidence", default=DEFAULT_EVIDENCE)
    d.add_argument("--tokenizer", required=True)
    d.add_argument("--tasks", help="comma-separated subset")
    d.add_argument("--out")
    args = parser.parse_args(argv)
    if args.command == "select":
        content = sample_file(args.evidence, args.train, default_protected(repo), args.task_fixes)
        Path(args.out).write_text(json.dumps(content, indent=1) + "\n")
        print(json.dumps(content["counts"], indent=1))
        return 0
    from transformers import AutoTokenizer

    sample = json.loads(Path(args.sample).read_text())
    folders = {label: Path(args.evidence) / s["run"] / "run" / label for label, s in sample["sources"].items()}
    plan = preflight(args.sample, folders)
    counts = dry_run(plan, AutoTokenizer.from_pretrained(args.tokenizer),
                     keep_tasks=set(args.tasks.split(",")) if args.tasks else None)
    text = json.dumps(counts, indent=1, default=str)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
