"""The two checks before any training (`--phase precheck`): sample, teacher sequences, and both checks' logic.

Registered definitions (fixed before the job; this module implements them and nothing else):

Sample: failed attempts of the recorded run `sdpo-ab-20261104-t63card-control-r1/run/control-t63card` (63 tasks x 16
attempts, untrained model, card sampling) that (a) are not infrastructure failures, (b) left required work undone
(outcomes `required_passed < required_total`), (c) end with a final assistant reply that has no tool call, after at
least one tool call, and (owner decision 2026-11-04) (d) get a final note from the R1 teacher (they fail at least
one scored benchmark required check the engine lists there; siblings and pool as R0/R1 training builds them). Up to 4
per task: `random.Random(SAMPLE_SALT + task).sample(sorted(ids), min(4, n))`. Both pass rules use only this sample;
the same draw from the (a)-(c) population (note-less attempts included) also runs and is reported as a side figure
only. Computed on
the laptop (`python sdpo_ab_precheck.py select ...`) and shipped as /stage/precheck-sample.json; the job recomputes it
from the recorded files and refuses to run if the two differ.

Teacher: the untrained model (the zero-initialized LoRA adapter every train phase starts from, `Runtime.start`), with
the teacher sequence built as training builds it (`EngineContext`, arm B: plain style, placement both, expected values
shown, the verified same-task sibling solution when there is one, fix hints on, no line or character limit,
not_done_source "benchmark"), cut where the recorded final reply begins (`engine_teacher_prefix`), in two conditions:
final_note off ("no_note") and on ("note").

Note wordings (an input of the phase, /stage/precheck-wordings.json): the engine's own final-note text and two other
templates, each with the engine's item list inserted; a "note_<name>" condition per wording.

Check 1 (stopping point), per attempt and condition: (i) the teacher's next-token probability of the tool-call opener
at that position; (ii) the share of 8 sampled continuations (temperature 1.0, top-p 1.0, to the end of the turn or
1,024 tokens) that contain a well-formed tool call (primary). Pass, per wording: mean(note) >= 3 x mean(no_note) and
mean(note) >= 0.20 on (ii). Check 2 uses the wording with the highest mean (ii) (ties: file order).

Coverage (descriptive, no pass bar): per attempt, under the no-note teacher and the chosen-wording note teacher, the
share of the attempt's failed required checks with at least one trained token (the call-body tokens of the call the
check is located at; in the note condition also the first 16 tokens of the final reply for every item the final note
lists) where the teacher's top-1 token differs from the student's sampled token.

Check 2 (continuation), "note" only: the teacher continues the attempt from that position as an agent in the attempt's
own world (rebuilt by replaying its recorded calls with the replica code, `sdpo_ab_world`, and verified to score
exactly the recorded partial credit and success), for at most 10 tool calls, with the real rollouts' turn and reply
limits and tool-call parsing, then the final world is graded. improved: partial credit above the recorded one and no
must-not check newly failing; worse: partial credit below it, or a must-not check newly failing; else same.
Pass: improved share >= 0.30 and worse share <= 0.20.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

SAMPLE_SALT = "sdpo-ab-20261104-precheck:"
PER_TASK = 4
SOURCE_RUN = "sdpo-ab-20261104-t63card-control-r1"
SOURCE_LABEL = "control-t63card"
CONTAINER_SOURCE = f"/evidence/{SOURCE_RUN}/run/{SOURCE_LABEL}"
SOURCE_FILES = ("episodes.jsonl.gz", "chains.jsonl.gz", "outcomes.json")
CONDITIONS = ("no_note", "note")  # the two engine calls; the note wordings are derived from "note"
# the registered teacher options (EngineContext keyword arguments; final_note per condition)
TEACHER_OPTIONS = {"expected_values": True, "fix_hints": ["call_fix"], "line_budget": None, "char_budget": None,
                   "not_done_source": "benchmark"}
TEACHER_ARM = "B"  # the sibling solution, where training would use one (EngineContext: arm B)
# R0 is recipe V's engine-train phase: an attempt's siblings are its own batch of 8 rollouts (in the recorded run,
# repetition indices 0-7 and 8-15 of a task are the two batches) and the solution pool holds every successful attempt
# of the recorded run (as --solution-pool --solution-pool-from would give)
BATCH_SIZE = 8

CHECK1_SAMPLES = 8
CHECK1_MAX_TOKENS = 1024
CHECK1_SAMPLING = {"temperature": 1.0, "top_p": 1.0}
CHECK1_PASS = {"ratio": 3.0, "floor": 0.20}
CHECK2_MAX_CALLS = 10
CHECK2_SAMPLING = {"temperature": 1.0, "top_p": 1.0}
CHECK2_PASS = {"improved_at_least": 0.30, "worse_at_most": 0.20}


# ------------------------------------------------------------------ sample

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def messages_of(record):
    return [node["message"] for node in record["traces"][0]["nodes"] if node.get("message")]


def final_reply_shape(messages):
    """None when the messages end with a final assistant reply without a tool call after at least one tool call
    (criterion c), else the reason they do not."""
    assistants = [m for m in messages if m.get("role") == "assistant"]
    if not assistants:
        return "no_assistant_message"
    if messages[-1].get("role") != "assistant":
        return "does_not_end_with_an_assistant_message"
    if assistants[-1].get("tool_calls"):
        return "final_assistant_message_has_a_tool_call"
    if not any(m.get("tool_calls") for m in assistants[:-1]):
        return "no_tool_call_before_the_final_reply"
    return None


def eligibility(outcome, record):
    """None when the attempt is in the sample population, else the first criterion it fails (in the registered
    order: not an infrastructure failure, failed, required work undone, final reply after a call)."""
    if outcome.get("infrastructure_failure") or outcome.get("success") is None:
        return "a:infrastructure_failure"
    if outcome["success"] != 0.0:
        return "succeeded"
    passed, total = outcome.get("required_passed"), outcome.get("required_total")
    if passed is None or total is None or not passed < total:
        return "b:no_required_work_undone"
    shape = final_reply_shape(messages_of(record))
    return f"c:{shape}" if shape else None


def choose(eligible_by_task):
    """{task: [episode ids]} in the registered draw: up to 4 per task, seeded by the task name."""
    out = {}
    for task in sorted(eligible_by_task):
        ids = sorted(eligible_by_task[task])
        if ids:
            out[task] = random.Random(SAMPLE_SALT + task).sample(ids, min(PER_TASK, len(ids)))
    return out


def iter_jsonl_gz(path, wanted=None):
    """Stream a .jsonl.gz file; with `wanted` (ids), only lines whose head names one of them are parsed."""
    with gzip.open(path, "rt") as stream:
        for line in stream:
            if wanted is not None and not any(i in line[:200] for i in wanted):
                continue
            value = json.loads(line)
            yield value


def chain_summary(chain, record):
    """Facts about the recorded token chain that the precheck relies on (the final reply is the last action)."""
    if chain is None:
        return {"chain_joined": False}
    actions = [s for s in chain["segments"] if s["kind"] == "action"]
    assistants = [m for m in messages_of(record) if m.get("role") == "assistant"]
    last = actions[-1] if actions else None
    return {"chain_joined": not chain.get("mismatch"), "chain_actions": len(actions),
            "actions_match_messages": len(actions) == len(assistants),
            "final_reply_complete": bool(last and last.get("complete")),
            "final_reply_limit": last.get("limit_reason") if last else None,
            "final_reply_tokens": len(last["ids"]) if last else None,
            "final_reply_has_call_ids": bool(last and last.get("tool_call_ids"))}


def population(folder):
    """One streamed pass: the attempts meeting (a)-(c) per task, with their shapes, and the reason counts."""
    folder = Path(folder)
    outcomes = json.loads((folder / "outcomes.json").read_text())
    by_id = {o["episode_id"]: o for o in outcomes}
    reasons, eligible, shapes, seen = Counter(), defaultdict(list), {}, 0
    for record in iter_jsonl_gz(folder / "episodes.jsonl.gz"):
        seen += 1
        outcome = by_id.get(record["id"])
        if outcome is None:
            raise ValueError(f"episode {record['id']} has no outcome")
        reason = eligibility(outcome, record)
        reasons[reason or "eligible"] += 1
        if reason is None:
            eligible[outcome["task"]].append(record["id"])
            messages = messages_of(record)
            shapes[record["id"]] = {
                "assistant_turns": sum(m.get("role") == "assistant" for m in messages),
                "tool_calls": sum(len(m.get("tool_calls") or []) for m in messages if m.get("role") == "assistant"),
                "partial_credit": record["traces"][0]["rewards"]["partial_credit"]["score"],
                "success": record["traces"][0]["metrics"]["task_completed_correctly"]}
    if seen != len(outcomes):
        raise ValueError(f"{seen} episodes but {len(outcomes)} outcomes")
    return {t: sorted(v) for t, v in sorted(eligible.items())}, shapes, reasons, seen, by_id


def note_eligibility(folder, population_by_task, engine=None):
    """{episode id: "final_note" or the reason it gets none}: whether the R1 teacher (the "note" EngineContext with
    the registered options, the attempt's own batch as siblings and every successful recorded attempt of the task in
    the solution pool, as R0/R1 training builds it) gives the attempt a final note, i.e. it fails at least one scored
    benchmark required check the engine lists there. Streams the run task by task (the file is grouped by task)."""
    engine = engine or teacher_engines()["note"]
    wanted = {i for ids in population_by_task.values() for i in ids}
    out, finished, current, records = {}, set(), None, []

    def flush():
        if current is None or current not in population_by_task:
            return
        engine.add_to_pool(records)
        for record in records:
            if record["id"] not in wanted:
                continue
            try:
                context = engine(record, batch_siblings(record, records), TEACHER_ARM, "both")
            except ValueError as error:
                out[record["id"]] = f"untrue: {error}"
                continue
            except Exception as error:  # noqa: BLE001 - recorded; such an attempt gets no note in training either
                out[record["id"]] = f"builder_error: {type(error).__name__}: {error}"
                continue
            final = (context or {}).get("final_note") or {}
            if context is None:
                out[record["id"]] = "no_context"
            elif final.get("turn") is None:
                out[record["id"]] = f"no_final_note: {final.get('reason')}"
            else:
                out[record["id"]] = "final_note"

    for record in iter_jsonl_gz(Path(folder) / "episodes.jsonl.gz"):
        task = record["traces"][0]["task"]["data"]["name"]
        if task != current:
            flush()
            finished.add(current)
            if task in finished:
                raise ValueError(f"episodes of {task} are not contiguous")
            current, records = task, []
        records.append(record)
    flush()
    missing = wanted - set(out)
    if missing:
        raise ValueError(f"{len(missing)} attempts got no note verdict")
    return out


def select_sample(folder, engine=None):
    """The registered sample (owner decision 2026-11-04): from the attempts meeting (a)-(c), those the R1 teacher
    gives a final note, up to 4 per task by the seeded draw ("selected_by_task", the pass rules use only these). The
    earlier draw from all attempts meeting (a)-(c) ("side_selected_by_task", note-less ones included) is run too and
    reported as a side figure only. Returns the sample file's selection part."""
    folder = Path(folder)
    population_by_task, shapes, reasons, seen, by_id = population(folder)
    verdicts = note_eligibility(folder, population_by_task, engine)
    eligible = {t: [i for i in ids if verdicts[i] == "final_note"] for t, ids in population_by_task.items()}
    eligible = {t: v for t, v in eligible.items() if v}
    chosen, side = choose(eligible), choose(population_by_task)
    primary = {i for ids in chosen.values() for i in ids}
    secondary = {i for ids in side.values() for i in ids}
    wanted = primary | secondary
    chains = {c["episode_id"]: c for c in iter_jsonl_gz(folder / "chains.jsonl.gz", wanted)
              if c["episode_id"] in wanted}
    records = {r["id"]: r for r in iter_jsonl_gz(folder / "episodes.jsonl.gz", wanted) if r["id"] in wanted}
    attempts = []
    for episode_id in sorted(wanted, key=lambda i: (by_id[i]["task"], i)):
        outcome = by_id[episode_id]
        attempts.append({"episode_id": episode_id, "task": outcome["task"], "primary": episode_id in primary,
                         "side": episode_id in secondary, "note_verdict": verdicts[episode_id], **shapes[episode_id],
                         "required_passed": outcome["required_passed"],
                         "required_total": outcome["required_total"],
                         "stop_condition": outcome.get("stop_condition"),
                         **chain_summary(chains.get(episode_id), records[episode_id])})
    note_reasons = Counter(v.split(":")[0] for v in verdicts.values())
    return {"population_by_task": population_by_task, "eligible_by_task": eligible, "selected_by_task": chosen,
            "side_selected_by_task": side, "note_verdicts": dict(sorted(verdicts.items())), "attempts": attempts,
            "counts": {"episodes": seen, "by_reason": dict(sorted(reasons.items())),
                       "population": sum(map(len, population_by_task.values())),
                       "population_note_verdicts": dict(sorted(note_reasons.items())),
                       "eligible": sum(map(len, eligible.values())), "tasks_with_eligible": len(chosen),
                       "selected": len(primary), "selected_per_task": {t: len(v) for t, v in chosen.items()},
                       "side_selected": len(secondary), "side_tasks": len(side), "attempts_run": len(attempts)}}


def sample_file(folder, run_root):
    """The shipped input: the definition, the source files' hashes, the environment commit and world date of the
    recorded run, and the selection with its counts."""
    folder, run_root = Path(folder), Path(run_root)
    inputs = json.loads((folder.parent / "inputs-collect.json").read_text())
    selection = select_sample(folder)
    return {
        "definition": {
            "population": "failed attempts that (a) are not infrastructure failures, (b) left required work undone "
                          "(outcomes required_passed < required_total), (c) end with a final assistant reply that has "
                          "no tool call, after at least one tool call",
            "eligible": "the population attempts the R1 teacher gives a final note (they fail at least one scored "
                        "benchmark required check the engine lists in it; note EngineContext with the registered "
                        "options, own batch of 8 as siblings, solution pool of the task's successful attempts)",
            "pass_rules_use": "selected_by_task only; side_selected_by_task (the same draw from the whole population, "
                              "note-less attempts included) is reported as a side figure",
            "draw": f"random.Random({SAMPLE_SALT!r} + task).sample(sorted(ids), min({PER_TASK}, n))",
            "teacher_options": TEACHER_OPTIONS, "teacher_arm": TEACHER_ARM,
            "conditions": "no_note and one note condition per wording of the wordings input",
            "check1": {"samples": CHECK1_SAMPLES, "max_tokens": CHECK1_MAX_TOKENS, **CHECK1_SAMPLING,
                       "pass": CHECK1_PASS},
            "check2": {"max_calls": CHECK2_MAX_CALLS, **CHECK2_SAMPLING, "pass": CHECK2_PASS}},
        "source": {"run": SOURCE_RUN, "label": SOURCE_LABEL, "container_dir": CONTAINER_SOURCE,
                   "sha256": {name: sha256_file(folder / name) for name in SOURCE_FILES},
                   "env_commit": (run_root / "env-commit.txt").read_text().strip(),
                   "code_sha": (run_root / "code-sha.txt").read_text().strip(),
                   "world_date": inputs.get("world_date"), "tz": inputs.get("tz"),
                   "sampling": inputs.get("collect_sampling"), "attempts": inputs.get("attempts")},
        **selection}


def check_shipped(shipped, population_by_task, tasks=None):
    """The job's guard: the (a)-(c) population the recorded files give equals the shipped one, the note-eligible set
    is inside it, and both draws are the registered draw of their sets (all tasks, or `tasks`). Note eligibility itself
    is recomputed on the laptop (select_sample, and a test); in the job every primary attempt must get its final note
    again (run_precheck excludes it otherwise)."""
    tasks = sorted(set(shipped["population_by_task"]) | set(population_by_task)) if tasks is None else sorted(tasks)
    for task in tasks:
        if shipped["population_by_task"].get(task) != population_by_task.get(task):
            raise ValueError(f"the shipped population of {task} differs from the one the recorded run gives")
        eligible = shipped["eligible_by_task"].get(task, [])
        if not set(eligible) <= set(population_by_task.get(task, [])):
            raise ValueError(f"the shipped note-eligible attempts of {task} are not in its population")
        if choose({task: eligible}).get(task) != shipped["selected_by_task"].get(task):
            raise ValueError(f"the shipped sample of {task} is not the registered draw")
        if choose({task: population_by_task.get(task, [])}).get(task) != shipped["side_selected_by_task"].get(task):
            raise ValueError(f"the shipped side sample of {task} is not the registered draw")


# ------------------------------------------------------------------ teacher sequences

def chain_problem(chain, record):
    """None when the recorded token chain is usable (joined, one action per assistant message, the final action
    without a call), else the reason."""
    if chain is None:
        return "chain_missing"
    if chain.get("mismatch"):
        return "chain_mismatch"
    actions = [s for s in chain["segments"] if s["kind"] == "action"]
    assistants = [m for m in messages_of(record) if m.get("role") == "assistant"]
    if not actions or len(actions) != len(assistants):
        return "chain_actions_differ_from_messages"
    if actions[-1].get("tool_call_ids"):
        return "final_action_has_calls"
    return None


def final_position(chain):
    """(index in chain["segments"] of the final reply, its assistant turn number)."""
    actions = [i for i, s in enumerate(chain["segments"]) if s["kind"] == "action"]
    return actions[-1], len(actions)


def tool_call_opener(tokenizer):
    """The first token of the model's tool-call format, derived from the chat template: an assistant message with no
    text and one call is rendered, its role header (and an empty thinking block, if the template adds one) removed,
    and the rest tokenized. Asserted to be the single token "<tool_call>" that the template's own format
    instructions name. Returns (token id, token text)."""
    import os

    tool = {"type": "function", "function": {"name": "probe_tool", "description": "probe",
                                              "parameters": {"type": "object", "properties": {"a": {"type": "string"}},
                                                             "required": ["a"]}}}
    user = [{"role": "user", "content": "u"}]
    call = {"role": "assistant", "content": "",
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "probe_tool", "arguments": {"a": "b"}}}]}

    def render(messages, generation):
        return tokenizer.apply_chat_template(messages, tools=[tool], tokenize=False, add_generation_prompt=generation,
                                             enable_thinking=False)

    base, prompt, full = render(user, False), render(user, True), render(user + [call], False)
    if not (prompt.startswith(base) and full.startswith(base)):
        raise ValueError("chat template does not extend the conversation prefix")
    generation, turn = prompt[len(base):], full[len(base):]
    header = os.path.commonprefix([generation, turn])
    body = turn[len(header):]
    if body.startswith("<think>"):
        body = body.split("</think>", 1)[1].lstrip("\n")
    first = tokenizer.encode(body, add_special_tokens=False)[0]
    text = tokenizer.decode([first])
    expected = tokenizer.convert_tokens_to_ids("<tool_call>")
    if text != "<tool_call>" or first != expected or "<function=probe_tool>" not in body:
        raise ValueError(f"tool-call opener {first} {text!r} is not the single token <tool_call> ({expected})")
    if "<tool_call>" not in base:
        raise ValueError("the chat template's tool instructions do not name <tool_call>")
    return first, text


def note_insertion(plain, note, turn):
    """The text a note condition (context `note`) adds before the final reply at `turn`, compared with the no-note
    context `plain` ("" when there is no final note). Raises when the two contexts differ anywhere else (first
    message, notes of other turns)."""
    if plain["first_message"] != note["first_message"]:
        raise ValueError("first messages differ")
    a, b = plain["notes"], note["notes"]
    if {t: v for t, v in a.items() if t != turn} != {t: v for t, v in b.items() if t != turn}:
        raise ValueError("notes of other turns differ")
    if a.get(turn) == b.get(turn):
        return ""
    if b.get(turn) is None:
        raise ValueError("the note condition dropped a note")
    if a.get(turn) is None:
        return "\n\n" + b[turn]
    if not b[turn].startswith(a[turn]):
        raise ValueError("the note condition changed an existing note")
    return b[turn][len(a[turn]):]


def note_placement(tokenizer, chain, plain, note, prefix_plain, prefix_note, render_delta):
    """Checks that the two teacher prefixes differ only by the note text, appended to the last tool (or user) message
    before the final reply, i.e. immediately before the final reply's generation prompt. Returns the inserted text."""
    import copy

    position, turn = final_position(chain)
    inserted = note_insertion(plain, note, turn)
    if not inserted:
        if prefix_plain != prefix_note:
            raise ValueError("teacher prefixes differ without a final note")
        return inserted
    segment = chain["segments"][position - 1]
    if segment["kind"] != "context" or segment.get("messages") is None:
        raise ValueError("the final reply does not follow a rendered message")
    messages = copy.deepcopy(segment["messages"])
    target = max(i for i, m in enumerate(messages) if m["role"] in ("tool", "user"))
    body = messages[target].get("content")
    body = body if isinstance(body, str) else json.dumps(body)
    marker = "⁣PRECHECK⁣"
    messages[target]["content"] = marker
    rendered = tokenizer.decode(render_delta(tokenizer, messages))
    if rendered.count(marker) != 1:
        raise ValueError("cannot locate the message end in the rendering")
    tail = rendered.split(marker)[1]
    plain_note = plain["notes"].get(turn)
    content = body + ("\n\n" + plain_note if plain_note is not None else "")
    text_plain, text_note = tokenizer.decode(prefix_plain), tokenizer.decode(prefix_note)
    if not text_plain.endswith(content + tail):
        raise ValueError("no-note teacher does not end with the final reply's preceding message")
    head = text_plain[:len(text_plain) - len(content + tail)]
    if text_note != head + content + inserted + tail:
        raise ValueError("note teacher differs from the no-note teacher by more than the note before the final reply")
    return inserted


# ------------------------------------------------------------------ note wordings (check 1 compares them)

PLACEHOLDER = "<items>"


def load_wordings(path):
    """The wordings file: {"placeholder": "<items>", "wordings": [{"name", "template"}, ...]}, the first being the
    engine's own text. Each template holds the placeholder once; names are unique."""
    content = json.loads(Path(path).read_text())
    if content.get("placeholder", PLACEHOLDER) != PLACEHOLDER:
        raise ValueError(f"the wordings' placeholder must be {PLACEHOLDER!r}")
    wordings = content["wordings"]
    names = [w["name"] for w in wordings]
    if not wordings or len(set(names)) != len(names) or any(not n.isidentifier() for n in names):
        raise ValueError("wordings need unique identifier names")
    for w in wordings:
        if w["template"].count(PLACEHOLDER) != 1:
            raise ValueError(f"wording {w['name']} must contain {PLACEHOLDER} exactly once")
    return wordings


def note_conditions(wordings):
    return ["no_note"] + [f"note_{w['name']}" for w in wordings]


def engine_numbering():
    """The engine's own numbering of the final-note items (abce.render.plain.final_note_line without its head): one
    line "(1) A; (2) B." or, when an item contains "; ", one item per line ("\n1. A\n2. B")."""
    from sdpo_ab_engine import load_engine

    load_engine()
    from abce.render.plain import FINAL_NOTE_HEAD, final_note_line

    return lambda texts: final_note_line(list(texts))[len(FINAL_NOTE_HEAD):]


def reworded_contexts(note, turn, wordings, numbering=None):
    """{condition: context} for every wording: the note context with its final-note line (the last text of the note
    at `turn`, as the engine appends it) replaced by the wording, its placeholder filled with the engine's items
    (`final_note["texts"]`) numbered by the engine's own rule. The first wording must reproduce the engine's line
    exactly. Without a final note every wording's context is the note context itself."""
    import copy

    final = note.get("final_note") or {}
    line = final.get("line")
    if final.get("turn") is None or not line:
        return {f"note_{w['name']}": note for w in wordings}
    texts = final.get("texts")
    if not texts:
        raise ValueError("the engine's final note carries no item texts")
    text = note["notes"].get(turn) or ""
    if not text.endswith(line):
        raise ValueError("the final-note line is not the end of the note at the final reply")
    items = (numbering or engine_numbering())(texts)
    lines = {w["name"]: w["template"].replace(PLACEHOLDER, items) for w in wordings}
    if lines[wordings[0]["name"]] != line:
        raise ValueError(f"the engine's final note {line!r} is not the engine wording {lines[wordings[0]['name']]!r}")
    out = {}
    for w in wordings:
        context = copy.deepcopy(note)
        context["notes"][turn] = text[:len(text) - len(line)] + lines[w["name"]]
        context["final_note"] = {**final, "line": lines[w["name"]], "wording": w["name"]}
        out[f"note_{w['name']}"] = context
    return out


# ------------------------------------------------------------------ coverage (descriptive)

COVERAGE_FINAL_TOKENS = 16


def failed_required(context):
    """The attempt's failed required checks as the engine lists them: facts of kind "missing" (one per check)."""
    return [f for f in context.get("facts") or [] if isinstance(f, dict) and f.get("kind") == "missing"]


def call_body_groups(tokenizer, student):
    """{(turn, 0-based call index in the turn): [indices into the student's assistant tokens]} for the tokens of
    each tool-call body (after <tool_call> through </tool_call>: the tokens `restrict_to_calls` trains)."""
    from sdpo_ab_runner import call_body_indices

    opener = tokenizer.convert_tokens_to_ids("<tool_call>")
    keep = set(call_body_indices(tokenizer, student))
    groups, current, counts, turn = defaultdict(list), None, Counter(), None
    for index, (token, t) in enumerate(zip(student["targets"], student["turns"])):
        if t != turn:
            turn, current = t, None
        if token == opener and index not in keep:
            current = (t, counts[t])
            counts[t] += 1
        elif index in keep and current is not None:
            groups[current].append(index)
    return dict(groups)


def coverage_tokens(tokenizer, student, context, note):
    """{fact id: [student token indices]} the training would teach for each failed required check: the call-body
    tokens of the call the fact is located at; with `note`, also the first 16 tokens of the final reply for every
    item the final note lists."""
    groups = call_body_groups(tokenizer, student)
    final_turn = max(student["turns"]) if student["turns"] else None
    final_tokens = [i for i, t in enumerate(student["turns"]) if t == final_turn][:COVERAGE_FINAL_TOKENS]
    listed = set(((context.get("final_note") or {}).get("items") or [])) if note else set()
    out = {}
    for fact in failed_required(context):
        tokens = []
        if fact.get("location") == "at_call" and fact.get("turn") is not None and fact.get("call") is not None:
            tokens += groups.get((fact["turn"], fact["call"]), [])
        if fact.get("id") in listed:
            tokens += final_tokens
        out[fact["id"]] = sorted(set(tokens))
    return out


def coverage(links, student_targets, teacher_top1):
    """Share of failed required checks with at least one linked token where the teacher's top-1 differs from the
    student's sampled token. `teacher_top1`: {student token index: teacher argmax}."""
    covered = {fid: any(teacher_top1[i] != student_targets[i] for i in tokens) for fid, tokens in links.items()}
    return {"checks": len(covered), "covered": sum(covered.values()),
            "share": sum(covered.values()) / len(covered) if covered else None,
            "linked_checks": sum(bool(t) for t in links.values()), "per_check": covered}


# ------------------------------------------------------------------ check 2: the continuation

# the real rollouts' limits: Agent (65,536-token context, 4,096-token replies) and the environment
# (sdpo_ab_replay.environment: max_turns 50, max_output_tokens 65,536, checked before each turn)
ROLLOUT_LIMITS = {"context": 65536, "reply": 4096, "max_turns": 50, "max_output_tokens": 65536}


async def continue_attempt(prefix, *, tokenizer, generate, parse, execute, render, turns_before, output_before,
                           max_calls=CHECK2_MAX_CALLS, limits=None):
    """The teacher continues the attempt from `prefix` (its sequence up to where the recorded final reply began) as
    the real rollout's agent would: each turn is sampled with the rollout's reply budget (Agent.completions), parsed
    for tool calls only when complete, closed with <|im_end|> when cut; calls run in order in the attempt's world and
    their results are rendered as the next context (render_delta); the episode ends on a reply without calls, on a
    cut reply, or at the environment's turn / output-token limits. At most `max_calls` calls run in total; calls of a
    turn beyond that are recorded as not run and the continuation stops.

    generate(ids, max_tokens, turn_index) -> (ids, finish_reason); parse(text) -> calls in the OpenAI shape;
    execute([(name, arguments text)]) -> [(content, status)]; render(tool messages) -> ids."""
    limits = {**ROLLOUT_LIMITS, **(limits or {})}
    tokens, turns, executed, output, stop = list(prefix), [], 0, output_before, None
    eos = tokenizer.eos_token_id
    while stop is None:
        if turns_before + len(turns) >= limits["max_turns"]:
            stop = "max_turns"
            break
        if output >= limits["max_output_tokens"]:
            stop = "max_output_tokens"
            break
        budget = min(limits["reply"], limits["context"] - len(tokens))
        if budget < 1:
            ids, finish = [], "context_limit"
        else:
            ids, finish = await generate(tokens, budget, len(turns))
        complete = finish == "stop"
        limit = None if complete else ("context_limit" if finish == "context_limit"
                                       else "reply_limit" if budget == limits["reply"] else "context_limit")
        raw = tokenizer.decode(ids, skip_special_tokens=True)
        calls = parse(raw) if complete else []
        tokens += ids
        if ids and ids[-1] != eos:
            tokens.append(eos)
        output += len(ids)
        turn = {"tokens": len(ids), "finish": finish, "limit": limit, "text": raw, "calls": []}
        turns.append(turn)
        if not calls:
            stop = "final_reply" if complete else limit
            break
        run, skipped = calls[:max_calls - executed], calls[max_calls - executed:]
        results = await execute([(c["function"]["name"], c["function"]["arguments"]) for c in run]) if run else []
        messages = []
        for call, (content, status) in zip(run, results, strict=True):
            turn["calls"].append({"id": call["id"], "name": call["function"]["name"],
                                  "arguments": call["function"]["arguments"], "executed": True, "status": status,
                                  "result": content})
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": content,
                             "name": call["function"]["name"]})
        for call in skipped:
            turn["calls"].append({"id": call["id"], "name": call["function"]["name"],
                                  "arguments": call["function"]["arguments"], "executed": False,
                                  "reason": "call_cap"})
        executed += len(run)
        if executed >= max_calls:
            stop = "call_cap"
            break
        tokens += render(messages)
    return {"turns": turns, "executed_calls": executed, "stop": stop, "final_tokens": len(tokens)}


def must_not_newly_failing(before, after):
    """Indices and types of scored must-not checks (the negative assertion family, sdpo_ab_metrics.negative_family)
    that passed before the continuation and fail after it."""
    from sdpo_ab_metrics import negative_family

    if len(before) != len(after):
        raise ValueError("assertion lists differ in length")
    out = []
    for index, (a, b) in enumerate(zip(before, after)):
        if a.get("type") != b.get("type") or a.get("params") != b.get("params"):
            raise ValueError(f"assertion {index} differs in type or parameters")
        if (negative_family(str(b.get("type"))) and a.get("passed") and not a.get("excluded")
                and not b.get("passed") and not b.get("excluded")):
            out.append({"index": index, "type": b.get("type")})
    return out


def classify(before_credit, after_credit, newly_failing):
    """The registered verdict of one continuation."""
    if after_credit < before_credit or newly_failing:
        return "worse"
    if after_credit > before_credit:
        return "improved"
    return "same"


def assertion_changes(before, after):
    return [{"index": i, "type": b.get("type"), "before": {"passed": a.get("passed"), "excluded": a.get("excluded")},
             "after": {"passed": b.get("passed"), "excluded": b.get("excluded")}}
            for i, (a, b) in enumerate(zip(before, after))
            if (a.get("passed"), a.get("excluded")) != (b.get("passed"), b.get("excluded"))]


# ------------------------------------------------------------------ check 1 helpers and the verdicts

def well_formed_call(calls, tool_names):
    """Whether the parsed calls contain one that names an offered tool with a JSON-object argument text."""
    for call in calls:
        function = call.get("function") or {}
        if function.get("name") not in tool_names:
            continue
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except (TypeError, ValueError):
            continue
        if isinstance(arguments, dict):
            return True
    return False


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def check1_verdict(rows, conditions):
    """rows: per included attempt {condition: {"act_share", "p_opener", ...}}; the rule is applied to every note
    wording against no_note. `best` is the wording with the highest mean act share (ties: file order), the one
    check 2 uses."""
    out = {}
    for key in ("act_share", "p_opener", "act_share_complete"):
        out[key] = {c: mean(r[c][key] for r in rows if r[c].get(key) is not None) for c in conditions}
    acts, base = out["act_share"], out["act_share"]["no_note"]
    verdicts = {c: (acts[c] is not None and base is not None and acts[c] >= CHECK1_PASS["ratio"] * base
                    and acts[c] >= CHECK1_PASS["floor"]) for c in conditions if c != "no_note"}
    with_note = [r for r in rows if r.get("final_note_present")]
    out["with_final_note_only"] = {"attempts": len(with_note), **{
        key: {c: mean(r[c][key] for r in with_note if r[c].get(key) is not None) for c in conditions}
        for key in ("act_share", "p_opener", "act_share_complete")}}
    notes = [c for c in conditions if c != "no_note"]
    scored = [c for c in notes if acts[c] is not None]
    best = max(scored, key=lambda c: (acts[c], -notes.index(c))) if scored else None
    return {"attempts": len(rows), "means": out, "rule": "per wording: mean act_share(note) >= 3 x mean "
            "act_share(no_note) and mean act_share(note) >= 0.20 (act_share = share of 8 samples with a well-formed "
            "tool call)", "passed_by_wording": verdicts, "passed_any": any(verdicts.values()), "best_wording": best}


def check2_verdict(rows):
    counts = Counter(r["verdict"] for r in rows)
    n = len(rows)
    shares = {k: (counts[k] / n if n else None) for k in ("improved", "same", "worse")}
    noted = [r for r in rows if r.get("final_note_present")]
    note_counts = Counter(r["verdict"] for r in noted)
    with_note = {"attempts": len(noted), "counts": dict(note_counts),
                 "shares": {k: (note_counts[k] / len(noted) if noted else None) for k in ("improved", "same", "worse")}}
    passed = bool(n) and shares["improved"] >= CHECK2_PASS["improved_at_least"] and \
        shares["worse"] <= CHECK2_PASS["worse_at_most"]
    return {"attempts": n, "counts": dict(counts), "shares": shares, "with_final_note_only": with_note,
            "rule": "improved share >= 0.30 and worse share <= 0.20", "passed": passed}


# ------------------------------------------------------------------ the job

SEED_SALT = "sdpo-ab-20261104-precheck"


def teacher_engines(engine_class=None):
    """One EngineContext per engine call, with the registered teacher options: final_note off ("no_note") and on
    ("note", the engine's own wording; the other wordings are derived from it, `reworded_contexts`)."""
    if engine_class is None:
        from sdpo_ab_engine import EngineContext as engine_class
    options = {k: v for k, v in TEACHER_OPTIONS.items() if k not in ("expected_values", "fix_hints")}
    return {c: engine_class(expected_values=TEACHER_OPTIONS["expected_values"], pool=True,
                            fix_hints=TEACHER_OPTIONS["fix_hints"], options={**options, "final_note": c == "note"})
            for c in CONDITIONS}


def preflight(sample_path, wordings_path, source=None, pinned=None, check_engine=True):
    """Before the model is loaded: the shipped sample and wordings, the recorded run's files (sha256 as shipped), the
    environment commit and world date of the recorded run, the engine's support for the registered options, and the
    replica's imports. Returns the plan `run_precheck` takes."""
    import datetime as dt
    import os

    sample = json.loads(Path(sample_path).read_text())
    wordings = load_wordings(wordings_path)
    folder = Path(source or sample["source"]["container_dir"])
    for name, digest in sample["source"]["sha256"].items():
        got = sha256_file(folder / name)
        if got != digest:
            raise ValueError(f"{folder / name}: sha256 {got} is not the shipped {digest}")
    env_commit = os.environ.get("SDPO_AB_ENV_COMMIT")
    if env_commit and env_commit != sample["source"]["env_commit"]:
        raise ValueError(f"environment {env_commit} is not the recorded run's {sample['source']['env_commit']} "
                         "(launch with SDPO_AB_ENV_COMMIT set to it)")
    day = sample["source"]["world_date"]
    if pinned and pinned != day:
        raise ValueError(f"world clock pinned to {pinned}, the recorded run used {day}")
    from sdpo_ab_execution import pin_world_clock

    pin_world_clock(day)
    if check_engine:
        from sdpo_ab_engine import ENGINE_OPTIONS, engine_supports, engine_version

        if not engine_supports(ENGINE_OPTIONS):
            raise ValueError(f"the context engine ({engine_version()}) does not take {ENGINE_OPTIONS}; re-vendor the "
                             "corrected engine (correction r0) first")
        from sdpo_ab_engine import missing_manifests

        missing = missing_manifests(sorted(sample["selected_by_task"]))
        if missing:
            raise ValueError(f"task manifests do not load for {missing} (the engine would silently drop their "
                             "rule-file facts; check ABCE_ENVIRONMENT / the shipped environment)")
    import sdpo_ab_world

    sdpo_ab_world.check_imports()
    return {"sample": sample, "wordings": wordings, "folder": folder, "day": dt.date.fromisoformat(day),
            "sample_path": str(sample_path), "wordings_path": str(wordings_path),
            "env_commit": env_commit or "unchecked (SDPO_AB_ENV_COMMIT not set)"}


def load_source(folder, keep_tasks):
    """Stream the recorded run once: the registered selection recomputed over the whole run, and every record of the
    tasks in `keep_tasks` (the siblings the engine sees, in file order)."""
    folder = Path(folder)
    by_id = {o["episode_id"]: o for o in json.loads((folder / "outcomes.json").read_text())}
    eligible, records = defaultdict(list), defaultdict(list)
    for record in iter_jsonl_gz(folder / "episodes.jsonl.gz"):
        outcome = by_id[record["id"]]
        if eligibility(outcome, record) is None:
            eligible[outcome["task"]].append(record["id"])
        if outcome["task"] in keep_tasks:
            records[outcome["task"]].append(record)
    return records, {t: sorted(v) for t, v in sorted(eligible.items())}


def batch_of(record):
    """The training batch a recorded attempt belongs to (repetition index // 8)."""
    index = (record.get("run") or {}).get("repetition_index")
    if not isinstance(index, int):
        raise ValueError(f"episode {record.get('id')} has no repetition index")
    return index // BATCH_SIZE


def batch_siblings(record, task_records):
    """The attempt's own batch: the task's recorded attempts with the same batch index (the attempt included)."""
    return [r for r in task_records if batch_of(r) == batch_of(record)]


def tool_names(chain):
    return {(t.get("function") or t).get("name") for t in chain["tools"]}


def prepare_item(tokenizer, attempt, record, chain, siblings, engines, wordings, render_delta, saved):
    """Contexts, teacher prefixes and their checks for one attempt: (item, {condition: exclusion reason})."""
    from sdpo_ab_runner import context_record, engine_teacher_prefix

    conditions = note_conditions(wordings)
    position, turn = final_position(chain)
    built, reasons = {}, {}
    for condition, engine in engines.items():
        try:
            context = engine(record, siblings, TEACHER_ARM, "both")
        except ValueError as error:  # the engine's truth check failed
            reasons[condition] = f"untrue: {error}"
            continue
        except Exception as error:  # noqa: BLE001 - an engine crash excludes the attempt, as in training
            reasons[condition] = f"builder_error: {type(error).__name__}: {error}"
            continue
        if context is None:
            reasons[condition] = "no_context"
            continue
        built[condition] = context
        saved.append({**context_record(context, TEACHER_ARM), "condition": condition,
                      "final_note": context.get("final_note")})
    contexts = {"no_note": built["no_note"]} if "no_note" in built else {}
    final = (built.get("note") or {}).get("final_note") or {}
    if "note" in built:
        if final.get("turn") not in (None, turn):
            reasons["note"] = f"final_note_turn {final.get('turn')} is not the final reply's turn {turn}"
        else:
            try:
                contexts.update(reworded_contexts(built["note"], turn, wordings))
            except ValueError as error:
                reasons["note"] = f"wording: {error}"
    for condition in conditions[1:]:
        if condition not in contexts:
            reasons.setdefault(condition, reasons.get("note", "no note context"))
    reasons.pop("note", None)
    prefixes = {c: engine_teacher_prefix(tokenizer, chain, position, ctx) for c, ctx in contexts.items()}
    for condition, ids in prefixes.items():
        if len(ids) >= ROLLOUT_LIMITS["context"]:
            reasons[condition] = "teacher_prefix_over_context_limit"
    inserted = {}
    if "no_note" in contexts and "no_note" not in reasons:
        for condition in conditions[1:]:
            if condition in reasons:
                continue
            try:
                inserted[condition] = note_placement(tokenizer, chain, contexts["no_note"], contexts[condition],
                                                     prefixes["no_note"], prefixes[condition], render_delta)
            except ValueError as error:
                reasons[condition] = f"note_placement: {error}"
    actions = [s for s in chain["segments"] if s["kind"] == "action"]
    item = {"attempt": attempt, "record": record, "chain": chain, "position": position, "turn": turn,
            "prefixes": prefixes, "contexts": contexts, "inserted": inserted, "final_note": final or None,
            "final_note_present": final.get("turn") is not None,
            "recorded_first_token": actions[-1]["ids"][0] if actions[-1]["ids"] else None,
            "output_before": sum(len(a["ids"]) for a in actions[:-1]), "check1": not reasons, "reasons": reasons}
    return item


async def run_precheck(plan, *, tokenizer, generate, parse_for, score_next, top1, suspend, out, concurrency=32,
                       engines=None, keep_tasks=None, world=None, log=None, engine_version=None):
    """Both checks and the coverage measure on the shipped sample. The model-facing parts are injected so the whole
    flow runs on CPU in tests:

    generate(ids, max_tokens, seed) -> (ids, finish_reason): one sample from the untrained model at temperature 1.0,
    top-p 1.0 (the rollout sampling); parse_for(chain) -> parse(text) -> calls (the rollout's tool-call parser for that
    chain's tools); score_next(ids, tokens) -> ({token: logprob}, top) or None (out of memory): the teacher
    adapter's next-token log-probabilities after `ids`; top1(ids, positions) -> [argmax token] or None: the teacher
    adapter's top-1 token at each position; suspend(): awaited after all sampling, before scoring. `world` is
    sdpo_ab_world (rebuild / execute / grade, in one worker thread)."""
    import asyncio
    import concurrent.futures
    import time

    from sdpo_ab_runner import (render_delta, request_seed, student_sequence, teacher_sequence, write,
                                write_jsonl_gz)

    if world is None:
        import sdpo_ab_world as world
    log = log or (lambda **fields: print(json.dumps(fields, default=str), flush=True))
    started = time.monotonic()
    out = Path(out)
    sample, folder, day, wordings = plan["sample"], Path(plan["folder"]), plan["day"], plan["wordings"]
    conditions = note_conditions(wordings)
    attempts = [a for a in sample["attempts"] if keep_tasks is None or a["task"] in keep_tasks]
    tasks = {a["task"] for a in attempts}
    records_by_task, recomputed = load_source(folder, tasks)
    check_shipped(sample, recomputed, None if keep_tasks is None else tasks)
    by_id = {r["id"]: r for rs in records_by_task.values() for r in rs}
    wanted = {a["episode_id"] for a in attempts}
    chains = {c["episode_id"]: c for c in iter_jsonl_gz(folder / "chains.jsonl.gz", wanted)
              if c["episode_id"] in wanted}
    engines = engines or teacher_engines()
    for engine in engines.values():  # the solution pool: every successful recorded attempt of the sampled tasks
        if hasattr(engine, "add_to_pool"):
            for task_records in records_by_task.values():
                engine.add_to_pool(task_records)
    opener, opener_text = tool_call_opener(tokenizer)
    log(stage="precheck_loaded", attempts=len(attempts), tasks=len(tasks), chains=len(chains), opener=opener,
        opener_text=opener_text, conditions=conditions)

    exclusions, items, saved = [], [], []

    def exclude(attempt, check, reason):
        exclusions.append({"episode_id": attempt["episode_id"], "task": attempt["task"], "check": check,
                           "reason": reason})

    for attempt in attempts:
        record, chain = by_id[attempt["episode_id"]], chains.get(attempt["episode_id"])
        problem = chain_problem(chain, record)
        if problem:
            exclude(attempt, "all", problem)
            continue
        item = prepare_item(tokenizer, attempt, record, chain, batch_siblings(record, records_by_task[attempt["task"]]),
                            engines, wordings,
                            render_delta, saved)
        if attempt.get("primary", True) and not item["reasons"] and not item["final_note_present"]:
            # the sample admitted it because the R1 teacher gives it a final note; the job's must too
            item["reasons"]["final_note"] = "a primary attempt got no final note in the job"
            item["check1"] = False
        if item["reasons"]:
            exclude(attempt, "check1+coverage", "; ".join(f"{c}: {r}" for c, r in sorted(item["reasons"].items())))
        items.append(item)
    write_jsonl_gz(out / "precheck-contexts.jsonl.gz", saved)
    log(stage="precheck_contexts", check1=sum(i["check1"] for i in items), exclusions=len(exclusions))

    loop = asyncio.get_running_loop()
    worker = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="precheck-world")
    # every attempt with a note context may be continued (the wording is chosen after check 1)
    rebuilds = {i["attempt"]["episode_id"]: loop.run_in_executor(worker, world.rebuild, i["record"], day)
                for i in items if any(c in i["prefixes"] for c in conditions[1:])}

    # check 1 (ii): 8 samples per attempt and condition
    gate = asyncio.Semaphore(max(1, concurrency))
    samples = []

    async def draw(item, condition, k):
        prefix = item["prefixes"][condition]
        budget = min(CHECK1_MAX_TOKENS, ROLLOUT_LIMITS["context"] - len(prefix))
        key = f"{item['attempt']['episode_id']}|{condition}|{k}"
        async with gate:
            ids, finish = await generate(prefix, budget, request_seed(SEED_SALT, key))
        text = tokenizer.decode(ids, skip_special_tokens=True)
        try:
            calls, error = item["parse"](text), None
        except Exception as problem:  # noqa: BLE001 - a text the parser cannot read has no well-formed call
            calls, error = [], f"{type(problem).__name__}: {problem}"
        samples.append({"episode_id": item["attempt"]["episode_id"], "condition": condition, "k": k,
                        "tokens": len(ids), "finish": finish, "complete": finish == "stop",
                        "opener_in_text": opener in ids, "acts": well_formed_call(calls, item["tools"]),
                        "parse_error": error,
                        "calls": [{"name": (c.get("function") or {}).get("name"),
                                   "arguments": (c.get("function") or {}).get("arguments")} for c in calls],
                        "text": text})

    jobs = []
    for item in items:
        item["parse"], item["tools"] = parse_for(item["chain"]), tool_names(item["chain"])
        if item["check1"]:
            jobs += [draw(item, c, k) for c in conditions for k in range(CHECK1_SAMPLES)]
    await asyncio.gather(*jobs)
    write_jsonl_gz(out / "precheck-samples.jsonl.gz", sorted(samples, key=lambda r: (r["episode_id"],
                                                                                     r["condition"], r["k"])))
    by_sample = defaultdict(list)
    for row in samples:
        by_sample[row["episode_id"], row["condition"]].append(row)
    check1 = []
    for item in items:
        if not item["check1"]:
            continue
        row = {}
        for condition in conditions:
            drawn = by_sample[item["attempt"]["episode_id"], condition]
            row[condition] = {"teacher_tokens": len(item["prefixes"][condition]), "samples": len(drawn),
                              "acts": sum(r["acts"] for r in drawn),
                              "act_share": sum(r["acts"] for r in drawn) / len(drawn) if drawn else None,
                              "act_share_complete": (sum(r["acts"] and r["complete"] for r in drawn) / len(drawn)
                                                     if drawn else None),
                              "opener_sampled": sum(r["opener_in_text"] for r in drawn)}
        item["check1_row"] = row
        if item["attempt"].get("primary", True):
            check1.append(row)
    # the wording for check 2: highest mean act share over the primary attempts (the registered sample)
    sampled_verdict = check1_verdict([{**r, "final_note_present": False} for r in check1], conditions)
    chosen = sampled_verdict["best_wording"] or conditions[1]
    log(stage="precheck_check1_sampled", samples=len(samples), act_share=sampled_verdict["means"]["act_share"],
        chosen=chosen, elapsed=round(time.monotonic() - started))

    # check 2: the teacher with the chosen wording continues each attempt in its rebuilt world
    continuations = []

    async def run_in_world(function, *arguments):
        return await loop.run_in_executor(worker, function, *arguments)

    async def carry_on(item):
        attempt, eid = item["attempt"], item["attempt"]["episode_id"]
        try:
            built, report = await rebuilds[eid]
        except Exception as error:  # noqa: BLE001 - recorded, counted and excluded
            exclude(attempt, "check2", f"replay_error: {type(error).__name__}: {error}")
            return
        if chosen not in item["prefixes"] or chosen in item["reasons"]:
            exclude(attempt, "check2", f"{chosen}: {item['reasons'].get(chosen, 'no context')}")
            return
        if report["variant"] == "unmatched":
            exclude(attempt, "check2", "replay_variant_unmatched: the recorded initial world is neither setup variant")
            return
        if not report["verified"]:
            exclude(attempt, "check2", f"replay_mismatch: replayed {report['replayed']['partial_credit']}/"
                                       f"{report['replayed']['task_completed_correctly']}, recorded "
                                       f"{report['recorded']['partial_credit']}/"
                                       f"{report['recorded']['task_completed_correctly']}")
            return

        async def turn_sample(ids, budget, index):
            async with gate:
                return await generate(ids, budget, request_seed(SEED_SALT, f"{eid}|continue|{index}"))

        result = await continue_attempt(
            item["prefixes"][chosen], tokenizer=tokenizer, generate=turn_sample, parse=item["parse"],
            execute=lambda pairs: run_in_world(world.execute, built, pairs),
            render=lambda messages: render_delta(tokenizer, messages), turns_before=item["turn"] - 1,
            output_before=item["output_before"])
        after = await run_in_world(world.grade, built)
        before = report["assertion_results"]
        newly = must_not_newly_failing(before, after["assertion_results"])
        recorded_credit = report["recorded"]["partial_credit"]
        continuations.append({
            "episode_id": eid, "task": attempt["task"], "condition": chosen,
            "primary": attempt.get("primary", True), "side": attempt.get("side", False),
            "final_note_present": item["final_note_present"], "stop": result["stop"],
            "executed_calls": result["executed_calls"], "turns": result["turns"],
            "before": {**report["replayed"], "recorded": report["recorded"]},
            "after": {k: after[k] for k in ("partial_credit", "task_completed_correctly", "assertions_passed",
                                            "assertions_scored", "assertions_excluded")},
            "assertions_before": before, "assertions_after": after["assertion_results"],
            "assertion_changes": assertion_changes(before, after["assertion_results"]),
            "must_not_newly_failing": newly,
            "verdict": classify(recorded_credit, after["partial_credit"], newly),
            "replay": {k: report[k] for k in ("variant", "calls_replayed", "unanswered_calls", "assertions_equal",
                                              "result_text_diffs")}})
        log(stage="precheck_continued", episode=eid, stop=result["stop"], calls=result["executed_calls"],
            before=recorded_credit, after=after["partial_credit"], verdict=continuations[-1]["verdict"])

    await asyncio.gather(*(carry_on(i) for i in items if i["attempt"]["episode_id"] in rebuilds))
    worker.shutdown(wait=True)
    for item in items:
        if item["attempt"]["episode_id"] not in rebuilds:
            exclude(item["attempt"], "check2", "no note context: " + "; ".join(
                f"{c}: {r}" for c, r in sorted(item["reasons"].items()) if c != "no_note"))
    log(stage="precheck_check2_done", continuations=len(continuations), elapsed=round(time.monotonic() - started))

    # after all sampling (the rollout engine sleeps): check 1 (i) and coverage, both from the teacher adapter
    await suspend()
    rows1, rows_coverage = [], []
    for item in items:
        if not item["check1"]:
            continue
        attempt, row = item["attempt"], item["check1_row"]
        wanted_tokens = [opener] + ([item["recorded_first_token"]] if item["recorded_first_token"] is not None else [])
        for condition in conditions:
            scored = score_next(item["prefixes"][condition], wanted_tokens)
            row[condition].update({
                "p_opener": math_exp(scored[0][opener]) if scored else None,
                "logp_opener": scored[0][opener] if scored else None,
                "logp_recorded_first_token": (scored[0].get(item["recorded_first_token"])
                                              if scored and item["recorded_first_token"] is not None else None),
                "top": scored[1] if scored else None})
            if scored is None:
                exclude(attempt, "check1(i)", f"{condition}: out_of_memory")
        rows1.append({"episode_id": attempt["episode_id"], "task": attempt["task"], "final_turn": item["turn"],
                      "primary": attempt.get("primary", True), "side": attempt.get("side", False),
                      "final_note_present": item["final_note_present"], "final_note": item["final_note"],
                      "inserted_text": item["inserted"], "recorded_partial_credit": attempt["partial_credit"],
                      "solution_episode": item["contexts"]["no_note"].get("solution_episode"), **row})
        # coverage: the R0 teacher (no note) and the chosen-wording note teacher over the whole attempt
        student = student_sequence(item["chain"])
        entry = {"episode_id": attempt["episode_id"], "task": attempt["task"],
                 "primary": attempt.get("primary", True), "side": attempt.get("side", False)}
        for condition, note in (("no_note", False), (chosen, True)):
            context = item["contexts"][condition]
            links = coverage_tokens(tokenizer, student, context, note)
            needed = sorted({i for tokens in links.values() for i in tokens})
            teacher = teacher_sequence(tokenizer, item["chain"], context)
            if teacher["targets"] != student["targets"]:
                raise ValueError(f"teacher and student targets differ: {attempt['episode_id']}")
            top = top1(teacher["ids"], [teacher["positions"][i] for i in needed]) if needed else []
            if top is None:
                exclude(attempt, "coverage", f"{condition}: out_of_memory")
                entry[condition] = None
                continue
            entry[condition] = coverage(links, student["targets"], dict(zip(needed, top)))
            entry[condition]["linked_tokens"] = len(needed)
        rows_coverage.append(entry)
    primary = lambda rows: [r for r in rows if r["primary"]]
    side = lambda rows: [r for r in rows if r["side"]]
    verdict1 = check1_verdict(primary(rows1), conditions)

    def coverage_of(rows):
        summary = {}
        for condition in ("no_note", chosen):
            values = [r[condition] for r in rows if r.get(condition)]
            summary[condition] = {
                "attempts": len(values), "mean_share": mean(v["share"] for v in values if v["share"] is not None),
                "checks": sum(v["checks"] for v in values), "covered": sum(v["covered"] for v in values),
                "pooled_share": (sum(v["covered"] for v in values) / sum(v["checks"] for v in values)
                                 if sum(v["checks"] for v in values) else None)}
        return summary

    coverage_summary = coverage_of(primary(rows_coverage))
    result = {
        "definition": sample["definition"], "source": sample["source"], "sample_counts": sample["counts"],
        "wordings": wordings, "conditions": conditions,
        "plan": {"sample_path": plan.get("sample_path"), "wordings_path": plan.get("wordings_path"),
                 "folder": str(folder), "world_date": str(day), "env_commit": plan.get("env_commit"),
                 "engine_version": engine_version, "seed_salt": SEED_SALT, "rollout_limits": ROLLOUT_LIMITS,
                 "kept_tasks": sorted(keep_tasks) if keep_tasks else None},
        "opener": {"id": opener, "text": opener_text},
        "attempts": len(attempts), "exclusions": exclusions,
        "exclusion_counts": dict(Counter(f"{e['check']}: {e['reason'].split(':')[0]}" for e in exclusions)),
        "engine_stats": {c: dict(getattr(e, "stats", {})) for c, e in engines.items()},
        "check1": {**verdict1, "set": "primary: the registered sample (attempts the R1 teacher gives a final note)",
                   "side_pooled": {k: v for k, v in check1_verdict(side(rows1), conditions).items()
                                   if k not in ("passed_by_wording", "passed_any", "best_wording")},
                   "rows": rows1},
        "check2": {**check2_verdict(primary(continuations)), "wording": chosen,
                   "side_pooled": {k: v for k, v in check2_verdict(side(continuations)).items() if k != "passed"},
                   "wording_rule": "the note wording with the highest check-1 mean act share (ties: file order)",
                   "rows": sorted(continuations, key=lambda r: r["episode_id"])},
        "coverage": {"definition": "share of the attempt's failed required checks (engine facts of kind missing) "
                                   "with at least one trained token where the teacher's top-1 differs from the "
                                   "student's sampled token; trained tokens: call-body tokens of the call the check "
                                   f"is located at, plus (note condition) the first {COVERAGE_FINAL_TOKENS} tokens of "
                                   "the final reply for every item the final note lists; descriptive, no pass bar",
                     "conditions": ["no_note", chosen], "summary": coverage_summary,
                     "side_pooled": coverage_of(side(rows_coverage)), "rows": rows_coverage},
        "elapsed_s": time.monotonic() - started}
    write(out / "precheck.json", result)
    print(summary_text(result), flush=True)
    return result


def math_exp(value):
    import math

    return math.exp(value)


def summary_text(result):
    c1, c2, cov = result["check1"], result["check2"], result["coverage"]["summary"]
    means = c1["means"]
    lines = [f"precheck: {result['attempts']} sampled attempts; opener {result['opener']['text']!r} "
             f"(id {result['opener']['id']}); exclusions {len(result['exclusions'])}: {result['exclusion_counts']}",
             f"check 1 (stopping point), {c1['attempts']} attempts:"]
    for condition in result["conditions"]:
        verdict = c1["passed_by_wording"].get(condition)
        lines.append(f"  {condition}: act share {fmt(means['act_share'][condition])} (complete turns "
                     f"{fmt(means['act_share_complete'][condition])}), P(opener) {fmt(means['p_opener'][condition])}, "
                     f"with a final note only {fmt(means['with_final_note_only']['act_share'][condition])}"
                     + ("" if verdict is None else f" -> {'PASS' if verdict else 'FAIL'}"))
    lines.append(f"  best wording {c1['best_wording']}; any wording passes: {c1['passed_any']}")
    lines.append(f"check 2 (continuation, {c2['wording']}), {c2['attempts']} attempts: {c2['counts']} shares "
                 f"{c2['shares']} -> {'PASS' if c2['passed'] else 'FAIL'}")
    side1, side2 = c1["side_pooled"], c2["side_pooled"]
    lines.append(f"side (pooled over the earlier population draw, note-less attempts included; not a pass rule): check 1 "
                 f"{side1['attempts']} attempts, act share " + ", ".join(
                     f"{c} {fmt(side1['means']['act_share'][c])}" for c in result["conditions"])
                 + f"; check 2 {side2['attempts']} attempts, shares {side2['shares']}")
    lines.append("coverage (descriptive): " + "; ".join(
        f"{c}: mean share {fmt(v['mean_share'])}, pooled {v['covered']}/{v['checks']}" for c, v in cov.items()))
    return "\n".join(lines)


def fmt(value):
    return "n/a" if value is None else f"{value:.3f}"


def main(argv=None):
    parser = argparse.ArgumentParser(description="precheck sample (laptop)")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("select", help="write the precheck sample file from the recorded run")
    p.add_argument("--run-root", required=True, help=f"the evidence folder of {SOURCE_RUN}")
    p.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    from sdpo_ab_engine import missing_manifests

    population_tasks = sorted(population(Path(args.run_root) / "run" / SOURCE_LABEL)[0])
    missing = missing_manifests(population_tasks)
    if missing:
        raise SystemExit(f"task manifests do not load: {missing} (set ABCE_ENVIRONMENT)")
    root = Path(args.run_root)
    content = sample_file(root / "run" / SOURCE_LABEL, root)
    Path(args.out).write_text(json.dumps(content, indent=1) + "\n")
    print(json.dumps(content["counts"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
