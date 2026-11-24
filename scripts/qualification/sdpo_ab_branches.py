"""R1 (plan D74): teacher takeover branches at mistake points, added to R0's loss behind `--r1-branches`.

Without the flag nothing here runs and R0 is unchanged (test_sdpo_ab_branches proves the runner's R0 path equal to
69e41d13/3f2c6ad5 with every flag block removed).

Rules (D74, items 1-6; readings that D74 leaves open are marked READING and listed in the report):

1. Mistake points of a failed training attempt (success attempts give none):
   (a) "stop": the attempt's final reply makes no call while the R1 teacher's final note (engine final_note,
       not_done_source "benchmark") lists benchmark required work undone at that reply's turn;
   (b) "harm": a whole call located by the engine (a rendered fact of kind prohibited or changed, location at_call,
       turn and call index) whose existence breaks a must-not check. READING: confirmed on replay - in the world just
       before that call, executing the recorded call alone flips at least one must-not check (negative family) from
       passed to not passed (flip-aware: the excluded flag is ignored). An engine fact the replay does not confirm is
       dropped and counted.
   READING: an attempt whose replay does not reproduce its recorded partial credit and success gives no point.
2. At each point the teacher (the task, the located facts, the final note; arm "A" = no shown solution; R0's engine
   options + final_note) continues in the replayed world: temperature 1.0, top-p 1.0, the rollouts' turn and reply
   limits, at most 10 calls, writing text as it naturally does. At (a) it starts where the final reply began; at (b)
   just before the harmful call: the student's own text (and earlier calls) of that reply stay, the teacher continues
   the same reply from the harmful call's <tool_call> position.
3. Kept: the longest prefix of the continuation's executed calls, at most 3, in which every call succeeds (harness
   status "ok" and no tool error by the engine's own rule `abce.record.calls.tool_error_text`) and, replayed one at a
   time (each call executed alone and the world graded after it), breaks no check that passed before (flip-aware:
   READING, the strictest: passed at the branch point OR just before that call, and not passed after). The first call
   that fails either test ends the kept prefix. No kept call -> the point trains nothing.
4. Branch sequence (student): the student's own prefix, then the kept calls (token for token as the teacher sampled
   them, <tool_call> through </tool_call>, "\n" between calls of one reply), each reply closed by <|im_end|> and
   followed by the real tool results of its calls; the teacher's text is removed. The teacher's sequence is its own
   continuation with its text, cut after the last kept call. Trained: every kept call's opener and call words
   (after <tool_call> through </tool_call>, as R0's call words). Call words: the loss's top-20 + tail reverse KL toward
   the teacher. Opener: one-sided Bernoulli KL on P(<tool_call>) toward q = min(teacher P, cap), active only when q is
   above the student's P (may only rise; READING of "capped": the target probability is capped at --r1-opener-cap).
   Importance weight 1. The student's log-probability of every inserted token is logged.
5. Each point is one unit: its loss is the mean over its trained tokens (each opener one token); the branch term of an
   optimizer step (one task) is --r1-branch-weight x the mean over that task's points. R0's term is unchanged; the
   gradient clip (1.0) applies to the sum, as R0's to its term. `--r1-branch-dry-run` measures, at the untrained
   adapter on round 1, the gradient norm of each term per step and trains nothing (D74 item 5).
6. Everything else is R0.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter

BRANCH_MAX_CALLS = 10
BRANCH_KEEP = 3
BRANCH_SAMPLING = {"temperature": 1.0, "top_p": 1.0}
BRANCH_ARM = "A"  # the engine's solution="none": no shown solution
HARM_KINDS = ("prohibited", "changed")
STOP, HARM = "stop", "harm"
ROLLOUT_LIMITS = {"context": 65536, "reply": 4096, "max_turns": 50, "max_output_tokens": 65536}


# ------------------------------------------------------------------ engine and attempts

def make_branch_engine(expected_values, fix_hints, options, engine_class=None):
    """The R1 teacher's engine: R0's settings (values, fix hints, options) plus the final note; no solution pool (arm A
    never shows one)."""
    if engine_class is None:
        from sdpo_ab_engine import EngineContext as engine_class
    return engine_class(expected_values=expected_values, pool=False, fix_hints=list(fix_hints or []),
                        options={**dict(options or {}), "final_note": True})


def succeeded(record):
    metrics = record["traces"][0].get("metrics") or {}
    return bool(metrics.get("task_completed_correctly"))


def assistant_messages(record):
    from sdpo_ab_world import path_nodes

    (trace,) = record["traces"]
    return [n["message"] for n in path_nodes(trace["nodes"])]


def chain_problem(chain, record):
    """None when the recorded chain is usable for a branch (joined, one action per assistant message)."""
    if chain is None:
        return "chain_missing"
    if chain.get("mismatch"):
        return "chain_mismatch"
    actions = [s for s in chain["segments"] if s["kind"] == "action"]
    assistants = [m for m in assistant_messages(record) if m.get("role") == "assistant"]
    if not actions or len(actions) != len(assistants):
        return "chain_actions_differ_from_messages"
    return None


def call_index(record):
    """({(turn, k): n}, {(turn, k): call id}): n = answered recorded calls before that call (the index into
    sdpo_ab_world.recorded_calls), for every answered call; turns count assistant messages from 1."""
    path = assistant_messages(record)
    answered = {m.get("tool_call_id") for m in path if m.get("role") == "tool"}
    index, ids, n, turn = {}, {}, 0, 0
    for message in path:
        if message.get("role") != "assistant":
            continue
        turn += 1
        for k, call in enumerate(message.get("tool_calls") or []):
            ids[(turn, k)] = call.get("id")
            if call.get("id") in answered:
                index[(turn, k)] = n
                n += 1
    return index, ids


def find_points(chain, record, context, opener, kinds=(HARM, STOP)):
    """(points, dropped): the attempt's mistake points of the given kinds from the R1 teacher's context, in attempt order
    (harm points by turn and call, then the stop point); a kind left out is neither found nor counted as dropped. A point: {"kind", "turn", "call", "position" (index of its action segment),
    "cut" (action tokens the student keeps), "recorded_index" (harm: answered calls before it), "facts"}."""
    from sdpo_ab_runner import rendered_facts

    actions = [i for i, s in enumerate(chain["segments"]) if s["kind"] == "action"]
    index, ids = call_index(record)
    harms, dropped = {}, []
    for fact in rendered_facts(context) if HARM in kinds else ():
        if fact.get("kind") not in HARM_KINDS or fact.get("location") != "at_call":
            continue
        turn, k = fact.get("turn"), fact.get("call")
        where = {"kind": HARM, "turn": turn, "call": k, "fact": fact.get("id")}
        if not turn or k is None or turn > len(actions):
            dropped.append({**where, "reason": "fact_not_located"})
            continue
        segment = chain["segments"][actions[turn - 1]]
        call_ids = segment.get("tool_call_ids") or []
        if k >= len(call_ids):
            dropped.append({**where, "reason": "call_not_in_action"})
            continue
        if (turn, k) not in index:
            dropped.append({**where, "reason": "call_unanswered"})
            continue
        if ids.get((turn, k)) != call_ids[k]:
            dropped.append({**where, "reason": "call_id_differs"})
            continue
        openers = [i for i, token in enumerate(segment["ids"]) if token == opener]
        if len(openers) != len(call_ids):
            dropped.append({**where, "reason": "openers_differ_from_calls"})
            continue
        point = harms.setdefault((turn, k), {"kind": HARM, "turn": turn, "call": k, "position": actions[turn - 1],
                                             "cut": openers[k], "recorded_index": index[(turn, k)], "facts": []})
        point["facts"].append(fact.get("id"))
    points = [harms[key] for key in sorted(harms)]
    final, turn = actions[-1], len(actions)
    note = context.get("final_note") or {}
    if STOP in kinds and not chain["segments"][final].get("tool_call_ids"):
        if note.get("turn") == turn and note.get("items"):
            points.append({"kind": STOP, "turn": turn, "call": None, "position": final, "cut": 0,
                           "recorded_index": None, "facts": list(note["items"])})
        elif note.get("turn") not in (None, turn):
            dropped.append({"kind": STOP, "turn": turn, "call": None, "reason": "final_note_turn_differs"})
    return points, dropped


# ------------------------------------------------------------------ checks

def flips(before, after):
    """Checks that passed before and do not pass after (the `passed` field, whatever the excluded flag)."""
    if len(before) != len(after):
        raise ValueError("assertion lists differ in length")
    out = []
    for index, (a, b) in enumerate(zip(before, after)):
        if a.get("type") != b.get("type") or a.get("params") != b.get("params"):
            raise ValueError(f"assertion {index} differs in type or parameters")
        if a.get("passed") is True and b.get("passed") is not True:
            out.append({"index": index, "type": b.get("type")})
    return out


def must_not_flips(before, after):
    from sdpo_ab_metrics import negative_family

    return [f for f in flips(before, after) if negative_family(str(f["type"]))]


def call_succeeded(content, status):
    """Harness status ok and no tool error by the engine's own rule."""
    if status != "ok":
        return False
    from sdpo_ab_engine import load_engine

    load_engine()
    from abce.record.calls import tool_error_text

    return tool_error_text(content) is None


def keep_decision(calls, keep=BRANCH_KEEP):
    """Indices (into `calls`, in execution order) of the kept calls: the longest prefix, at most `keep`, whose calls
    all succeeded, broke nothing and have a token span. Each call: {"success", "broken", "span"}."""
    kept = []
    for index, call in enumerate(calls):
        if len(kept) >= keep or not call["success"] or call["broken"] or call.get("span") is None:
            break
        kept.append(index)
    return kept


# ------------------------------------------------------------------ the continuation

def call_spans(ids, opener, closer):
    """[(start, end)] of each <tool_call> ... </tool_call> in `ids` (end exclusive); None when unbalanced."""
    spans, start = [], None
    for i, token in enumerate(ids):
        if token == opener:
            if start is not None:
                return None
            start = i
        elif token == closer:
            if start is None:
                return None
            spans.append((start, i + 1))
            start = None
    return None if start is not None else spans


async def continue_branch(prefix, *, tokenizer, generate, parse, run_one, render, base_assertions, opener, closer,
                          turns_before, output_before, partial=0, pending=(), max_calls=BRANCH_MAX_CALLS,
                          keep=BRANCH_KEEP, limits=None, succeeded_fn=None):
    """The teacher continues from `prefix` as the rollout's agent would (`sdpo_ab_precheck.continue_attempt`'s loop),
    with every call executed alone and the world graded after it (`run_one(name, arguments) -> (content, status,
    assertion results)`). `partial`: tokens of the current reply already in `prefix` (harm points; they count toward
    the reply limit); `pending`: tool messages of the student's earlier calls of that reply, rendered with the first
    turn's results. Returns turns (generated ids, text, calls with results, spans, rendered results), the executed
    calls in order, the kept indices and the stop reason."""
    limits = {**ROLLOUT_LIMITS, **(limits or {})}
    succeeded_fn = succeeded_fn or call_succeeded
    tokens, turns, executed, output, stop = list(prefix), [], [], output_before, None
    previous = base_assertions
    eos = tokenizer.eos_token_id
    while stop is None:
        if turns_before + len(turns) >= limits["max_turns"]:
            stop = "max_turns"
            break
        if output >= limits["max_output_tokens"]:
            stop = "max_output_tokens"
            break
        reply = limits["reply"] - (partial if not turns else 0)
        budget = min(reply, limits["context"] - len(tokens))
        if budget < 1:
            ids, finish = [], "context_limit"
        else:
            ids, finish = await generate(tokens, budget, len(turns))
        complete = finish == "stop"
        limit = None if complete else ("context_limit" if finish == "context_limit"
                                       else "reply_limit" if budget == reply else "context_limit")
        raw = tokenizer.decode(ids, skip_special_tokens=True)
        calls = parse(raw) if complete else []
        start = len(tokens)
        tokens += ids
        if ids and ids[-1] != eos:
            tokens.append(eos)
        output += len(ids)
        spans = call_spans(ids, opener, closer) if calls else []
        spans_ok = spans is not None and len(spans) == len(calls)
        turn = {"start": start, "ids": list(ids), "tokens": len(ids), "finish": finish, "limit": limit, "text": raw,
                "calls": [], "spans_ok": spans_ok, "rendered": None}
        turns.append(turn)
        if not calls:
            stop = "final_reply" if complete else limit
            break
        room = max_calls - len(executed)
        messages = [dict(m) for m in pending] if len(turns) == 1 else []
        for k, call in enumerate(calls):
            name, arguments = call["function"]["name"], call["function"]["arguments"]
            if k >= room:
                turn["calls"].append({"id": call["id"], "name": name, "arguments": arguments, "executed": False,
                                      "reason": "call_cap"})
                continue
            content, status, after = await run_one(name, arguments)
            broken = {f["index"]: f for f in flips(previous, after) + flips(base_assertions, after)}
            broken = [broken[i] for i in sorted(broken)]
            previous = after
            entry = {"id": call["id"], "name": name, "arguments": arguments, "executed": True, "status": status,
                     "result": content, "success": succeeded_fn(content, status), "broken": broken,
                     "turn": len(turns) - 1, "k": k, "span": spans[k] if spans_ok else None}
            turn["calls"].append(entry)
            executed.append(entry)
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": content, "name": name})
        turn["rendered"] = render(messages)
        if len(executed) >= max_calls:
            stop = "call_cap"
            break
        tokens += turn["rendered"]
    kept = keep_decision(executed, keep)
    for index, entry in enumerate(executed):
        entry["kept"] = index in kept
    return {"turns": turns, "executed": executed, "kept": kept, "stop": stop, "tokens": tokens}


# ------------------------------------------------------------------ sequences

def branch_sequences(student_prefix, teacher_prefix, continuation, opener, eos, newline):
    """The student's and the teacher's sequences of one branch, or None without kept calls. Targets are the kept
    calls' tokens (opener included); `openers` lists the indices (into targets) of the <tool_call> tokens."""
    executed, kept = continuation["executed"], continuation["kept"]
    if not kept:
        return None
    chosen = [executed[i] for i in kept]
    last_turn = chosen[-1]["turn"]
    turns = continuation["turns"]
    student, teacher = list(student_prefix), list(teacher_prefix)
    s_pos, t_pos, targets, openers = [], [], [], []
    for index in range(last_turn + 1):
        turn = turns[index]
        here = [c for c in chosen if c["turn"] == index]
        if index < last_turn and len(here) != sum(c["executed"] for c in turn["calls"]):
            raise ValueError("a turn before the last kept call has calls that were not kept")
        start = len(teacher)
        end = here[-1]["span"][1] if index == last_turn else len(turn["ids"])
        teacher += turn["ids"][:end]
        for j, call in enumerate(here):
            if j:
                student += newline
            a, b = call["span"]
            for offset in range(a, b):
                token = turn["ids"][offset]
                if token == opener:
                    openers.append(len(targets))
                s_pos.append(len(student) - 1)
                t_pos.append(start + offset - 1)
                targets.append(token)
                student.append(token)
        if index < last_turn:
            if not turn["ids"] or turn["ids"][-1] != eos:
                teacher.append(eos)
            student.append(eos)
            teacher += turn["rendered"]
            student += turn["rendered"]
    for p, q, token in zip(s_pos, t_pos, targets):
        if student[p + 1] != token or teacher[q + 1] != token:
            raise ValueError("branch target misaligned")
    return {"student": {"ids": student, "positions": s_pos, "targets": targets},
            "teacher": {"ids": teacher, "positions": t_pos, "targets": list(targets)},
            "openers": openers, "opener": opener, "calls": len(chosen), "tokens": len(targets)}


def student_prefix(chain, position, cut):
    ids = []
    for segment in chain["segments"][:position]:
        ids += segment["ids"]
    return ids + list(chain["segments"][position]["ids"][:cut])


def pending_messages(chain, position, k):
    """The tool messages of the student's first k calls of the action at `position` (harm points with k > 0)."""
    if k == 0:
        return []
    call_ids = chain["segments"][position].get("tool_call_ids") or []
    following = chain["segments"][position + 1] if position + 1 < len(chain["segments"]) else None
    messages = [m for m in ((following or {}).get("messages") or []) if m.get("role") == "tool"]
    found = {m.get("tool_call_id"): m for m in messages}
    if any(c not in found for c in call_ids[:k]):
        return None
    return [found[c] for c in call_ids[:k]]


# ------------------------------------------------------------------ building every branch of a batch

async def build_branches(records, joined, *, tokenizer, engine, generate, parse_for, world, day, run_in_world,
                         siblings_of, seed_for, log=None, concurrency=32, keep_tasks=None, on_point=None,
                         point_kinds=(HARM, STOP)):
    """Every mistake point of the batch's failed attempts, its continuation and its sequences.

    generate(ids, max_tokens, seed) -> (ids, finish): one teacher sample (temperature 1.0, top-p 1.0); parse_for(chain)
    -> parse(text) -> calls; world: sdpo_ab_world (or a stand-in); run_in_world(function, *args): awaits it in the
    world worker; siblings_of(record): the attempt's batch; seed_for(key) -> seed.
    on_point (the check before training, sdpo_ab_r1check; None in training): awaited once per continued point with
    record, point, built (the replayed world), snapshots, recorded (calls), result (the continuation), row and
    sequences (None without a kept call); it may add keys to the row.
    Returns {"branches": {task: [branch]}, "rows": [one row per point or drop], "counts": Counter}."""
    import asyncio

    from sdpo_ab_engine import load_engine  # noqa: F401 - import check before any work
    from sdpo_ab_precheck import tool_call_opener
    from sdpo_ab_runner import engine_teacher_prefix, render_delta, task_of

    log = log or (lambda **fields: None)
    opener, _ = tool_call_opener(tokenizer)
    closer = tokenizer.convert_tokens_to_ids("</tool_call>")
    eos, newline = tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)
    rows, counts, jobs = [], Counter(), []
    gate = asyncio.Semaphore(max(1, concurrency))

    def drop(record, reason, **extra):
        counts["drop:" + reason.split(":")[0]] += 1
        rows.append({"episode_id": record["id"], "task": task_of(record), "dropped": reason, **extra})

    for record in records:
        if keep_tasks is not None and task_of(record) not in keep_tasks:
            continue
        if succeeded(record):
            continue
        counts["failed_attempts"] += 1
        chain = joined.get(record["id"])
        problem = chain_problem(chain, record)
        if problem:
            drop(record, problem)
            continue
        try:
            context = engine(record, siblings_of(record), BRANCH_ARM, "both")
        except ValueError as error:
            drop(record, f"untrue: {error}")
            continue
        except Exception as error:  # noqa: BLE001 - an engine crash excludes the attempt, as in training
            drop(record, f"builder_error: {type(error).__name__}: {error}")
            continue
        if context is None:
            drop(record, "no_context")
            continue
        points, dropped = find_points(chain, record, context, opener, point_kinds)
        for entry in dropped:
            counts["drop_point:" + entry["reason"]] += 1
            rows.append({"episode_id": record["id"], "task": task_of(record), "dropped": entry["reason"], **entry})
        if not points:
            counts["attempts_without_point"] += 1
            continue
        counts["attempts_with_point"] += 1
        jobs.append((record, chain, context, points))

    async def attempt(record, chain, context, points):
        before = {p["recorded_index"] for p in points if p["kind"] == HARM}
        try:
            built, report, snapshots = await run_in_world(world.rebuild_with_snapshots, record, day, before)
        except Exception as error:  # noqa: BLE001
            for p in points:
                drop(record, f"replay_error: {type(error).__name__}: {error}", kind=p["kind"], turn=p["turn"])
            return
        if report["variant"] == "unmatched" or not report["verified"]:
            for p in points:
                drop(record, "replay_mismatch", kind=p["kind"], turn=p["turn"], call=p["call"])
            return
        from sdpo_ab_world import recorded_calls

        recorded, _ = recorded_calls(record)
        await asyncio.gather(*(one_point(record, chain, context, point, built, snapshots, recorded)
                               for point in points))

    async def one_point(record, chain, context, point, built, snapshots, recorded):
        where = {"kind": point["kind"], "turn": point["turn"], "call": point["call"], "facts": point["facts"]}
        if point["kind"] == HARM:
            n = point["recorded_index"]
            call = recorded[n]
            arguments = call["arguments"] if call["arguments"] is None or isinstance(call["arguments"], str) \
                else json.dumps(call["arguments"])

            def verify():
                probe = world.fork(built, snapshots[n])
                before = world.grade(probe)["assertion_results"]
                world.execute(probe, [(call["tool"], arguments)])
                return must_not_flips(before, world.grade(probe)["assertion_results"])

            broken = await run_in_world(verify)
            if not broken:
                drop(record, "not_harmful_on_replay", **where)
                return
            where["must_not_broken"] = broken
            start = await run_in_world(world.fork, built, snapshots[n])
        else:
            start = await run_in_world(world.fork, built)
        pending = pending_messages(chain, point["position"], point["call"] or 0)
        if pending is None:
            drop(record, "pending_results_missing", **where)
            return
        cut_ids = list(chain["segments"][point["position"]]["ids"][:point["cut"]])
        teacher_prefix = engine_teacher_prefix(tokenizer, chain, point["position"], context) + cut_ids
        if len(teacher_prefix) >= ROLLOUT_LIMITS["context"]:
            drop(record, "teacher_prefix_over_context_limit", **where)
            return
        prefix_student = student_prefix(chain, point["position"], point["cut"])
        base = (await run_in_world(world.grade, start))["assertion_results"]
        actions = [s for s in chain["segments"][:point["position"]] if s["kind"] == "action"]
        key = f"{record['id']}|{point['kind']}|{point['turn']}|{point['call']}"

        async def turn_sample(ids, budget, index):
            async with gate:
                return await generate(ids, budget, seed_for(f"{key}|{index}"))

        async def run_one(name, arguments):
            def go():
                ((content, status),) = world.execute(start, [(name, arguments)])
                return content, status, world.grade(start)["assertion_results"]
            return await run_in_world(go)

        result = await continue_branch(
            teacher_prefix, tokenizer=tokenizer, generate=turn_sample, parse=parse_for(chain), run_one=run_one,
            render=lambda messages: render_delta(tokenizer, messages), base_assertions=base, opener=opener,
            closer=closer, turns_before=point["turn"] - 1,
            output_before=sum(len(a["ids"]) for a in actions) + len(cut_ids), partial=len(cut_ids), pending=pending)
        sequences = branch_sequences(prefix_student, teacher_prefix, result, opener, eos, newline)
        row = {"episode_id": record["id"], "task": task_of(record), **where, "stop": result["stop"],
               "executed_calls": len(result["executed"]), "kept": len(result["kept"]),
               "calls": [{k: c.get(k) for k in ("name", "status", "success", "broken", "kept", "turn", "k")}
                         for c in result["executed"]],
               "first_turn_words": len(result["turns"][0]["text"].split()) if result["turns"] else 0,
               "trained_tokens": sequences["tokens"] if sequences else 0,
               "openers": len(sequences["openers"]) if sequences else 0,
               "student_tokens": len(sequences["student"]["ids"]) if sequences else None,
               "teacher_tokens": len(sequences["teacher"]["ids"]) if sequences else None,
               "solution_shown": bool(context.get("solution_episode"))}
        rows.append(row)
        counts[f"points:{point['kind']}"] += 1
        counts[f"trained_points:{point['kind']}"] += bool(sequences)
        log(stage="r1_branch", episode=record["id"], kind=point["kind"], turn=point["turn"], stop=result["stop"],
            calls=len(result["executed"]), kept=len(result["kept"]))
        if on_point is not None:
            await on_point(record=record, point=point, built=built, snapshots=snapshots, recorded=recorded,
                           result=result, row=row, sequences=sequences)
        if sequences:
            sequences.update(episode_id=record["id"], task=task_of(record), kind=point["kind"], turn=point["turn"],
                             call=point["call"], row=row)
            branches.setdefault(task_of(record), []).append(sequences)

    branches = {}
    await asyncio.gather(*(attempt(*job) for job in jobs))
    return {"branches": branches, "rows": rows, "counts": counts}


def summarize(rows):
    """Per kind: points, trained points, kept calls, stop reasons, drops (for branches.json and the log)."""
    out = {}
    for kind in (STOP, HARM):
        mine = [r for r in rows if r.get("kind") == kind and "dropped" not in r]
        out[kind] = {"points": len(mine), "trained": sum(r["kept"] > 0 for r in mine),
                     "kept_calls": sum(r["kept"] for r in mine),
                     "mean_kept": statistics.mean([r["kept"] for r in mine]) if mine else None,
                     "stops": dict(Counter(r["stop"] for r in mine)),
                     "first_call_kept_share": (sum(bool(r["calls"]) and bool(r["calls"][0]["kept"]) for r in mine)
                                               / len(mine)) if mine else None}
    out["drops"] = dict(Counter(r["dropped"].split(":")[0] for r in rows if "dropped" in r))
    return out


# ------------------------------------------------------------------ loss (reference math on floats; torch below)

def opener_target(log_p_teacher, cap):
    """q = min(teacher P(<tool_call>), cap)."""
    return min(math.exp(log_p_teacher), cap)


def opener_active(log_p_student, q):
    """One-sided: the student's P may only rise toward q."""
    return q > math.exp(log_p_student)


def opener_reference(log_p_student, log_p_teacher, cap):
    """KL(Bernoulli(p_student) || Bernoulli(q)) when active, else 0 (float reference of `opener_term`)."""
    from sdpo_ab_runner import bernoulli_kl

    q = opener_target(log_p_teacher, cap)
    if not opener_active(log_p_student, q):
        return 0.0
    return bernoulli_kl(log_p_student, math.log(q))


def point_mean(body_losses, opener_losses):
    """A point's loss: the mean over its trained tokens (each opener one token)."""
    values = list(body_losses) + list(opener_losses)
    return sum(values) / len(values)


def branch_term(point_losses, weight):
    """The step's branch term: weight x the mean over the step's points (each point one unit)."""
    point_losses = list(point_losses)
    return weight * sum(point_losses) / len(point_losses) if point_losses else 0.0


def opener_term(s_logits, t_logits, opener, cap):
    """Torch: per opener row, the one-sided capped Bernoulli KL (differentiable in the student's logits)."""
    import torch

    log_s, log_t = s_logits.float().log_softmax(-1), t_logits.float().log_softmax(-1)
    lp_s, lp_t = log_s[:, opener], log_t[:, opener].detach()
    q = lp_t.exp().clamp(max=cap)
    p_s = lp_s.exp()
    l1_s = torch.log1p(-p_s.clamp(max=1 - 1e-7))
    kl = p_s * (lp_s - q.log()) + (1 - p_s) * (l1_s - torch.log1p(-q))
    active = (q > p_s).detach()
    return kl * active.float(), {"p_student": p_s.detach().tolist(), "p_teacher": lp_t.exp().tolist(),
                                 "target": q.tolist(), "active": active.tolist()}


def branch_point_loss(model, branch, cap, topk=20, chunk=1024):
    """One point's loss (mean over its trained tokens; body: top-20 + tail reverse KL, importance weight 1; openers:
    `opener_term`) and its stats, including the student's log-probability of every inserted token."""
    import torch
    from torch.utils.checkpoint import checkpoint
    from trl.experimental.sdpo.loss_utils import compute_topk_self_distillation_loss

    from sdpo_ab_runner import inner_hidden, lm_head

    student, teacher = branch["student"], branch["teacher"]
    with torch.no_grad():
        model.set_adapter("sdpo_teacher")
        model.eval()
        ht = inner_hidden(model, teacher["ids"], teacher["positions"]).detach()
    model.set_adapter("default")
    model.train()
    hs = inner_hidden(model, student["ids"], student["positions"])
    head = lm_head(model)
    targets = torch.tensor(student["targets"], device=hs.device)
    openers = set(branch["openers"])
    body = [i for i in range(len(student["targets"])) if i not in openers]

    def body_chunk(h_s, h_t):
        return compute_topk_self_distillation_loss(head(h_s).float(), head(h_t).float().detach(),
                                                   distillation_topk=topk, distillation_alpha=1.0,
                                                   distillation_add_tail=True).sum()

    total = hs.new_zeros((), dtype=torch.float32)
    if body:
        index = torch.tensor(body, device=hs.device)
        hb, tb = hs.index_select(0, index), ht.index_select(0, index)
        for s in range(0, len(body), chunk):
            total = total + checkpoint(body_chunk, hb[s:s + chunk], tb[s:s + chunk], use_reentrant=False)
    stats = {}
    if openers:
        index = torch.tensor(sorted(openers), device=hs.device)
        terms, stats = opener_term(head(hs.index_select(0, index)), head(ht.index_select(0, index)),
                                   branch["opener"], cap)
        total = total + terms.sum()
        stats["opener_loss"] = float(terms.detach().sum())
    with torch.no_grad():
        logp = torch.cat([head(hs[s:s + chunk].detach()).float().log_softmax(-1).gather(
            -1, targets[s:s + chunk, None]).flatten() for s in range(0, len(student["targets"]), chunk)])
    loss = total / len(student["targets"])
    stats.update(tokens=len(student["targets"]), opener_tokens=len(openers),
                 student_logp_inserted_mean=float(logp.mean()), student_p_inserted_mean=float(logp.exp().mean()),
                 student_logp_inserted_min=float(logp.min()))
    return loss, stats


def branch_parity(generator, cap=0.5):
    """`opener_term` against `opener_reference` on random logits (both sides of the cap and of the student's P);
    returns the largest absolute difference."""
    import torch

    opener = 3
    s, t = torch.randn(12, 20, generator=generator) * 3, torch.randn(12, 20, generator=generator) * 3
    t[:4, opener] += 8.0  # teacher above the cap
    s[4:8, opener] += 8.0  # student above the teacher (inactive)
    terms, _ = opener_term(s, t, opener, cap)
    log_s, log_t = s.log_softmax(-1)[:, opener].tolist(), t.log_softmax(-1)[:, opener].tolist()
    reference = [opener_reference(a, b, cap) for a, b in zip(log_s, log_t)]
    return max(abs(float(x) - y) for x, y in zip(terms, reference))


def branch_parity_check(output, cap):
    """Run in the GPU job before training (flag only): writes parity-branches.json, refuses on failure."""
    import torch

    from sdpo_ab_runner import write

    difference = branch_parity(torch.Generator().manual_seed(20261105), cap)
    result = {"opener_vs_reference": difference, "passed": difference < 1e-5, "cap": cap}
    write(output / "parity-branches.json", result)
    if not result["passed"]:
        raise ValueError(f"branch parity failed: {result}")
    return result


def weight_suggestion(steps):
    """D74 item 5 from the dry run's per-step gradient norms ({"r0_norm", "branch_norm"}, both terms present): the
    weight at which the branch term's norm is at most R0's, by two summaries; the smaller is the proposal."""
    both = [s for s in steps if s.get("r0_norm") and s.get("branch_norm")]
    if not both:
        return {"steps": 0, "proposal": None}
    ratio_of_means = statistics.mean(s["r0_norm"] for s in both) / statistics.mean(s["branch_norm"] for s in both)
    median_ratio = statistics.median(s["r0_norm"] / s["branch_norm"] for s in both)
    return {"steps": len(both), "r0_norm_mean": statistics.mean(s["r0_norm"] for s in both),
            "branch_norm_mean": statistics.mean(s["branch_norm"] for s in both),
            "ratio_of_means": ratio_of_means, "median_ratio": median_ratio,
            "proposal": min(ratio_of_means, median_ratio),
            "rule": "weight = min(mean R0 norm / mean branch norm, median per-step ratio): the branch term's gradient "
                    "norm at most equal to R0's (D74 item 5); recorded in the plan before launch"}


# ------------------------------------------------------------------ CPU dry run (laptop; no model)

def render_calls_text(calls):
    """Calls in the model's own format (Qwen3.5: <tool_call><function=...><parameter=...>), "\\n" between calls."""
    blocks = []
    for name, arguments in calls:
        values = json.loads(arguments) if isinstance(arguments, str) and arguments.strip() else (arguments or {})
        body = "".join(f"<parameter={k}>\n{v if isinstance(v, str) else json.dumps(v)}\n</parameter>\n"
                       for k, v in values.items())
        blocks.append(f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>")
    return "\n".join(blocks)


def parse_calls_text(text):
    """A stand-in for the rollout's parser on that format (values read as JSON where they parse)."""
    import re

    out = []
    for i, block in enumerate(re.findall(r"<tool_call>(.*?)</tool_call>", text, re.S)):
        name = re.search(r"<function=([^>\n]+)>", block)
        if not name:
            continue
        values = {}
        for key, value in re.findall(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", block, re.S):
            try:
                values[key] = json.loads(value)
            except ValueError:
                values[key] = value
        out.append({"id": f"mock-{i}", "type": "function",
                    "function": {"name": name.group(1), "arguments": json.dumps(values)}})
    return out


def dry_run(sample_path, evidence, tokenizer_path, attempts=6, out=None, seed="r1-branch-dry-run"):
    """Mistake points, replays, continuations and branch sequences for a few recorded failed attempts, on CPU with
    the real engine, tokenizer and replayed worlds. The teacher is mocked: its continuation re-issues, one call per
    turn after a line of text, the calls of a successful attempt of the same task (recorded runs) that the student
    had not made (by name and arguments), then ends with a short final reply; with no success available it only
    writes a final reply (the point then trains nothing)."""
    import asyncio
    import random
    import time
    from pathlib import Path

    from transformers import AutoTokenizer

    import sdpo_ab_refine_check as rc
    import sdpo_ab_world as world
    from sdpo_ab_runner import task_of

    started = time.monotonic()
    sample = json.loads(Path(sample_path).read_text())
    folders = {label: Path(evidence) / s["run"] / "run" / label for label, s in sample["sources"].items()}
    plan = rc.preflight(sample_path, folders)
    pool = [a for a in sample["attempts"] if not a["task_fixed_after_recording"]]
    random.Random(seed).shuffle(pool)
    chosen, tasks = [], set()
    for attempt in pool:  # distinct tasks
        if attempt["task"] not in tasks:
            chosen.append(attempt)
            tasks.add(attempt["task"])
        if len(chosen) == attempts:
            break
    plan["sample"] = {**sample, "attempts": chosen}
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    loaded = time.monotonic()
    picked, records, labels, chains = rc.load_sources(plan, tasks)
    load_s = time.monotonic() - loaded
    engine = make_branch_engine(True, ["call_fix"], rc.R0_OPTIONS)
    by_id = {r["id"]: r for rs in records.values() for r in rs}
    targets = [by_id[a["episode_id"]] for a in picked]
    successes = {t: [r for r in rs if succeeded(r)] for t, rs in records.items()}
    from sdpo_ab_world import recorded_calls

    plans = {}
    for record in targets:
        done = {(c["tool"], json.dumps(c["arguments"]) if not isinstance(c["arguments"], str) else c["arguments"])
                for c in recorded_calls(record)[0]}
        mocks = []
        for success in successes.get(task_of(record), [])[:1]:
            for c in recorded_calls(success)[0]:
                arguments = c["arguments"] if isinstance(c["arguments"], str) else json.dumps(c["arguments"])
                if (c["tool"], arguments) not in done:
                    mocks.append((c["tool"], arguments))
        plans[record["id"]] = {"calls": mocks[:BRANCH_MAX_CALLS],
                               "success": successes[task_of(record)][0]["id"] if successes.get(task_of(record)) else None}

    async def generate(ids, budget, key):
        episode, *_rest, index = key.split("|")
        calls = plans[episode]["calls"]
        index = int(index)
        if index < len(calls):
            text = f"I still need to do this step ({calls[index][0]}).\n\n" + render_calls_text([calls[index]])
        else:
            text = "All required work is now done."
        return tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id], "stop"

    timings = []

    async def run_in_world(function, *arguments):
        t0 = time.monotonic()
        try:
            return await asyncio.to_thread(function, *arguments)
        finally:
            timings.append((getattr(function, "__name__", "call"), time.monotonic() - t0))

    def siblings_of(record):
        key = rc.batch_key(record, labels[record["id"]])
        return [r for r in records[task_of(record)] if rc.batch_key(r, labels[r["id"]]) == key]

    built = time.monotonic()
    result = asyncio.run(build_branches(
        targets, {a["episode_id"]: chains.get(a["episode_id"]) for a in picked}, tokenizer=tokenizer, engine=engine,
        generate=generate, parse_for=lambda chain: parse_calls_text, world=world, day=plan["day"],
        run_in_world=run_in_world, siblings_of=siblings_of, seed_for=lambda key: key, concurrency=4))
    build_s = time.monotonic() - built
    examples = []
    for task, items in sorted(result["branches"].items()):
        for b in items[:1]:
            s, t = b["student"], b["teacher"]
            first = s["positions"][0] + 1
            examples.append({
                "episode_id": b["episode_id"], "kind": b["kind"], "turn": b["turn"], "call": b["call"],
                "student_tokens": len(s["ids"]), "teacher_tokens": len(t["ids"]), "trained_tokens": b["tokens"],
                "openers": len(b["openers"]),
                "student_at_junction": tokenizer.decode(s["ids"][max(0, first - 40):first + 40]),
                "teacher_before_first_kept_call": tokenizer.decode(t["ids"][max(0, t["positions"][0] - 60):
                                                                            t["positions"][0] + 1]),
                "targets_equal": s["targets"] == t["targets"]})
    per_function = {}
    for name, seconds in timings:
        entry = per_function.setdefault(name, {"calls": 0, "seconds": 0.0})
        entry["calls"] += 1
        entry["seconds"] += seconds
    report = {"attempts": [{"episode_id": a["episode_id"], "task": a["task"], "source": a["source"],
                            "partial_credit": a["partial_credit"], "mock_teacher_calls": len(plans[a["episode_id"]]
                                                                                              ["calls"]),
                            "mock_from_success": plans[a["episode_id"]]["success"]} for a in picked],
              "counts": dict(result["counts"]), "summary": summarize(result["rows"]), "rows": result["rows"],
              "examples": examples, "engine_stats": dict(engine.stats),
              "seconds": {"load_records": load_s, "build_branches": build_s, "total": time.monotonic() - started,
                          "world": per_function}}
    if out:
        Path(out).write_text(json.dumps(report, indent=1, default=str) + "\n")
    return report


def main(argv=None):
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="R1 branches: CPU dry run on recorded failed attempts (no model)")
    parser.add_argument("--sample", required=True, help="refine-check-sample.json (failed attempts of training tasks)")
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--attempts", type=int, default=6)
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    report = dry_run(args.sample, args.evidence, args.tokenizer, args.attempts, args.out)
    print(json.dumps({k: report[k] for k in ("attempts", "counts", "summary", "seconds")}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
