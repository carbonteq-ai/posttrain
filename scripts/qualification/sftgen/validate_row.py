"""Per-row gates for Route B SFT rows; every gate is reported separately for every row.

Gates (a row is admitted only if every gate passes; `call_numbers_shown` is advisory and `final_reply` review flags
never fail a row):
  structure            system, user, then turns of (assistant with tool_calls, one tool message per call with matching
                       ids, in order), then one final assistant message; trainable = exactly the assistant messages
  teacher_request_exact (teacher-in-runner rows) episode_rows' cross-check passed: the proxy saw a request with this
                       system, user and byte-equal tools, and every tool message answers the preceding calls in order
  prompt_exact         (with --runner-prompts) messages[:2] and tools byte-equal to the runner's first prompt
  replay_score         abtool2 replay of the call list at the pinned environment (bd655be, checked: commit and a clean
                       checkout): task_completed_correctly 1.0 and every assertion row passed (excluded rows
                       included), as J-sft data/admit.py. World: the row's world_date/fresh_setup; rows without them
                       (the runner's own episodes) replay the later-rollout world, then the first-rollout world, and
                       use the one whose results match (named in the report)
  results_equal_replay every tool message is byte-equal to the replayed result, same count
  note_before_calls    every calling assistant turn has non-empty visible text
  grounding            (thinking: reasoning so far, incl. its correct arithmetic, counts as prior context)
                       every number in a note or the reply is in the prior context (system, user, earlier notes, earlier
                       call arguments, tool results; not the tool schemas) or results from arithmetic written in that
                       same note/reply (evaluated here, chains a = b = c pair by pair, intermediate results count; a
                       wrong expression fails). A percentage grounds its fraction and back ("6%" <-> 0.06); ordered-list
                       markers at a line start ("1. ") are layout and not checked; ISO dates are never arithmetic
  no_harness_words     no word that reveals the generation harness (rowlib.HARNESS_WORDS / LITERALS) in a note or the
                       reply (thinking: or the reasoning), unless the task's own visible context (incl. the tool
                       definitions) uses it
  final_reply          last message is assistant, no tool calls, non-empty, no tool markup
  reply_names_no_skipped_items  no phrase that names skipped/excluded items in the reply (system prompt: "list only
                       items you acted on"); failing since 13 Nov (plan v4: "the reply follows the system prompt")
  zero_tool_errors     no TRAINED failed call (13 Nov: a failed call may stay in a row only in a masked, untrained
                       turn listed in metadata.masked_turns); no result matching the runner's tool_error_text and no
                       non-ok harness status (teacher
                       trajectories with any errored call are rejected; the count is reported)
  (no call cap)        the benchmark limits turns (50, enforced by the runner and checked by `length`), not calls;
                       the Phase 1 45-call cap (one call per turn there) was dropped 12 Nov (Ali); calls are reported
                       in no_identical_call_5
  no_identical_call_5  no identical call (name + sorted arguments) 5 or more times
  length               rendered row <= 32,768 tokens and every assistant message <= 32,768 tokens (the runner's reply
                       limit) and at most 50 assistant messages, with the real Qwen3.5-4B tokenizer and chat template,
                       thinking off
  call_numbers_shown   (advisory) every number inside a call's text arguments is in the prior context or in the same
                       turn's note (computed numbers must be shown before the call); count-like keys (max/limit/page)
                       are ignored

Run with the environment interpreter (replay needs automationbench_v1 at bd655be; tokenizer via transformers):
  CAPPED_MEM=4G .../bin/capped .../sdpo-work/tools/pyfix/python validate_row.py ROWS.jsonl --report REPORT.json \
      [--tokenizer DIR] [--runner-prompts runner_prompts.jsonl] [--no-replay] [--admitted-out ADMITTED.jsonl]
Exit 0 only if every row passes every non-advisory gate.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from rowlib import (IDENTICAL_CALL_LIMIT, LIMIT_KEYS, MARKUP, MAX_REPLY_TOKENS,  # noqa: E402
                    MAX_ROW_TOKENS, MAX_TURNS, SKIP_PHRASES, argument_strings, content_text, date_expressions,
                    expressions, grounding, harness_hits, number_values, numbers, row_calls, serialize_row,
                    tool_error_text, words_in)

from settings import TOKENIZER as DEFAULT_TOKENIZER  # noqa: E402
ADVISORY = {"call_numbers_shown"}


def gate(ok: bool, **details) -> dict:
    return {"pass": bool(ok), **details}


def structure(row: dict) -> dict:
    m = row["messages"]
    problems = []
    if len(m) < 4 or m[0].get("role") != "system" or m[1].get("role") != "user":
        problems.append("does not start with system, user")
    i = 2
    while i < len(m) - 1:
        a = m[i]
        if a.get("role") != "assistant" or not a.get("tool_calls"):
            problems.append(f"message {i} is not a calling assistant turn")
            break
        ids = [c.get("id") for c in a["tool_calls"]]
        if len(set(ids)) != len(ids):
            problems.append(f"message {i}: duplicate call ids")
        for k, call in enumerate(a["tool_calls"]):
            t = m[i + 1 + k] if i + 1 + k < len(m) else {}
            if t.get("role") != "tool" or t.get("tool_call_id") != call.get("id") or \
                    t.get("name") != call["function"]["name"]:
                problems.append(f"message {i + 1 + k}: tool message does not answer call {call.get('id')}")
            if not isinstance(call["function"].get("arguments"), dict):
                problems.append(f"message {i}: call {k + 1} arguments are not an object")
        i += 1 + len(a["tool_calls"])
    last = m[-1]
    if i != len(m) - 1 or last.get("role") != "assistant" or last.get("tool_calls"):
        problems.append("does not end with one final assistant message after the last tool results")
    masked = sorted((row.get("metadata") or {}).get("masked_turns") or [])
    assistant = [k for k, x in enumerate(m) if x.get("role") == "assistant"]
    if not set(masked) <= set(assistant[:-1]):
        problems.append("masked_turns must be assistant turns before the final reply")
    trainable = [k for k in assistant if k not in masked]
    if row.get("trainable_message_indices") != trainable:
        problems.append("trainable_message_indices are not exactly the assistant messages minus masked_turns")
    if not row.get("tools"):
        problems.append("no tools")
    return gate(not problems, problems=problems)


def prompt_exact(row: dict, runner: dict | None) -> dict:
    if runner is None:
        return gate(False, reason="task missing from the runner prompts file")
    msgs = json.dumps(row["messages"][:2], ensure_ascii=False) == json.dumps(runner["messages0"], ensure_ascii=False)
    tools = json.dumps(row["tools"], ensure_ascii=False) == json.dumps(runner["tools"], ensure_ascii=False)
    return gate(msgs and tools, messages_equal=msgs, tools_equal=tools, runner_episode=runner.get("episode_id"))


def replay_gates(row: dict) -> tuple[dict, dict, dict]:
    """Replay the row's calls with abtool2 at the pinned environment.

    World: the row's metadata `world_date` (default the study pin 2026-10-28) and `fresh_setup`. Rows recorded by the
    evaluation runner (own successes) do not say which world they ran in: the runner's first rollout of a task in a
    process uses the first-rollout world and every later rollout the later-rollout world (abtool2.initial_state). For
    those rows both worlds are replayed, later-rollout first, and the one whose results are byte-equal is used (the
    report names it); no other freedom is allowed."""
    import abtool2

    md = row["metadata"]
    calls = row_calls(row)
    pin = abtool2.environment_pin()
    try:
        abtool2.check_pin(pin)
    except abtool2.Refusal as refusal:
        failed = gate(False, reason=str(refusal), env_commit=pin["env_commit"])
        return failed, failed, {"replay_statuses": []}
    day = abtool2._parse_day(md.get("world_date"))
    worlds = [bool(md["fresh_setup"])] if "fresh_setup" in md else [False, True]
    results = [content_text(m["content"]) for m in row["messages"] if m["role"] == "tool"]
    out = replayed = diff = None
    fresh = worlds[0]
    for fresh in worlds:
        try:
            out = abtool2.replay(md["task"], calls, day, fresh=fresh, cap=False)
        except Exception as error:  # noqa: BLE001 - a replay that raises fails both gates with its reason
            failed = gate(False, reason=f"replay raised {type(error).__name__}: {error}")
            return failed, failed, {"replay_statuses": []}
        replayed = [content_text(c["result"]) for c in out["calls"]]
        diff = [k + 1 for k, (a, b) in enumerate(zip(results, replayed)) if a != b]
        if len(results) == len(replayed) and not diff:
            break
    world = {"world_date": (day or abtool2.STUDY_WORLD_DATE).isoformat(),
             "world_setup": "first-rollout (fresh)" if fresh else "later-rollout",
             "world_setup_source": "row metadata" if len(worlds) == 1 else "inferred: the world whose results match"}
    failed_rows = [{"type": a.get("type"), "excluded": a.get("excluded")} for a in out["assertion_results"]
                   if not a.get("passed")]
    score_gate = gate(out["task_completed_correctly"] == 1.0 and not failed_rows,
                      task_completed_correctly=out["task_completed_correctly"],
                      partial_credit=out["partial_credit"], assertions_passed=out["assertions_passed"],
                      assertions_scored=out["assertions_scored"], assertions_excluded=out["assertions_excluded"],
                      failed_rows=failed_rows, env_commit=pin["env_commit"], **world)
    equal_gate = gate(len(results) == len(replayed) and not diff, calls=len(results), replayed=len(replayed),
                      differing_calls=diff, **world)
    return score_gate, equal_gate, {"replay_statuses": [c["status"] for c in out["calls"]],
                                    "replay_results": replayed}


def visible_context(row: dict) -> tuple[str, set[str]]:
    """Text of the task's own visible context (system, user, tool names and definitions, tool results) and its words.
    The tool definitions (descriptions, parameter schemas) are shown to the agent, so their words are not harness
    words (e.g. "JSON" in a tool's argument description)."""
    m = row["messages"]
    parts = [content_text(x.get("content")) for x in m if x["role"] in ("system", "user", "tool")]
    parts += [t["function"]["name"].replace("_", " ") for t in row["tools"]]
    parts += [json.dumps(t["function"], ensure_ascii=False) for t in row["tools"]]
    text = "\n".join(parts)
    return text, words_in(text)


THINKING = False  # set by --thinking: rows carry reasoning_content and render with thinking on

# 13 Nov (Ali approved, P7 failure analysis): a teacher-in-runner row's context is exactly the runner's prompt
# (teacher_request_exact), so it cannot leak our generation harness; these everyday words ("send a brief follow-up",
# "the checks", "zoom sessions") were false alarms there. Evaluation-awareness words (grader, rubric, benchmark,
# assertion, ...) still count.
TEACHER_GENERIC_WORDS = {"brief", "session", "sessions", "command", "commands", "checks", "shell", "bash", "json",
                         "transcript", "stdout", "stderr"}
# 13 Nov: an identifier with digits copied verbatim from earlier context ("HVAC-NW-303", "rm_1007") is grounded
IDENTIFIER = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)+\b")


def strip_known_identifiers(text: str, context_text: str) -> str:
    """Remove from `text` every identifier that contains a digit and appears verbatim in the earlier context."""
    return IDENTIFIER.sub(lambda m: " " if any(c.isdigit() for c in m.group(0)) and m.group(0) in context_text
                          else m.group(0), text)


def is_a3(row: dict) -> bool:
    return (row.get("metadata") or {}).get("source") == A3_SOURCE


def a3_prefix_turns(row: dict) -> set[int]:
    """The masked student turns of an A3 continuation row (each with A3_PREFIX_REASON); empty for every other row."""
    if not is_a3(row):
        return set()
    meta = row.get("metadata") or {}
    reasons = {int(k): v for k, v in (meta.get("masked_reasons") or {}).items()}
    return {t for t in meta.get("masked_turns") or [] if reasons.get(t) == A3_PREFIX_REASON}


def text_gates(row: dict) -> dict:
    """note_before_calls (thinking: reasoning_every_turn), grounding, no_harness_words, final_reply,
    call_numbers_shown (one pass over the row)."""
    m = row["messages"]
    _, allowed = visible_context(row)
    if (row.get("metadata") or {}).get("source") == "teacher-in-runner":
        allowed = allowed | TEACHER_GENERIC_WORDS
    context: set[Decimal] = number_values(content_text(m[0]["content"])) | number_values(content_text(m[1]["content"]))
    context_text = content_text(m[0]["content"]) + "\n" + content_text(m[1]["content"])
    missing_notes, ungrounded, wrong, harness, unshown = [], [], [], [], []
    prefix = a3_prefix_turns(row)
    for i, x in enumerate(m[2:], 2):
        if x["role"] == "tool":
            context |= number_values(content_text(x["content"]))
            context_text += "\n" + content_text(x["content"])
            continue
        if x["role"] != "assistant":
            continue
        text = x.get("content") or ""
        if i in prefix:  # an A3 student turn: context only (it is never trained)
            if THINKING:
                context |= number_values(x.get("reasoning_content") or "")
            context |= number_values(text)
            context_text += "\n" + text + "\n" + (x.get("reasoning_content") or "")
            for call in x.get("tool_calls") or []:
                context |= number_values(json.dumps(call["function"]["arguments"], ensure_ascii=False))
                context_text += "\n" + json.dumps(call["function"]["arguments"], ensure_ascii=False)
            continue
        if THINKING:
            if not (x.get("reasoning_content") or "").strip():
                missing_notes.append(i)
        elif x.get("tool_calls") and not text.strip():
            missing_notes.append(i)
        if THINKING:  # the turn's reasoning precedes its text and stays in the rendered context: its numbers and
            # its correct arithmetic ground what follows (as an earlier note would)
            reasoning = x.get("reasoning_content") or ""
            context |= number_values(reasoning)
            for e in expressions(reasoning):
                if e["ok"]:
                    context |= {abs(v).normalize() for v in e["values"]}
        g = grounding(strip_known_identifiers(text, context_text), context)
        if g["ungrounded"]:
            ungrounded.append({"message": i, "numbers": g["ungrounded"], "text": text[:300]})
        if g["wrong_expressions"]:
            wrong.append({"message": i, "expressions": g["wrong_expressions"]})
        hits = harness_hits(text + "\n" + (x.get("reasoning_content") or "") if THINKING else text, allowed)
        if hits:
            harness.append({"message": i, "words": hits})
        note_values = number_values(text)
        for e in expressions(text) + date_expressions(text):
            if e["ok"]:
                note_values |= {abs(v).normalize() for v in e["values"]}
        for call in x.get("tool_calls") or []:
            for key, value in argument_strings(call["function"]["arguments"]):
                if LIMIT_KEYS.search(key):
                    continue
                for n in numbers(value):
                    v = Decimal(n.replace(",", "")).normalize()
                    if v not in context and not any(abs(v - w) <= Decimal("0.005") for w in note_values):
                        unshown.append({"message": i, "tool": call["function"]["name"], "key": key, "number": n})
        context |= number_values(text)
        context_text += "\n" + text + "\n" + (x.get("reasoning_content") or "")
        for call in x.get("tool_calls") or []:
            context |= number_values(json.dumps(call["function"]["arguments"], ensure_ascii=False))
            context_text += "\n" + json.dumps(call["function"]["arguments"], ensure_ascii=False)
    last = m[-1]
    reply = last.get("content") if last.get("role") == "assistant" else None
    problems = []
    if last.get("role") != "assistant" or last.get("tool_calls"):
        problems.append("last message is not an assistant message without tool calls")
    if not (reply or "").strip():
        problems.append("empty reply")
    markup = [s for s in MARKUP if s in (reply or "")]
    if markup:
        problems.append(f"tool markup in reply: {markup}")
    flags = [f"names skipped/excluded items: {s.group(0)!r}" for s in SKIP_PHRASES.finditer(reply or "")]
    return {
        ("reasoning_every_turn" if THINKING else "note_before_calls"): gate(
            not missing_notes, **({"turns_without_reasoning": missing_notes} if THINKING
                                  else {"calling_turns_without_text": missing_notes})),
        "grounding": gate(not ungrounded and not wrong, ungrounded=ungrounded, wrong_expressions=wrong),
        "no_harness_words": gate(not harness, hits=harness),
        "final_reply": gate(not problems, problems=problems, review_flags=flags, chars=len(reply or "")),
        # plan v4 check "the reply follows the system prompt": "list only items you acted on. Do not name, enumerate,
        # or explain items you skipped, excluded, or rejected". The 397B's replies often do (P7 chunk 0: 5 of 8
        # otherwise admitted rows); the benchmark does not score the chat reply, so only this gate keeps the 4B from
        # being trained to do it (the same habit fails its emails' must-not checks). Fails the row (13 Nov).
        "reply_names_no_skipped_items": gate(not flags, flags=flags),
        "call_numbers_shown": gate(not unshown, advisory=True, unshown=unshown[:30], count=len(unshown)),
    }


MASK_REASONS = {"names left-out items in a message", "wrong action, corrected later"}
# SDPO plan v4 phase A3 (13 Nov): a continuation row is the student's own turns up to the continuation point, then the
# teacher's continuation. The student's turns are masked (never trained) with this reason; only rows whose source is
# A3_SOURCE may use it. Their text gates (reasoning, grounding, harness words) are not applied (the student's own slips
# are not training material), while structure, replay, results-equal and every gate on the trained turns are unchanged.
A3_SOURCE = "sdpo-a3-continuation"
A3_PREFIX_REASON = "student turn before the continuation point (SDPO A3, not trained)"


def error_gate(row: dict, statuses: list[str]) -> dict:
    """No failed call is trained. Without masked_turns: no failed call at all. With metadata.masked_turns (13 Nov, Ali
    approved): every failed call (error text or non-ok replay status) was issued by a masked (untrained) assistant
    turn, and every masked turn issued at least one failed call; the recovery that follows stays trained."""
    results = [content_text(x["content"]) for x in row["messages"] if x["role"] == "tool"]
    errors = [{"call": k + 1, "error": e[:200]} for k, r in enumerate(results) if (e := tool_error_text(r))]
    bad_status = [k + 1 for k, s in enumerate(statuses) if s != "ok"]
    masked = set((row.get("metadata") or {}).get("masked_turns") or [])
    issuer = [k for k, x in enumerate(row["messages"]) if x["role"] == "assistant" for _ in x.get("tool_calls") or []]
    failed = sorted({e["call"] for e in errors} | set(bad_status))
    if not masked:
        return gate(not failed, errors=errors, count=len(errors), non_ok_status_calls=bad_status)
    failed_turns = {issuer[c - 1] for c in failed if c - 1 < len(issuer)}
    # every masked turn needs a recorded reason: "failed call" (must really have one), "names left-out items in a
    # message" (13 Nov: a turn whose message narrates exclusions and cannot be cleanly edited stays untrained) or "wrong
    # action, corrected later" (13 Nov v3 review: a successful but wrong action the trajectory itself undoes later;
    # the wrong turn stays untrained, the correction is trained)
    reasons = {int(k): v for k, v in ((row.get("metadata") or {}).get("masked_reasons") or {}).items()}
    allowed_reasons = MASK_REASONS | ({A3_PREFIX_REASON} if is_a3(row) else set())
    unexplained = sorted(t for t in masked if t not in failed_turns and reasons.get(t) not in allowed_reasons)
    ok = failed_turns <= masked and not unexplained
    return gate(ok, errors=errors, count=len(errors), non_ok_status_calls=bad_status, masked_turns=sorted(masked),
                failed_call_turns=sorted(failed_turns), trained_failed_calls=sorted(failed_turns - masked),
                masked_without_reason=unexplained)


def call_gates(row: dict) -> tuple[dict, dict]:
    calls = row_calls(row)
    keys = Counter(c["tool"] + "|" + json.dumps(c["arguments"], sort_keys=True) for c in calls)
    top = keys.most_common(1)[0] if keys else ("", 0)
    return gate(top[1] < IDENTICAL_CALL_LIMIT, max_identical=top[1], call=top[0][:200], calls=len(calls))


def length_gate(row: dict, tokenizer) -> dict:
    m, tools = row["messages"], row["tools"]
    full = tokenizer.apply_chat_template(m, tools=tools, tokenize=True, enable_thinking=THINKING, return_dict=True)
    total = len(full["input_ids"])
    longest, over = 0, []
    for k, x in enumerate(m):
        if x["role"] != "assistant":
            continue
        prefix = tokenizer.apply_chat_template(m[:k], tools=tools, tokenize=False, add_generation_prompt=True,
                                               enable_thinking=THINKING)
        upto = tokenizer.apply_chat_template(m[:k + 1], tools=tools, tokenize=False, enable_thinking=THINKING)
        target = upto[len(prefix):].rstrip("\n") if upto.startswith(prefix) else None
        if target is None:
            over.append({"message": k, "reason": "render of the history is not a prefix extension"})
            continue
        n = len(tokenizer(target, add_special_tokens=False)["input_ids"])
        longest = max(longest, n)
        if n > MAX_REPLY_TOKENS:
            over.append({"message": k, "tokens": n})
    assistants = sum(x["role"] == "assistant" for x in m)
    ok = total <= MAX_ROW_TOKENS and not over and assistants <= MAX_TURNS
    return gate(ok, tokens=total, longest_assistant_tokens=longest, assistant_messages=assistants, over=over)


def validate(row: dict, *, tokenizer=None, runner: dict | None = None, use_runner: bool = False,
             replay: bool = True) -> dict:
    gates: dict[str, dict] = {"structure": structure(row)}
    if use_runner:
        gates["prompt_exact"] = prompt_exact(row, runner)
    meta = row.get("metadata") or {}
    if "exact" in meta:  # teacher-in-runner rows (episode_rows.py): the proxy saw this system/user/tools, calls answered
        gates["teacher_request_exact"] = gate(meta["exact"] is True, cross_check=meta.get("cross_check"),
                                              structure_problems=meta.get("structure_problems"))
    statuses: list[str] = []
    if replay:
        score_gate, equal_gate, extra = replay_gates(row)
        gates["replay_score"], gates["results_equal_replay"] = score_gate, equal_gate
        statuses = extra["replay_statuses"]
    gates.update(text_gates(row))
    gates["zero_tool_errors"] = error_gate(row, statuses)
    gates["no_identical_call_5"] = call_gates(row)
    if tokenizer is not None:
        gates["length"] = length_gate(row, tokenizer)
    passed = all(g["pass"] for name, g in gates.items() if name not in ADVISORY)
    return {"id": row["id"], "task": row["metadata"].get("task"), "pass": passed, "gates": gates}


def load_tokenizer(path: Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(path), local_files_only=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("rows")
    parser.add_argument("--report", required=True)
    parser.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    parser.add_argument("--no-length", action="store_true")
    parser.add_argument("--runner-prompts")
    parser.add_argument("--no-replay", action="store_true")
    parser.add_argument("--admitted-out", help="write the rows that pass every non-advisory gate, unchanged")
    parser.add_argument("--thinking", action="store_true",
                        help="thinking-on rows: reasoning on every assistant turn; lengths rendered with thinking on")
    args = parser.parse_args(argv)
    global THINKING
    THINKING = args.thinking
    rows = [json.loads(line) for line in Path(args.rows).read_text().splitlines() if line.strip()]
    runner = None
    if args.runner_prompts:
        runner = {r["task"]: r for r in map(json.loads, Path(args.runner_prompts).open())}
    tokenizer = None if args.no_length else load_tokenizer(Path(args.tokenizer))
    results = []
    for row in rows:
        r = validate(row, tokenizer=tokenizer, runner=(runner or {}).get(row["metadata"].get("task")),
                     use_runner=runner is not None, replay=not args.no_replay)
        results.append(r)
        print(r["id"], "PASS" if r["pass"] else "FAIL",
              " ".join(f"{k}={'ok' if g['pass'] else 'FAIL'}" for k, g in r["gates"].items()), flush=True)
    names = sorted({k for r in results for k in r["gates"]})
    summary = {"rows": len(rows), "passed": sum(r["pass"] for r in results),
               "per_gate_pass": {k: sum(r["gates"][k]["pass"] for r in results if k in r["gates"]) for k in names},
               "advisory": sorted(ADVISORY),
               "tool_error_rows": sum(not r["gates"]["zero_tool_errors"]["pass"] for r in results),
               "tool_errors_total": sum(r["gates"]["zero_tool_errors"]["count"] for r in results),
               "final_reply_review_flags": sum(bool(r["gates"]["final_reply"]["review_flags"]) for r in results),
               "tokenizer": None if tokenizer is None else args.tokenizer}
    Path(args.report).write_text(json.dumps({"summary": summary, "rows": results}, indent=1, ensure_ascii=False,
                                            default=str))
    print(json.dumps(summary, indent=1))
    if args.admitted_out:
        with open(args.admitted_out, "w") as stream:
            for row, r in zip(rows, results):
                if r["pass"]:
                    stream.write(serialize_row(row))
    return 0 if summary["passed"] == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
