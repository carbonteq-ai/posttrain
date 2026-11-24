"""S-final (Phase 1 item 9, 10 November 2026): R0's recipe with every Track C finding applied, behind `--sf-final`.

Without `--sf-final` nothing here runs: the runner's R0 path (calls+anchor, 69e41d13) and R1's flagged blocks are
unchanged (test_sdpo_final proves the runner's functions equal the pinned code with every `--sf-*` block removed).

What the flag changes (each item names the Track C close-out finding, `scratch/reconcile/D-phase1/C-closeout/closeout.md`):

1. Decision term (F1, F2): a two-sided, capped Bernoulli KL on P(<tool_call>) (or P(<|im_end|>) inside a reply) at
   grounded decision points only, with those positions taken out of the anchor (the `exclude` rows of recipe R,
   `sdpo_ab_runner.episode_loss`). A point pushes in one direction only, the direction its evidence supports
   (`DIRECTION`), toward a constant target: the cap (0.5) for an up push, the floor (0.2) for a down push; the
   teacher's P is a reading only (E-review 2, blocker 1). The guarded run (Ali, 11 November 2026;
   `D-final/guarded/design-guarded.md`) trains `TRAINED_KINDS` only, every one at a reply's first token:
     - branch_first (up): the first kept call of a stop branch, at the student's final-reply start (the missed call);
     - branch_stop (down): after the kept call's result, when the replayed world has no failed required check left;
     - success_stop (down): the final reply (no call) of a successful attempt, grounded by the grader;
     - harm_opener (down): the <tool_call> of the first located harmful call, when it is the reply's first token.
   Counted, never trained: a harm opener after the student's own text (it fights the anchored preamble: E-review 3,
   issues 1-2), harm_in_reply (it ends a batch reply the student announced: E-review 4, issue 3), the redundant-call kinds (after text, and "stop at the graded subset" on batch tasks: E-review 3,
   issue 4; their replay is not run), branch_later (none with `--sf-keep 1`).
2. Branches (F2, F3): a kept call must succeed, break nothing and flip at least one failed required check in replay
   (`keep_rule`); the continuation ends at the first call that is not kept; the flip is recorded per call; at most
   `--sf-branch-candidates` candidates per task are continued and at most `--sf-points-per-task` with a kept call
   are trained; the teacher's stop after the kept calls is trained when the world shows the work done (branch_stop).
3. Admission (F4, F5): successful attempts are built with arm A (their own facts; never a sibling's solution);
   attempts with an incomplete reply, a runaway reply (`sdpo_ab_runner.runaway`) or one identical call (same name and
   arguments) made `--sf-max-repeat` times are dropped and counted (a tool name repeated with different arguments is
   batch work: E-review 3, issue 4).
4. Mixed-task steps (F6): the round's admitted rows are shuffled across tasks and stepped `--rollouts-per-task`
   rows at a time; branch points are spread over the steps round-robin.
5. Weights: R0's term is unchanged (weight 1); the branch body term and the decision term carry fixed weights set
   from `--sf-dry-run` (gradient norm of each term at the untrained adapter) by `weight_suggestion`, so that each
   term's norm is `--sf-branch-share` / `--sf-decision-share` of R0's.
6. Monitors (closeout 2.6, 5.8): per round, written in advance (`ROUND_RULES`), checked in the run from round 2:
   turn-limit stops, identical calls, calls per attempt, reply-limit share, doubt; a fixed-set P(call) probe after
   every round's checkpoint; the structure bars S1-S3 every round; guard10 after every round with a drop rule. A stop writes sf-stop.json and ends
   the run (the queue then takes checkpoint-(K-1), as D81).
7. Housekeeping (F11, F12): `--sf-final` logs through a broken-pipe-safe writer; chains whose re-render check failed
   are excluded.
"""

from __future__ import annotations

import json
import math
import random
import statistics
from collections import Counter, defaultdict

from sdpo_ab_branches import (BRANCH_MAX_CALLS, HARM_KINDS, ROLLOUT_LIMITS, assistant_messages, call_index,
                              call_spans, call_succeeded, chain_problem, flips, make_branch_engine,
                              pending_messages, student_prefix, succeeded)

# ------------------------------------------------------------------ fixed names and defaults

SF_FLAGS = ("sf_final", "sf_branch_weight", "sf_decision_weight", "sf_cap", "sf_floor", "sf_dry_run", "sf_branch_share",
            "sf_decision_share", "sf_points_per_task", "sf_branch_candidates", "sf_keep", "sf_max_repeat",
            "sf_guard_tasks", "sf_guard_attempts", "sf_guard_reference", "sf_guard_drop", "sf_guard_loops_reference",
            "sf_norm_probe_steps", "sf_mask_values", "sf_no_round_stops")

# Decision kinds. DIRECTION is the behaviour: +1 = "call here", -1 = "stop here". TOKEN says which token's probability
# the push acts on: "opener" = P(<tool_call>) at a reply start (sign = DIRECTION); "end" = P(<|im_end|>) right after a
# </tool_call> inside a reply (the only way to stop there is to end the reply, so a stop is an UP push on <|im_end|>).
HARM_OPENER, HARM_IN_REPLY, BRANCH_FIRST, BRANCH_LATER, BRANCH_STOP, SUCCESS_STOP, REDUNDANT_CALL, REDUNDANT_IN_REPLY = (
    "harm_opener", "harm_in_reply", "branch_first", "branch_later", "branch_stop", "success_stop", "redundant_call",
    "redundant_in_reply")
BRANCH_LATER_SAME_REPLY = "branch_later_same_reply"  # counted, never trained (not a decision position: P(call) ~ 1)
HARM_OPENER_AFTER_TEXT = "harm_opener_after_text"  # counted, never trained (after the student's own anchored text)
DIRECTION = {HARM_OPENER: -1, HARM_IN_REPLY: -1, BRANCH_FIRST: 1, BRANCH_LATER: 1, BRANCH_STOP: -1, SUCCESS_STOP: -1,
             REDUNDANT_CALL: -1, REDUNDANT_IN_REPLY: -1, HARM_OPENER_AFTER_TEXT: -1}
TOKEN = {HARM_OPENER: "opener", HARM_IN_REPLY: "end", BRANCH_FIRST: "opener", BRANCH_LATER: "opener",
         BRANCH_STOP: "opener", SUCCESS_STOP: "opener", REDUNDANT_CALL: "opener", REDUNDANT_IN_REPLY: "end",
         HARM_OPENER_AFTER_TEXT: "opener"}
# The guarded run's trained kinds (design-guarded 1.1); any other kind in a training record fails bar S1. harm_in_reply
# is counted, not trained: ~95% of its points end a batch reply the student announced (E-review 4, issue 3).
TRAINED_KINDS = (BRANCH_FIRST, BRANCH_STOP, SUCCESS_STOP, HARM_OPENER)
# Position classes, named by where the point sits for the model (E-review 3, issue 1): first_token = a reply's first
# token (with thinking on, the first token after `</think>`); after_text = an opener after the student's own text in
# the same reply (never trained); within_reply = the <|im_end|> after a </tool_call>. The balance is reported per
# class; the 20% balance bar is waived by Ali (11 November 2026) and replaced by the bars S1-S3 (`structure_bars`).
POSITION_CLASS = {BRANCH_FIRST: "first_token", BRANCH_LATER: "first_token", BRANCH_STOP: "first_token",
                  SUCCESS_STOP: "first_token", HARM_OPENER: "first_token", HARM_OPENER_AFTER_TEXT: "after_text",
                  REDUNDANT_CALL: "after_text", HARM_IN_REPLY: "within_reply", REDUNDANT_IN_REPLY: "within_reply"}
UP_KINDS = tuple(k for k, s in DIRECTION.items() if s > 0)
DOWN_KINDS = tuple(k for k, s in DIRECTION.items() if s < 0)
STOP_KINDS = tuple(k for k in TRAINED_KINDS if DIRECTION[k] < 0)  # the stop pushes bar S2 counts

DEFAULTS = {"cap": 0.5, "floor": 0.2, "points_per_task": 2, "branch_candidates": 3, "keep": 1, "max_repeat": 5,
            "branch_share": 0.25, "decision_share": 0.25, "guard_attempts": 8, "guard_drop": 0.15,
            "norm_probe_steps": 3, "mask_values": True, "stop_min": 20, "replay_drop_max": 0.05, "probe_attempts": 64,
            "probe_after_result_up": 0.03, "probe_final_done_up": 0.05, "probe_down_ratio": 0.5}

# Round rules (closeout 5.8; R0's D80 stops kept, R1's final-reply doubt stop kept). Round K >= 2 against round 1.
ROUND_RULES = {
    "turn_limit": "attempts ending at the turn limit > max(3 x round 1, 2% of attempts)",
    "identical_calls": "share of attempts with one identical call (name and arguments) made >= max_repeat times > "
                       "max(3 x round 1 share, round 1 share + 0.005)",
    "calls_per_attempt": "calls per attempt outside 0.75-1.33 x round 1 (R0, D80)",
    "reply_limit": "reply-limit share of attempts >= 10% (R0, D80)",
    "doubt": "doubt phrases per reply >= 0.45 (R0, D80)",
    "final_doubt": "doubt phrases per final reply > 2 x round 1 (R1, D74)",
    "p_call_probe": "every round, after training: on a fixed set of round-1 attempts (half successes), mean "
                    "P(<tool_call>) at the non-final reply starts after a tool result rose > 0.03 above, or fell below "
                    "0.5 x, its untrained reading; or at the final-reply starts of the successes rose > 0.05 above it",
    "later_opener_drift": "mean student P at the branch_stop points moved > 0.15 from round 1",
    "S1_structure": "every round: a trained decision point of a kind outside TRAINED_KINDS",
    "S2_stop_supply": "every round: active trained stop pushes < 20",
    "S3_replay_health": "every round, before training: branch candidates dropped by replay_error or replay_mismatch "
                        "> 5% of the continued candidates",
    "guard": "guard tasks' task-mean success < reference - guard_drop",
    "guard_loops": "guard attempts at the turn limit or with an identical call made >= max_repeat times > "
                   "max(2 x the untrained guard count, untrained + 4)",
}


# ------------------------------------------------------------------ options

def check_final_options(args):
    """`--sf-*` flags act only on engine-train with R0's loss; the run's numbers must be stated or measured."""
    if not getattr(args, "sf_final", False):
        raise ValueError("the --sf-* options need --sf-final")
    if args.phase != "engine-train" or args.loss_tokens != "calls+anchor" or args.arm is None:
        raise ValueError("--sf-final needs --phase engine-train --loss-tokens calls+anchor --arm (R0's loss)")
    if getattr(args, "r1_branches", False):
        raise ValueError("--sf-final replaces --r1-branches; give one of them")
    if not 0.5 <= args.sf_cap <= 1.0:
        raise ValueError("--sf-cap is in [0.5, 1]: the constant target of an up push")
    if not 0.0 < args.sf_floor <= 0.5:
        raise ValueError("--sf-floor is in (0, 0.5]: the constant target of a down push")
    if args.sf_dry_run:
        if args.sf_branch_weight is not None or args.sf_decision_weight is not None:
            raise ValueError("--sf-dry-run measures at weight 1; it takes no --sf-branch-weight/--sf-decision-weight")
    elif args.sf_branch_weight is None or args.sf_decision_weight is None:
        raise ValueError("--sf-final needs --sf-branch-weight and --sf-decision-weight (from the dry run)")
    elif args.sf_branch_weight < 0 or args.sf_decision_weight < 0:
        raise ValueError("--sf-branch-weight and --sf-decision-weight are >= 0")
    for name in ("sf_points_per_task", "sf_branch_candidates", "sf_keep", "sf_max_repeat", "sf_guard_attempts"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} is at least 1")
    if args.sf_keep != 1:
        raise ValueError("--sf-keep is 1 in the guarded run: a second kept call would train branch_later, not a "
                         "trained kind (design-guarded 1.2)")
    if args.sf_branch_candidates < args.sf_points_per_task:
        raise ValueError("--sf-branch-candidates is at least --sf-points-per-task")
    if not (0 < args.sf_branch_share <= 1 and 0 < args.sf_decision_share <= 1):
        raise ValueError("--sf-branch-share and --sf-decision-share are in (0, 1]")
    if args.sf_guard_tasks and args.sf_guard_reference is None:
        raise ValueError("--sf-guard-tasks needs --sf-guard-reference (the untrained guard mean at the same settings)")
    if args.sf_guard_loops_reference is not None and (args.sf_guard_loops_reference < 0 or not args.sf_guard_tasks):
        raise ValueError("--sf-guard-loops-reference is >= 0 and needs --sf-guard-tasks")
    if args.sf_norm_probe_steps < 0:
        raise ValueError("--sf-norm-probe-steps is >= 0")
    from sdpo_ab_execution import world_date

    if not world_date():
        raise ValueError("--sf-final replays attempts: pin the world date (SDPO_AB_WORLD_DATE)")


# ------------------------------------------------------------------ admission (F4, F5, F11)

def _calls(record):
    """(name, arguments as sorted-key JSON) of every call of the attempt, in order."""
    out = []
    for m in assistant_messages(record):
        if m.get("role") != "assistant":
            continue
        for c in m.get("tool_calls") or []:
            function = c.get("function") or {}
            arguments = function.get("arguments", c.get("arguments"))
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except ValueError:
                    pass
            out.append((function.get("name") or c.get("name"), json.dumps(arguments, sort_keys=True, default=str)))
    return out


def identical_call_count(record):
    """The most times one identical call (same name, same arguments) is made in the attempt: the loop reading (a
    tool name repeated with different arguments is batch work, E-review 3 issue 4; guarded/loop_args.json)."""
    counts = Counter(_calls(record))
    return max(counts.values()) if counts else 0


def name_run(record):
    """The longest run of one tool name in consecutive calls (the old F5 reading; reported only)."""
    names = [name for name, _ in _calls(record)]
    best = run = 0
    for i, name in enumerate(names):
        run = run + 1 if i and name == names[i - 1] else 1
        best = max(best, run)
    return best


def attempt_problems(record, chain, max_repeat, runaway_fn):
    """F5 and F11: why an attempt is not trained ([] when it is): incomplete replies, runaway replies (the runner's
    rule), one identical call made max_repeat times, a chain whose re-render check failed."""
    actions = [s for s in chain["segments"] if s["kind"] == "action"]
    problems = []
    if any(not s.get("complete", True) for s in actions):
        problems.append("incomplete_reply")
    if runaway_fn(actions):
        problems.append("runaway")
    if identical_call_count(record) >= max_repeat:
        problems.append("identical_calls")
    if chain.get("rerender_mismatch"):
        problems.append("rerender_mismatch")
    return problems


def cut_chain_after_turn(chain, turn):
    """The chain up to and including the action segment of `turn` (1-based assistant turn)."""
    seen = 0
    for index, segment in enumerate(chain["segments"]):
        if segment["kind"] == "action":
            seen += 1
            if seen == turn:
                return {**chain, "segments": chain["segments"][:index + 1]}
    raise ValueError(f"turn {turn} not in the chain")


def arm_for(args, success):
    """F4: a successful attempt is built with arm A (its own facts, no sibling solution); failures use --arm."""
    return "A" if success else args.arm


def neutral_context_for(chain):
    """A success the engine has nothing to say about still gives its stop point: the teacher then sees the
    student's own prompt (no facts, no notes)."""
    user = next(m for m in chain["messages0"] if m["role"] == "user")
    content = user.get("content")
    return {"first_message": content if isinstance(content, str) else json.dumps(content), "notes": {},
            "facts": [], "solution_episode": None, "coverage": {}, "neutral": True}


# ------------------------------------------------------------------ decision points of the student's own attempt

def reply_start_index(tokenizer, targets, turns, turn, thinking, think_close=None):
    """Index (into the assistant tokens) of the token that decides call-or-text at the start of `turn`'s reply:
    the reply's first token, or, with thinking on, the first non-blank token after `</think>` (None when the
    reply has no closed thinking block)."""
    indices = [i for i, t in enumerate(turns) if t == turn]
    if not indices:
        return None
    if not thinking:
        return indices[0]
    close = think_close if think_close is not None else tokenizer.convert_tokens_to_ids("</think>")
    after = None
    for i in indices:
        if targets[i] == close:
            after = i + 1
            break
    if after is None:
        return None
    for i in indices:
        if i >= after and tokenizer.decode([targets[i]]).strip():
            return i
    return None


def call_token_indices(targets, turns, opener, closer):
    """Per turn: the indices of the <tool_call> tokens and of the </tool_call> tokens, in order."""
    openers, closers = defaultdict(list), defaultdict(list)
    for index, (token, turn) in enumerate(zip(targets, turns)):
        if token == opener:
            openers[turn].append(index)
        elif token == closer:
            closers[turn].append(index)
    return openers, closers


def call_point(targets, turns, opener, closer, turn, k, kind_at_start, kind_in_reply):
    """The decision point for call k of `turn`: at k = 0 the opener itself (P(<tool_call>) at the reply start); at
    k > 0 the token after the previous call's </tool_call> (the "\n" before the next <tool_call>), where the only way
    not to make the call is to end the reply, so the trained token is <|im_end|> (E-review issue 2). None when the
    call is not in the chain."""
    openers, closers = call_token_indices(targets, turns, opener, closer)
    if k < len(openers.get(turn, [])):
        if k == 0:
            return {"index": openers[turn][0], "kind": kind_at_start, "turn": turn, "call": k}
        if k - 1 < len(closers.get(turn, [])) and closers[turn][k - 1] + 1 < len(targets):
            return {"index": closers[turn][k - 1] + 1, "kind": kind_in_reply, "turn": turn, "call": k}
    return None


def own_points(tokenizer, student, context, success, final_complete, thinking, redundant=None):
    """The attempt's own grounded decision points as indices into its assistant tokens, each with its kind:
    - the first located harmful call (failed attempts; `rendered_facts`): harm_opener at k = 0 when its <tool_call>
      is the reply's first token (`reply_start_index`), harm_opener_after_text at k = 0 after the student's own text
      (untrained), harm_in_reply at k > 0 (the <|im_end|> decision after the previous call);
    - the final reply's start of a successful attempt whose final reply made no call (success_stop);
    - the first redundant call (`redundant`: {"turn", "call"} from the replay, every required check already passing
      and no must-not broken before it): redundant_call at k = 0, redundant_in_reply at k > 0.
    Returns (points, unmatched harm facts)."""
    from sdpo_ab_runner import decision_tokens, rendered_facts

    opener, _ = decision_tokens(tokenizer)
    closer = tokenizer.convert_tokens_to_ids("</tool_call>")
    targets, turns = student["targets"], student["turns"]
    openers, _closers = call_token_indices(targets, turns, opener, closer)
    points, unmatched = [], 0
    if redundant:
        point = call_point(targets, turns, opener, closer, redundant["turn"], redundant["call"], REDUNDANT_CALL,
                           REDUNDANT_IN_REPLY)
        if point:
            points.append(point)
    if success:
        final = max(turns) if turns else None
        if final is not None and not openers.get(final) and final_complete:
            index = reply_start_index(tokenizer, targets, turns, final, thinking)
            if index is not None:
                points.append({"index": index, "kind": SUCCESS_STOP})
        return points, unmatched
    located = []
    for fact in rendered_facts(context or {}):
        if fact.get("kind") in HARM_KINDS and fact.get("location") == "at_call" and fact.get("turn"):
            call = fact.get("call")
            if call is not None and call < len(openers.get(fact["turn"], [])):
                located.append((fact["turn"], call))
            else:
                unmatched += 1
    if located:
        turn, call = min(located)
        point = call_point(targets, turns, opener, closer, turn, call, HARM_OPENER, HARM_IN_REPLY)
        if point and point["kind"] == HARM_OPENER and point["index"] != reply_start_index(tokenizer, targets, turns,
                                                                                        turn, thinking):
            point["kind"] = HARM_OPENER_AFTER_TEXT  # after the student's own text: counted, never trained
        if point and all(point["index"] != q["index"] for q in points):
            points.append(point)
    return points, unmatched


def cap_per_task(items, cap, rng):
    """At most `cap` items per task, chosen by the seeded rng when there are more (the order is kept)."""
    if len(items) <= cap:
        return list(items)
    chosen = set(rng.sample(range(len(items)), cap))
    return [item for i, item in enumerate(items) if i in chosen]


# ------------------------------------------------------------------ branches (F2, F3)

def required_failed(assertions):
    """Indices of required checks (outside the must-not family, not excluded: the grader's own population,
    `sdpo_ab_metrics.required_checks`) that do not pass."""
    from sdpo_ab_metrics import negative_family

    return [i for i, a in enumerate(assertions)
            if not negative_family(str(a.get("type"))) and not a.get("excluded") and a.get("passed") is not True]


def must_not_failed(assertions):
    from sdpo_ab_metrics import negative_family

    return [i for i, a in enumerate(assertions)
            if negative_family(str(a.get("type"))) and not a.get("excluded") and a.get("passed") is False]


def recorded_assertions(record):
    return (record["traces"][0].get("info") or {}).get("automationbench", {}).get("assertions") or []


def redundant_candidate(record):
    """Only an attempt whose recorded end has no failed required check can hold a redundant call (a required check
    that passed and later failed is not looked for)."""
    return not required_failed(recorded_assertions(record))


def first_redundant_call(world, record, day):
    """Runs in the world thread. The index (into the recorded answered calls), turn and call of the first recorded
    call made when every required check already passed and no must-not check had failed, or None. Grounded by the
    grader alone: the student's own call after the work was done (E-review section 5, item 1)."""
    calls, _ = __import__("sdpo_ab_world").recorded_calls(record)
    if not calls:
        return None
    built, report, snapshots = world.rebuild_with_snapshots(record, day, set(range(len(calls))))
    if report["variant"] == "unmatched" or not report["verified"]:
        return {"error": "replay_mismatch"}
    index, _ids = call_index(record)
    where = {n: (turn, k) for (turn, k), n in index.items()}
    for i in range(len(calls)):
        graded = world.grade(world.fork(built, snapshots[i]))["assertion_results"]
        if not required_failed(graded) and not must_not_failed(graded):
            turn, k = where[i]
            return {"call_index": i, "turn": turn, "call": k, "calls": len(calls)}
    return None


def required_flips(before, after):
    """Required checks (outside the must-not family) that did not pass before and pass after."""
    if len(before) != len(after):
        raise ValueError("assertion lists differ in length")
    failed = set(required_failed(before))
    return [{"index": i, "type": after[i].get("type")} for i in sorted(failed) if after[i].get("passed") is True]


def keep_rule(call, kept_so_far, keep):
    """A call is kept when it succeeded, broke nothing, flipped a failed required check, has a token span and the
    cap is not reached (closeout 5.2a)."""
    return (len(kept_so_far) < keep and call["success"] and not call["broken"] and bool(call.get("fixed"))
            and call.get("span") is not None)


def stop_candidates(records, joined, outcomes_by_id, max_repeat, runaway_fn):
    """Failed attempts whose final reply made no call and whose chain is usable and not dropped by F5."""
    out = []
    for record in records:
        if succeeded(record):
            continue
        chain = joined.get(record["id"])
        if chain_problem(chain, record):
            continue
        if attempt_problems(record, chain, max_repeat, runaway_fn):
            continue
        actions = [s for s in chain["segments"] if s["kind"] == "action"]
        if actions[-1].get("tool_call_ids") or not actions[-1].get("complete", True):
            continue
        if not required_failed(recorded_assertions(record)):
            continue  # nothing to flip (only a must-not failed): no call could be kept (E-review 2, issue 3)
        out.append(record)
    return out


async def continue_final(prefix, *, tokenizer, generate, parse, run_one, render, base_assertions, opener, closer,
                         turns_before, output_before, keep, max_calls=BRANCH_MAX_CALLS, limits=None,
                         succeeded_fn=None, visible=None, render_student=None):
    """The teacher continues from the student's final-reply start (as `sdpo_ab_branches.continue_branch`), each call
    executed alone and graded; a call is kept by `keep_rule` (it must flip a failed required check); the first call
    that is not kept ends the continuation. Returns the turns, executed calls, kept indices, the stop reason,
    whether the world after the kept calls has no failed required check (`done`), and whether the teacher's reply
    right after the last kept call made no call (`teacher_stopped`). `render_student` (thinking on): how the real
    results are rendered in the student's sequence (an empty thinking block, as the thinking-off template), while
    `render` is the teacher's prompt rendering."""
    limits = {**ROLLOUT_LIMITS, **(limits or {})}
    render_student = render_student or render
    succeeded_fn = succeeded_fn or call_succeeded
    tokens, turns, executed, kept, output, stop = list(prefix), [], [], [], output_before, None
    previous, after_kept = base_assertions, base_assertions
    eos = tokenizer.eos_token_id
    teacher_stopped = False
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
        shown = visible(raw) if visible else raw
        calls = parse(shown) if complete else []
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
            if complete:
                stop = "final_reply"
                # the teacher stopped right after the kept calls: every executed call so far is kept
                teacher_stopped = bool(kept) and len(kept) == len(executed) and len(turns) >= 2
            else:
                stop = limit
            break
        messages, kept_messages = [], []
        for k, call in enumerate(calls):
            name, arguments = call["function"]["name"], call["function"]["arguments"]
            if stop is not None or len(executed) >= max_calls:
                turn["calls"].append({"id": call["id"], "name": name, "arguments": arguments, "executed": False,
                                      "reason": "after_unkept" if stop else "call_cap"})
                continue
            content, status, after = await run_one(name, arguments)
            broken = {f["index"]: f for f in flips(previous, after) + flips(base_assertions, after)}
            entry = {"id": call["id"], "name": name, "arguments": arguments, "executed": True, "status": status,
                     "result": content, "success": succeeded_fn(content, status),
                     "broken": [broken[i] for i in sorted(broken)], "fixed": required_flips(previous, after),
                     "turn": len(turns) - 1, "k": k, "span": spans[k] if spans_ok else None}
            previous = after
            entry["kept"] = keep_rule(entry, kept, keep)
            turn["calls"].append(entry)
            executed.append(entry)
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": content, "name": name})
            if entry["kept"]:
                kept.append(len(executed) - 1)
                after_kept = after
                kept_messages.append(messages[-1])
            else:
                stop = "unkept_call" if len(kept) < keep else "keep_cap"
        turn["rendered"] = render(messages) if messages else None  # the teacher's own continuation: every result
        # the student's sequence: the kept calls' results only (an unkept call in the same reply is never the
        # student's; E-review 2, issue 4)
        turn["rendered_student"] = render_student(kept_messages) if kept_messages else None
        turn["unkept_in_reply"] = len(messages) - len(kept_messages)
        if stop is None and len(executed) >= max_calls:
            stop = "call_cap"
        if stop is None:
            tokens += turn["rendered"]
    done = bool(kept) and not required_failed(after_kept)
    return {"turns": turns, "executed": executed, "kept": kept, "stop": stop, "tokens": tokens, "done": done,
            "teacher_stopped": teacher_stopped}


def final_sequences(student_prefix_ids, teacher_prefix, continuation, opener, eos, newline, train_stop,
                    tokenizer=None):
    """The student's and the teacher's sequences of one branch, or None without kept calls (as
    `sdpo_ab_branches.branch_sequences`). Targets are the kept calls' tokens; `openers` indexes the <tool_call>
    tokens among them that start a reply (decision positions); `same_reply_openers` the later calls of one reply
    (after `</tool_call>\n`, P(call) ~ 1: counted, never trained, E-review issue 8); `value_indices` the tokens of
    the calls' parameter values (`sdpo_ab_runner.parameter_spans`), masked from the body term when
    `--sf-mask-values` is on (E-review issue 4). With `train_stop`, both sequences continue with the last kept
    reply's <|im_end|> and the real results of its calls, and `stop_positions` gives the (student, teacher)
    positions whose next token is the next-reply decision (no target: only P(<tool_call>) there is trained). With
    thinking on the student side uses each turn's `rendered_student` (results followed by an empty thinking block),
    so every student position is a call-or-text decision after `</think>`; the teacher side keeps its own reasoning."""
    executed, kept = continuation["executed"], continuation["kept"]
    if not kept:
        return None
    chosen = [executed[i] for i in kept]
    last_turn = chosen[-1]["turn"]
    turns = continuation["turns"]
    student, teacher = list(student_prefix_ids), list(teacher_prefix)
    s_pos, t_pos, targets, openers, same_reply, values = [], [], [], [], [], []
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
            value_offsets = set()
            if tokenizer is not None:
                from sdpo_ab_runner import parameter_spans

                for found in parameter_spans(tokenizer, turn["ids"][a:b]):
                    for lo, hi in found.values():
                        value_offsets.update(range(lo, hi))
            for offset in range(a, b):
                token = turn["ids"][offset]
                if token == opener:
                    (openers if j == 0 else same_reply).append(len(targets))
                if offset - a in value_offsets:
                    values.append(len(targets))
                s_pos.append(len(student) - 1)
                t_pos.append(start + offset - 1)
                targets.append(token)
                student.append(token)
        if index < last_turn or train_stop:
            if index < last_turn and (not turn["ids"] or turn["ids"][-1] != eos):
                teacher.append(eos)
            if index == last_turn:
                teacher += turn["ids"][end:]  # the rest of the teacher's reply (nothing after its last kept call)
                if not turn["ids"] or turn["ids"][-1] != eos:
                    teacher.append(eos)
            student.append(eos)
            teacher += turn["rendered"]
            student += turn["rendered_student"]
    for p, q, token in zip(s_pos, t_pos, targets):
        if student[p + 1] != token or teacher[q + 1] != token:
            raise ValueError("branch target misaligned")
    out = {"student": {"ids": student, "positions": s_pos, "targets": targets},
           "teacher": {"ids": teacher, "positions": t_pos, "targets": list(targets)},
           "openers": openers, "same_reply_openers": same_reply, "value_indices": values, "opener": opener,
           "calls": len(chosen), "tokens": len(targets), "stop_positions": None}
    if train_stop:
        out["stop_positions"] = (len(student) - 1, len(teacher) - 1)
    return out


def branch_point_kinds(branch):
    """The decision kinds of a branch's openers (first, later...) and its stop, in order."""
    kinds = [BRANCH_FIRST if i == 0 else BRANCH_LATER for i in range(len(branch["openers"]))]
    if branch.get("stop_positions"):
        kinds.append(BRANCH_STOP)
    return kinds


def think_pad(tokenizer):
    """Thinking on: how a thinking-on generation prompt (ending `<think>\n`) becomes the thinking-off one (ending
    `<think>\n\n</think>\n\n`): drop its trailing whitespace tokens and append an empty reasoning block. Applied to
    the student's final-reply start in a branch, so the student's decision positions sit after `</think>` (review
    finding 2); the results inside a branch are rendered thinking-off for the student. Returns {"drop", "add"};
    refused when the two renderings do not share every token but trailing whitespace."""
    from sdpo_ab_runner import render_delta

    probe = [{"role": "tool", "tool_call_id": "probe", "content": "ok", "name": "probe"}]
    on, off = render_delta(tokenizer, probe, thinking=True), render_delta(tokenizer, probe, thinking=False)
    common = 0
    while common < min(len(on), len(off)) and on[common] == off[common]:
        common += 1
    dropped = on[common:]
    if not dropped or tokenizer.decode(dropped).strip() or len(off) <= common:
        raise ValueError("thinking-off prompt is not the thinking-on prompt minus trailing whitespace plus an empty "
                         "thinking block")
    return {"drop": len(dropped), "add": off[common:]}


def with_think_pad(prefix, pad):
    """The student's prefix (a thinking-on generation prompt) rewritten to end with an empty thinking block."""
    if not pad:
        return list(prefix)
    return list(prefix[:len(prefix) - pad["drop"]]) + list(pad["add"])


async def build_final_branches(records, joined, *, tokenizer, engine, generate, parse_for, world, day, run_in_world,
                               siblings_of, seed_for, outcomes_by_id, runaway_fn, candidates_per_task, points_per_task,
                               keep, max_repeat, log=None, concurrency=32, visible=None, thinking=False):
    """Stop branches of the batch under the S-final rules. Returns {"branches": {task: [branch]}, "rows": [...],
    "counts": Counter}; each row records per call `fixed` (the required checks it flipped), `kept`, and the
    point's `done` and `teacher_stopped` readings (F3: the flip is in the training row)."""
    import asyncio

    from sdpo_ab_precheck import tool_call_opener
    from sdpo_ab_runner import engine_teacher_prefix, render_delta, task_of

    log = log or (lambda **fields: None)
    opener, _ = tool_call_opener(tokenizer)
    closer = tokenizer.convert_tokens_to_ids("</tool_call>")
    eos, newline = tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)
    pad = think_pad(tokenizer) if thinking else None
    rows, counts, jobs = [], Counter(), []
    gate = asyncio.Semaphore(max(1, concurrency))
    by_task = defaultdict(list)
    for record in stop_candidates(records, joined, outcomes_by_id, max_repeat, runaway_fn):
        by_task[task_of(record)].append(record)
    counts["failed_attempts"] = sum(not succeeded(r) for r in records)
    counts["stop_candidates"] = sum(len(v) for v in by_task.values())

    def drop(record, reason, **extra):
        counts["drop:" + reason.split(":")[0]] += 1
        rows.append({"episode_id": record["id"], "task": task_of(record), "dropped": reason, **extra})

    for task in sorted(by_task):
        # the attempts closest to done first (fewest failed required checks at the recorded end: E-review 5.3), id
        # as the tie-break; a seeded draw only among equals is not needed
        ranked = sorted(by_task[task], key=lambda r: (len(required_failed(recorded_assertions(r))), r["id"]))
        chosen = ranked[:candidates_per_task]
        counts["candidates"] += len(chosen)
        for record in chosen:
            chain = joined[record["id"]]
            try:
                context = engine(record, siblings_of(record), "A", "both")
            except ValueError as error:
                drop(record, f"untrue: {error}")
                continue
            except Exception as error:  # noqa: BLE001 - an engine crash excludes the attempt, as in training
                drop(record, f"builder_error: {type(error).__name__}: {error}")
                continue
            if context is None:
                drop(record, "no_context")
                continue
            actions = [i for i, s in enumerate(chain["segments"]) if s["kind"] == "action"]
            note = context.get("final_note") or {}
            if note.get("turn") != len(actions) or not note.get("items"):
                drop(record, "no_final_note")
                continue
            jobs.append((task, record, chain, context, {"position": actions[-1], "turn": len(actions),
                                                        "facts": list(note["items"])}))

    async def one(task, record, chain, context, point):
        where = {"kind": "stop", "turn": point["turn"], "call": None, "facts": point["facts"]}
        try:
            built, report, snapshots = await run_in_world(world.rebuild_with_snapshots, record, day, set())
        except Exception as error:  # noqa: BLE001
            drop(record, f"replay_error: {type(error).__name__}: {error}", **where)
            return
        if report["variant"] == "unmatched" or not report["verified"]:
            drop(record, "replay_mismatch", **where)
            return
        start = await run_in_world(world.fork, built)
        teacher_prefix = engine_teacher_prefix(tokenizer, chain, point["position"], context)
        if len(teacher_prefix) >= ROLLOUT_LIMITS["context"]:
            drop(record, "teacher_prefix_over_context_limit", **where)
            return
        prefix_student = with_think_pad(student_prefix(chain, point["position"], 0), pad)
        base = (await run_in_world(world.grade, start))["assertion_results"]
        actions = [s for s in chain["segments"][:point["position"]] if s["kind"] == "action"]
        key = f"{record['id']}|stop|{point['turn']}"

        async def turn_sample(ids, budget, index):
            async with gate:
                return await generate(ids, budget, seed_for(f"{key}|{index}"))

        async def run_one(name, arguments):
            def go():
                ((content, status),) = world.execute(start, [(name, arguments)])
                return content, status, world.grade(start)["assertion_results"]
            return await run_in_world(go)

        result = await continue_final(
            teacher_prefix, tokenizer=tokenizer, generate=turn_sample, parse=parse_for(chain), run_one=run_one,
            render=lambda messages: render_delta(tokenizer, messages), base_assertions=base, opener=opener,
            closer=closer, turns_before=point["turn"] - 1, output_before=sum(len(a["ids"]) for a in actions),
            keep=keep, visible=visible,
            render_student=(lambda messages: render_delta(tokenizer, messages, thinking=False)) if thinking else None)
        # the stop is trained whenever the replayed world shows the work done after the kept calls (the grader
        # grounds it, as success_stop); whether the teacher itself stopped is recorded (E-review 5.2)
        train_stop = result["done"]
        sequences = final_sequences(prefix_student, teacher_prefix, result, opener, eos, newline, train_stop,
                                    tokenizer=tokenizer)
        row = {"episode_id": record["id"], "task": task, **where, "stop": result["stop"],
               "executed_calls": len(result["executed"]), "kept": len(result["kept"]), "done": result["done"],
               "teacher_stopped": result["teacher_stopped"], "stop_trained": bool(sequences) and train_stop,
               "unkept_call_in_last_kept_reply": bool(result["kept"]) and bool(
                   result["turns"][result["executed"][result["kept"][-1]]["turn"]].get("unkept_in_reply")),
               "calls": [{k: c.get(k) for k in ("name", "status", "success", "broken", "fixed", "kept", "turn", "k")}
                         for c in result["executed"]],
               "first_turn_words": len(result["turns"][0]["text"].split()) if result["turns"] else 0,
               "trained_tokens": sequences["tokens"] if sequences else 0,
               "openers": len(sequences["openers"]) if sequences else 0,
               "same_reply_openers": len(sequences["same_reply_openers"]) if sequences else 0,
               "value_tokens": len(sequences["value_indices"]) if sequences else 0,
               "student_tokens": len(sequences["student"]["ids"]) if sequences else None,
               "teacher_tokens": len(sequences["teacher"]["ids"]) if sequences else None,
               "solution_shown": bool(context.get("solution_episode"))}
        rows.append(row)
        counts["points"] += 1
        counts["points_with_kept_call"] += bool(sequences)
        counts["points_done_after_kept"] += bool(result["done"])
        counts["points_stop_trained"] += bool(sequences) and train_stop
        counts["points_stop_trained_teacher_stopped"] += bool(sequences) and train_stop and result["teacher_stopped"]
        counts["points_done_teacher_called_on"] += bool(result["done"]) and not result["teacher_stopped"]
        log(stage="sf_branch", episode=record["id"], turn=point["turn"], stop=result["stop"],
            calls=len(result["executed"]), kept=len(result["kept"]), done=result["done"],
            stop_trained=bool(sequences) and train_stop)
        if sequences:
            sequences.update(episode_id=record["id"], task=task, kind="stop", turn=point["turn"], call=None, row=row,
                             final_position=point["position"], kinds=branch_point_kinds(sequences))
            candidates_out[task].append(sequences)

    candidates_out = defaultdict(list)
    await asyncio.gather(*(one(*job) for job in jobs))
    branches = {}
    for task, items in candidates_out.items():
        items.sort(key=lambda b: b["episode_id"])
        trained = draw_trained(items, points_per_task, random.Random(seed_for(f"sf-points|{task}")))
        for b in items:
            b["row"]["trained"] = b in trained
        counts["trained_points"] += len(trained)
        counts["trained_stops"] += sum(bool(b["stop_positions"]) for b in trained)
        counts["kept_calls_trained"] += sum(b["calls"] for b in trained)
        counts["same_reply_openers_untrained"] += sum(len(b["same_reply_openers"]) for b in trained)
        counts["value_tokens_in_trained_bodies"] += sum(len(b["value_indices"]) for b in trained)
        counts["body_tokens_in_trained"] += sum(b["tokens"] - len(b["openers"]) - len(b["same_reply_openers"])
                                                for b in trained)
        branches[task] = trained
    return {"branches": branches, "rows": rows, "counts": counts}


async def find_redundant_calls(records, *, world, day, run_in_world, log=None):
    """For every attempt whose recorded end has no failed required check (`redundant_candidate`), the first
    redundant call by replay (`first_redundant_call`). Returns ({episode_id: {...}}, Counter)."""
    import asyncio

    from sdpo_ab_runner import task_of

    log = log or (lambda **fields: None)
    found, counts = {}, Counter()
    candidates = [r for r in records if redundant_candidate(r)]
    counts["candidates"] = len(candidates)

    async def one(record):
        try:
            result = await run_in_world(first_redundant_call, world, record, day)
        except Exception as error:  # noqa: BLE001
            counts["replay_error"] += 1
            log(stage="sf_redundant", episode=record["id"], error=f"{type(error).__name__}: {error}")
            return
        if result is None:
            counts["no_redundant_call"] += 1
        elif "error" in result:
            counts[result["error"]] += 1
        else:
            found[record["id"]] = {**result, "task": task_of(record), "success": succeeded(record)}
            counts["found"] += 1
            counts["found_in_reply"] += result["call"] > 0
    await asyncio.gather(*(one(r) for r in candidates))
    return found, counts


def draw_trained(items, cap, rng):
    """The trained branches of a task: those with a stop first (a branch whose world is done after the kept calls
    is a stop example), then the others, each group drawn by the seeded rng when over the cap (E-review 2, issue 7)."""
    with_stop = [b for b in items if b.get("stop_positions")]
    without = [b for b in items if not b.get("stop_positions")]
    chosen = cap_per_task(with_stop, cap, rng)
    if len(chosen) < cap:
        chosen += cap_per_task(without, cap - len(chosen), rng)
    return chosen


def summarize_branches(rows):
    mine = [r for r in rows if "dropped" not in r]
    trained = [r for r in mine if r.get("trained")]
    return {"points": len(mine), "with_kept_call": sum(r["kept"] > 0 for r in mine), "trained": len(trained),
            "trained_stops": sum(bool(r.get("stop_trained")) for r in trained),
            "trained_stops_teacher_stopped": sum(bool(r.get("stop_trained")) and bool(r.get("teacher_stopped"))
                                                for r in trained),
            "same_reply_openers_untrained": sum(r.get("same_reply_openers", 0) for r in trained),
            "value_tokens_in_trained_bodies": sum(r.get("value_tokens", 0) for r in trained),
            "kept_calls_trained": sum(r["kept"] for r in trained),
            "kept_per_trained_point": dict(Counter(r["kept"] for r in trained)),
            "stops": dict(Counter(r["stop"] for r in mine)),
            "done_after_kept": sum(bool(r["done"]) for r in mine),
            "done_but_teacher_called_on": sum(bool(r["done"]) and not r["teacher_stopped"] for r in mine),
            "fixed_checks_trained": sum(len(c["fixed"] or []) for r in trained for c in r["calls"] if c["kept"]),
            "drops": dict(Counter(r["dropped"].split(":")[0] for r in rows if "dropped" in r))}


# ------------------------------------------------------------------ loss: the two-sided capped decision term

def decision_target(p_teacher, sign, cap, floor=None):
    """The constant target of a push: cap for an up push, floor for a down push (floor = 1 - cap unless given).
    Every point kind is grounded by the grader (a replay flip, the work done, a success) or by a located fact, never
    by the teacher's confidence, so the teacher's probability is a reading only (E-review 2, blocker 1: on the rows
    that supply the stops the teacher sees the student's own view, and a teacher-bounded target was inert there).
    `p_teacher` is accepted and ignored."""
    floor = 1.0 - cap if floor is None else floor
    return cap if sign > 0 else floor


def decision_active(p_student, q, sign):
    """A point contributes only on its own side: up when the student is below q, down when above."""
    return q > p_student if sign > 0 else q < p_student


def forward_bernoulli_kl(q, log_p):
    """KL(Bernoulli(q) || Bernoulli(p)) from q and log p: the cross-entropy of p under q minus H(q). Its gradient in
    the logit of p is p - q: bounded by 1 and non-vanishing at small p (the reverse KL's gradient p(1-p)(logit p -
    logit q) is ~1e-3 at p = 1e-4, which is where R1's first openers sat and stayed: closeout 2.3; review finding 1)."""
    p = math.exp(log_p)
    one_p = -math.expm1(log_p)
    kl = 0.0
    if q > 0.0:
        kl += q * (math.log(q) - log_p)
    if q < 1.0:
        kl += (1.0 - q) * (math.log(1.0 - q) - (math.log(one_p) if one_p > 0.0 else -math.inf))
    return kl


def decision_reference(log_p_student, log_p_teacher, sign, cap, floor=None):
    """Float reference of `two_sided_term` for one point: KL(Bernoulli(q) || Bernoulli(p_s)) toward the constant
    target q when active, else 0 (`log_p_teacher` is a reading, not used)."""
    q = min(max(decision_target(math.exp(log_p_teacher), sign, cap, floor), 1e-6), 1.0 - 1e-6)
    if not decision_active(math.exp(log_p_student), q, sign):
        return 0.0
    return forward_bernoulli_kl(q, log_p_student)


def decision_gradient(p_student, q, sign):
    """|d loss / d logit| of one point = |p_s - q| when active, else 0 (the per-point gradient mass the balance
    monitor and the dry-run bars read; exact for the opener logit)."""
    return abs(p_student - q) if decision_active(p_student, q, sign) else 0.0


def two_sided_term(s_logits, t_logits, tokens, signs, cap, floor=None):
    """Torch: per decision row, KL(Bernoulli(q) || Bernoulli(p_s)) of the student's P(token) toward the **constant**
    target q (up: cap; down: floor), active on the point's own side only (up when p_s < q, down when p_s > q); exact
    log(1 - p) through the masked log-sum-exp (recipe R). The teacher's P(token) is returned as a reading.
    `tokens`: one token id per row (<tool_call> at a reply start, <|im_end|> inside a reply) or a single id for
    every row. Gradient in the token's logit: p_s - q."""
    import torch
    from sdpo_ab_runner import DECISION_MASK_LOGIT

    floor = 1.0 - cap if floor is None else floor
    s, t = s_logits.float(), t_logits.float().detach()
    ids = torch.tensor(tokens if isinstance(tokens, (list, tuple)) else [tokens] * s.shape[0], device=s.device)
    log_s, log_t = s.log_softmax(-1), t.log_softmax(-1)
    lp_s, lp_t = log_s.gather(-1, ids[:, None]).flatten(), log_t.gather(-1, ids[:, None]).flatten()
    masked = s.scatter(-1, ids[:, None], DECISION_MASK_LOGIT)
    l1_s = torch.logsumexp(masked, -1) - torch.logsumexp(s, -1)
    sign = torch.tensor(signs, device=s.device, dtype=torch.float32)
    p_t = lp_t.exp()
    q = torch.where(sign > 0, torch.full_like(p_t, cap), torch.full_like(p_t, floor)).clamp(1e-6, 1.0 - 1e-6)
    p_s = lp_s.exp()
    kl = q * (q.log() - lp_s) + (1.0 - q) * (torch.log1p(-q) - l1_s)
    active = torch.where(sign > 0, q > p_s, q < p_s).detach()
    per = kl * active.float()
    grad = ((p_s - q).abs() * active.float()).detach()
    stats = {"p_student": p_s.detach().tolist(), "p_teacher": p_t.tolist(), "target": q.tolist(),
             "active": active.tolist(), "kl": kl.detach().tolist(), "grad": grad.tolist(), "signs": list(signs),
             "tokens": ids.tolist()}
    return per, stats


def decision_parity(generator, cap=0.5, floor=None):
    """`two_sided_term` against `decision_reference` on random logits, both signs, both sides of the cap and of the
    student's P, and the torch gradient in the opener logit against p_s - q; the largest absolute difference."""
    import torch

    tokens = [3, 5] * 8  # two token ids per row set (opener-like and end-like)
    s, t = torch.randn(16, 20, generator=generator) * 3, torch.randn(16, 20, generator=generator) * 3
    for i in range(16):
        if i < 4:
            t[i, tokens[i]] += 8.0
        elif i < 8:
            s[i, tokens[i]] += 8.0
        elif i < 12:
            t[i, tokens[i]] -= 8.0
        else:
            s[i, tokens[i]] -= 8.0
    s.requires_grad_(True)
    signs = [1, -1] * 8
    terms, stats = two_sided_term(s, t, tokens, signs, cap, floor)
    log_s = [s.log_softmax(-1)[i, tokens[i]].item() for i in range(16)]
    log_t = [t.log_softmax(-1)[i, tokens[i]].item() for i in range(16)]
    reference = [decision_reference(a, b, sign, cap, floor) for a, b, sign in zip(log_s, log_t, signs)]
    worst = max(abs(float(x) - y) for x, y in zip(terms, reference))
    terms.sum().backward()
    grad = [s.grad[i, tokens[i]].item() for i in range(16)]  # d/d logit of the Bernoulli = p_s - q (active)
    expected = [(p - q) * a for p, q, a in zip(stats["p_student"], stats["target"], stats["active"])]
    return max(worst, max(abs(g - e) for g, e in zip(grad, expected)))


def parity_check(output, cap, floor=None):
    """Run in the GPU job before training: writes parity-final.json, refuses on failure."""
    import torch

    from sdpo_ab_runner import write

    difference = decision_parity(torch.Generator().manual_seed(20261110), cap, floor)
    result = {"two_sided_vs_reference": difference, "passed": difference < 1e-5, "cap": cap, "floor": floor}
    write(output / "parity-final.json", result)
    if not result["passed"]:
        raise ValueError(f"final decision parity failed: {result}")
    return result


# ------------------------------------------------------------------ loss: rows and branches

def point_token_and_sign(kind, opener, end):
    """(token id, sign on that token): a stop inside a reply is an up push on <|im_end|>."""
    if TOKEN[kind] == "end":
        return end, 1
    return opener, DIRECTION[kind]


def row_decision(student, teacher, points, opener, end):
    """The attempt's own points as positions in its student and teacher sequences (before `restrict_to_calls`),
    with the token each push acts on and its sign on that token."""
    tokens_signs = [point_token_and_sign(p["kind"], opener, end) for p in points]
    return {"positions": [student["positions"][p["index"]] for p in points],
            "teacher_positions": [teacher["positions"][p["index"]] for p in points],
            "kinds": [p["kind"] for p in points], "signs": [sign for _, sign in tokens_signs],
            "tokens": [token for token, _ in tokens_signs]}


def chunk_loss_tokens(head, hs, ht, target, old, topk, exclude_ids):
    """`sdpo_ab_runner.chunk_loss` with one excluded token id per row (-1 = none): the anchor at a decision position
    holds the distribution over the other tokens only, whichever token the push acts on."""
    import torch
    from trl.experimental.sdpo.loss_utils import compute_topk_self_distillation_loss

    from sdpo_ab_runner import DECISION_MASK_LOGIT

    s = head(hs).float()
    with torch.no_grad():
        t = head(ht).float()
    s_kl, t_kl = s, t
    rows = exclude_ids >= 0
    if bool(rows.any()):
        mask = torch.zeros_like(s, dtype=torch.bool)
        mask[rows] = mask[rows].scatter(-1, exclude_ids[rows][:, None], True)
        s_kl, t_kl = s.masked_fill(mask, DECISION_MASK_LOGIT), t.masked_fill(mask, DECISION_MASK_LOGIT)
    per = compute_topk_self_distillation_loss(s_kl, t_kl, distillation_topk=topk, distillation_alpha=1.0,
                                              distillation_add_tail=True)
    selected = s.log_softmax(-1).gather(-1, target[:, None]).flatten()
    ratio = (selected.detach() - old).clamp(-20, 20).exp().clamp(max=2.0)
    return (per * ratio).sum(), ratio.detach()


def reply_start_rows(student, anchor, tokenizer=None, thinking=False):
    """Anchor rows at the call-or-text decision of every reply after a tool result (turn >= 2; with thinking on,
    after that reply's `</think>`): the P(<tool_call>) monitor positions."""
    starts = []
    for turn in sorted(set(student["turns"])):
        if turn < 2:
            continue
        index = reply_start_index(tokenizer, student["targets"], student["turns"], turn, thinking) if thinking \
            else next(i for i, t in enumerate(student["turns"]) if t == turn)
        if index is not None:
            starts.append(student["positions"][index])
    where = {p: i for i, p in enumerate(anchor["positions"])}
    return [where[p] for p in starts if p in where]


def final_episode_loss(model, student, teacher, anchor, decision, opener, cap, exclude_positions=(), topk=20,
                       chunk=1024, monitor_rows=None, floor=None):
    """R0's row loss (distillation on the call bodies + the anchor) with the decision positions excluded from the
    anchor, plus the row's decision points (returned separately: their sum and count, so the step can average over
    all its points). `exclude_positions`: further anchored positions held out of the anchor (a branch's first
    opener sits at the row's final-reply start). `monitor_rows`: anchor rows whose P(<tool_call>) is read (no
    gradient). Returns (r0_loss, decision_sum, n_points, stats); stats carries the distill and anchor parts as tensors
    (`parts`) for the dry run's split."""
    import torch
    from torch.utils.checkpoint import checkpoint

    from sdpo_ab_runner import chunk_loss, inner_hidden, lm_head

    n_points = len(decision["signs"]) if decision else 0
    with torch.no_grad():
        model.set_adapter("sdpo_teacher")
        model.eval()
        extra = decision["teacher_positions"] if decision else []
        both = inner_hidden(model, teacher["ids"], teacher["positions"] + extra).detach()
        ht, ht_decision = both[:len(teacher["positions"])], both[len(teacher["positions"]):]
        model.set_adapter("sdpo_anchor")
        ha = inner_hidden(model, student["ids"], anchor["positions"]).detach()
    model.set_adapter("default")
    model.train()
    count = len(student["targets"])
    hs = inner_hidden(model, student["ids"], student["positions"] + anchor["positions"])
    head = lm_head(model)

    def mean_loss(hidden, reference, sequence, exclude_ids=None):
        target = torch.tensor(sequence["targets"], device="cuda")
        old = torch.tensor(sequence["old"], device="cuda", dtype=torch.float32)
        total, ratios = hidden.new_zeros((), dtype=torch.float32), []
        for start in range(0, len(sequence["targets"]), chunk):
            if exclude_ids is None:
                part, ratio = checkpoint(chunk_loss, head, hidden[start:start + chunk], reference[start:start + chunk],
                                         target[start:start + chunk], old[start:start + chunk], topk,
                                         use_reentrant=False)
            else:
                part, ratio = checkpoint(chunk_loss_tokens, head, hidden[start:start + chunk],
                                         reference[start:start + chunk], target[start:start + chunk],
                                         old[start:start + chunk], topk, exclude_ids[start:start + chunk],
                                         use_reentrant=False)
            total = total + part
            ratios += ratio.tolist()
        return total / len(sequence["targets"]), ratios

    stats = {}
    if count:
        loss, ratios = mean_loss(hs[:count], ht, student)
    else:
        loss, ratios = hs.new_zeros((), dtype=torch.float32), []
    stats["distill"] = float(loss.detach())
    index = {p: i for i, p in enumerate(anchor["positions"])}
    held_out = [(p, tok) for p, tok in zip(decision["positions"], decision["tokens"])] if decision else []
    held_out += [(p, opener) for p in exclude_positions]  # a branch's first opener at the row's final-reply start
    exclude_ids = None
    if held_out:
        exclude_ids = torch.full((len(anchor["positions"]),), -1, dtype=torch.long, device=hs.device)
        for p, tok in held_out:
            if p not in index:
                raise ValueError("a decision point is not an anchored token")
            exclude_ids[index[p]] = tok
    held, _ = mean_loss(hs[count:], ha, anchor, exclude_ids)
    loss = loss + held
    stats["anchor"] = float(held.detach())
    stats["ratios"] = ratios
    decision_sum = hs.new_zeros((), dtype=torch.float32)
    if n_points:
        d_rows = [index[p] for p in decision["positions"]]
        per, d_stats = two_sided_term(head(hs[count:][d_rows]), head(ht_decision), decision["tokens"],
                                      decision["signs"], cap, floor)
        decision_sum = per.sum()
        stats["decision"] = {**d_stats, "kinds": decision["kinds"], "loss": float(per.detach().sum())}
    stats["parts"] = {"distill": loss - held, "anchor": held}
    if monitor_rows:
        with torch.no_grad():
            logits = head(hs[count:][monitor_rows].detach()).float().log_softmax(-1)
            stats["reply_start_p_call"] = logits[:, opener].exp().tolist()
    return loss, decision_sum, n_points, stats


def final_branch_loss(model, branch, opener, cap, topk=20, chunk=1024, floor=None, mask_values=True):
    """One branch: the kept calls' body words (mean over the body tokens, top-20 + tail reverse KL toward the
    teacher scored in its own context, importance weight 1; the parameter-value tokens left out when `mask_values`,
    and the same-reply later openers always) and its decision points (reply-start openers up, the stop down),
    returned separately as (body_loss, decision_sum, n_points, stats)."""
    import torch
    from torch.utils.checkpoint import checkpoint
    from trl.experimental.sdpo.loss_utils import compute_topk_self_distillation_loss

    from sdpo_ab_runner import inner_hidden, lm_head

    student, teacher = branch["student"], branch["teacher"]
    s_extra = [student["positions"][i] for i in branch["openers"]]
    t_extra = [teacher["positions"][i] for i in branch["openers"]]
    signs = [1] * len(branch["openers"])
    if branch.get("stop_positions"):
        s_extra.append(branch["stop_positions"][0])
        t_extra.append(branch["stop_positions"][1])
        signs.append(-1)
    with torch.no_grad():
        model.set_adapter("sdpo_teacher")
        model.eval()
        ht_all = inner_hidden(model, teacher["ids"], teacher["positions"] + t_extra).detach()
    model.set_adapter("default")
    model.train()
    hs_all = inner_hidden(model, student["ids"], student["positions"] + s_extra)
    n = len(student["positions"])
    hs, ht, hs_d, ht_d = hs_all[:n], ht_all[:n], hs_all[n:], ht_all[n:]
    head = lm_head(model)
    skip = set(branch["openers"]) | set(branch.get("same_reply_openers") or [])
    if mask_values:
        skip |= set(branch.get("value_indices") or [])
    body = [i for i in range(len(student["targets"])) if i not in skip]

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
    body_loss = total / max(1, len(body))
    per, d_stats = two_sided_term(head(hs_d), head(ht_d), opener, signs, cap, floor)
    with torch.no_grad():
        targets = torch.tensor(student["targets"], device=hs.device)
        logp = torch.cat([head(hs[s:s + chunk].detach()).float().log_softmax(-1).gather(
            -1, targets[s:s + chunk, None]).flatten() for s in range(0, len(student["targets"]), chunk)])
    stats = {"body_tokens": len(body), "value_tokens_masked": len(branch.get("value_indices") or []) if mask_values else 0,
             "body_loss": float(body_loss.detach()),
             "decision": {**d_stats, "kinds": branch["kinds"], "loss": float(per.detach().sum())},
             "student_logp_inserted_mean": float(logp.mean()), "student_p_inserted_mean": float(logp.exp().mean()),
             "student_logp_inserted_min": float(logp.min())}
    return body_loss, per.sum(), len(signs), stats


# ------------------------------------------------------------------ steps (F6) and weights

def plan_steps(rows_by_task, branches_by_task, rows_per_step, rng, tasks=None):
    """Mixed-task steps: every admitted row of the round, shuffled across tasks, in steps of ceil(rows / tasks) rows
    (so the round has about one step per task, R0's step count and EMA lag; `rows_per_step` is the fallback when
    `tasks` is not given); the round's branches shuffled and spread over the steps round-robin. Each row is
    (task, row); each branch is the branch dict. Returns [{"rows": [...], "branches": [...]}]."""
    rows = [(task, row) for task in sorted(rows_by_task) for row in rows_by_task[task]]
    branches = [b for task in sorted(branches_by_task) for b in branches_by_task[task]]
    if tasks:
        rows_per_step = max(1, math.ceil(len(rows) / tasks))
    rng.shuffle(rows)
    rng.shuffle(branches)
    steps = [{"rows": rows[i:i + rows_per_step], "branches": []} for i in range(0, len(rows), rows_per_step)]
    if not steps and branches:
        steps = [{"rows": [], "branches": []} for _ in range(max(1, math.ceil(len(branches) / 4)))]
    for i, branch in enumerate(branches):
        steps[i % len(steps)]["branches"].append(branch)
    return steps


def weight_suggestion(steps, branch_share, decision_share):
    """From the dry run's per-step gradient norms ({"r0_norm", "branch_norm", "decision_norm"} at weight 1): the
    weight that makes each term's norm the given share of R0's, by the smallest of the ratio of means, the median
    per-step ratio, and the **max-norm bound** mean R0 norm / max per-step term norm (so the term's largest step
    stays at most `share` of R0's mean: review finding 5). Steps whose decision term has no active point are left
    out of the decision rule and counted."""
    out = {"rule": "weight = share x min(mean R0 norm / mean term norm, median per-step ratio, mean R0 norm / max "
                   "per-step term norm); the term's gradient norm is then at most `share` of R0's at the untrained "
                   "adapter, in the mean and at its largest step"}
    for term, share in (("branch", branch_share), ("decision", decision_share)):
        both = [s for s in steps if s.get("r0_norm") and s.get(f"{term}_norm")]
        if term == "decision":
            both = [s for s in both if s.get("active_points", 1)]
        if not both:
            out[term] = {"steps": 0, "proposal": None, "share": share}
            continue
        r0_mean = statistics.mean(s["r0_norm"] for s in both)
        ratio_of_means = r0_mean / statistics.mean(s[f"{term}_norm"] for s in both)
        median_ratio = statistics.median(s["r0_norm"] / s[f"{term}_norm"] for s in both)
        max_bound = r0_mean / max(s[f"{term}_norm"] for s in both)
        out[term] = {"steps": len(both), "r0_norm_mean": r0_mean,
                     f"{term}_norm_mean": statistics.mean(s[f"{term}_norm"] for s in both),
                     f"{term}_norm_max": max(s[f"{term}_norm"] for s in both),
                     "ratio_of_means": ratio_of_means, "median_ratio": median_ratio, "max_norm_bound": max_bound,
                     "share": share, "proposal": share * min(ratio_of_means, median_ratio, max_bound)}
    return out


# ------------------------------------------------------------------ monitors (closeout 2.6, 5.8)

def direction_balance(records):
    """Per kind and per position class over a round's decision records (rows and branches): points, active points,
    active loss, and the active **gradient mass** (sum of |p_s - q|, the per-point gradient in the pushed token's
    logit). Shares are read within a class (`POSITION_CLASS`: first_token, after_text, within_reply), reported only
    (the balance bar is waived; the enforced bars are `structure_bars`)."""
    by_kind = defaultdict(lambda: {"points": 0, "active": 0, "loss": 0.0, "grad": 0.0, "p_student": 0.0,
                                   "p_teacher": 0.0})
    for record in records:
        grads = record.get("grad") or [abs(p - q) * a for p, q, a in zip(record["p_student"], record["target"],
                                                                         record["active"])]
        for kind, active, kl, g, ps, pt in zip(record["kinds"], record["active"], record["kl"], grads,
                                                record["p_student"], record["p_teacher"]):
            entry = by_kind[kind]
            entry["points"] += 1
            entry["active"] += int(active)
            entry["loss"] += kl if active else 0.0
            entry["grad"] += g
            entry["p_student"] += ps
            entry["p_teacher"] += pt
    for entry in by_kind.values():
        entry["p_student"] /= entry["points"]
        entry["p_teacher"] /= entry["points"]
        entry["active_share"] = entry["active"] / entry["points"]
    classes = {}
    members = defaultdict(list)
    for kind, cls in POSITION_CLASS.items():
        members[cls].append(kind)
    for cls, kinds in members.items():
        up = sum(by_kind[k]["grad"] for k in kinds if k in by_kind and DIRECTION[k] > 0)
        down = sum(by_kind[k]["grad"] for k in kinds if k in by_kind and DIRECTION[k] < 0)
        total = up + down
        classes[cls] = {"kinds": kinds, "up_grad": up, "down_grad": down,
                        "up_active": sum(by_kind[k]["active"] for k in kinds if k in by_kind and DIRECTION[k] > 0),
                        "down_active": sum(by_kind[k]["active"] for k in kinds if k in by_kind and DIRECTION[k] < 0),
                        "up_share": up / total if total else None, "down_share": down / total if total else None}
    up = sum(v["grad"] for k, v in by_kind.items() if DIRECTION[k] > 0)
    down = sum(v["grad"] for k, v in by_kind.items() if DIRECTION[k] < 0)
    total = up + down
    return {"by_kind": dict(by_kind), "classes": classes, "up_grad": up, "down_grad": down,
            "up_loss": sum(v["loss"] for k, v in by_kind.items() if DIRECTION[k] > 0),
            "down_loss": sum(v["loss"] for k, v in by_kind.items() if DIRECTION[k] < 0),
            "up_active": sum(v["active"] for k, v in by_kind.items() if DIRECTION[k] > 0),
            "down_active": sum(v["active"] for k, v in by_kind.items() if DIRECTION[k] < 0),
            "up_share": up / total if total else None, "down_share": down / total if total else None,
            "kind_grad_share": {k: v["grad"] / total for k, v in by_kind.items()} if total else {},
            "branch_stops_active": by_kind[BRANCH_STOP]["active"] if BRANCH_STOP in by_kind else 0,
            "trained_stops_active": sum(v["active"] for k, v in by_kind.items() if k in STOP_KINDS),
            "untrained_kinds": sorted(k for k in by_kind if k not in TRAINED_KINDS),
            "p_student_by_kind": {k: v["p_student"] for k, v in by_kind.items()}}


def structure_bars(balance, branch_counts, stop_min=DEFAULTS["stop_min"], replay_drop_max=DEFAULTS["replay_drop_max"]):
    """The guarded run's enforced bars (design-guarded 1.5), read every round and by the dry run. Returns
    {"S1_structure", "S2_stop_supply", "S3_replay_health"}: each {"value", "pass"}, plus "failed": the stop lines."""
    untrained = list((balance or {}).get("untrained_kinds") or [])
    stops = int((balance or {}).get("trained_stops_active") or 0)
    continued = int(branch_counts.get("candidates", 0))
    replay = int(branch_counts.get("drop:replay_error", 0)) + int(branch_counts.get("drop:replay_mismatch", 0))
    share = replay / continued if continued else 0.0
    bars = {"S1_structure": {"value": untrained, "pass": not untrained},
            "S2_stop_supply": {"value": stops, "min": stop_min, "pass": stops >= stop_min},
            "S3_replay_health": {"value": share, "dropped": replay, "continued": continued, "max": replay_drop_max,
                                 "pass": share <= replay_drop_max}}
    bars["failed"] = [f"{name}: {bar['value']}" for name, bar in bars.items() if not bar["pass"]]
    return bars


def reply_start_positions(student, tokenizer=None, thinking=False):
    """[(sequence position, final)] for every reply start after a tool result (turn >= 2; with thinking on, after that
    reply's `</think>`): the position whose next token is the reply's call-or-text decision; `final` marks the
    attempt's last reply."""
    out, last = [], max(student["turns"]) if student["turns"] else None
    for turn in sorted(set(student["turns"])):
        if turn < 2:
            continue
        index = reply_start_index(tokenizer, student["targets"], student["turns"], turn, thinking) if thinking \
            else next(i for i, t in enumerate(student["turns"]) if t == turn)
        if index is not None:
            out.append((student["positions"][index], turn == last))
    return out


PROBE_GROUPS = ("after_result", "final_done", "final_failed")


def select_probe(rows_by_task, n, rng, tokenizer=None, thinking=False):
    """The fixed P(call) probe (design-guarded 1.6, E-review 4 issue 1): n attempts from round 1's admitted rows, half
    successes and half failures where there are enough (seeded draws), each with its reply starts after a tool result
    grouped as after_result (a non-final reply: the student went on), final_done (a success's final reply: it should
    stop there) and final_failed (a failure's final reply: where branch_first pushes up; reported only)."""
    pools = {True: [], False: []}
    for task in sorted(rows_by_task):
        for row in sorted(rows_by_task[task], key=lambda r: r["record"]["id"]):
            if row.get("loop_row"):
                continue
            starts = reply_start_positions(row["full"], tokenizer, thinking)
            if not starts:
                continue
            success = bool(row.get("success"))
            groups = defaultdict(list)
            for position, final in starts:
                groups[("final_done" if success else "final_failed") if final else "after_result"].append(position)
            pools[success].append({"episode_id": row["record"]["id"], "task": task, "success": success,
                                   "ids": list(row["full"]["ids"]), "groups": dict(groups)})
    wins = min(len(pools[True]), n // 2)
    fails = min(len(pools[False]), n - wins)
    wins = min(len(pools[True]), n - fails)
    pick = lambda pool, k: [pool[i] for i in sorted(rng.sample(range(len(pool)), k))]  # noqa: E731
    return pick(pools[True], wins) + pick(pools[False], fails)


def p_call_probe(rt, probe):
    """Mean P(<tool_call>) of the current adapter per probe group (no gradient, eval mode, train mode restored). Run
    inside the training phase (rollout engine asleep); an out-of-memory item is skipped and counted."""
    import torch

    from sdpo_ab_runner import decision_tokens, inner_hidden, lm_head

    opener, _ = decision_tokens(rt.tokenizer)
    model = rt.model
    model.set_adapter("default")
    model.eval()
    values, oom, items = defaultdict(list), 0, {}
    try:
        with torch.no_grad():
            head = lm_head(model)
            for item in probe:
                order = [(g, p) for g in PROBE_GROUPS for p in item["groups"].get(g, [])]
                try:
                    hidden = inner_hidden(model, item["ids"], [p for _, p in order])
                    probs = head(hidden).float().log_softmax(-1)[:, opener].exp().tolist()
                except torch.OutOfMemoryError:
                    oom += 1
                    torch.cuda.empty_cache()
                    continue
                mine = defaultdict(list)
                for (group, _), value in zip(order, probs):
                    values[group].append(value)
                    mine[group].append(value)
                items[item["episode_id"]] = dict(mine)
    finally:
        model.train()
    return {"attempts": len(probe), "oom_skipped": oom, "items": items,
            **{g: {"mean": statistics.mean(values[g]) if values[g] else None, "positions": len(values[g]),
                   "above_half": sum(v > 0.5 for v in values[g])} for g in PROBE_GROUPS}}


def _common_means(probe, untrained):
    """Per group, the two readings' means over the attempts read in both (an out-of-memory skip in one reading must
    not move the other's mean: E-review 4 re-review, N2). Falls back to the stored means without per-attempt values."""
    a_items, b_items = (untrained or {}).get("items"), (probe or {}).get("items")
    out = {}
    for g in PROBE_GROUPS:
        if a_items is None or b_items is None:
            out[g] = ((untrained.get(g) or {}).get("mean"), (probe.get(g) or {}).get("mean"))
            continue
        common = sorted(set(a_items) & set(b_items))
        a = [v for e in common for v in a_items[e].get(g, [])]
        b = [v for e in common for v in b_items[e].get(g, [])]
        out[g] = (statistics.mean(a) if a else None, statistics.mean(b) if b else None)
    return out


def probe_stops(probe, untrained, up_after=DEFAULTS["probe_after_result_up"],
                up_done=DEFAULTS["probe_final_done_up"], down_ratio=DEFAULTS["probe_down_ratio"]):
    """The p_call_probe stop lines (ROUND_RULES) of one reading against the untrained one."""
    stops = []
    if not probe or not untrained:
        return stops
    means = _common_means(probe, untrained)
    a, b = means["after_result"]
    if a is not None and b is not None and (b > a + up_after or b < down_ratio * a):
        stops.append(f"p_call_probe[after_result]: {b:.4f} vs untrained {a:.4f} (up line +{up_after}, down line "
                     f"{down_ratio} x)")
    a, b = means["final_done"]
    if a is not None and b is not None and b > a + up_done:
        stops.append(f"p_call_probe[final_done]: {b:.4f} vs untrained {a:.4f} (up line +{up_done})")
    return stops


def round_readings(outcomes, records, chains_path, max_repeat, p_call_values, balance, guard=None):
    """The round's monitor readings (every number the rules below read)."""
    from sdpo_ab_hesitation import HESITATION, stats
    import gzip

    attempts = [o for o in outcomes if not o.get("infrastructure_failure")]
    n = max(1, len(attempts))
    by_id = {r["id"]: r for r in records}
    repeated = sum(identical_call_count(by_id[o["episode_id"]]) >= max_repeat for o in attempts
                   if o["episode_id"] in by_id)
    name_runs = sum(name_run(by_id[o["episode_id"]]) >= 10 for o in attempts if o["episode_id"] in by_id)
    hes = stats(chains_path)
    calls = finals = final_doubt = call_replies = 0
    for line in gzip.open(chains_path, "rt"):
        actions = [s for s in json.loads(line)["segments"] if s["kind"] == "action"]
        calls += sum((s.get("raw") or "").count("<tool_call>") for s in actions)
        call_replies += sum("<tool_call>" in (s.get("raw") or "") for s in actions)
        if actions and "<tool_call>" not in (actions[-1].get("raw") or ""):
            finals += 1
            final_doubt += len(HESITATION.findall(actions[-1].get("raw") or ""))
    return {"attempts": len(attempts),
            "turn_limit": sum(o.get("stop_condition") == "max_turns" for o in attempts),
            "identical_calls": repeated, "identical_calls_share": repeated / n,
            "name_run_ge10": name_runs, "name_run_ge10_share": name_runs / n,
            "calls_per_attempt": calls / n, "calls_per_call_reply": calls / max(1, call_replies),
            "reply_limit_share": hes["reply_limit_episode_share"], "doubt_per_reply": hes["hesitation_per_reply"],
            "final_doubt_per_reply": final_doubt / max(1, finals),
            "p_call_after_result_mean": statistics.mean(p_call_values) if p_call_values else None,
            "p_call_after_result_n": len(p_call_values), "balance": balance, "guard": guard}


def round_stops(round1, current, k, guard_reference=None, guard_drop=DEFAULTS["guard_drop"], guard_loops_reference=None,
                probe_untrained=None):
    """The stop lines of round k (ROUND_RULES), [] when every rule passes. Rules against round 1 apply for k >= 2;
    the structure bars, the P(call) probe (against `probe_untrained`) and the guard rules apply every round."""
    stops = []
    if k >= 2 and round1:
        n = current["attempts"]
        if current["turn_limit"] > max(3 * round1["turn_limit"], 0.02 * n):
            stops.append(f"turn_limit: {current['turn_limit']} > max(3 x {round1['turn_limit']}, 2% of {n})")
        a, b = round1["identical_calls_share"], current["identical_calls_share"]
        if b > max(3 * a, a + 0.005):
            stops.append(f"identical_calls: {b:.4f} > max(3 x {a:.4f}, {a:.4f} + 0.005)")
        ratio = current["calls_per_attempt"] / round1["calls_per_attempt"] if round1["calls_per_attempt"] else 1.0
        if not 0.75 <= ratio <= 1.33:
            stops.append(f"calls_per_attempt: ratio {ratio:.3f} outside 0.75-1.33")
        if current["reply_limit_share"] >= 0.10:
            stops.append(f"reply_limit: {current['reply_limit_share']:.3f} >= 0.10")
        if current["doubt_per_reply"] >= 0.45:
            stops.append(f"doubt: {current['doubt_per_reply']:.3f} >= 0.45")
        if current["final_doubt_per_reply"] > 2 * round1["final_doubt_per_reply"]:
            stops.append(f"final_doubt: {current['final_doubt_per_reply']:.3f} > 2 x {round1['final_doubt_per_reply']:.3f}")
        for kind in (BRANCH_STOP,):
            a = ((round1.get("balance") or {}).get("p_student_by_kind") or {}).get(kind)
            b = ((current.get("balance") or {}).get("p_student_by_kind") or {}).get(kind)
            if a is not None and b is not None and abs(b - a) > 0.15:
                stops.append(f"later_opener_drift: {kind} {b:.3f} vs round 1 {a:.3f}")
    stops += (current.get("bars") or {}).get("failed") or []
    stops += probe_stops(current.get("p_call_probe"), probe_untrained)
    guard = current.get("guard")
    if guard is not None and guard_reference is not None and guard["mean_task_success"] < guard_reference - guard_drop:
        stops.append(f"guard: {guard['mean_task_success']:.3f} < {guard_reference:.3f} - {guard_drop}")
    if guard is not None and guard_loops_reference is not None:
        loops = guard.get("loop_attempts")
        if loops is not None and loops > max(2 * guard_loops_reference, guard_loops_reference + 4):
            stops.append(f"guard_loops: {loops} > max(2 x {guard_loops_reference}, {guard_loops_reference} + 4)")
    return stops


# ------------------------------------------------------------------ safe log (F12)

def safe_log(**fields):
    """`sdpo_ab_runner.log` that survives a closed stdout (R0 lost its 4th round to EPIPE, closeout F12)."""
    import sys

    try:
        print(json.dumps({"t": __import__("time").time(), **fields}, default=str), flush=True)
    except (BrokenPipeError, OSError):
        try:
            sys.stdout = open("/dev/null", "w")  # noqa: SIM115 - keep the process alive, drop the log
        except OSError:
            pass


# ------------------------------------------------------------------ GPU orchestration (called from the runner's
# `--sf-final` blocks; every function here mirrors the R0/R1 code it replaces, file:line in the comments)

async def build_round_branches(rt, args, records, joined, label, outcomes):
    """The round's stop branches (as `sdpo_ab_runner.r1_build_branches`, 1954-2005): the EMA teacher served by the
    rollout engine at temperature 1.0, top-p 1.0, worlds rebuilt in one worker thread. Writes branch-build.json."""
    import asyncio
    import concurrent.futures
    import datetime as dt
    import time

    import sdpo_ab_world as world
    from sdpo_ab_branches import BRANCH_SAMPLING
    from sdpo_ab_execution import world_date
    from sdpo_ab_runner import (THINKING, log, parsed_calls, request_seed, runaway, task_of, tool_call_parser,
                                visible_reply, write)

    by_task = defaultdict(list)
    for record in records:
        by_task[task_of(record)].append(record)
    loop = asyncio.get_running_loop()
    worker = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="sf-world")
    started = time.monotonic()

    async def run_in_world(function, *arguments):
        return await loop.run_in_executor(worker, function, *arguments)

    async def generate(ids, max_tokens, seed):
        ids, _logps, finish = await rt.rollout.sample(list(ids), max_tokens, seed=seed)
        return ids, finish

    def parse_for(chain):
        def parse(text):
            native, parser = tool_call_parser(rt.tokenizer, chain["messages0"], chain["tools"])
            return parsed_calls(parser.extract_tool_calls(text, native))
        return parse

    saved = rt.rollout.temperature, rt.rollout.top_p, rt.rollout.extra_sampling
    rt.rollout.temperature, rt.rollout.top_p = BRANCH_SAMPLING["temperature"], BRANCH_SAMPLING["top_p"]
    rt.rollout.extra_sampling = {}
    await rt.rollout.load_teacher(rt.model)
    try:
        result = await build_final_branches(
            records, joined, tokenizer=rt.tokenizer, engine=rt.branch_engine, generate=generate, parse_for=parse_for,
            world=world, day=dt.date.fromisoformat(world_date()), run_in_world=run_in_world,
            siblings_of=lambda record: by_task[task_of(record)],
            seed_for=lambda key: request_seed(f"{args.seed_salt}|sf-branch", key),
            outcomes_by_id={o["episode_id"]: o for o in outcomes}, runaway_fn=runaway,
            candidates_per_task=args.sf_branch_candidates, points_per_task=args.sf_points_per_task, keep=args.sf_keep,
            max_repeat=args.sf_max_repeat, log=log, concurrency=args.concurrency,
            visible=(lambda raw: visible_reply(raw)[1]) if THINKING else None, thinking=THINKING)
    finally:
        await rt.rollout.restore_student()
        rt.rollout.temperature, rt.rollout.top_p, rt.rollout.extra_sampling = saved
        worker.shutdown(wait=True)
    summary = summarize_branches(result["rows"])
    # the guarded run trains no redundant-call point (design-guarded 1.1), so their replay is not run
    redundant, r_counts = {}, Counter(not_run=1)
    result["redundant"], result["redundant_counts"] = redundant, r_counts
    write(rt.out / label / "branch-build.json", {"summary": summary, "counts": dict(result["counts"]),
                                                  "redundant_counts": dict(r_counts),
                                                  "redundant": {k: {x: v[x] for x in ("turn", "call", "call_index", "success")}
                                                                for k, v in redundant.items()},
                                                  "elapsed_s": time.monotonic() - started,
                                                  "engine_stats": dict(rt.branch_engine.stats)})
    log(stage="sf_branches", label=label, elapsed=round(time.monotonic() - started), **dict(result["counts"]),
        **{"redundant_" + k: v for k, v in r_counts.items()})
    return result


def prepare_task(rt, args, name, task_records, joined, build_context, contracts, placement, variant, saved,
                 timings, admission, redundant=None):
    """Admission and sequences of one task's attempts (replaces `train_arm`'s loop body, sdpo_ab_runner 1706-1767):
    F5 drops, arm A for successes (F4), R0's sequences, the attempt's own decision points, the monitor rows.
    `redundant`: {episode_id: {"turn", "call"}} from `find_redundant_calls`. An attempt that F5 drops but holds a
    redundant call is admitted as a decision-only "loop row" (anchor + that point, no distillation): the student's
    own loop gets its corrective push where the grader grounds it (E-review issue 10).
    Returns the task's rows: dicts with record, context, student, teacher, anchor, points (uncapped), monitor."""
    import time

    from sdpo_ab_runner import (THINKING, context_record, pick, restrict_to_calls, runaway, student_sequence,
                                teacher_sequence, text_anchor)

    redundant = redundant or {}
    rows = []
    for record in task_records:
        chain = joined.get(record["id"])
        entry = {"episode_id": record["id"], "task": name}
        if chain is None or chain.get("mismatch"):
            admission.append({**entry, "admitted": False, "reason": "chain_not_joined"})
            continue
        success = succeeded(record)
        problems = attempt_problems(record, chain, args.sf_max_repeat, runaway)
        loop_row = bool(problems) and record["id"] in redundant and "rerender_mismatch" not in problems
        if problems and not loop_row:
            admission.append({**entry, "admitted": False, "reason": "sf:" + ",".join(problems), "success": success})
            continue
        if loop_row:  # anchor + the redundant-call point only; the engine is not consulted; the chain is cut after
            # the redundant call's reply, so the loop that followed is neither anchored nor seen (E-review 2, issue 5)
            cut = cut_chain_after_turn(chain, redundant[record["id"]]["turn"])
            student_full = student_sequence(cut)
            anchor = text_anchor(rt.tokenizer, student_full) if student_full["targets"] else None
            points, _ = own_points(rt.tokenizer, student_full, None, False, True, THINKING, redundant[record["id"]])
            points = [q for q in points if q["kind"] in (REDUNDANT_CALL, REDUNDANT_IN_REPLY)]
            if anchor is None or not points:
                admission.append({**entry, "admitted": False, "reason": "sf:" + ",".join(problems), "success": success})
                continue
            context = neutral_context_for(chain)
            teacher_full = teacher_sequence(rt.tokenizer, cut, context)
            rows.append({"record": record, "context": context, "student": pick(student_full, [], ("positions", "targets", "old", "turns")),
                         "teacher": pick(teacher_full, [], ("positions", "targets")), "anchor": anchor,
                         "full": student_full, "teacher_full": teacher_full, "points": points, "unmatched": 0,
                         "monitor": [], "exclude": [], "success": success, "neutral": True, "loop_row": True,
                         "final_start_position": None, "final_has_call": True})
            admission.append({**entry, "admitted": True, "loop_row": True, "problems": problems, "success": success,
                              "tokens": len(student_full["ids"]), "assistant_tokens": len(student_full["targets"]),
                              "loss_tokens": 0, "anchor_tokens": len(anchor["targets"]),
                              "own_points": dict(Counter(q["kind"] for q in points))})
            continue
        started = time.monotonic()
        arm = arm_for(args, success)
        try:
            context = build_context(record, task_records, arm, placement, contracts[name], variant=variant)
            timings.append({"episode_id": record["id"], "task": name, "arm": arm, "error": False,
                            "seconds": time.monotonic() - started,
                            "solution": bool(context and context.get("solution_episode"))})
        except ValueError as error:
            admission.append({**entry, "admitted": False, "reason": f"builder: {error}"})
            continue
        except Exception as error:  # noqa: BLE001 - D45: an engine crash excludes, never stops
            rt.engine_context.stats["builder_error"] += 1
            admission.append({**entry, "admitted": False, "reason": f"builder_error: {type(error).__name__}: {error}"})
            continue
        neutral = False
        if context is None:
            if not success:
                admission.append({**entry, "admitted": False, "reason": "no_context"})
                continue
            context, neutral = neutral_context_for(chain), True  # a success's stop point needs no feedback
        else:
            saved.append(context_record(context, arm))
        student_full = student_sequence(chain)
        if not student_full["targets"]:
            admission.append({**entry, "admitted": False, "reason": "no_assistant_tokens"})
            continue
        teacher_full = teacher_sequence(rt.tokenizer, chain, context)
        anchor = text_anchor(rt.tokenizer, student_full)
        actions = [s for s in chain["segments"] if s["kind"] == "action"]
        points, unmatched = own_points(rt.tokenizer, student_full, None if neutral else context, success,
                                       bool(actions[-1].get("complete", True)), THINKING, redundant.get(record["id"]))
        untrained = Counter(p["kind"] for p in points if p["kind"] not in TRAINED_KINDS)
        points = [p for p in points if p["kind"] in TRAINED_KINDS]  # design-guarded 1.1
        if neutral:
            calls, teacher_calls = None, None  # a success the engine says nothing about: anchor + stop point only
        else:
            calls, teacher_calls = restrict_to_calls(rt.tokenizer, student_full, teacher_full)
        if calls is None:
            if not points or anchor is None:
                admission.append({**entry, "admitted": False,
                                  "reason": "no_call_tokens" if not neutral else "success_without_feedback_or_solution"})
                continue
            calls = pick(student_full, [], ("positions", "targets", "old", "turns"))
            teacher_calls = pick(teacher_full, [], ("positions", "targets"))
        if anchor is None:
            admission.append({**entry, "admitted": False, "reason": "no_anchor_tokens"})
            continue
        final_turn = max(student_full["turns"])
        final_start = next(i for i, t in enumerate(student_full["turns"]) if t == final_turn)
        rows.append({"record": record, "context": context, "student": calls, "teacher": teacher_calls, "anchor": anchor,
                     "full": student_full, "teacher_full": teacher_full, "points": points, "unmatched": unmatched,
                     "monitor": reply_start_rows(student_full, anchor, rt.tokenizer, THINKING), "exclude": [],
                     "success": success, "neutral": neutral,
                     # the branch's first opener sits at the final-reply start only with thinking off; with thinking
                     # on it follows an empty thinking block, a different position, so nothing is held out
                     "final_start_position": None if THINKING else student_full["positions"][final_start],
                     "final_has_call": bool(actions[-1].get("tool_call_ids"))})
        admission.append({**entry, "admitted": True, "tokens": len(calls["ids"]), "success": success,
                          "assistant_tokens": len(student_full["targets"]), "loss_tokens": len(calls["targets"]),
                          "anchor_tokens": len(anchor["targets"]), "teacher_tokens": len(teacher_full["ids"]),
                          "solution": context.get("solution_episode"), "arm": arm, "neutral_context": neutral,
                          "kinds": sorted({f["kind"] if isinstance(f, dict) else str(f) for f in context.get("facts") or []}),
                          "own_points": dict(Counter(p["kind"] for p in points)), "untrained_points": dict(untrained),
                          "unmatched_harm_facts": unmatched})
    return rows


def cap_points(rows_by_task, branches_by_task, cap, seed_for):
    """At most `cap` points of each kind per task: the attempts' own points (harm openers, success stops) are kept
    or dropped per row (the row still trains R0's term); branches are already capped by `build_final_branches`.
    A trained branch's first opener sits at its attempt's final-reply start, so that row holds the position out of
    its anchor (`exclude`). A decision-only success row (neutral context) that loses its point is removed: it would
    train nothing but its anchor (review finding 11). Returns the counts of points kept and dropped per kind."""
    counts = Counter()
    for task, rows in rows_by_task.items():
        for kind in (HARM_OPENER, HARM_IN_REPLY, SUCCESS_STOP, REDUNDANT_CALL, REDUNDANT_IN_REPLY):
            holders = [r for r in rows if any(p["kind"] == kind for p in r["points"])]
            keep = set(id(r) for r in cap_per_task(holders, cap, random.Random(seed_for(f"sf-own-points|{kind}|{task}"))))
            for row in holders:
                if id(row) in keep:
                    counts[f"kept:{kind}"] += 1
                else:
                    row["points"] = [p for p in row["points"] if p["kind"] != kind]
                    counts[f"dropped:{kind}"] += 1
        counts["neutral_rows_removed"] += sum(bool(r.get("neutral")) and not r["points"] for r in rows)
        rows[:] = [r for r in rows if not (r.get("neutral") and not r["points"])]
        by_id = {r["record"]["id"]: r for r in rows}
        for branch in branches_by_task.get(task, []):
            row = by_id.get(branch["episode_id"])
            if row is not None and not row["final_has_call"] and row.get("final_start_position") is not None:
                row["exclude"].append(row["final_start_position"])
                counts["branch_first_excluded_from_anchor"] += 1
            else:
                counts["branch_first_without_row"] += 1
    for task in [t for t, rows in rows_by_task.items() if not rows]:
        del rows_by_task[task]
    return counts


def ema_step(rt, parameters, optimizer):
    """R0's optimizer step: clip 1.0 with a finite check, AdamW, teacher EMA 0.01 (sdpo_ab_runner 1838-1847)."""
    import torch
    from peft import get_peft_model_state_dict, set_peft_model_state_dict

    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
    if not torch.isfinite(norm):
        raise ValueError("nonfinite gradient")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    student_state = get_peft_model_state_dict(rt.model, adapter_name="default")
    teacher_state = get_peft_model_state_dict(rt.model, adapter_name="sdpo_teacher")
    set_peft_model_state_dict(rt.model, {k: v.lerp(student_state[k].to(v.device), 0.01)
                                         for k, v in teacher_state.items()}, adapter_name="sdpo_teacher")
    rt.model.set_adapter("default")
    return float(norm)


def grad_norm(parameters):
    import torch

    norms = [p.grad.detach().float().norm() for p in parameters if p.grad is not None]
    return float(torch.stack(norms).norm()) if norms else 0.0


def probe_passes(dry, probe):
    """The backward passes of a step: the dry run measures each term alone (distill, anchor, branch, decision, R0's
    whole term) and trains nothing; a probe step of a training round measures the same and then runs the full "all"
    pass last, whose gradient the step uses (E-review 2, issue 2: the first version cleared it); an ordinary step
    runs "all" only."""
    measured = ("distill", "anchor", "branch", "decision", "r0")
    if dry:
        return measured
    return measured + ("all",) if probe else ("all",)


def train_round(rt, args, rows_by_task, branches_by_task, parameters, optimizer, rng, folder, batch, n_tasks=None):
    """The round's mixed-task steps (F6). Per step: R0's term (mean over the step's rows), the branch body term
    (weight w_b, mean over the step's branches) and the decision term (weight w_d, mean over every decision point of
    the step, rows and branches alike); then R0's clip, step and EMA. With `--sf-dry-run`: the gradient norm of each
    term alone at weight 1 (three passes per step), no step. Returns (steps, decision_records, p_call_values,
    dry_steps)."""
    import torch

    from sdpo_ab_runner import decision_tokens, log

    opener, end = decision_tokens(rt.tokenizer)
    cap, floor, dry = args.sf_cap, args.sf_floor, args.sf_dry_run
    probe_steps = 0 if dry else args.sf_norm_probe_steps
    w_b = 1.0 if dry else args.sf_branch_weight
    w_d = 1.0 if dry else args.sf_decision_weight
    steps_plan = plan_steps(rows_by_task, branches_by_task, max(1, args.rollouts_per_task), rng, tasks=n_tasks)
    steps, decision_records, p_call, dry_steps = [], [], [], []
    for number, plan in enumerate(steps_plan, 1):
        rows, branches = plan["rows"], plan["branches"]
        n_points = sum(len(r["points"]) for _, r in rows) + sum(len(b["kinds"]) for b in branches)
        # the dry run measures each term alone; in training the first `probe_steps` steps of the round do the same
        # before their real step (the mid-training distill/anchor split: E-review issue 5)
        probe = dry or number <= probe_steps
        passes = probe_passes(dry, probe)
        norms, oom = {}, 0
        for term in passes:
            optimizer.zero_grad(set_to_none=True)
            losses, ratios, anchored, distilled, decided = [], [], [], [], []
            for task, row in rows:
                if term == "branch":
                    continue
                try:
                    decision = row_decision(row["full"], row["teacher_full"], row["points"], opener, end) \
                        if row["points"] else None
                    loss, d_sum, n_row_points, stats = final_episode_loss(
                        rt.model, row["student"], row["teacher"], row["anchor"], decision, opener, cap,
                        exclude_positions=row["exclude"], monitor_rows=row["monitor"], floor=floor)
                    total = (loss if term in ("all", "r0") else stats["parts"][term] if term in ("distill", "anchor")
                             else loss.new_zeros(())) / len(rows)
                    if n_row_points and term in ("all", "decision"):
                        total = total + w_d * d_sum / n_points
                    if total.requires_grad:
                        total.backward()
                    losses.append(float(loss.detach()))
                    anchored.append(stats["anchor"])
                    distilled.append(stats["distill"])
                    ratios += stats["ratios"]
                    if term == ("distill" if dry else "all"):  # recorded once per step
                        p_call += stats.get("reply_start_p_call", [])
                    if n_row_points and term in ("all", "decision"):
                        decided.append(stats["decision"]["loss"])
                        if term == ("decision" if dry else "all"):
                            decision_records.append({"episode_id": row["record"]["id"], "task": task, "source": "row",
                                                     "_step": number, **stats["decision"]})
                except torch.OutOfMemoryError:
                    oom += 1
                    torch.cuda.empty_cache()
            for branch in branches:
                if term in ("distill", "anchor", "r0"):
                    continue
                try:
                    body, d_sum, n_b, stats = final_branch_loss(rt.model, branch, opener, cap, floor=floor,
                                                                mask_values=args.sf_mask_values)
                    total = w_b * body / len(branches) if term in ("all", "branch") else body.new_zeros(())
                    if term in ("all", "decision"):
                        total = total + w_d * d_sum / n_points
                    if total.requires_grad:
                        total.backward()
                    if term in ("all", "decision"):
                        decided.append(stats["decision"]["loss"])
                        if term == ("decision" if dry else "all"):
                            decision_records.append({"episode_id": branch["episode_id"], "task": branch["task"],
                                                     "source": "branch", "_step": number, **stats["decision"]})
                    branch["row"].update(body_loss=stats["body_loss"], body_tokens=stats["body_tokens"],
                                         student_p_inserted_mean=stats["student_p_inserted_mean"],
                                         student_logp_inserted_min=stats["student_logp_inserted_min"],
                                         decision=stats["decision"])
                except torch.OutOfMemoryError:
                    oom += 1
                    torch.cuda.empty_cache()
                    branch["row"]["oom"] = True
            if probe and term != "all":  # a measured pass: read the norm and clear; the "all" pass (last) keeps its
                norms[term] = grad_norm(parameters)  # gradient for the step below
                optimizer.zero_grad(set_to_none=True)
        if dry:
            active = sum(sum(r["active"]) for r in decision_records if r.get("_step") == number)
            dry_steps.append({"step": number, "rows": len(rows), "branches": len(branches), "points": n_points,
                              "active_points": active,
                              "distill_norm": norms.get("distill") if rows else None,
                              "anchor_norm": norms.get("anchor") if rows else None,
                              "r0_norm": norms.get("r0") if rows else None,
                              "branch_norm": norms.get("branch") if branches else None,
                              "decision_norm": norms.get("decision") if n_points else None})
            log(stage="sf_dry_step", folder=folder, batch=batch, **dry_steps[-1])
            continue
        if not losses and not branches:
            steps.append({"step": number, "rows": len(rows), "skipped": True, "oom": oom})
            continue
        norm = ema_step(rt, parameters, optimizer)
        probe_norms = {f"{k}_norm": v for k, v in norms.items()} if probe else {}
        steps.append({"step": number, "rows": len(rows), "tasks": len({t for t, _ in rows}), "branches": len(branches),
                      "points": n_points, "oom": oom,
                      **({"loss": sum(losses) / len(losses), "distill_loss": sum(distilled) / len(distilled),
                          "anchor_loss": sum(anchored) / len(anchored)} if losses else {}),
                      **({"decision_loss": sum(decided) / n_points} if n_points and decided else {}),
                      "grad_norm": norm, "mean_importance_weight": sum(ratios) / max(1, len(ratios)),
                      "clipped_share": sum(r >= 2.0 for r in ratios) / max(1, len(ratios)),
                      "branch_weight": w_b, "decision_weight": w_d, **probe_norms})
        log(stage="step", folder=folder, batch=batch, **steps[-1])
    return steps, decision_records, p_call, dry_steps


async def guard_round(rt, args, folder, batch):
    """guard10 (D72) after the round's checkpoint, at the test settings (card sampling, `--sf-guard-attempts`),
    served by the refreshed student; returns the collection's summary."""
    import json as _json
    from pathlib import Path

    from sdpo_frozen_environment import CARD_SAMPLING

    names = _json.loads(Path(args.sf_guard_tasks).read_text())["task_names"]
    records, _, outcomes, summary = await rt.collect(f"{folder}/guard-{batch}", names, args.sf_guard_attempts,
                                                     CARD_SAMPLING["temperature"], CARD_SAMPLING["top_p"],
                                                     sampling=dict(CARD_SAMPLING))
    return {"mean_task_success": summary["mean_task_success"], "tasks": len(names), "attempts": args.sf_guard_attempts,
            "label": f"{folder}/guard-{batch}", **guard_loops(records, outcomes, args.sf_max_repeat)}


def guard_loops(records, outcomes, max_repeat):
    """Loop readings of a guard collection at card sampling (the setting where R1's loops showed, closeout 2.6):
    attempts at the turn limit, attempts with an identical call made >= max_repeat times, and their union."""
    by_id = {r["id"]: r for r in records}
    live = [o for o in outcomes if not o.get("infrastructure_failure")]
    turn_limit = {o["episode_id"] for o in live if o.get("stop_condition") == "max_turns"}
    repeated = {o["episode_id"] for o in live if o["episode_id"] in by_id
                and identical_call_count(by_id[o["episode_id"]]) >= max_repeat}
    return {"turn_limit": len(turn_limit), "identical_calls": len(repeated), "loop_attempts": len(turn_limit | repeated),
            "scored_attempts": len(live)}


def run_round(rt, args, prepared, branch_build, parameters, optimizer, rng, folder, batch, label, outcomes, records,
              history):
    """One round's S-final training, called by `train_arm` after admission: (1) the rollout readings and the
    pre-training round rules (a stop here keeps checkpoint-(batch-1) untouched); (2) the per-task point cap and the
    anchor exclusions; (3) the mixed-task steps; (4) the round's files (branches.jsonl.gz, decision.json,
    sf-readings.json; with --sf-dry-run, sf-gradnorms.json and the run ends). Returns {"steps": [...]} or
    {"steps": [...], "stopped": <train_arm's return value>}."""
    from sdpo_ab_runner import log, request_seed, write, write_jsonl_gz

    rounds = rt.__dict__.setdefault("sf_rounds", {})
    seed_for = lambda key: request_seed(f"{args.seed_salt}|sf", key)  # noqa: E731
    readings = round_readings(outcomes, records, rt.out / label / "chains.jsonl.gz", args.sf_max_repeat, [], None)
    stops = [] if args.sf_no_round_stops else round_stops(rounds.get(1), readings, batch)
    write(rt.out / label / "sf-readings-pre.json", {"readings": readings, "stops": stops, "rules": ROUND_RULES})
    log(stage="sf_readings_pre", folder=folder, batch=batch, stops=stops,
        **{k: v for k, v in readings.items() if k not in ("balance", "guard")})
    if stops:
        # round K's rollouts come from checkpoint-(K-1): that checkpoint is the flagged one (E-review 4, issue 2)
        write(rt.out / folder / "sf-stop.json", {"batch": batch, "when": "after_collection", "stops": stops,
                                                 "use_checkpoint": batch - 2, "flagged_checkpoint": batch - 1})
        return {"steps": [], "stopped": {"gate_c": None, "history": history, "stopped": "sf_stop",
                                         "last_passing": batch - 2, "sf_stop": stops}}
    rows_by_task = {name: rows for name, rows in prepared.items() if rows}
    if batch == 1:  # the fixed P(call) probe and its untrained reading, before any training (design-guarded 1.6)
        from sdpo_ab_runner import THINKING

        rt.sf_probe = select_probe(rows_by_task, DEFAULTS["probe_attempts"], random.Random(seed_for("sf-probe")),
                                   rt.tokenizer, THINKING)
        # held out of round 1's training, so the probe reads drift on positions the anchor never held in place (E-review 4
        # re-review, N4); their branches still train. The dry run trains nothing: it keeps every row for the norms
        held = set() if args.sf_dry_run else {p["episode_id"] for p in rt.sf_probe}
        rows_by_task = {name: [r for r in rows if r["record"]["id"] not in held] for name, rows in rows_by_task.items()}
        rows_by_task = {name: rows for name, rows in rows_by_task.items() if rows}
        rounds["probe_untrained"] = p_call_probe(rt, rt.sf_probe)
        write(rt.out / folder / "sf-probe.json", {"untrained": rounds["probe_untrained"],
                                                  "episodes": [p["episode_id"] for p in rt.sf_probe]})
        log(stage="sf_probe", folder=folder, batch=batch,
            **{k: v for k, v in rounds["probe_untrained"].items() if k != "items"})
    counts = cap_points(rows_by_task, branch_build["branches"], args.sf_points_per_task, seed_for)
    replay = structure_bars({}, branch_build["counts"])["S3_replay_health"]
    if not replay["pass"] and not args.sf_no_round_stops and not args.sf_dry_run:  # S3 before any training step
        stops = [f"S3_replay_health: {replay['value']}"]
        write(rt.out / folder / "sf-stop.json", {"batch": batch, "when": "before_training", "stops": stops,
                                                 "use_checkpoint": batch - 1, "flagged_checkpoint": None})
        return {"steps": [], "stopped": {"gate_c": None, "history": history, "stopped": "sf_stop",
                                         "last_passing": batch - 1, "sf_stop": stops}}
    steps, decision_records, p_call, dry_steps = train_round(rt, args, rows_by_task, branch_build["branches"],
                                                             parameters, optimizer, rng, folder, batch,
                                                             n_tasks=len(prepared))
    balance = direction_balance(decision_records)
    bars = structure_bars(balance, branch_build["counts"])
    write_jsonl_gz(rt.out / label / "branches.jsonl.gz", branch_build["rows"])
    write(rt.out / label / "decision.json", {"balance": balance, "bars": bars, "point_counts": dict(counts),
                                             "branch_counts": dict(branch_build["counts"]), "records": decision_records})
    log(stage="sf_decision", folder=folder, batch=batch, point_counts=dict(counts),
        **{k: v for k, v in balance.items() if k != "by_kind"})
    if args.sf_dry_run:
        reading = {"batch": batch, "steps": dry_steps,
                   "suggestion": weight_suggestion(dry_steps, args.sf_branch_share, args.sf_decision_share),
                   "balance": balance, "bars": bars, "probe_untrained": rounds.get("probe_untrained"),
                   "point_counts": dict(counts), "branch_counts": dict(branch_build["counts"])}
        write(rt.out / folder / "sf-gradnorms.json", reading)
        log(stage="sf_dry_run", **{k: v for k, v in reading["suggestion"].items() if k != "rule"})
        return {"steps": steps, "stopped": {"gate_c": None, "history": history, "stopped": "sf_dry_run",
                                            "dry_run": reading}}
    readings.update(p_call_after_result_mean=statistics.mean(p_call) if p_call else None,
                    p_call_after_result_n=len(p_call), balance=balance, bars=bars,
                    # the round's trained adapter (= checkpoint-batch), read while the rollout engine sleeps
                    p_call_probe=p_call_probe(rt, rt.sf_probe))
    rounds[batch] = readings
    return {"steps": steps}


async def after_checkpoint(rt, args, folder, batch, label):
    """After the round's checkpoint and the rollout refresh: guard10 (if given) and the post-training round rules
    (the P(call) probe read at the end of the round's training, bars S1-S3, guard). Returns the stop lines ([] to continue); a stop keeps
    checkpoint-(batch-1) (D81's rule)."""
    from sdpo_ab_runner import log, write

    rounds = rt.sf_rounds
    readings = rounds[batch]
    guard = await guard_round(rt, args, folder, batch) if args.sf_guard_tasks else None
    readings["guard"] = guard
    stops = [] if args.sf_no_round_stops else round_stops(rounds.get(1), readings, batch, args.sf_guard_reference,
                                                           args.sf_guard_drop, args.sf_guard_loops_reference,
                                                           rounds.get("probe_untrained"))
    write(rt.out / label / "sf-readings.json", {"readings": readings, "stops": stops, "rules": ROUND_RULES})
    log(stage="sf_readings", folder=folder, batch=batch, stops=stops, guard=guard,
        p_call_after_result_mean=readings.get("p_call_after_result_mean"),
        p_call_probe={k: v for k, v in (readings.get("p_call_probe") or {}).items() if k != "items"},
        p_call_probe_untrained={k: v for k, v in (rounds.get("probe_untrained") or {}).items() if k != "items"},
        up_share=(readings.get("balance") or {}).get("up_share"),
        down_share=(readings.get("balance") or {}).get("down_share"),
        trained_stops_active=(readings.get("balance") or {}).get("trained_stops_active"))
    if stops:
        write(rt.out / folder / "sf-stop.json", {"batch": batch, "when": "after_training", "stops": stops,
                                                 "use_checkpoint": batch - 1})
    return stops
