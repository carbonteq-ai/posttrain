"""The handover pool124 SDPO recipe ("H"), reproduced inside the D24 runner for a like-for-like comparison.

Lines between the COPIED markers are verbatim from the handover commit 979c8fe0,
scripts/qualification/sdpo_agent_config.py (aligned_messages :59, feedback_text :68, successful_reference :84,
observed_tool_error :94, balanced_rows :110, admission :167). The only edit: balanced_rows has no default task
list (the handover defaulted to its 8-task pilot list), so `tasks` is required.

Recipe constants follow 979c8fe0:scripts/qualification/sdpo_pool124_config.py (profile, collection_budget,
task_batch with mode "failure-frontier") and sdpo_agent_ablation.py (associate :197, update loop :586-700).
The glue below (`handover_rows`) rebuilds the handover's per-call rows from our recorded episodes and token
chains instead of the handover's live call log.
"""

import copy
import json
from collections import defaultdict

# 979c8fe0:sdpo_pool124_config.py collection_budget("online_train", "pool124-50") and profile(): two rollouts per
# task, 8 turns, 16,384-token context, 2,048-token replies, 12,288-token episodes; LR 1e-5 held for updates 1-10
# (hold10-decay20-flat), EMA 0.01, top-20 + tail, alpha 1.0, failure-frontier selection.
TRAINING_BUDGET = {"context": 16384, "reply": 2048, "turns": 8, "episode": 12288, "replicas": 2}
UPDATES = 10
SELECTION_MODE = "failure-frontier"

import hashlib


# ---- COPIED from 979c8fe0:scripts/qualification/sdpo_pool124_config.py ----
def step_tasks(tasks, step):
    if len(tasks) != 124 or len(set(tasks)) != 124 or not 1 <= step <= 250:
        raise ValueError("invalid pool or update")
    ordered = sorted(tasks, key=lambda task: hashlib.sha256(
        ("sdpo-pool124-20261018-v1:" + task).encode()).hexdigest())
    start = (step - 1) * 32
    return [ordered[(start + i) % 124] for i in range(32)]
# ---- END COPIED ----


def update_tasks(tasks, step):
    """The tasks rolled out and trained in one update: the handover's 32-of-124 rotation on the 124-task pool,
    otherwise every task (fit-20)."""
    return step_tasks(tasks, step) if len(tasks) == 124 else list(tasks)


# ---- COPIED from 979c8fe0:scripts/qualification/sdpo_agent_config.py ----
def aligned_messages(messages):
    value = copy.deepcopy(messages)
    for message in value:
        if message["role"] == "system":
            message["content"] = message["content"].replace(
                "a budget of ~50 tool-using turns", "a budget of at most 8 assistant turns")
    return value


def feedback_text(assertions, observations, successful_attempt=None):
    """Native facts only, teacher-only; no guessed repair or externally supplied answer."""
    if successful_attempt is not None:
        return ("A previous attempt on this same task satisfied all native checks. "
                "Use it as a reference, not as a new observation in the student's history.\n"
                "Correct solution:\n" + successful_attempt +
                "\nReconsider the next action at the original decision point.")
    factual = [{k: a[k] for k in ("type", "passed", "excluded", "params") if k in a} for a in assertions]
    return ("The following is native environment feedback from this earlier attempt. "
            "These are hindsight observations, not actions already executed at the original decision point.\n"
            + json.dumps({"native_checks": factual, "observed_tool_responses": observations}, sort_keys=True)
            + "\nCorrectly continue the original task from the original decision point. "
              "Do not repeat a search unchanged when its response supplies no matching evidence. "
              "Do not invent identifiers or claim completion without evidence.")


def successful_reference(episode, siblings):
    """Prefer this verified successful rollout; otherwise a verified sibling."""
    def successful(item):
        return item["traces"][0].get("metrics", {}).get("task_completed_correctly") == 1

    if successful(episode):
        return episode
    return next((item for item in siblings if successful(item)), None)


def observed_tool_error(responses):
    """Factual transport/validation errors, not inferred goal failures."""
    for message in responses:
        content = message.get("content", "")
        if isinstance(content, str):
            if content.startswith("Error executing tool "):
                return True
            try:
                content = json.loads(content)
            except (ValueError, TypeError):
                continue
        if isinstance(content, dict) and content.get("success") is False:
            return True
    return False


def balanced_rows(rows, batch=16, offset=0, tasks=None, mode="legacy-balanced"):
    tasks = list(tasks or ())
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("training task order must be nonempty and unique")
    groups = defaultdict(list)
    for row in rows:
        groups[row["task"]].append(row)
    if set(groups) != set(tasks):
        raise ValueError("every training task must have an eligible complete assistant action")
    result = []
    if mode not in ("legacy-balanced", "distinct-balanced", "native-error-priority", "failure-frontier",
                    "first-error-frontier", "first-risk-frontier"):
        raise ValueError("unknown action selection treatment")
    used = set()
    for i in range(batch):
        # Wider cohorts rotate task coverage across updates with fixed batch16.
        # Eight-task historical controls retain their original selection exactly.
        task = tasks[(i + (offset * batch if len(tasks) > batch else 0)) % len(tasks)]
        choices = groups[task]
        terminal = [r for r in choices if r.get("terminal_text")]
        ordinary = [r for r in choices if not r.get("terminal_text")]
        # Cover stopping as well as tool actions. Do not manufacture a terminal
        # target for an episode that never produced one.
        cycle = offset // max(1, (len(tasks) + batch - 1) // batch)
        terminal_slot = i >= len(tasks) or (len(tasks) >= batch and (i + cycle) % 2 == 1)
        selected = terminal if terminal_slot and terminal else ordinary or choices
        if mode == "native-error-priority" and not terminal_slot:
            selected = [r for r in choices if r.get("observed_tool_error")] or selected
        if mode in ("failure-frontier", "first-error-frontier", "first-risk-frontier") and not terminal_slot:
            frontier = {}
            first_errors = {}
            for row in choices:
                if ((row.get("native_success") == 0 or row.get("observed_obligation_failure") is True)
                        and not row.get("terminal_text")
                        and row.get("executed_tool_responses", 0) > 0):
                    key = row["episode_id"]
                    if key not in frontier or row["generation_index"] > frontier[key]["generation_index"]:
                        frontier[key] = row
                    risky = row.get("observed_tool_error") or (
                        mode == "first-risk-frontier" and row.get("observed_action_risks"))
                    if risky and (key not in first_errors or
                            row["generation_index"] < first_errors[key]["generation_index"]):
                        first_errors[key] = row
            selected = ((list(first_errors.values()) if mode in ("first-error-frontier", "first-risk-frontier") else [])
                        or list(frontier.values()) or selected)
        if mode != "legacy-balanced":
            def identity(row):
                return (row.get("episode_id"), row.get("generation_index", id(row)))
            fresh = [r for r in selected if identity(r) not in used]
            fresh = fresh or [r for r in choices if identity(r) not in used]
            selected = fresh or selected  # explicitly auditable if only one action exists
        chosen = selected[offset % len(selected)]
        result.append(chosen)
        used.add((chosen.get("episode_id"), chosen.get("generation_index", id(chosen))))
    return result


def admission(row, context=16384):
    if not row["complete"]:
        return "incomplete_generation"
    if not row["action_ids"]:
        return "empty_action"
    if max(len(row[k]) + len(row["action_ids"]) for k in ("student_prefix_ids", "teacher_prefix_ids")) > context:
        return "overlong_whole_transition"
    return None
# ---- end COPIED ----


def _trace(record):
    return record["traces"][0]


def _messages(record):
    return [n.get("message") for n in _trace(record)["nodes"] if n.get("message")]


def solution_json(reference):
    """The handover's "Correct solution" payload: the reference rollout's assistant and tool messages
    (979c8fe0:sdpo_agent_ablation.py:248-249)."""
    return json.dumps([n["message"] for n in _trace(reference)["nodes"]
                       if (n.get("message") or {}).get("role") in ("assistant", "tool")], sort_keys=True)


def teacher_prefix(tokenizer, chain, action_index, text, first_prompt, render_delta):
    """Tokens the teacher reads before the selected action: the student's own prefix with the feedback as a separate
    user message right before the decision (979c8fe0:sdpo_agent_ablation.py:266, `call["messages"] + [user]`).

    Turn 1 re-renders the first prompt with the extra message. Later turns keep the student's tokens up to the last
    tool results and re-render only those results plus the feedback message and the generation prompt."""
    feedback = {"role": "user", "content": text}
    segments = chain["segments"]
    tokens, last_context = [], None
    for index, segment in enumerate(segments[:action_index]):
        if segment["kind"] == "context" and segment["messages"] is not None:
            last_context = index
    actions_before = sum(s["kind"] == "action" for s in segments[:action_index])
    if actions_before == 0:
        return first_prompt(tokenizer, copy.deepcopy(chain["messages0"]) + [feedback], chain["tools"])
    if last_context is None or any(s["kind"] == "action" for s in segments[last_context + 1:action_index]):
        return None
    for segment in segments[:last_context]:
        tokens += segment["ids"]
    return tokens + render_delta(tokenizer, copy.deepcopy(segments[last_context]["messages"]) + [feedback])


def handover_rows(tokenizer, records, joined, first_prompt, render_delta, context_limit=TRAINING_BUDGET["context"],
                  feedback=None):
    """One row per sampled assistant action, as the handover's `associate` (979c8fe0:sdpo_agent_ablation.py:197-327)
    builds them for the "solution-or-feedback" mode, with the student and teacher token sequences for the loss.

    `generation_index` orders actions by episode (in collection order) and then by turn; the handover numbered calls
    in arrival order, which agrees within an episode, the only place the failure frontier compares indices.

    `feedback`, when given, replaces the handover's text: feedback(record, task_records, action_ordinal, default)
    returns the teacher text for the episode's `action_ordinal`-th assistant action (0-based); `default` is the
    handover's `feedback_text` for the same action."""
    by_task = defaultdict(list)
    for record in records:
        by_task[_trace(record)["task"]["data"]["name"]].append(record)
    rows, excluded, index = [], [], 0
    for record in records:
        trace = _trace(record)
        task = trace["task"]["data"]["name"]
        chain = joined.get(record["id"])
        if chain is None or chain.get("mismatch"):
            excluded.append({"episode_id": record["id"], "task": task, "reason": "chain_not_joined"})
            continue
        assertions = trace.get("info", {}).get("automationbench", {}).get("assertions")
        if assertions is None:
            raise ValueError("missing native factual check results")
        messages = _messages(record)
        reference = successful_reference(record, by_task[task])
        solution = solution_json(reference) if reference is not None else None
        prefix, history, ordinal = [], [], -1
        for position, segment in enumerate(chain["segments"]):
            if segment["kind"] != "action":
                prefix += segment["ids"]
                history += [m for m in segment["messages"] or [] if m.get("role") == "tool"]
                continue
            current = set(segment.get("tool_call_ids") or [])
            responses = [m for m in messages if m["role"] == "tool" and m.get("tool_call_id") in current]
            if not current and history:
                responses = history[-1:]
            observations = [{"tool": m.get("name"), "response": m.get("content")} for m in responses]
            ordinal += 1
            text = feedback_text(assertions, observations, solution)
            if feedback is not None:
                text = feedback(record, by_task[task], ordinal, text)
            teacher_ids = teacher_prefix(tokenizer, chain, position, text, first_prompt, render_delta)
            action = list(segment["ids"])
            row = {"task": task, "episode_id": record["id"], "generation_index": index,
                   "student_prefix_ids": list(prefix), "teacher_prefix_ids": teacher_ids or [],
                   "action_ids": action, "complete": bool(segment.get("complete")),
                   "sampled_logprobs": list(segment.get("logprobs") or []),
                   "terminal_text": not bool(current), "teacher_feedback": text,
                   "solution_used": solution is not None,
                   "reference_episode_id": reference["id"] if reference is not None else None,
                   "native_success": trace.get("metrics", {}).get("task_completed_correctly"),
                   "executed_tool_responses": len(responses),
                   "observed_tool_error": bool(current) and observed_tool_error(responses)}
            index += 1
            prefix += segment["ids"]
            reason = "no_teacher_prefix" if teacher_ids is None else admission(row, context=context_limit)
            if reason:
                excluded.append({"episode_id": record["id"], "task": task, "generation_index": row["generation_index"],
                                 "reason": reason})
            else:
                rows.append(row)
    return rows, excluded


def loss_sequences(row):
    """Student and teacher sequences for `episode_loss`: every token of the one selected action, nothing else."""
    student, teacher, action = row["student_prefix_ids"], row["teacher_prefix_ids"], row["action_ids"]
    return ({"ids": student + action, "positions": [len(student) + k - 1 for k in range(len(action))],
             "targets": list(action), "old": list(row["sampled_logprobs"])},
            {"ids": teacher + action, "positions": [len(teacher) + k - 1 for k in range(len(action))],
             "targets": list(action)})


def update(batch, loss_of, clip, step, ema, isfinite):
    """One handover optimizer update (979c8fe0:sdpo_agent_ablation.py:633-700): every selected action's loss divided
    by the batch size and back-propagated, one clipped step, then the EMA teacher update. Returns (losses, norm)."""
    losses = []
    for row in batch:
        loss = loss_of(row)
        if not isfinite(loss):
            raise ValueError("nonfinite SDPO loss")
        (loss / len(batch)).backward()
        losses.append(float(loss.detach()))
    norm = clip()
    if not isfinite(norm) or norm <= 0:
        raise ValueError("nonfinite/zero SDPO gradient")
    step()
    ema()
    return losses, norm


def select(rows, tasks, offset):
    """`balanced_rows` over the tasks that have an admitted action; the rest are an explicit shortfall for this update
    (979c8fe0:docs/plan/sdpo-pool124-workstation.md decision log: never duplicate or substitute a task). The handover
    code raised instead (task_batch); skipping keeps one over-long task from ending the run. Returns (batch, shortfall)."""
    present = {row["task"] for row in rows}
    eligible = [task for task in tasks if task in present]
    shortfall = [task for task in tasks if task not in present]
    if not eligible:
        raise ValueError("no task has an admitted action")
    return balanced_rows(rows, len(eligible), offset=offset, tasks=eligible, mode=SELECTION_MODE), shortfall
