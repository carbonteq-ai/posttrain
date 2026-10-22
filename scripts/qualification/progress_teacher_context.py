"""Teacher-only, progress-aware context; raw hidden assertion parameters stay out."""

import copy
import hashlib
import json
from collections import Counter

VERSION = "system-progress-feedback-v1"

DECISION_BRIEFS = {
    "finance.intercompany_balance_recon": "Reconcile each entity pair in BOTH directions: HQ receivable against that division's payable to HQ; HQ payable against that division's receivable from HQ. Do not compare receivable against receivable merely because both columns have that word. Use the actual entity relationship in the returned rows. Preserve source amount strings and separately label computed differences. Report only discrepant entities to the public recipient. Silently omit fully matched entities from every outgoing section, including summaries and matched/excluded lists. Do not invent aggregate counts.",
    "hr.offboarding_automation": "Apply the ACTUALLY READ policy separately to each departure row, including Type, Last Day and Notes. A voluntary-resignation farewell rule must not be applied to an involuntary termination. Follow the observed future-date deferral and preservation rules. The request's additional severance instruction does not erase the current policy's Payroll/legal restriction; do not directly process severance with an unrelated tool. Do not invent a current date or channel ID. Only public names or previously observed resolved IDs are grounded.",
    "marketing.campaign_calendar": "Read the current conflict-detection policy if not already observed. Compare the actual date intervals AND audience/priority/status exceptions in that policy, not date overlap alone. Do not announce a postponed or otherwise ineligible campaign as a current conflict. Omit excluded campaign names rather than adding explanations about their exclusion. Resolve the audit recipient from an actual policy/source read. The public #campaign-ops name is supplied; do not invent an ID.",
    "marketing.guest_post_outreach": "Use the observed target rows and current outreach guidelines to distinguish eligible, already-pending, recently rejected and recently contacted publications. Read the guidelines if still missing; do not invent a time window, sender or policy title. Personalize only for eligible publications using their observed recent topics. Do not mark a pitch sent until an actual send has succeeded in the decision-time history. Do not search fabricated noise contacts or write status updates as a substitute for sending.",
    "operations.energy_consumption_anomaly": "Use the latest month per building, compare with that building's historical baseline, and apply the ACTUALLY OBSERVED config threshold and strictness. Do not invent the formula or count an exact boundary as above it. Keep incomplete and non-anomalous buildings out of outgoing reports entirely, including skipped/within-threshold appendices. Report actual usage, baseline and deviation only for eligible anomalies. Use a channel name/ID and director recipient supported by the public task or actual returned rows/config.",
    "sales.soft_matching": "Read the applicable contact-matching guidelines if not already observed, using public guideline terms rather than hidden expected IDs. Match the request's person, company/account, title and domain using the observed guidelines and record relationships together. Similar spelling alone does not make another person's Contact the target. Do not update the current candidate merely because its phone field exists. Once identity and requested phone are supported, update that Contact and follow the observed note requirements, preserving other contacts and unrelated fields.",
    "support.gorgias_order_lookup": "Use exact order-number matches from each actual open order-inquiry ticket; similar prefixes are not the same order. Read VIP and refund-policy evidence before claiming priority/refund eligibility. If a ticket has multiple orders, handle the required orders individually. A lookup by Customer Email is not a substitute for the order-number relationship unless the observed schema proves it. Reply with supported tracking/ship/refund information or the required escalation. Log only the successfully looked-up orders using the public Action Taken values and compute outcome counts from actual supported actions, not a planned batch.",
    "simple.gmail_reply_meeting_request": "Find the actual meeting email if not already read. Use the message/thread identifier returned by that read, not a guessed ID or unrelated message. Reply confirming the public date/time and preparation of quarterly review slides. Do not invent a personal name or additional sender fields. Preserve a valid already-supported reply instead of starting another search.",
    "simple.trello_catering_contract": "The public task supplies the board ID, not the list ID. First use the board-list tool to resolve the To Do list and inspect its actual returned ID. Do not skip that prerequisite by guessing a predictable list ID, even if a guessed ID would happen to pass a task check. If that list ID is already observed, create the requested card with the exact public title and due date. Do not create duplicate cards or unnecessarily repeat the list lookup.",
    "simple.mailchimp_full_contact": "The public request already supplies the list ID, subscriber email and first/last names. Use the declared subscriber tool and its actual field schema to add that contact. Preserve all supplied contact fields. Do not invent unrelated profile details or refuse a supported subscription action because a later result says it already succeeded in the separate recorded attempt.",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def tool_outcome(raw):
    """Keep explicit native text/MCP errors; never infer correctness from text."""
    value = raw
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except ValueError:
            if raw.startswith(("Unsupported arguments for ", "Error executing tool ")):
                return {"success": False, "error": raw, "empty": False}
            value = None
    if isinstance(value, dict):
        error = value.get("error")
        if value.get("isError") is True:
            error = value.get("content") or error or "Native MCP tool error"
        return {"success": value.get("success"), "error": error,
                "failed": value.get("success") is False or bool(error),
                "empty": any(value.get(k) == [] for k in ("results", "messages", "rows"))}
    return {"success": None, "error": None, "empty": False}


def retrieval_summary(tool, raw):
    """An empty read is an observation, never a failure/goal judgement."""
    if not any(word in tool for word in ("find", "get", "lookup", "list", "search", "query")):
        return {"empty_returned_collection": False, "returned_collection_sizes": {}}
    for _ in range(3):
        if not isinstance(raw, str):
            break
        try:
            raw = json.loads(raw)
        except ValueError:
            break
    sizes = {k: len(v) for k, v in raw.items() if k in (
        "results", "messages", "rows", "records", "tickets", "event_types", "channels", "matches")
        and isinstance(v, list)} if isinstance(raw, dict) else {}
    return {"empty_returned_collection": bool(sizes) and not any(sizes.values()),
            "returned_collection_sizes": sizes}


def repeated_successful_read(call, tool):
    """Flag exact repeated reads without intervening writes; not action credit."""
    name = tool["function"]["name"]
    if not any(word in name for word in ("find", "get", "lookup", "list", "search", "query")):
        return False

    def arguments(value):
        try:
            return json.loads(value) if isinstance(value, str) else value
        except ValueError:
            return value

    candidate, repeated = None, False
    for message in call["messages"]:
        for previous in message.get("tool_calls") or []:
            function = previous["function"]
            if any(word in function["name"] for word in ("send", "create", "update", "delete", "add", "reply", "post", "move")):
                candidate, repeated = None, False
            elif function["name"] == name and arguments(function["arguments"]) == arguments(tool["function"]["arguments"]):
                candidate = previous["id"]
        if message.get("role") == "tool" and message.get("tool_call_id") == candidate:
            outcome = tool_outcome(message.get("content"))
            repeated = outcome["success"] is True and not retrieval_summary(name, message.get("content"))["empty_returned_collection"]
    return repeated


def selected_observations(call, trace_messages):
    attempted = {t["id"]: t["function"] for t in call["message"].get("tool_calls") or []}
    return [{"tool": attempted[m["tool_call_id"]]["name"], "response": m.get("content"),
             "attempted_arguments": attempted[m["tool_call_id"]]["arguments"]}
            for m in trace_messages if m.get("role") == "tool" and m.get("tool_call_id") in attempted]


def observation_ledger(messages):
    """Cite prior tool messages without declaring successful goal completion."""
    ledger = []
    requests = {}
    for message in messages:
        for call in message.get("tool_calls") or []:
            function = call.get("function", call)
            args = function.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    pass
            requests[call["id"]] = {"tool": function.get("name"), "arguments": args}
    for index, message in enumerate(messages):
        if message.get("role") != "tool":
            continue
        raw = message.get("content")
        outcome = tool_outcome(raw)
        failed = outcome.get("failed") or outcome["success"] is False or bool(outcome["error"])
        empty = outcome["empty"]
        ledger.append({"message_index": index, "tool": message.get("name"),
                       "tool_call_id": message.get("tool_call_id"), "response_sha256": digest(raw),
                       "observed_request": requests.get(message.get("tool_call_id")),
                       "status": "explicit_error" if failed else "returned_empty" if empty else "returned",
                       "meaning": "A returned response is evidence available in the history, not a task-success claim."})
    return ledger


def observed_policy_candidates(messages):
    """Quote only policy-like bodies actually returned before this decision."""
    found, seen = [], set()

    def walk(value, index, path):
        if isinstance(value, dict):
            item = None
            if "Policy" in value and "Steps" in value:
                item = {"kind": "policy_row", "source": copy.deepcopy(value)}
            elif "subject" in value and any(w in str(value["subject"]).casefold() for w in ("policy", "guideline", "rules")):
                fields = ("id", "message_id", "subject", "from", "date", "internal_date", "body_plain", "body_html")
                item = {"kind": "policy_email_candidate", "source": {k: value[k] for k in fields if k in value},
                        "body_available": bool(value.get("body_plain") or value.get("body_html"))}
            if item is not None and digest(item) not in seen:
                seen.add(digest(item))
                found.append({"message_index": index, "source_path": path, **item})
            for key, child in value.items():
                walk(child, index, path + [key])
        elif isinstance(value, list):
            for offset, child in enumerate(value):
                walk(child, index, path + [offset])

    for index, message in enumerate(messages):
        if message.get("role") != "tool":
            continue
        raw = message.get("content")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except ValueError:
                continue
        walk(raw, index, [])
    return found


def compose(messages, sheet, assertions, observations):
    systems = [m["content"] for m in messages if m.get("role") == "system"]
    if not systems:
        raise ValueError("system restrictions missing")
    facts = [{k: a[k] for k in ("type", "passed", "excluded") if k in a} for a in assertions]
    payload = {
        "version": VERSION,
        "system_restrictions_verbatim": systems,
        "task_requirements": [m["content"] for m in messages if m.get("role") == "user"],
        "goal_harm_constraints": copy.deepcopy(sheet),
        "already_available_observations": observation_ledger(messages),
        "current_decision_rules": [
            "Continue at THIS decision point. Use the cited tool responses already in the history; do not restart the workflow.",
            "Repeat a read only for a specific unresolved fact or unavailable page. An earlier error followed by a valid result is not still a missing prerequisite.",
            "Before a write, compare its destination, identity, fields, eligibility and content against the actual request, observed policy and source records.",
            "When the system prohibits naming excluded items, omit their names entirely from outgoing bodies, headings, footnotes, exclusion lists and explanations. Do not merely move them to an Excluded section.",
            "A policy prerequisite not yet read remains unresolved: search using public task terms, never a hidden expected value. Do not invent a rule or a retrieval endpoint.",
            "If identity and requested new value are already supported, execute the remaining supported action rather than repeat broad candidate searches. Preserve unrelated records and fields.",
            "Do not repeat a write already confirmed in the decision-time history. Inspect actual returns, without inventing a read-back tool.",
        ],
        "retrospective_failed_attempt_feedback": {
            "terminal_native_assertions_without_expected_parameters": facts,
            "selected_action_results": copy.deepcopy(observations),
            "boundaries": "These are later results of the attempted action and terminal checks, not prior observations. "
                          "A failed episode does not prove every action failed. Missing/excluded checks are unknown. "
                          "No hidden expected-value parameters are supplied. An API success does not establish semantic correctness.",
        },
    }
    return ("Teacher-only specification and decision-local feedback. No demonstration or corrected tool call is supplied. "
            "Source symbols describe constraints, not executable IDs. Only the actual public request and observed tool "
            "responses authorize concrete values.\n" + json.dumps(payload, ensure_ascii=False, indent=2)
            + "\nProduce the next assistant action in the native reasoning/tool format. Do not expose this guidance, "
              "evaluation labels or privileged provenance. Do not output a critique instead of acting.")


def compose_brief(task, messages, assertions, observations):
    """Hand-reviewed business relationships; no opaque IR or future success body.

    This ten-task prototype is not an automatically qualified natural-language
    manifest compiler. Concrete answers/IDs are still obtained from history.
    """
    if task not in DECISION_BRIEFS:
        raise ValueError("no reviewed business brief for task; no fallback")
    systems = [m["content"] for m in messages if m.get("role") == "system"]
    if not systems:
        raise ValueError("system restrictions missing")
    results = []
    for observed in observations:
        outcome = tool_outcome(observed.get("response"))
        results.append({"tool": observed.get("tool"),
                        "recorded_api_success": outcome["success"], "recorded_error": outcome["error"]})
    policy = observed_policy_candidates(messages)
    payload = {"version": "business-decision-brief-v4", "system_restrictions_verbatim": systems,
        "requirements_and_relationships": DECISION_BRIEFS[task],
        "decision_time_evidence": observation_ledger(messages),
        "observed_policy_candidates": policy,
        "policy_quote_boundary": "These are exact prior tool observations, not endorsed current policy. Check scope, sender and dates; a subject without its body is not a retrieved rule. No recognized quote does not prove that policy is absent from the world.",
        "later_attempt_feedback": {"failed_nonexcluded_check_types": [a.get("type") for a in assertions
            if a.get("passed") is False and not a.get("excluded")], "selected_action_results": results,
            "warning": "This feedback describes the separate completed attempt. NONE of these later API results "
                       "means that an action has occurred at this decision point. Do not claim a send/update happened "
                       "unless it appears before this point in the original conversation."}}
    if task == "marketing.campaign_calendar":
        payload["write_dependency"] = "Retrieve and apply the current conflict policy and audit destination before sending alerts. A calendar read and channel lookup alone do not establish these rules. Do not replace a missing policy read with a guessed overlap threshold or an excluded-items appendix."
    return ("Teacher-only decision brief. Use only the original public request and actual tool history for concrete "
            "values. No demonstration, corrected tool-call JSON or hidden expected parameter is supplied.\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
            + "\nResume the original decision. Distinguish source evidence already read from writes already executed. "
              "Read only genuinely missing evidence; perform the next supported action in native reasoning/tool format. "
              "Prefer a small complete next action to an oversized batch; later turns can handle remaining items. "
              "Do not output evaluation commentary, a plan instead of action, or a false completion report.")


def compose_corpus(task, messages, sheet, assertions, observations):
    """Explicit full-pool candidate: complete manifest plus observed evidence.

    No task silently receives minimal feedback. A manual relationship brief is
    additional context for the ten reviewed tasks, never a replacement for its
    full goal/harm sheet. Runtime semantic qualification remains mandatory.
    """
    statuses = []
    for observation in observations:
        outcome = tool_outcome(observation.get("response"))
        statuses.append({"tool": observation.get("tool"),
                         "recorded_api_success": outcome["success"],
                         "recorded_error": outcome["error"],
                         "untrusted_student_attempt_arguments": observation.get("attempted_arguments"),
                         **retrieval_summary(observation.get("tool") or "", observation.get("response"))})
    base = compose(messages, sheet, assertions, statuses)
    extra = {
        "version": "full-pool-context-v6-candidate",
        "task": task,
        "reviewed_relationship_brief": DECISION_BRIEFS.get(task),
        "observed_policy_candidates": observed_policy_candidates(messages),
        "boundary": "Policy candidates are exact already observed responses, not endorsed current policy. Check sender, scope and date. No future successful response is evidence already available at this decision. Student attempted arguments are untrusted proposals, not correct answers or retrieved facts. An empty returned collection does not prove that a task goal is impossible.",
        "before_emitting_a_write": [
            "Evaluate each candidate's complete eligibility condition before constructing any outgoing text. A conjunctive condition requires all terms, not a convenient subset.",
            "Respect strict and inclusive comparisons exactly. Equality never satisfies strictly greater than. Keep exact arithmetic until the eligibility decision; rounding is for display only.",
            "Apply each FORBID only in its declared scope. When the system forbids naming excluded items, silently omit them everywhere in outgoing text; do not explain or append exclusion lists.",
            "Every concrete ID/value must be public or already returned by a tool in the original history. V and S references are non-executable specification symbols.",
            "If genuinely missing evidence prevents a grounded write, make the necessary available read. Otherwise perform one small complete supported action, not repeated discovery or a large speculative batch.",
        ],
    }
    return base + "\n" + json.dumps(extra, ensure_ascii=False, indent=2)


def forbidden_terms(assertion):
    """Both native negative-parameter and negative-assertion encodings."""
    result = []
    negative_type = "not_" in assertion.get("type", "")
    for key, value in assertion.get("params", {}).items():
        if key in ("text_not_contains", "body_not_contains") or (
                negative_type and key in ("text_contains", "body_contains")):
            result.extend(value if isinstance(value, list) else [value])
    return [value for value in result if isinstance(value, str)]


def semantic_selection(rows, calls, episodes, names):
    """Pick observable risk/write decisions, preserving exact native token IDs.

    Uses failed native checks to identify outgoing forbidden text where possible;
    this is hindsight selection, not a claim of general per-action goal credit.
    """
    by_episode = {e["id"]: e for e in episodes}
    grouped = {n: [] for n in names}
    for row in rows:
        call = calls[row["generation_index"]]
        if call["prefix_ids"] != row["student_prefix_ids"] or call["action_ids"] != row["action_ids"]:
            raise ValueError("original token association differs")
        tools = call["message"].get("tool_calls") or []
        identities = {t["id"] for t in tools}
        native_messages = [n.get("message") or {} for n in by_episode[row["episode_id"]]["traces"][0].get("nodes", [])]
        errors = [tool_outcome(m.get("content")) for m in native_messages
                  if m.get("role") == "tool" and m.get("tool_call_id") in identities]
        explicit_error = any(o["success"] is False or bool(o["error"]) for o in errors)
        read_risk = False
        if row.get("native_success") == 0:
            for tool in tools:
                responses = [m.get("content") for m in native_messages
                             if m.get("role") == "tool" and m.get("tool_call_id") == tool["id"]]
                read_risk |= any(retrieval_summary(tool["function"]["name"], raw)["empty_returned_collection"]
                                 for raw in responses) or repeated_successful_read(call, tool)
        serialized = json.dumps([t["function"] for t in tools], ensure_ascii=False).casefold()
        assertions = by_episode[row["episode_id"]]["traces"][0].get("info", {}).get("automationbench", {}).get("assertions", [])
        forbidden = []
        for assertion in assertions:
            if assertion.get("passed") is not False or assertion.get("excluded"):
                continue
            forbidden.extend(forbidden_terms(assertion))
        matched = [x for x in forbidden if isinstance(x, str) and x.casefold() in serialized]
        write_action = any(any(w in t["function"]["name"] for w in
                               ("send", "create", "update", "delete", "add", "reply", "post", "move")) for t in tools)
        rank = (not bool(matched), row.get("native_success") != 0,
                not (explicit_error or row.get("observed_tool_error", False)), not write_action,
                not read_risk, row["generation_index"])
        grouped[row["task"]].append((rank, row))
    if any(not v for v in grouped.values()):
        raise ValueError("missing complete decision for one or more frozen tasks")
    return [min(grouped[n], key=lambda x: x[0])[1] for n in names]


def balanced_batch(rows, primary, names, size=32):
    selected = list(primary)
    used = {(r["episode_id"], r["generation_index"]) for r in selected}
    candidates = {n: [r for r in rows if r["task"] == n] for n in names}
    while len(selected) < size:
        counts = Counter(r["task"] for r in selected)
        progress = False
        for name in sorted(names, key=lambda n: (counts[n], names.index(n))):
            options = [r for r in candidates[name] if (r["episode_id"], r["generation_index"]) not in used]
            if options:
                row = options[0]
                selected.append(row)
                used.add((row["episode_id"], row["generation_index"]))
                progress = True
                if len(selected) == size:
                    break
        if not progress:
            raise ValueError("insufficient distinct complete actions; no duplicate or truncated fallback")
    return selected
