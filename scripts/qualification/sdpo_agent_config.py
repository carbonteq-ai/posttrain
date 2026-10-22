"""Pure selection, feedback and admission rules for bounded agent SDPO research."""

import copy
import hashlib
import json
import math
from collections import defaultdict

MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
TRAIN = ["hr.offboarding_automation", "simple.email_sf_contact_mobile_update",
         "simple.gmail_reply_meeting_request", "simple.slack_ceo_announcement",
         "simple.mailchimp_add_subscriber", "simple.sf_lead_revenue_update",
         "simple.trello_catering_contract", "finance.po_email_logging"]
DEV = ["hr.handbook_acknowledgment", "simple.email_sheets_update_reply",
       "simple.sf_case_in_progress", "support.zendesk_hubspot_org_sync"]


def update_learning_rate(peak, step, steps, schedule="constant", end_ratio=.1):
    """Applied rate, indexed by actual optimizer updates, with explicit endpoints."""
    if (not math.isfinite(peak) or peak <= 0 or steps < 1 or not 1 <= step <= steps
            or not 0 < end_ratio <= 1 or schedule not in ("constant", "cosine-decay")):
        raise ValueError("invalid optimizer learning-rate schedule")
    if schedule == "constant":
        return peak
    if steps < 2:
        raise ValueError("decay needs at least two updates")
    fraction = (step - 1) / (steps - 1)
    return peak * (end_ratio + (1 - end_ratio) * (1 + math.cos(math.pi * fraction)) / 2)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def population_rollouts(cohort, online_rollouts=2):
    if online_rollouts not in (2, 4, 8):
        raise ValueError("unqualified online rollout group size")
    return online_rollouts if cohort == "online_train" else 2


def cohorts(protocol, pool, heldout, old_reserve):
    pilot = set(protocol["cohorts"]["pilot"]["task_names"])
    development = set(protocol["cohorts"]["development"]["task_names"])
    population = set(pool["task_names"])
    forbidden = set(heldout["task_names"])
    if not set(TRAIN) <= pilot or not set(DEV) <= development or population & forbidden:
        raise ValueError("cohort authority mismatch")
    unused = population - pilot - development - set(old_reserve["task_names"])
    # Freeze a new research reserve before any SDPO outcomes. Historical use
    # after the Oct11 audit remains unknown; this is not a pristine global test.
    reserve = sorted(sorted(unused, key=lambda n: fingerprint(["sdpo-reserve-20261018-v1", n]))[:24])
    if len(unused) != 44 or len(reserve) != 24:
        raise ValueError("unexpected remaining research population")
    return {"train": TRAIN, "development": DEV, "reserve": reserve,
            "reserve_sha256": fingerprint(reserve), "heldout62_sealed": True,
            "scope": "SDPO research reserve unused by prior SELF pilot/dev/reserve; later historical use not reaudited"}


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
    tasks = list(TRAIN if tasks is None else tasks)
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
