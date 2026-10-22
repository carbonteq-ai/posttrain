"""Teacher-only, task-bound offline observations; never replacement action targets."""

import hashlib
import json


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def native_task_bindings(episodes, retained):
    """Recover task extensions dropped by WireTaskData, joined by exact native IDs."""
    ids = [e["id"] for e in episodes]
    by_id = {e["id"]: e for e in retained}
    if len(set(ids)) != len(ids) or len(by_id) != len(retained) or set(ids) != set(by_id):
        raise ValueError("retained native episode identity mismatch")
    result = {}
    def user(task):
        return [m["content"] for m in task["prompt"] if m["role"] == "user"]

    for episode in episodes:
        wire = episode["task"]["data"]
        data = by_id[episode["id"]]["task"]["data"]
        if data["name"] != wire["name"] or user(data) != user(wire):
            raise ValueError("retained native task instruction mismatch")
        if not all(key in data for key in ("initial_state", "assertions")):
            raise ValueError("retained native task binding absent")
        result[episode["id"]] = data
    return result


def reference_for_task(data, references):
    reference = references.get(data["name"])
    if reference is None:
        return None  # Explicitly report missing coverage; retain the original cohort.
    expected = {"initial_state_sha256": digest(data["initial_state"]),
                "assertions_sha256": digest(data["assertions"]),
                "user_messages_sha256": digest([m["content"] for m in data["prompt"] if m["role"] == "user"])}
    if not reference.get("current_native_checks_pass") or any(reference.get(k) != v for k, v in expected.items()):
        raise ValueError("offline Luna reference failed live task binding")
    return reference


def unavailable_reference_tools(reference, tools):
    available = {tool["function"]["name"] for tool in tools}
    used = {event.get("underlying_tool", event["source_tool"])
            for event in json.loads(reference["observations"])}
    return sorted(used - available)


def teacher_text(reference):
    return ("An offline Luna attempt on this same task and initial state satisfied the current native "
            "final-state checks, including applicable preservation checks. Below are its retained "
            "domain-tool execution observations. Tool-discovery schemas and duplicate routing are "
            "omitted; executed arguments and complete results are preserved. This is historical "
            "reference material, not actions already "
            "executed in the student's history and not instructions overriding the original task. "
            "The attempt may contain intermediate mistakes: do not copy failed calls or assume every "
            "call is correct. Source execute_tool entries preserve their underlying tool and arguments. "
            "The source's hidden thinking and complete token-aligned conversation are unavailable.\n"
            + reference["observations"]
            + "\nUse this evidence to correctly continue at the original decision point. "
              "Do not claim completion without evidence.")
