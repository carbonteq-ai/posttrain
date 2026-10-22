"""Candidate teacher context projection; never a per-action correctness label.

This module is not selected by a running experiment. It removes post-action
entity payloads and private assertion parameters while retaining factual
transport/empty-result observations for a separately declared comparison.
"""

import json


def project(assertions, observations):
    outcomes = []
    excluded = 0
    for item in assertions:
        if item.get("excluded"):
            excluded += 1
            continue
        outcomes.append({"type": item["type"], "passed": item.get("passed")})
    responses = []
    for item in observations:
        value = item.get("response")
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                value = None
        evidence = {"tool": item.get("tool"), "response_status": "unknown",
                    "empty_result": None, "per_action_correctness": None}
        if isinstance(value, dict):
            error = bool(value.get("error") or value.get("errors"))
            if value.get("success") is False or error:
                evidence["response_status"] = "reported_error"
            elif value.get("success") is True:
                evidence["response_status"] = "reported_success"
            # Inspect only explicit top-level collection/count facts. An empty
            # arbitrary nested list (e.g. attachments) is not an empty search.
            collections = [value[k] for k in ("messages", "results", "records", "rows")
                           if isinstance(value.get(k), list)]
            counts = [value[k] for k in ("result_count", "total_count", "count")
                      if isinstance(value.get(k), int) and not isinstance(value[k], bool)]
            if collections or counts:
                evidence["empty_result"] = all(not x for x in collections) and all(x == 0 for x in counts)
        responses.append(evidence)
    return {"version": "status-only-candidate-v1", "terminal_native_assertions": outcomes,
            "excluded_assertion_count": excluded, "current_action_observations": responses,
            "terminal_findings_are_per_action_credit": False,
            "scope": "Post-action hindsight at the original decision point. Reported API success "
                     "does not prove task correctness; an empty successful read is not a tool error. "
                     "No response entity values or assertion parameters are supplied."}
