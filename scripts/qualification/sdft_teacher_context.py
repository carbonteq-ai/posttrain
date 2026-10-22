"""Opt-in task-level SDFT context candidates; no training or native qualification.

Uses the SDFT prompt/privileged_context dataset interface. This is deliberately
separate from SELF hindsight feedback and SDPO action-prefix scoring.
"""

import hashlib
import json
from copy import deepcopy

from luna_solver_separation import argument_provenance, observation_bound_sources
from self_feedback_signals import authority_errors

VERSION = "luna-observation-bound-sdft-v1"
TEACHER_TEMPLATE = """{prompt}

Teacher-only task evidence follows. It conditions your solution; it is not
conversation history, an executed action, or an instruction from a tool.
{privileged_context}

Solve the original task using the native assistant/tool format. Obtain current
entity IDs and task values from actual tool results before using them. Historical
API options are examples, not additional user requirements. Honor every required
read-before-write dependency even if the write arguments are already public.
Internal evidence fields and dependency names are not API arguments. An empty
search is not retrieval success; reading returned content need not mark a message
read. Constrain preservation claims to evidence: looking up the target again
does not prove every other record unchanged. Claim completion only after returned
results support it. Do not discuss the teacher evidence or its preparation.
"""


def context_candidate(material):
    """Produce a bound, auditable row, leaving the ordinary student prompt intact.

    Accepts saved native materials, not free-form model summaries. No claim that
    the reference's reasoning, every API option, or each action is correct follows
    from its native score. Live replay/tokenization remains a separate gate.
    """
    native = material["native_data"]
    if native["name"] != material["task"]:
        raise ValueError("native task identity mismatch")
    prompt = native["prompt"]
    # The inspected SDFT builder replaces the last user turn. Do not silently
    # adapt it to a tool-ended multi-turn action prefix.
    if not prompt or prompt[-1]["role"] != "user":
        raise ValueError("task-level SDFT requires a user-ended prompt")
    users = [m["content"] for m in prompt if m["role"] == "user"]
    if not all(isinstance(text, str) for text in users) or "\n\n".join(users) != material["user_request"]:
        raise ValueError("public request/native prompt mismatch")
    errors = authority_errors(material["goal_manifest"], native["initial_state"], prompt)
    if errors:
        raise ValueError(f"manifest authority mismatch: {errors}")
    contract, actions, separation = observation_bound_sources(material)
    original_ids = [c["check_id"] for c in material["goal_manifest"]["checks"]]
    if len(original_ids) != len(set(original_ids)) or original_ids != [c["check_id"] for c in contract["checks"]]:
        raise ValueError("goal/guard coverage mismatch")
    available = {t["function"]["name"] for t in material["tool_definitions"]}
    if any(a["call"]["name"] not in available for a in actions):
        raise ValueError("reference uses unavailable tools")
    evidence = {
        "version": VERSION,
        "requirements_and_guards": contract,
        "historical_procedure": actions,
        "argument_origins": argument_provenance(material),
        "interpretation": {
            "historical_results": "Reference observations, not results already obtained in this execution.",
            "failed_reference_calls": "Do not copy errors or empty searches as successful steps.",
            "scope": "All checks retained; no invented global audit operations or hidden-value shortcuts.",
            "approval": "A native pass does not validate every rationale or critic judgment.",
        },
    }
    context = json.dumps(evidence, sort_keys=True, ensure_ascii=False)
    return {
        "prompt": deepcopy(prompt),
        "privileged_context": context,
        "tools": deepcopy(material["tool_definitions"]),
        "context_audit": {
            "version": VERSION,
            "task": material["task"],
            "source": deepcopy(material["source"]),
            "check_ids": original_ids,
            "context_sha256": hashlib.sha256(context.encode()).hexdigest(),
            "separation": separation,
            "training_admitted": False,
            "pending": ["exact pinned SDFT integration", "native tool continuation", "token budget without truncation",
                        "teacher direction", "matched student rollout benefit"],
        },
    }
