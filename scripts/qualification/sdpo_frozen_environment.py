"""Frozen AutomationBench environment for the SDPO teacher-context study (plan D9, D10).

One configuration serves rollouts, evaluations and heldout62. The native taskset
rejects tool additions for tasks that are not selected, so additions are scoped
to the population of each run.
"""

from __future__ import annotations

ENVIRONMENT_REPOSITORY = "carbonteq-ai/verifiers-environments"
ENVIRONMENT_BRANCH = "feat/sdpo-automationbench"
ENVIRONMENT_BASE_COMMIT = "c0f0baa3032b3787de674e9a20d68219fcca7eed"
AUTOMATIONBENCH_DATA_COMMIT = "b3d26ab9bbb6a1b54bb6724f56ca560265e64274"
MODEL = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"

DOMAINS = ("simple", "sales", "marketing", "operations", "support", "finance", "hr")

# Upstream prompt ("~50 tool-using turns") stays unchanged, so no turn_budget.
TASK_OPTIONS = {
    "toolset": "limited_zapier",
    "search_top_k": 20,
    "capture_actions": True,
    "manifest_assessments": True,
    "world_time_context_if_declared": True,
    "strict_tool_arguments": True,
    "deterministic_world": True,
}

TOOL_ADDITIONS = {
    "finance.ap_turnover_analysis": ("slack_list_channels",),  # D25: the Slack correction channel must be discoverable
    "finance.sales_tax_remittance": ("gmail_find_email", "gmail_get_email_by_id"),
    "finance.prepaid_amortization": ("slack_list_channels",),
    "operations.calendly_equipment_inspection": ("calendly_list_event_types",),
    "support.hiver_escalation_patterns": ("jira_project",),
}

LIMITS = {"max_turns": 50, "max_reply_tokens": 4096, "max_context_tokens": 65536}
ROLLOUT_SAMPLING = {"temperature": 1.0, "top_p": 1.0}
EVALUATION_SAMPLING = {"temperature": 0.6, "top_p": 0.95}
# Correction b (2026-11-03), evaluation only (`--phase collect --sampling card`): the Qwen3.5-4B model card's
# recommended non-thinking settings for general tasks, recorded verbatim from the card. The protocol default above
# is unchanged; training rollouts keep ROLLOUT_SAMPLING.
CARD_SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5,
                 "repetition_penalty": 1.0}
# Phase 2A plan v4 (12 Nov 2026), evaluation only (`--phase collect --sampling card-thinking --thinking on`): the
# Qwen3.5-4B model card's recommended thinking-mode settings for general tasks, recorded verbatim from the card.
CARD_THINKING_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5,
                          "repetition_penalty": 1.0}
COLLECT_SAMPLINGS = {"protocol": EVALUATION_SAMPLING, "card": CARD_SAMPLING, "card-thinking": CARD_THINKING_SAMPLING}
ATTEMPTS_PER_TASK = 8


def taskset_config(task_names):
    """Verifiers taskset configuration for one run's population."""
    names = list(task_names)
    if not names or len(names) != len(set(names)):
        raise ValueError("task_names must be non-empty and unique")
    unknown = sorted({name.split(".", 1)[0] for name in names} - set(DOMAINS))
    if unknown:
        raise ValueError(f"unknown domains: {', '.join(unknown)}")
    config = {
        "id": "automationbench-v1",
        "domains": list(DOMAINS),
        "task_names": names,
        "task": dict(TASK_OPTIONS),
    }
    additions = {name: list(TOOL_ADDITIONS[name]) for name in names if name in TOOL_ADDITIONS}
    if additions:
        config["additional_tools_by_task"] = additions
    return config
