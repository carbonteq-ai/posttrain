"""CPU-only, hand-selected two-per-domain Luna reasoning-generation pilot."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from luna_ssr_sft import DOMAINS, annotation_requests, exclusions, prepare_case, sha

# Selection is based on source quality and workflow diversity, never generated
# outcomes or heldout performance. Sources/failed checks remain auditable.
SELECTION = {
    "finance.cost_center_allocation": "Allocation, sheet writes and email; zero omitted legacy assertions.",
    "finance.financial_close_checklist": "Multi-source reconciliation and Slack/email; zero omitted legacy assertions.",
    "hr.docusign_offer_letters": "Template discovery, DocuSign creation and tracking updates.",
    "hr.twilio_emergency_notification": "SMS plus email/Slack notification and spreadsheet lookup.",
    "marketing.ad_platform_audit": "Campaign lookup and status mutation, with email reporting.",
    "marketing.email_blast_suppression": "Subscriber lookup/archive with suppression constraints.",
    "operations.access_request_validation": "Conditional access workflow, Asana creation and sheet updates.",
    "operations.housekeeping_schedule": "Calendar lookup/create and scheduling with email reporting.",
    "sales.five_level_conditional": "Conditional spreadsheet/CRM lookup and Salesforce task creation.",
    "sales.zoom_calendar_conflict": "Cross-calendar conflict resolution and Zoom rescheduling.",
    "support.freshdesk_contract_renewal": "Perfect recorded source: ticket lookup/create and renewal notification.",
    "support.hiver_workload_forecast": "Near-perfect diagnostic: conversation aggregation and workload forecast.",
    "simple.email_hubspot_deal_slack": "Email retrieval -> HubSpot deal -> Slack; cross-service handoff.",
    "simple.event_asana_calendar": "Asana task and calendar discovery/create; distinct tools from other Simple task.",
}


def service(tool):
    for prefix in ("google_sheets", "google_drive", "google_calendar", "google_ads"):
        if tool.startswith(prefix + "_"):
            return prefix
    return tool.split("_", 1)[0]


def scope_rationale_schemas(requests, definitions):
    """Retain all observed facts; omit only unrelated, unused tool schemas."""
    known = {definition["function"]["name"]: definition for definition in definitions}
    previous = set()
    for request in requests:
        target = request["target_action"]["name"]
        relevant = previous | {target}
        request["prefix_evidence"] = [
            {
                **entry,
                "value": {
                    "available_tool_names": sorted(known),
                    "relevant_native_definitions": [known[name] for name in sorted(relevant & known.keys())],
                    "missing_native_definitions": sorted(relevant - known.keys()),
                },
            }
            if entry["id"] == "tools"
            else entry
            for entry in request["prefix_evidence"]
        ]
        previous.add(target)
    return requests


def prepare(project, inventory_path, output):
    from automationbench_v1.contracts.loader import load_task_contract
    from automationbench_v1.limited_tools import selected_tool_definitions
    from automationbench_v1.taskset import AutomationBenchConfig, AutomationBenchTaskConfig, AutomationBenchTaskset
    from self_feedback_signals import authority_errors

    if output.exists():
        raise ValueError("use a fresh output directory; preserve prior attempts")
    candidates = json.loads(inventory_path.read_text())
    rows = {row["task"]: row for row in candidates["tasks"]}
    excluded, exclusion_hashes = exclusions(project)
    names = list(SELECTION)
    if set(names) & set(excluded):
        raise ValueError("pilot overlaps protected tasks")
    if Counter(name.split(".")[0] for name in names) != Counter({domain: 2 for domain in DOMAINS}):
        raise ValueError("pilot must contain two unique tasks per category")
    tasks = AutomationBenchTaskset(
        AutomationBenchConfig(
            domains=list(DOMAINS),
            task_names=names,
            task=AutomationBenchTaskConfig(toolset="limited_zapier", world_time_context_if_declared=True),
        )
    ).load()
    native = {task.data.name: task for task in tasks}
    output.mkdir(parents=True)
    summaries = []
    for name in names:
        task = native[name]
        data = task.data.model_dump(mode="json")
        contract = load_task_contract(name).model_dump(mode="json", exclude_none=True)
        case = prepare_case(rows[name], data, contract)
        case["binding"]["manifest_authority_errors"] = authority_errors(contract, data["initial_state"], data["prompt"])
        tools = selected_tool_definitions(data["zapier_tools"])
        actions = case["teacher_only"]["reference_actions_and_observations"]
        observed_tools = sorted({action["call"]["name"] for action in actions})
        # Provide native definitions, not invented argument examples. Keep all
        # factual prefix observations; do not shorten long trajectories silently.
        case["tool_definitions"] = tools
        case["skeleton_request"]["available_tool_definitions"] = tools
        case["rationale_request_templates"] = annotation_requests(
            data["prompt"],
            contract,
            actions,
            {"status": "pending local skeleton generation; not a generated skeleton"},
            tools,
        )
        case["rationale_request_templates"] = scope_rationale_schemas(case["rationale_request_templates"], tools)
        case["rationale_templates_ready_for_submission"] = False
        case["selection_reason"] = SELECTION[name]
        case["verification_policy"] = (
            "Reference is an example, not gold. Check every goal and harm; identify "
            "unsupported or failed reference steps. A repair requires new execution "
            "and fresh observations. Do not rewrite recorded results. No future "
            "facts in per-action rationales. Full native replay required before SFT."
        )
        case["skeleton_request"]["verification_policy"] = case["verification_policy"]
        case["skeleton_request"]["reference_compatibility"] = case["binding"]
        case["skeleton_request"]["reference_partial_credit"] = case["source_partial_credit"]
        for request in case["rationale_request_templates"]:
            request["teacher_only_guidance"]["reference_compatibility"] = case["binding"]
            request["teacher_only_guidance"]["reference_partial_credit"] = case["source_partial_credit"]
        payload = json.dumps(case, indent=2) + "\n"
        filename = name + ".json"
        (output / filename).write_text(payload)
        summaries.append(
            {
                "task": name,
                "domain": name.split(".")[0],
                "selection_reason": SELECTION[name],
                "source_partial_credit": case["source_partial_credit"],
                "assertions_excluded": case["source_metrics"].get("assertions_excluded"),
                "actions": len(actions),
                "observed_tools": observed_tools,
                "services": sorted({service(tool) for tool in observed_tools}),
                "binding": case["binding"],
                "case_file": filename,
                "case_sha256": hashlib.sha256(payload.encode()).hexdigest(),
                "skeleton_request_characters": len(json.dumps(case["skeleton_request"])),
                "largest_rationale_request_characters": max(
                    (len(json.dumps(request)) for request in case["rationale_request_templates"]),
                    default=0,
                ),
                "sft_qualified": False,
            }
        )
    manifest = {
        "selection": "two per domain; Simple pair selected for five distinct services",
        "task_identity_sha256": sha(names),
        "tasks": summaries,
        "candidate_inventory_sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
        "exclusion_file_sha256": exclusion_hashes,
        "generator": {
            "model": "Qwen/Qwen3.5-4B",
            "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            "thinking": True,
        },
        "generation_started": False,
        "native_replay_qualified": False,
        "sft_qualified": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "tasks": len(summaries),
                "domains": dict(Counter(row["domain"] for row in summaries)),
                "cases": [
                    {"task": row["task"], "actions": row["actions"], "binding": row["binding"]} for row in summaries
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.project, args.inventory, args.output)
