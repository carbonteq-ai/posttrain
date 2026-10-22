"""Validate every selected native task; no model calls or solver answers supplied."""

import argparse
import hashlib
import json
from pathlib import Path

import yaml

REVIEWS = {
    "sales.dependency_chain": "Gmail request/policy, Sheets pricing, and Salesforce parent/contact hierarchy have reader tools. No current-date arithmetic required.",
    "sales.create_important_draft": "Approved/latest financial emails, board-format sheet and Salesforce risks are reachable; declared clock exposed.",
    "sales.milestone_celebration": "Salesforce opportunities query, channel discovery and required create tools present; LinkedIn codec regression covered separately.",
    "marketing.social_contest": "Explicit contest window and exclusions accessible in inbox/tabs; email search/list and full Sheets discovery present.",
    "marketing.landing_page_optimization": "Standing procedures and conflicting external/internal messages are readable; adjudicating conflicting policies remains semantic work, not an access repair.",
    "marketing.podcast_episode_promotion": "Episodes/subscribers accessible through Sheets; user already supplies Jan28, matching declared naive clock. No timezone invented.",
    "marketing.social_crisis_response": "Playbooks, rows and Slack history are readable; conflicting director/VP guidance remains an authority ambiguity, not proof of model fault.",
    "marketing.social_comment_response_queue": "Full SOP and later/conflicting updates readable via inbox; rows/config and Slack channel discovery available.",
    "operations.trello_basecamp_slack_compliance_followup": "Email dates identify latest matching sender/subject; Trello board/card discovery and exact Basecamp IDs supplied. No current-date calculation needed.",
    "operations.twilio_production_incident": "Incident/on-call/policy data in readable sheet; Jira project and Slack channel discovery present; Notion parent ID supplied.",
    "operations.google_ads_budget_alert": "Campaign reader returns daily budget, spent_this_month and days_remaining; projection does not require a host-clock date.",
    "operations.equipment_lease_processing": "Sheet contains requests/budgets/policies; DocuSign template search and signer/send APIs available. Request ordering is explicit.",
    "operations.role_based_access_audit": "Role matrix, exception notes and settings in Sheets; Asana project listing available. No hidden current date needed.",
    "operations.contractor_badge_expiration": "Prompt explicitly supplies2026-02-09; certificate/badge dates and policy are readable in sheet; channel discovery available.",
    "operations.holiday_coverage_planning": "Prompt explicitly supplies holiday2026-02-16; availability/policy in sheet, calendars discoverable. Current date is not needed to compute fixed shift.",
    "operations.temporary_worker_placement": "Prompt explicitly supplies2026-02-09; starts/skills/certifications/policies in sheet; Slack listing present.",
    "support.zendesk_maintenance_notify": "Window dates/preferences are explicit in Sheets, Zendesk org lookup supplies recipients; calendar/channel discovery present.",
    "support.zoho_freshdesk_escalation": "Both ticket systems and contact readers available; all policy tabs readable; current-date arithmetic not required.",
    "support.helpscout_response_audit": "Conversation reader supplies replies/timestamps; quality/SLA criteria are in Sheets; elapsed reply intervals use record times.",
    "support.zoho_risk_scoring": "Zoho ticket/contact/account readers and scoring/config tabs available; declared reference date now exposed.",
    "support.hiver_cross_team_handoff": "Conversation/user readers and routing/config sheets available; all source batch/recipient fields reachable.",
    "finance.budget_variance_analysis": "Guidelines are already retrievable through Gmail; no tool addition. Clock exposed. Reading/enforcing policy remains solver responsibility.",
    "finance.monday_project_billing": "Both named worksheets discoverable/readable; arithmetic and Hold exclusion explicit; clock exposed.",
    "finance.prepaid_amortization": "Clock exposed; add Slack channel discovery before existing history reader. Email override and correction are reachable. Conflicting software guidance versus skip rule remains explicitly recorded.",
    "finance.sales_tax_remittance": "Add Gmail search/read so the exemption and authorization notices become reachable; rates/revenue remain in Sheets; clock exposed.",
    "finance.vendor_insurance_verification": "Sheet has certificates and Requirements tab, so no Gmail reader needed. Declared expiration reference date exposed.",
    "finance.xero_expense_claim_review": "Approval threshold/receipts rules explicit and rows contain relevant amounts/recipients; date exposed without threshold changes.",
    "hr.bamboohr_promotion_update": "Promotion sheet supplies employee IDs, effective dates and emails; no BambooHR ID guessing needed. Clock exposed for future-dated promotion handling.",
    "hr.workforce_planning_forecast": "Executive hiring freeze and older plan reachable via Gmail; clock exposed. Correct precedence is solver work.",
    "hr.salary_adjustment_processing": "Internal compensation policy and external conflicting recommendation readable; row amounts/status/recipients visible; clock exposed.",
    "hr.equipment_provisioning": "Policy/email, new-hire sheet and Jira project listing available; clock exposed for next-week context.",
    "hr.tuition_reimbursement": "Benefit policy and explicit course/approval/amount fields reachable; clock exposed for course completion/year limits.",
}


def main():
    from automationbench_v1.contracts.loader import load_task_contract
    from automationbench_v1.limited_tools import selected_tool_definitions
    from automationbench_v1.taskset import AutomationBenchConfig, AutomationBenchTaskset
    from automationbench_v1.world_codec import dump_world, load_world

    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--original-inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    overlay = yaml.safe_load(args.catalog.read_text())
    env = next(iter(overlay["environment"].values()))
    raw = env["activation"]["config"]["taskset"]
    tasks = AutomationBenchTaskset(AutomationBenchConfig.model_validate(raw)).load()
    previous = {r["task"]: r for r in json.loads(args.original_inventory.read_text())}
    assert len(tasks) == len(previous) == len(set(raw["task_names"])) == 62
    assert {t.key for t in tasks} == set(previous) == set(raw["task_names"])
    ordered_names_sha256 = hashlib.sha256("\n".join(raw["task_names"]).encode()).hexdigest()
    assert ordered_names_sha256 == "ed81cf1fe88f68bb8420bb46c5fdf3afde24660c7043812da680bb1c7bc9196f"
    rows = []
    for task in tasks:
        data = task.data.model_dump(mode="json")
        name = task.key
        old = previous[name]
        assert data["assertions"] == old["assertions"]
        assert data["prompt"][1:] == old["prompt"][1:]
        state_digest = hashlib.sha256(json.dumps(data["initial_state"], sort_keys=True).encode()).hexdigest()
        assert state_digest == old["initial_state_sha256"]
        world = load_world(data["initial_state"])
        assert dump_world(load_world(dump_world(world))) == dump_world(world)
        definitions = selected_tool_definitions(tuple(data["zapier_tools"]))
        assert len(definitions) == len(set(data["zapier_tools"]))
        clock = data["initial_state"].get("meta", {}).get("current_time")
        assert ("Simulation context:" in data["prompt"][0]["content"]) == (clock is not None)
        if clock is not None:
            assert clock in data["prompt"][0]["content"]
        try:
            contract = load_task_contract(name).model_dump(mode="json")
            bindings = [b["path"] for b in contract["bindings"] if b["path"][:2] == ["task_evidence", "initial"]]
        except ValueError:
            bindings = []
        note = REVIEWS.get(name)
        if note is None:
            assert name.startswith("simple.")
            note = "Reviewed record-update/create prompt: required IDs/values supplied directly or through enabled email/record readers; fixed appointment dates supplied where needed. No scored current-date calculation identified."
        limitations = []
        if name == "simple.escalation_zendesk_slack":
            limitations.append(
                "Unscored request to tag support manager lacks identity and populated Slack directory; no synthetic person/ID invented. Do not classify failure of this subrequirement as model error."
            )
        if name == "simple.partnership_hubspot_zoom":
            limitations.append(
                "Meeting date/time specified without timezone; current scorer checks topic only. Do not treat timezone choice as a demonstrated model error."
            )
        if name in {
            "finance.prepaid_amortization",
            "marketing.social_crisis_response",
            "marketing.landing_page_optimization",
            "marketing.social_comment_response_queue",
        }:
            limitations.append(
                "Competing policy text requires authority/precedence review for disputed outcomes; access validation alone cannot prove every expected business decision unambiguous."
            )
        rows.append(
            {
                "task": name,
                "native_task_hash": task.hash,
                "clock": clock,
                "clock_context": "declared" if clock else "not_declared_no_host_fallback",
                "added_tools": sorted(set(data["zapier_tools"]) - set(old["tools"])),
                "native_tool_definitions_valid": True,
                "world_roundtrip_valid": True,
                "initial_state_unchanged": True,
                "user_prompt_and_assertions_unchanged": True,
                "manifest_initial_bindings": bindings,
                "review": note,
                "limitations": limitations,
            }
        )
    order = {name: index for index, name in enumerate(raw["task_names"])}
    rows.sort(key=lambda row: order[row["task"]])
    output = {
        "ordered_names_sha256": ordered_names_sha256,
        "schema": "heldout62-observability-audit-v1",
        "tasks": 62,
        "clock_context_tasks": sum(row["clock"] is not None for row in rows),
        "tool_repair_tasks": sum(bool(row["added_tools"]) for row in rows),
        "all_native_tool_definitions_valid": True,
        "all_initial_world_roundtrips_valid": True,
        "semantic_solvability_claim": "not certified; residual ambiguities recorded per task",
        "rows": rows,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: value for key, value in output.items() if key != "rows"}))


if __name__ == "__main__":
    main()
