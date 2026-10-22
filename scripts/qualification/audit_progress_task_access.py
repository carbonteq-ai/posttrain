"""Ten-task read-path evidence, isolated from models and training state."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

ROUTES = {
    "finance.intercompany_balance_recon": [("google_drive_find_multiple_files", {"title": "", "max_results": 20})],
    "hr.offboarding_automation": [("google_drive_find_multiple_files", {"title": "", "max_results": 20})],
    "marketing.campaign_calendar": [("google_sheets_get_spreadsheet_by_id", {"spreadsheet_id": "ss_calendar", "includeGridData": True}),
                                    ("gmail_find_email", {"query": "conflict", "max_results": 20})],
    "marketing.guest_post_outreach": [("google_sheets_get_spreadsheet_by_id", {"spreadsheet_id": "ss_targets", "includeGridData": True}),
                                      ("gmail_find_email", {"query": "guidelines", "max_results": 20})],
    "operations.energy_consumption_anomaly": [("google_drive_find_multiple_files", {"title": "", "max_results": 20})],
    "sales.soft_matching": [("gmail_find_email", {"query": "matching", "max_results": 20}),
                             ("gmail_find_email", {"query": "HR", "max_results": 20})],
    "support.gorgias_order_lookup": [("gorgias_get_tickets", {}),
                                     ("google_sheets_get_spreadsheet_by_id", {"spreadsheet_id": "ss_orders", "includeGridData": True})],
    "simple.gmail_reply_meeting_request": [("gmail_find_email", {"query": "meeting", "max_results": 20})],
    "simple.trello_catering_contract": [("trello_board_list", {"board": "brd_events", "name": "To Do"})],
    "simple.mailchimp_full_contact": [],
}


def audit(cases):
    from automationbench_v1.limited_tools import AutomationBenchLimitedToolset, AutomationBenchLimitedToolsetConfig
    from automationbench_v1.simulation import simulated_setup, world_clock_base
    from automationbench_v1.tools import AutomationBenchState
    from automationbench_v1.world_codec import dump_world, load_world

    result = []
    for case in cases:
        data = case["episode"]["task"]["data"]
        name = data["name"]
        state = copy.deepcopy(data["initial_state"])
        clock = world_clock_base(state)
        with simulated_setup(state, clock):
            world = dump_world(load_world(copy.deepcopy(state)))
        tools = AutomationBenchLimitedToolset(AutomationBenchLimitedToolsetConfig(
            allowed_tools=tuple(data["zapier_tools"]), strict_tool_arguments=True))
        tools._inert_state = AutomationBenchState(world=world, initial_state=copy.deepcopy(state),
            deterministic_world=True, capture_actions=True, world_clock=clock.isoformat())
        rows = []
        routes = list(ROUTES[name])
        for tool, arguments in routes:
            if tool not in data["zapier_tools"]:
                rows.append({"tool": tool, "arguments": arguments, "error": "required read tool not exposed"})
                continue
            try:
                raw = tools.invoke(tool, **arguments)
                value = json.loads(raw) if isinstance(raw, str) else raw
                rows.append({"tool": tool, "arguments": arguments, "result": value})
                if tool == "google_drive_find_multiple_files" and "google_sheets_get_spreadsheet_by_id" in data["zapier_tools"]:
                    for file in value.get("results", []):
                        if file.get("id"):
                            routes.append(("google_sheets_get_spreadsheet_by_id", {
                                "spreadsheet_id": file["id"], "includeGridData": True}))
            except Exception as error:
                rows.append({"tool": tool, "arguments": arguments, "error": type(error).__name__, "message": str(error)})
        result.append({"task": name, "public_request": [m["content"] for m in data["prompt"] if m["role"] == "user"],
                       "tools": data["zapier_tools"], "read_routes": rows,
                       "declared_time": state.get("meta", {}).get("current_time"),
                       "initial_state_sha256": hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest(),
                       "note": "Mailchimp is a direct public-argument write task; use retained native baseline execution as proof." if not rows else None})
    return {"tasks": result, "model_calls": 0, "optimizer_updates": 0,
            "scope": "Real pinned native invoke on isolated states, not a model success or full MCP transport gate. Queries use public task IDs/terms or available discovery; no hidden expected rubric answers supplied. Inspect actual results before declaring access adequate."}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = audit(json.loads(args.cases.read_text())["cases"])
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"tasks": len(result["tasks"]), "read_calls": sum(len(r["read_routes"]) for r in result["tasks"]),
                      "errors": sum(bool(c.get("error")) for r in result["tasks"] for c in r["read_routes"])}))
