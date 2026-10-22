"""Pinned-image native proof for all seven Jira-create training fixtures."""

import argparse
import copy
import json
from pathlib import Path


def verify():
    from automationbench_v1.contracts.engine import binding_reason
    from automationbench_v1.contracts.loader import load_task_contract
    from automationbench_v1.limited_tools import AutomationBenchLimitedToolset, AutomationBenchLimitedToolsetConfig
    from automationbench_v1.taskset import AutomationBenchConfig, AutomationBenchTaskConfig, AutomationBenchTaskset
    from automationbench_v1.tools import AutomationBenchState
    from automationbench_v1.world_codec import dump_world, load_world

    rows = []
    for name, project in [
        ("operations.chemical_inventory_safety_audit", "SAFETY"),
        ("operations.fire_suppression_testing", "FIRE"),
        ("operations.twilio_afterhours_incident", "Incident Management"),
        ("simple.api_key_jira_slack", "INFRA"),
        ("simple.email_jira_bug_slack", "PLAT"),
        ("simple.jira_auth_improvements", "PROD"),
        ("support.hiver_escalation_patterns", "Support Issues"),
    ]:
        [task] = AutomationBenchTaskset(AutomationBenchConfig(domains=[name.split(".")[0]], task_names=[name],
            task=AutomationBenchTaskConfig(toolset="limited_zapier", strict_tool_arguments=True))).load()
        data = task.data.model_dump(mode="json")
        initial = copy.deepcopy(data["initial_state"])
        world = dump_world(load_world(initial))
        assert binding_reason({"task_evidence": {"initial": initial, "prompt": data["prompt"]}}, load_task_contract(name)) is None
        tools = AutomationBenchLimitedToolset(AutomationBenchLimitedToolsetConfig(
            allowed_tools=task.data.zapier_tools, strict_tool_arguments=True))
        tools._inert_state = AutomationBenchState(world=world, initial_state=copy.deepcopy(initial), capture_actions=True)
        made = json.loads(tools.invoke("jira_create_issue", project=project, summary="Fixture proof", issuetype="Task"))
        assert made["success"] is True and len(tools._inert_state.world["jira"]["issues"]) == 1
        blocked = json.loads(tools.invoke("jira_create_issue", project="UNSEEDED_UNKNOWN", summary="No mutation", issuetype="Task"))
        assert blocked["success"] is False and blocked["error"] == "jira_project_not_found"
        assert len(tools._inert_state.world["jira"]["issues"]) == 1
        rows.append({"task": name, "binding_valid": True, "persisted_creation": made,
                     "unknown_project_rejection": blocked})
    return {"qualified": True, "tasks": rows, "optimizer_updates": 0, "model_calls": 0,
            "scope": "Private native fixture/access proof only. No demonstration is supplied to any teacher/student; not full-task model success."}


def verify_discovery():
    from automationbench_v1.limited_tools import AutomationBenchLimitedToolset, AutomationBenchLimitedToolsetConfig
    from automationbench_v1.taskset import AutomationBenchConfig, AutomationBenchTaskConfig, AutomationBenchTaskset
    from automationbench_v1.tools import AutomationBenchState
    from automationbench_v1.world_codec import dump_world, load_world
    from progress_environment_profile import ACCESS, DISCOVERY, task_options, taskset_options

    name = "support.hiver_escalation_patterns"
    def task(profile):
        return AutomationBenchTaskset(AutomationBenchConfig(domains=["support"], task_names=[name],
            task=AutomationBenchTaskConfig(toolset="limited_zapier", **task_options(profile)),
            **taskset_options(profile, [name]))).load()[0]
    old, new = task(ACCESS), task(DISCOVERY)
    assert old.data.initial_state == new.data.initial_state and old.data.assertions == new.data.assertions
    assert old.data.prompt == new.data.prompt
    assert set(new.data.zapier_tools) - set(old.data.zapier_tools) == {"jira_project"}
    initial = copy.deepcopy(new.data.initial_state)
    tools = AutomationBenchLimitedToolset(AutomationBenchLimitedToolsetConfig(
        allowed_tools=new.data.zapier_tools, strict_tool_arguments=True))
    tools._inert_state = AutomationBenchState(world=dump_world(load_world(initial)),
        initial_state=copy.deepcopy(initial), capture_actions=True)
    before = copy.deepcopy(tools._inert_state.world)
    # Search terms are not project identifiers. The exact key must come from
    # this real read, not the task's hidden expected-value assertion.
    found = json.loads(tools.invoke("jira_project", searchByParameter="support"))
    assert found["success"] and found["count"] == 1
    assert tools._inert_state.world == before
    project = found["results"][0]["project"]
    made = json.loads(tools.invoke("jira_create_issue", project=project, summary="Discovery proof", issuetype="Task"))
    assert made["success"] and len(tools._inert_state.world["jira"]["issues"]) == 1
    return {"qualified": True, "task": name, "read": found, "create": made,
            "read_only_discovery": True, "initial_state_assertions_prompt_unchanged": True,
            "additional_tool": "jira_project", "model_calls": 0, "optimizer_updates": 0,
            "scope": "Private native discovery-to-create access proof, not model success or a teacher demonstration."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--discovery-only", action="store_true")
    args = parser.parse_args()
    result = verify_discovery() if args.discovery_only else verify()
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({"qualified": True, "tasks": 1 if args.discovery_only else len(result["tasks"])}))
