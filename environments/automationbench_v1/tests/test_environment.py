from __future__ import annotations

import asyncio

import verifiers.v1 as vf
from automationbench.schema.world import WorldState
from automationbench_v1.api_tools import AutomationBenchApiToolset
from automationbench_v1.limited_tools import AutomationBenchLimitedToolset
from automationbench_v1.scoring import score_world
from automationbench_v1.taskset import AutomationBenchConfig, AutomationBenchTaskset
from automationbench_v1.tools import AutomationBenchState, AutomationBenchToolset


def test_simple_taskset_preserves_typed_world_and_assertions() -> None:
    taskset = AutomationBenchTaskset(AutomationBenchConfig(domains=["simple"]))
    task = taskset.select(1)[0]

    assert task.data.task_name == "simple.email_sf_contact_phone_update"
    assert task.data.zapier_tools == (
        "gmail_find_email",
        "gmail_get_email_by_id",
        "salesforce_find_records",
        "salesforce_contact_update",
    )
    assert task.data.assertions[0]["type"] == "salesforce_field_equals"
    assert task.data.prompt[0].content.startswith("You are a workflow automation agent")


def test_dense_and_strict_scores_use_upstream_assertion_registry() -> None:
    task = AutomationBenchTaskset(AutomationBenchConfig(domains=["simple"])).select(1)[0]
    initial = task.data.initial_state
    before = score_world(world=initial, initial_state=initial, assertions=task.data.assertions)
    assert before.partial_credit == 0.0
    assert before.task_completed_correctly == 0.0

    world = WorldState.model_validate(initial)
    world.salesforce.contacts[0].phone = "+1-555-0101"
    after = score_world(
        world=world.model_dump(mode="json"),
        initial_state=initial,
        assertions=task.data.assertions,
    )
    assert after.partial_credit == 1.0
    assert after.task_completed_correctly == 1.0
    assert after.assertions_passed == after.assertions_scored == 1


def test_default_toolset_matches_upstream_zapier_meta_tools() -> None:
    task = AutomationBenchTaskset(AutomationBenchConfig(domains=["simple"])).select(1)[0]
    state = AutomationBenchState(world=task.data.initial_state, initial_state=task.data.initial_state)
    toolset = AutomationBenchToolset(vf.ToolsetConfig())
    toolset._inert_state = state

    search = toolset.search_tools("salesforce update contact", top_k=5)
    assert "salesforce_contact_update" in search
    result = toolset.execute_tool(
        "salesforce_contact_update",
        '{"id":"003001","phone":"+1-555-0101"}',
    )
    assert "error" not in result.lower()
    world = WorldState.model_validate(toolset.state.world)
    assert world.salesforce.contacts[0].phone == "+1-555-0101"


def test_optional_api_toolset_searches_and_mutates_per_rollout_state() -> None:
    task = AutomationBenchTaskset(AutomationBenchConfig(domains=["simple"])).select(1)[0]
    state = AutomationBenchState(world=task.data.initial_state, initial_state=task.data.initial_state)
    toolset = AutomationBenchApiToolset(vf.ToolsetConfig())
    toolset._inert_state = state

    search = toolset.api_search("salesforce update contact", top_k=3)
    assert "salesforce" in search.lower()
    result = toolset.api_fetch(
        "PATCH",
        "https://example.my.salesforce.com/services/data/v61.0/sobjects/Contact/003001",
        body='{"Phone":"+1-555-0101"}',
    )
    assert "error" not in result.lower()
    world = WorldState.model_validate(toolset.state.world)
    assert world.salesforce.contacts[0].phone == "+1-555-0101"


def test_limited_zapier_mode_exposes_and_executes_only_task_tools() -> None:
    config = AutomationBenchConfig(
        domains=["simple"],
        task={"toolset": "limited_zapier"},
    )
    task = AutomationBenchTaskset(config).select(1)[0]
    [toolset] = task.tool_servers()
    assert isinstance(toolset, AutomationBenchLimitedToolset)
    assert toolset.config.allowed_tools == task.data.zapier_tools

    toolset._inert_state = AutomationBenchState(
        world=task.data.initial_state,
        initial_state=task.data.initial_state,
    )
    toolset.invoke("salesforce_contact_update", id="003001", phone="+1-555-0101")
    world = WorldState.model_validate(toolset.state.world)
    assert world.salesforce.contacts[0].phone == "+1-555-0101"


def test_task_setup_and_finalize_put_evaluation_detail_on_trace() -> None:
    task = AutomationBenchTaskset(AutomationBenchConfig(domains=["simple"])).select(1)[0]
    trace = vf.Trace(
        task=vf.TraceTask(type=type(task).__name__, data=task.data),
        state=AutomationBenchState(),
    )
    asyncio.run(task.setup(trace, None))  # type: ignore[arg-type]
    world = WorldState.model_validate(trace.state.world)
    world.salesforce.contacts[0].phone = "+1-555-0101"
    trace.state.world = world.model_dump(mode="json")
    asyncio.run(task.finalize(trace, None))  # type: ignore[arg-type]
    asyncio.run(task.score(trace))

    assert trace.reward == 1.0
    assert trace.metrics["task_completed_correctly"] == 1.0
    assert trace.info["automationbench"]["assertions"][0]["passed"] is True
