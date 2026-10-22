"""Pinned-image native MCP admission check without adding test dependencies."""

import asyncio
import copy
import hashlib
import json

import httpx
from automationbench_v1.limited_tools import RejectUnknownToolArguments
from mcp.server.mcpserver import MCPServer


async def verify():
    calls = []
    server = MCPServer("progress-strict-transport")

    def read(spreadsheet_id: str, includeGridData: bool = False):
        calls.append({"spreadsheet_id": spreadsheet_id, "includeGridData": includeGridData})
        return calls[-1]

    server.add_tool(read, name="read")
    server.middleware.append(RejectUnknownToolArguments({"read": frozenset({"spreadsheet_id", "includeGridData"})}))
    app = server.streamable_http_app(json_response=True, stateless_http=True)
    async with server.session_manager.run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost:8000") as client:
            async def call(identity, arguments):
                response = await client.post("/mcp", json={"jsonrpc": "2.0", "id": identity,
                    "method": "tools/call", "params": {"name": "read", "arguments": arguments}},
                    headers={"Accept": "application/json, text/event-stream", "MCP-Protocol-Version": "2025-06-18"})
                response.raise_for_status()
                return response.json()["result"]

            rejected = await call(1, {"spreadsheet_id": "private-value", "include_grid_data": "True"})
            if not rejected.get("isError") or calls or "private-value" in json.dumps(rejected):
                raise ValueError("unknown argument was executed, accepted or leaked")
            accepted = await call(2, {"spreadsheet_id": "private-value", "includeGridData": True})
            if accepted.get("isError") or calls != [{"spreadsheet_id": "private-value", "includeGridData": True}]:
                raise ValueError("declared argument did not execute exactly once")
    import verifiers.v1 as vf
    from automationbench_v1.contracts.loader import load_task_contract
    from automationbench_v1.taskset import AutomationBenchConfig, AutomationBenchTaskConfig, AutomationBenchTaskset
    from automationbench_v1.tools import AutomationBenchState

    profile_checks = []
    for name in ("sales.linkedin_job_change", "sales.linkedin_sales_prospecting"):
        [task] = AutomationBenchTaskset(AutomationBenchConfig(domains=["sales"], task_names=[name],
            task=AutomationBenchTaskConfig(deterministic_world=True))).load()
        original = copy.deepcopy(task.data.initial_state)
        trace = vf.Trace(agent=vf.AgentInfo(config=vf.AgentConfig()),
            task=vf.TraceTask(type=type(task).__name__, data=task.data), state=AutomationBenchState())
        await task.validate(None)
        await task.setup(trace, None)
        binding = next(b for b in load_task_contract(name).bindings if b.path == ("task_evidence", "initial"))
        digest = hashlib.sha256(json.dumps(task.data.initial_state, sort_keys=True,
            ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        if original != task.data.initial_state or trace.state.initial_state != original or digest != binding.canonical_sha256:
            raise ValueError("native setup rewrote LinkedIn source or broke unchanged authority")
        profile_checks.append({"task": name, "native_validate_and_setup_preserve_source": True,
                               "original_manifest_digest_matches": True})
    return {"qualified": True, "transport": "native MCP JSON-RPC over ASGI",
            "unknown_argument_rejected_without_execution": True, "corrected_call_executions": 1,
            "no_argument_value_leak": True, "model_calls": 0, "optimizer_updates": 0,
            "linkedin_source_preservation": profile_checks}


if __name__ == "__main__":
    print(json.dumps(asyncio.run(verify())))
