"""Task-scoped v1 MCP tools backed by AutomationBench's simulated world."""

from __future__ import annotations

import verifiers.v1 as vf
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.meta import execute_tool as upstream_execute_tool
from automationbench.tools.zapier.meta import search_tools as upstream_search_tools
from pydantic import Field


class AutomationBenchState(vf.State):
    world: dict[str, object] = Field(default_factory=dict)
    initial_state: dict[str, object] = Field(default_factory=dict)
    assertions: tuple[dict[str, object], ...] = ()
    search_top_k: int = 20


class AutomationBenchToolset(vf.Toolset[vf.ToolsetConfig, AutomationBenchState]):
    """AutomationBench's canonical Zapier meta-tool interface."""

    TOOL_PREFIX = None

    @vf.tool
    def search_tools(self, query: str, top_k: int = 5) -> str:
        """Find Zapier-style tools by service name, action, or description."""

        bounded = max(1, min(top_k, self.state.search_top_k))
        return upstream_search_tools(query, top_k=bounded)

    @vf.tool
    def execute_tool(self, tool_name: str, arguments: str) -> str:
        """Execute a tool found by ``search_tools`` against this rollout's world."""

        world = WorldState.model_validate(self.state.world)
        result = upstream_execute_tool(world, tool_name, arguments)
        self.state.world = world.model_dump(mode="json")
        return result


if __name__ == "__main__":
    AutomationBenchToolset.run()


__all__ = ["AutomationBenchState", "AutomationBenchToolset"]
