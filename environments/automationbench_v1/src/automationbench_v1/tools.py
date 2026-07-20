"""Task-scoped v1 MCP tools backed by AutomationBench's simulated APIs."""

from __future__ import annotations

import verifiers.v1 as vf
from automationbench.schema.world import WorldState
from automationbench.tools.api import api_fetch as upstream_api_fetch
from automationbench.tools.api import api_search as upstream_api_search
from automationbench.tools.api import base64_encode as upstream_base64_encode
from pydantic import Field


class AutomationBenchState(vf.State):
    world: dict[str, object] = Field(default_factory=dict)
    initial_state: dict[str, object] = Field(default_factory=dict)
    assertions: tuple[dict[str, object], ...] = ()
    search_top_k: int = 20


class AutomationBenchToolset(vf.Toolset[vf.ToolsetConfig, AutomationBenchState]):
    """Canonical API-mode discovery and execution tools."""

    TOOL_PREFIX = None

    @vf.tool
    def api_search(self, query: str, top_k: int = 5) -> str:
        """Search simulated SaaS API schemas for relevant endpoints."""

        bounded = max(1, min(top_k, self.state.search_top_k))
        return upstream_api_search(query, top_k=bounded)

    @vf.tool
    def api_fetch(
        self,
        method: str,
        url: str,
        params: str | None = None,
        body: str | None = None,
    ) -> str:
        """Call a discovered endpoint against this rollout's simulated world."""

        world = WorldState.model_validate(self.state.world)
        result = upstream_api_fetch(world, method, url, params=params, body=body)
        self.state.world = world.model_dump(mode="json")
        return result

    @vf.tool
    def base64_encode(self, text: str) -> str:
        """Encode text as base64url for API fields such as Gmail raw bodies."""

        return upstream_base64_encode(text)


if __name__ == "__main__":
    AutomationBenchToolset.run()


__all__ = ["AutomationBenchState", "AutomationBenchToolset"]
