"""AutomationBench's task-filtered concrete Zapier tool mode."""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any, get_type_hints

import verifiers.v1 as vf
from automationbench.schema.world import WorldState
from automationbench.tools import ALL_TOOLS

from .tools import AutomationBenchState

_ZAPIER_TOOLS = {tool.__name__: tool for tool in ALL_TOOLS}


class AutomationBenchLimitedToolsetConfig(vf.ToolsetConfig):
    allowed_tools: tuple[str, ...]


class AutomationBenchLimitedToolset(vf.Toolset[AutomationBenchLimitedToolsetConfig, AutomationBenchState]):
    """Expose only the concrete Zapier tools selected by an AutomationBench task."""

    TOOL_PREFIX = None

    def invoke(self, tool_name: str, *args: Any, **kwargs: Any) -> Any:
        """Invoke one configured concrete tool and persist its world mutation."""

        if tool_name not in self.config.allowed_tools:
            raise ValueError(f"tool {tool_name!r} is not enabled for this task")
        try:
            func = _ZAPIER_TOOLS[tool_name]
        except KeyError as error:
            raise ValueError(f"unknown AutomationBench tool {tool_name!r}") from error
        world = WorldState.model_validate(self.state.world)
        cleaned = {key: value for key, value in kwargs.items() if not (isinstance(value, dict) and not value)}
        result = func(*args, world=world, **cleaned)
        self.state.world = world.model_dump(mode="json")
        return result

    def _tool_wrapper(self, tool_name: str, func: Callable[..., Any]) -> Callable[..., Any]:
        signature = inspect.signature(func)
        visible = [parameter for name, parameter in signature.parameters.items() if name != "world"]

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return self.invoke(tool_name, *args, **kwargs)

        wrapper.__signature__ = signature.replace(parameters=visible)  # type: ignore[attr-defined]
        wrapper.__annotations__ = {name: value for name, value in get_type_hints(func).items() if name != "world"}
        return wrapper

    def _register(self, mcp: Any) -> None:
        for name in self.config.allowed_tools:
            try:
                func = _ZAPIER_TOOLS[name]
            except KeyError as error:
                raise ValueError(f"unknown AutomationBench tool {name!r}") from error
            wrapped = self._tool_wrapper(name, func)
            mcp.add_tool(
                self._with_state(wrapped),
                name=name,
                description=(inspect.getdoc(func) or "").strip() or None,
            )


if __name__ == "__main__":
    AutomationBenchLimitedToolset.run()


__all__ = ["AutomationBenchLimitedToolset", "AutomationBenchLimitedToolsetConfig"]
