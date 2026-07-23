"""AutomationBench datasets and deterministic world-state evaluation on Verifiers v1."""

from __future__ import annotations

import json
from typing import Any, Literal

import verifiers.v1 as vf
from automationbench.domains import get_available_domains, get_domain_dataset
from automationbench.rubric.registry import AssertionRegistry
from automationbench.schema.world import WorldState
from pydantic import Field

from .api_tools import AutomationBenchApiToolset
from .limited_tools import (
    AutomationBenchLimitedToolset,
    AutomationBenchLimitedToolsetConfig,
)
from .scoring import ScoreSnapshot, score_world
from .tools import AutomationBenchState, AutomationBenchToolset

type Domain = Literal["simple", "sales", "marketing", "operations", "support", "finance", "hr"]


def _strip_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strip_none(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_strip_none(item) for item in value if item is not None]
    return value


def _service_for_name(name: str) -> str | None:
    fields = sorted(
        (str(field) for field in WorldState.model_fields if field != "meta"),
        key=len,
        reverse=True,
    )
    return next((field for field in fields if name == field or name.startswith(field + "_")), None)


def _allowed_services(initial_state: dict, assertions: tuple[dict, ...], tools: tuple[str, ...]) -> list[str]:
    allowed = {key for key in initial_state if key != "meta" and key in WorldState.model_fields}
    for assertion in assertions:
        if service := _service_for_name(str(assertion.get("type", ""))):
            allowed.add(service)
    for tool in tools:
        if service := _service_for_name(tool):
            allowed.add(service)
    return sorted(allowed)


class AutomationBenchData(vf.TaskData):
    domain: Domain
    task_name: str
    initial_state: dict[str, Any]
    assertions: tuple[dict[str, Any], ...]
    zapier_tools: tuple[str, ...]


class AutomationBenchTaskConfig(vf.TaskConfig):
    tools: vf.ToolsetConfig = Field(default_factory=vf.ToolsetConfig)
    toolset: Literal["zapier", "limited_zapier", "api"] = "zapier"
    search_top_k: int = 20


class AutomationBenchTask(vf.Task[AutomationBenchData, AutomationBenchState, AutomationBenchTaskConfig]):
    tools = (AutomationBenchToolset,)

    def tool_servers(self) -> list[vf.Toolset]:
        if self.config.toolset == "api":
            return [AutomationBenchApiToolset(self.config.tools)]
        if self.config.toolset == "limited_zapier":
            config = AutomationBenchLimitedToolsetConfig.model_validate(
                {
                    **self.config.tools.model_dump(mode="python"),
                    "allowed_tools": self.data.zapier_tools,
                }
            )
            return [AutomationBenchLimitedToolset(config)]
        return super().tool_servers()

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        del runtime
        world = WorldState.model_validate(self.data.initial_state)
        world.meta.allowed_services = _allowed_services(
            self.data.initial_state,
            self.data.assertions,
            self.data.zapier_tools,
        )
        state = trace.state
        state.world = world.model_dump(mode="json")
        state.initial_state = self.data.initial_state
        state.assertions = self.data.assertions
        state.search_top_k = self.config.search_top_k

    def _snapshot(self, trace: vf.Trace) -> ScoreSnapshot:
        return score_world(
            world=trace.state.world,
            initial_state=trace.state.initial_state,
            assertions=trace.state.assertions,
        )

    async def finalize(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        del runtime
        snapshot = self._snapshot(trace)
        trace.info["automationbench"] = {
            "domain": self.data.domain,
            "task_name": self.data.task_name,
            "assertions": list(snapshot.assertion_results),
            "end_state": snapshot.end_state,
        }

    @vf.reward(weight=1.0)
    async def partial_credit(self, trace: vf.Trace) -> float:
        return self._snapshot(trace).partial_credit

    @vf.metric
    async def outcome_metrics(self, trace: vf.Trace) -> dict[str, float]:
        snapshot = self._snapshot(trace)
        return {
            "task_completed_correctly": snapshot.task_completed_correctly,
            "assertions_passed": float(snapshot.assertions_passed),
            "assertions_scored": float(snapshot.assertions_scored),
            "assertions_excluded": float(snapshot.assertions_excluded),
        }

    async def validate(self, runtime: vf.Runtime) -> bool:
        del runtime
        world = WorldState.model_validate(self.data.initial_state)
        for assertion in self.data.assertions:
            AssertionRegistry.check(world, dict(assertion))
        return True


class AutomationBenchConfig(vf.TasksetConfig):
    domains: list[Domain] = Field(default_factory=lambda: ["simple"])
    task: AutomationBenchTaskConfig = Field(default_factory=AutomationBenchTaskConfig)


class AutomationBenchTaskset(vf.Taskset[AutomationBenchTask, AutomationBenchConfig]):
    def load(self) -> list[AutomationBenchTask]:
        available = set(get_available_domains())
        unknown = set(self.config.domains) - available
        if unknown:
            raise ValueError(f"unknown AutomationBench domains: {', '.join(sorted(unknown))}")
        tasks: list[AutomationBenchTask] = []
        index = 0
        for domain in self.config.domains:
            for raw in get_domain_dataset(domain):
                info = raw.get("info", {})
                if isinstance(info, str):
                    info = json.loads(info)
                info = _strip_none(info)
                prompt = raw.get("prompt")
                initial_state = _strip_none(info.get("initial_state", {}))
                assertions = tuple(_strip_none(item) for item in info.get("assertions", []))
                zapier_tools = tuple(str(item) for item in info.get("zapier_tools", []))
                tasks.append(
                    AutomationBenchTask(
                        AutomationBenchData(
                            idx=index,
                            name=str(raw.get("task") or f"{domain}-{index}"),
                            prompt=prompt,
                            domain=domain,
                            task_name=str(raw.get("task") or f"{domain}-{index}"),
                            initial_state=initial_state,
                            assertions=assertions,
                            zapier_tools=zapier_tools,
                        ),
                        self.config.task,
                    )
                )
                index += 1
        return tasks


__all__ = ["AutomationBenchTaskset"]
