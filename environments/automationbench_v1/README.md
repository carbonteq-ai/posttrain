# AutomationBench v1 environment

This package is a native Verifiers v1 adapter for Zapier AutomationBench 1.0.5.
It reuses the upstream task builders, simulated SaaS world, API tools, and
assertion registry at immutable commit
`a321764ace3cfbe42289e6a13abef2f0f4f56fad`.

The adapter owns only the v1 boundary:

- typed `TaskData` and per-rollout state;
- a task-scoped MCP toolset exposing `api_search`, `api_fetch`, and
  `base64_encode`;
- deterministic final-state scoring as `partial_credit` reward and
  `task_completed_correctly` metric;
- trace metadata containing assertion results and the final world state.

The upstream project currently requires Python 3.13, so this environment is an
independent package rather than a member of the platform's Python 3.12 uv
workspace. The model endpoint and Trackio host do not need to run in the same
Python environment as this package.

