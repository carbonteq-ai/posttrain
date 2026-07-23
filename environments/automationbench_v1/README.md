# AutomationBench v1 environment

This package is a native Verifiers v1 adapter for Zapier AutomationBench 1.0.5.
It reuses the upstream task builders, simulated SaaS world, API tools, and
assertion registry through the CarbonTeq Python 3.12 compatibility fork at
immutable commit `d54dbebabdba6c6eda201694aee8ddcf36ccfc51`. That commit changes
the supported Python floor and lock resolution without changing benchmark
behavior.

The adapter owns only the v1 boundary:

- typed `TaskData` and per-rollout state;
- a task-scoped MCP toolset preserving upstream's default `search_tools` and
  `execute_tool` Zapier meta-tool interface;
- an optional API-mode toolset exposing `api_search`, `api_fetch`, and
  `base64_encode` for explicit API-mode compositions;
- upstream-compatible `limited_zapier` mode that exposes only each task's
  concrete Zapier tools, suitable for smaller-policy curricula;
- deterministic final-state scoring as `partial_credit` reward and
  `task_completed_correctly` metric;
- trace metadata containing assertion results and the final world state.

The adapter and its pinned AutomationBench dependency now share the platform's
Python 3.12 runtime. `posttrain-lab[gpu-posttrain]` installs this package, so
evaluation and environment-driven GRPO use the same recorded dependency graph
as the trainer instead of an untracked patched wheel.

## Validate and run

```bash
uv run --project environments/automationbench_v1 --python 3.12 \
  --with pytest --with pytest-asyncio \
  pytest -q environments/automationbench_v1/tests

LOCAL_INFERENCE_API_KEY=EMPTY \
uv run --project environments/automationbench_v1 --python 3.12 \
  eval automationbench-v1 \
  --harness.id null \
  --taskset.domains simple \
  --model Qwen/Qwen3.5-2B \
  --client.base-url http://127.0.0.1:8000/v1 \
  --client.api-key-var LOCAL_INFERENCE_API_KEY \
  --num-tasks 1 --num-rollouts 1 --max-concurrent 1 \
  --sampling.max-tokens 2048 --sampling.temperature 0 \
  --max-turns 50 --max-total-tokens 8192 \
  --rich false --push false --output-dir /tmp/automationbench-v1
```

The endpoint must already be running. The package writes native
`config.toml`, `traces.jsonl`, and `eval.log` evidence. The environment-only
GRPO composition uses the same package in-process and retains exact resolved
task identities in native traces. The environment itself has no Trackio
dependency.
