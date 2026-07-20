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

## Validate and run

```bash
uv run --project environments/automationbench_v1 --python 3.13 \
  --with pytest --with pytest-asyncio \
  pytest -q environments/automationbench_v1/tests

LOCAL_INFERENCE_API_KEY=EMPTY \
uv run --project environments/automationbench_v1 --python 3.13 \
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
`config.toml`, `traces.jsonl`, and `eval.log` evidence. Platform jobs run the
same operation through an isolated-worker adapter so their host can stream
completed trace lines and promote the directory; the environment itself has no
Trackio dependency.
