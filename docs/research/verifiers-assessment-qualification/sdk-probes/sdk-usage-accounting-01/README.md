# SDK output accounting qualification

Five actual unpaid executions of the pinned `openai-codex==0.160.0` worker passed. A loopback provider supplied known usage counters; no signed-in authentication or real model inference was used.

The SDK accumulated two responses reporting output 7 and 5 into output 12, while reasoning 3 and 2 became reasoning 5. It kept reasoning as a detail rather than adding it again. The pinned decoder and its upstream regression test use this same representation; their exact excerpts are in `decoder-source.json`.

The five cases cover completion below threshold, terminal completion at threshold, active-turn interruption at threshold, exact interruption at cumulative output 16,384, and a 16,384 threshold crossed by cumulative output 16,385. The last case preserves one token of observed overshoot. Active interruption finishes with `status=interrupted` and `ok=false`; passing this test means the interruption protocol worked, not that a benchmark task succeeded.

The first execution exposed a real race: terminal usage arrived after the turn had completed, and interruption returned JSON-RPC -32600, `no active turn to interrupt`. The worker now recognizes that exact typed error, retains the crossing evidence and consumes the real terminal notification. The original failed event journals and fixed executions are retained in `raw-evidence.zip` and the ignored filesystem locations listed in `summary.json`.

These results qualify the executable accounting and interruption path. They do not establish a hard pre-generation limit, signed-in provider usage completeness, generated-token alignment, Docker execution, or full campaign readiness. The provider counters were deliberately synthetic; no tokenizer estimate was substituted for reported usage.

Reproduce from `/home/hammad/projects/verifiers` with a fresh output directory:

```sh
VF_RUN_CODEX_SDK_USAGE=1 uv run --frozen pytest tests/v1/test_configs.py -k actual_unpaid_usage_accounting -q --basetemp=/tmp/sdk-usage-accounting-check
```

The opt-in fixture executes the production PEP 723 worker with its exact SDK requirement. Its subprocess uses `--no-config` to avoid the repository's historical dependency publication cutoff; it does not edit dependencies or lockfiles.
