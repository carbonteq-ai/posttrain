# Thread handoff

Handoff for work in `/home/hammad/projects/rl` as of 2026-07-20.

Read [the execution plan](../.agents/plan/posttraining-platform-refactor.md),
[platform architecture](./architecture.md), and
[post-training lifecycle](./functional/finetuning-lifecycle.md) first.

## Canonical implementation

```text
packages/common   framework-free identities, model/artifact types, execution context
packages/serve    typed vLLM profiles, workloads, prompts, benchmark operation
packages/eval     endpoint-neutral Verifiers v1 operation and code-defined programs
packages/train    typed SFT/DPO/GRPO operations and internal TRL adapters
packages/reports  read-only evidence/query boundary
apps/lab          code-defined jobs, execution host, Trackio observer
environments/automationbench_v1  independently locked native-v1 environment
```

The target is code-first. Reusable behavior lives in `posttrain.common`,
`posttrain.serve`, `posttrain.eval`, and `posttrain.train`. Jobs compose those
APIs in ordinary Python. Trackio observes attempts; it does not define jobs,
profiles, scheduling, or execution.

The new serving path has no dependency on Trackio, YAML, the legacy `common`
package, or the lab host. Its representative prompt corpus is a package
resource, so installing the wheel is sufficient to use it. Root serving YAML,
the old benchmark suite YAML, and the old serving CLIs were deleted rather than
retained as compatibility surfaces.

## Implemented slices

- Immutable foundation profiles for Qwen3.5-2B and LFM2.5-1.2B-Thinking.
- Typed vLLM standard, TurboQuant K8V4, and Qwen MTP profiles.
- A typed 1K-to-32K workload matrix with concurrency 1/2/4/8; this machine is
  intentionally limited to 1/2/4.
- Public `posttrain.serve.benchmark(context, request)` with an internal vLLM
  adapter, typed result, metrics, standard inference trace, and result artifact.
- Code-defined foundation-screening job and Trackio-backed lab execution host.
- Batched run-level metrics and mandatory job/action/invocation/attempt
  identity on every tracked attempt.
- Temporary execution workspaces; durable evidence belongs in Trackio.
- Import-linter boundaries preventing reusable packages from importing the lab
  and preventing serving from importing observation or legacy config systems.
- Public `posttrain.eval.evaluate(context, request)` with typed endpoint,
  program, environment-cell, context, reasoning, and invocation-budget inputs.
- Pinned upstream GSM8K and three other general environment packages, plus
  code-defined general, agentic-smoke, and AutomationBench domain programs.
- Native Verifiers v1 AutomationBench 1.0.5 port with per-rollout world state,
  API discovery/execution MCP tools, dense reward, strict outcome metrics, and
  full assertion/end-state trace detail.
- Public observer-neutral SFT, DPO, and GRPO operations with model renderers,
  QLoRA profiles, recovery checkpoints, selected adapters, and summaries.
- A backend-neutral `OnlineRLBridge` and `PolicyGenerator` contract. Native
  Verifiers episodes own multi-turn/tool execution and scoring; the private TRL
  adapter supplies the already-loaded policy and receives exact token
  sequences, logprobs, environment masks, rewards, and traces.
- Colocated Qwen GRPO with one trainable QLoRA model, an immutable vLLM
  bitsandbytes base, dynamic adapter synchronization, level-1 sleep, and no
  separately loaded Verifiers policy.

## GPU evidence

Comparable 128-input/32-output, concurrency-1 cells completed on the RTX 3070
Ti 8 GB using vLLM 0.25.1:

| Model | Trackio run | Output rate | TTFT | Peak VRAM |
| --- | --- | ---: | ---: | ---: |
| Qwen3.5-2B | `serve/qwen3.5-2b/short-interactive-ctx1024-c1-a7b17eb5-a1` | 75.66 tok/s | 35.5 ms | 6.99 GiB |
| LFM2.5-1.2B-Thinking | `serve/lfm2.5-1.2b-thinking/short-interactive-ctx1024-c1-13f0e79a-a1` | 170.90 tok/s | 14.2 ms | 6.35 GiB |

These are path-validation cells, not a completed capacity study. Qwen uses a
text-only, eager, no-offload 8 GB-safe profile. First-shape Triton JIT warnings
still appear during warmup and startup optimization remains open.

The managed LFM OpenAI endpoint also completed through canonical run
`serve-online/lfm2.5-1.2b-thinking-95a0371d-a1` at clean revision `371a49c`.
Health and model probes passed; a streamed reasoning response produced final
content `4.` with 22 input tokens, 169 output tokens, 42 ms TTFT, and 884 ms
end-to-end latency. The searchable trace is compact; the complete streamed
response and server log are separate Trackio artifacts. Reasoning-only
truncations are rejected rather than recorded as successful model responses.

The endpoint-neutral GSM8K cell then ran at an explicit 8,192-token evaluation
context with a 4,096-token response ceiling. These are one-task integration
checks, not capability estimates:

| Model | Trackio run | Reward | Trace outcome |
| --- | --- | ---: | --- |
| Qwen3.5-2B | `eval/general/qwen3.5-2b/math-gsm8k-05f21808-a1` | 1.0 | expected and produced `18`; complete, no error, not truncated |
| LFM2.5-1.2B-Thinking | `eval/general/lfm2.5-1.2b-thinking/math-gsm8k-8991dbf7-a1` | 0.0 | produced `32`; complete but truncated after repetitive 4,096-token generation |

Each run contains one queryable Verifiers trace, direct synchronization-health
metrics, the complete native evaluation directory, and its serving log. This
demonstrates why run completion and trace quality remain distinct facts.

## Training evidence

| Technique | Trackio run ID | Parent | Steps | Key signal |
| --- | --- | --- | ---: | --- |
| SFT | `b549afa7241942bfa6ed31cc4fdacffd` | Qwen3.5 foundation | 2 | final loss `0.9181`, grad norm `4.6875` |
| DPO | `9a89fda28de34c6d9254995402becba9` | SFT adapter `v0` | 2 | final loss `0.3474`, grad norm `0.1387` |
| GRPO | `07984dfc3feb44e1b34dcd5b92e2d850` | SFT adapter `v0` | 1 | native episode bridge, reward std `0.00139`, grad norm `0.05420` |

The canonical GRPO run is from clean revision `e7babfc`. Verifiers drove two
native episodes through the policy client; both were correct, completed with
`agent_completed`, and produced 251 and 232 sampled tokens. Their traces
preserve model identity, exact token IDs/logprobs, train masks, reward
components, and stop state. The run also produced the native trace artifact,
GRPO adapter, step-1 recovery checkpoint, and summary. Loss was `-3.997e-06`
and the nonzero gradient proves the requested backpropagation pass. The
sequence-level importance ratio reached the configured `0.1` floor, so that
profile remains a tuning target even though the architectural acceptance gate
passed.

A live `automationbench-v1` simple task also completed through the independent
Python 3.13 runtime against the Qwen endpoint. The MCP tool server started, Qwen
made correctly parsed `api_search` calls, final-state scoring ran, and the trace
retained assertion and end-world detail. Qwen repeated a schema search until
the six-turn diagnostic limit, made no Salesforce mutation, and scored 0. This
was a runtime qualification, not yet a canonical tracked job; the generic
isolated-worker composition is the remaining integration step.

## Commands

```bash
cd /home/hammad/projects/rl
uv sync --all-packages --group dev --python 3.12
uv run --group dev ruff check .
uv run --group dev pyright
uv run --group dev lint-imports
uv run --group dev pytest -q --cov --cov-fail-under=65

uv run --package posttrain-serve --extra vllm \
  posttrain-lab foundation-qwen-smoke --tracked --project posttrain-foundation
uv run --package posttrain-serve --extra vllm \
  posttrain-lab foundation-lfm-smoke --tracked --project posttrain-foundation
uv run --package posttrain-serve --extra vllm \
  posttrain-lab foundation-lfm-online-smoke --tracked --project posttrain-foundation
uv run --package posttrain-lab --extra gpu-eval \
  posttrain-lab foundation-qwen-gsm8k --tracked --project posttrain-foundation
uv run --package posttrain-lab --extra gpu-eval \
  posttrain-lab foundation-lfm-gsm8k --tracked --project posttrain-foundation

uv run --project environments/automationbench_v1 --python 3.13 \
  --with pytest --with pytest-asyncio pytest -q environments/automationbench_v1/tests

uv run --package posttrain-lab --extra gpu-posttrain \
  posttrain-lab gsm8k-qwen-sft-smoke --tracked --project posttrain-platform
uv run --package posttrain-lab --extra gpu-posttrain \
  posttrain-lab gsm8k-qwen-grpo-smoke --tracked \
  --project posttrain-platform --adapter-version v0
```

The module-backed CLI is required for GPU jobs because vLLM may use spawned
workers after CUDA initialization; do not launch those operations from stdin.

## Next slices

1. Add a generic isolated-environment executor so the proven Python 3.13
   AutomationBench runtime streams traces and promotes its native directory in
   the same Trackio attempt as the managed model endpoint.
2. Add report-side reward/pass/truncation calculations over trace populations;
   do not persist those derived summaries in eval runs.
3. Qualify the LFM SFT and DPO profiles; do not infer their renderer or kernel
   behavior from Qwen.
4. Extend GRPO when a concrete job requires multimodal trajectories, multiple
   terminal branches, or higher rollout concurrency. Linear multi-turn/tool-use
   episodes now run natively through Verifiers; ambiguous branch shapes are
   rejected explicitly.

Do not repair old YAML or CLI paths. Replace them at the package boundary and
delete them once the corresponding vertical slice is proven.
