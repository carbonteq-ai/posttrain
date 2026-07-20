# vLLM

vLLM is an optional backend of `packages/serve` and an optional rollout dependency of `packages/train`.

```bash
uv sync --package posttrain-serve --extra vllm --python 3.12
uv sync --package posttrain-train --extra vllm --python 3.12
```

The vLLM dependency set pins PyTorch and the CUDA compiler components to the
same CUDA minor version. This is required because FlashInfer compiles kernels
locally: the runtime, headers, NVCC, NVVM, CRT, and CCCL cannot safely float to
different CUDA releases.

Both `posttrain-serve[vllm]` and `posttrain-train[vllm]` pin vLLM 0.25.1. Training resolves TRL
1.8.0 from the immutable CarbonTeq fork commit documented in
[ADR 0007](../../decisions/0007-trl-vllm-025-fork.md), which raises TRL's
validated vLLM ceiling without changing its trainer or weight-sync logic.
Future vLLM versions remain unsupported until both the fork tests and the lab's
GPU rollout smoke pass and the pins are advanced together.

`posttrain.serve` exposes two execution paths. `benchmark` uses the offline
engine for exact token-shape throughput cells. `launch` manages an
OpenAI-compatible vLLM process; `probe` validates health and model exposure;
and `generate` streams one chat request while retaining TTFT, usage, reasoning,
tool-call deltas, and raw protocol events. The lab host adds Trackio observation
and stores the server log as an artifact.

The shared model profile selects the native chat template and tool grammar.
The serving profile selects vLLM's parser flags. See
[ADR 0008](../../decisions/0008-model-conversation-contracts.md), including the
tested LFM2.5 template override required for multi-turn OpenAI tool history.

NVIDIA's pip toolkit uses `lib` and versioned shared-object names, while CUDA JIT
builders commonly expect `CUDA_HOME/lib64` and linker names such as
`libcudart.so`. `packages/serve/src/posttrain/serve/cuda.py` validates the toolkit against
the active PyTorch build and creates a cache-local conventional view. It does
not alter the installed wheels and does not disable FlashInfer.

Run either current code-defined foundation smoke through the lab composition
root. The operation remains usable directly from Python through
`posttrain.serve.benchmark`; the CLI is only a job launcher:

```bash
uv run --package posttrain-serve --extra vllm --python 3.12 \
  posttrain-lab foundation-lfm-smoke --tracked --project posttrain-foundation
uv run --package posttrain-serve --extra vllm --python 3.12 \
  posttrain-lab foundation-qwen-smoke --tracked --project posttrain-foundation
```

The reusable matrix is a typed value and can be inspected without loading a
model:

```bash
uv run --package posttrain-serve python -c \
  'from posttrain.serve import CORE_INFERENCE_V1; print(*CORE_INFERENCE_V1.cells(max_concurrency=4), sep="\n")'
```

The checked-in suite contains concurrency 1, 2, 4, and 8 for portability. On
this RTX 3070 Ti, execute only through concurrency 4:

The lab host records each executed matrix cell as a separate Trackio run carrying the
code-defined job, action, invocation, and attempt IDs. The suite covers short interactive, decode-heavy, balanced,
and prefill-heavy shapes at 1K through 32K configured context. All 32K cells
resolve the model profile's `turboquant_k8v4` serve variant.

Model targets are selected through typed definitions in `posttrain.common.profiles`;
vLLM settings, MTP, TurboQuant, cache behavior, and compatibility declarations
belong in typed profiles shipped with `packages/serve`. Each benchmark records its resolved inputs, package and GPU
context, portable throughput metrics, native output, and result artifact in a
typed Trackio run.

The first launch includes model loading, Torch compilation, FlashInfer JIT, and
CUDA graph capture. Those setup costs must not be mixed into steady-state token
throughput. On an 8 GB display GPU, model-weight fit and KV-cache capacity must
also be measured separately: KV-cache quantization cannot make oversized model
weights fit.

## TurboQuant compatibility

vLLM 0.25.1 recognizes `turboquant_k8v4` and selects the TurboQuant attention
backend, but its hybrid-cache path can mark `TQFullAttentionSpec` as unquantized
and replace the cache dtype with `auto` inside a spawned worker. The serve
package installs a narrow, state-guarded `vllm.general_plugins` entry point that
supplies the missing non-per-token quantized marker. vLLM loads this plugin in
the controller, engine core, and worker processes. It does not patch the
installed wheel, alter TurboQuant's packed page-size calculation, or fall back
to a different cache dtype. The guard becomes a no-op when upstream reports a
non-`NONE` TurboQuant quantization mode.

## Validated smoke result

On 2026-07-20, `LiquidAI/LFM2.5-1.2B-Thinking` at the pinned profile revision
completed on the RTX 3070 Ti with vLLM 0.25.1, PyTorch 2.11.0+cu130,
FlashAttention 2, FlashInfer sampling, compiled graphs, chunked prefill, and
asynchronous scheduling enabled.

The one-request/32-output-token smoke run produced 177.46 output tokens/s and
used 6.66 GiB peak device memory (5.44 GiB above the display baseline). This
proves the execution and tracking path; it is not a serving capacity result.
The Trackio run is `lfm2.5-1.2b-vllm-smoke-8`. A workload matrix with multiple
prompt/decode lengths and concurrency levels is required before comparing model
profiles.

## Validated matrix cells

On 2026-07-20, the current controlled benchmark implementation also completed these LFM2.5 cells:

| Cell | Result |
| --- | --- |
| 128 input / 128 output, 32K context, c1, TurboQuant K8V4 | 172.67 output tok/s, 13.1 ms mean TTFT, 6.61 GiB peak VRAM |
| 128 input / 128 output, 4K context, c4 | 591.98 output tok/s, 41.6 ms mean TTFT, 6.89 GiB peak VRAM |
| 24,576 input / 128 output, 32K context, c1, TurboQuant K8V4 | 8,616.93 input tok/s, 44.88 output tok/s, 1.83 s mean TTFT, 7.11 GiB peak VRAM |

These validate execution and measurement coverage, not the complete comparison
matrix. Full base-model comparisons must use matching suite cells and package,
model, hardware, and configuration revisions.

On the same date, the original Qwen3.5-2B profile failed during CUDA-graph/KV
profiling after loading 4.25 GiB of weights. The tested text-only correction
disables multimodal request capacity, skips multimodal profiling, caps
`max_num_seqs` at 4, uses eager execution, and reserves 75% rather than 82% of
device memory. It does not offload weights or KV cache to host RAM. A 128-input,
32-output, concurrency-1 cell then completed at 65.05 output tok/s, 43.1 ms
TTFT, and 6.92 GiB peak VRAM. Cold start was 65.24 seconds, so this safe local
profile is not yet the optimized Qwen profile; compilation and graph-capture
variants must be evaluated independently rather than folded into this result.

The first canonical code-defined foundation-screening job is Trackio run
`serve/qwen3.5-2b/short-interactive-ctx1024-c1-a7b17eb5-a1` in project
`posttrain-foundation`, produced from clean Git revision `5b429cb`. It records
the mandatory job/action/invocation/attempt identities, one 23-field run-level
metric batch, one inference trace, and one versioned serving-result artifact.
It measured 75.66 output tok/s, 35.5 ms TTFT, 6.99 GiB peak VRAM, and a
cache-warm 19.51 second engine start. The earlier module-backed run remains
valid direct evidence, but its scalar metrics were fragmented across steps;
the canonical rerun verifies the corrected observation grain.

The matching LFM2.5 canonical job is
`serve/lfm2.5-1.2b-thinking/short-interactive-ctx1024-c1-13f0e79a-a1`
at clean Git revision `c13df39`. Its single metric batch records 170.90 output
tok/s, 14.2 ms TTFT, 6.35 GiB peak VRAM, and 22.55 seconds engine startup.
Both foundation models therefore pass the same typed 128-input/32-output/c1
cell through the same job, observer, trace, and artifact path.

The first successful managed online endpoint run is
`serve-online/lfm2.5-1.2b-thinking-b477f22b-a1` at clean revision `a8c1706`.
The health/model probe passed and a streamed response stopped normally with
final content after 169 output tokens. It measured 168 ms TTFT and 953 ms
end-to-end latency. Earlier capped attempts correctly exposed a product
distinction: endpoint health can succeed while a model response is truncated
inside reasoning, so the job now requires non-empty final content.
