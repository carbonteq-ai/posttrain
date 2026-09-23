# vLLM

vLLM is an optional backend of `packages/serve` and an optional rollout dependency of `packages/train`.

For runtime optimization work, follow the
[CUDA graph debugging and qualification guideline](./cuda-graph-debugging.md).
It defines upstream diagnostic controls, deterministic replay fixtures,
evidence requirements, and the current Uno investigation entry point.

```bash
uv sync --package posttrain-serve --extra vllm --python 3.12
uv sync --package posttrain-train --extra vllm --python 3.12
```

The vLLM dependency set pins PyTorch and the CUDA compiler components to the
same CUDA minor version. This is required because FlashInfer compiles kernels
locally: the runtime, headers, NVCC, NVVM, CRT, and CCCL cannot safely float to
different CUDA releases.

The selected development release is
[`carbonteq-v0.29.1.dev3`](https://github.com/carbonteq-ai/vllm/releases/tag/carbonteq-v0.29.1.dev3),
commit `564ff2b43d499f5d17bcee554d126360b768dd98`, based exactly on upstream
`44dd18fe0bb0f13157f97a5aa029b6604468fca6`. Its retained source archive has
SHA-256 `28d20ff20893e1570b3789fb1367af4ca78a8bd71e31c03d43476a6f89aa57d0`.
Over dev2 (`fbbba6698b2f8a912b94705cfc09eb4fd7243716`) it adds the generic
SM120 batch-invariant GEMM rule, batch-invariant split-KV attention, GDN chunk
alignment and invariant CUDA RMSNorm (see
[the optimization architecture](../../architecture/vllm-inference-optimization.md)),
and multi-turn prefix reuse for hybrid and sliding-window models: a decoding
request's last computed block stays reachable under sparse retention, so an
agentic turn no longer re-prefills the previous turn's generated tokens
(LFM2.5 AutomationBench replay at c16: 43% fewer prefilled tokens, 0.1% instead
of 9.6% of reusable context recomputed, collection 1.023x).
The independent SM120 kernel is
[`sm120-paged-attention` v0.1.0](https://github.com/carbonteq-ai/sm120-paged-attention/releases/tag/v0.1.0),
commit `99a6fe0acbb4756735aa8e47236f8b74e3f7c4be`, with wheel SHA-256
`3149a539c3296afbc56dfec88e86012fa00c8ac167eb828fa95efe90fdd38430`.

## Uno and SM120 release boundary

The fork provides the native Uno proposer, position-gated system LoRA overlay,
owner-scoped Punica metadata, policy-plus-Uno adapter composition, SM120 paged
FA4 backend, and shape-tuned batch-invariant linear configurations. Uno keeps
its proposer eager; target decode retains ordinary vLLM CUDA graphs. Rejected
authoritative-prefix, proposer-graph, deterministic-noise, and debug-switch
experiments are not part of the public configuration.

The release-clean RTX PRO c4 control produced 435.78 output tok/s with the
released kernel and zero preemptions. Focused configuration, attention,
invariant-linear, Uno, and LoRA tests pass. A real rank-8 policy-LoRA optimizer
step passed policy versions 0 and 1 with 32/32 finite target logprobs and a
maximum post-update logprob movement of 0.0655067.

Posttrain therefore admits native Uno inference and LoRA training only.
Full-weight and QLoRA Uno refresh are rejected during job compilation until
they pass independent live optimizer/update gates. Distributional comparison,
mixed-batch churn, and the long-prompt cell remain stable-promotion evidence;
they do not silently broaden the qualified training modes.

See the [optimization architecture](../../architecture/vllm-inference-optimization.md)
and [Uno rollout plan](../../plan/uno-vllm-rollout-integration.md) for the
diagnostic chain and retained artifacts.

The fork delta is Python-only. The veRL image verifies the upstream 0.25.1
x86_64 ABI3 wheel with SHA-256
`16fc7a28df1576eb6f7ca0455026551b8f9adb674c19c66059359ef3e964bd1e`
and uses those compiled extensions beneath the fork sources. This avoids an
implicit local C++/CUDA rebuild while preserving a content-addressed binary
base.

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
`libcudart.so`. `posttrain.common.cuda` validates the toolkit against the
active PyTorch build and creates a cache-local conventional view. Serving and
colocated training activate the same view before vLLM or FlashInfer starts. The
activation also prepends the active interpreter's scripts directory to `PATH`
so pip-installed JIT tools such as `ninja` resolve for EngineCore children
even when `vllm` was launched by absolute path. It does not alter the installed
wheels and does not disable FlashInfer.

### Direct RTX PRO development loop

Python, Triton, scheduler, and benchmark changes are iterated directly on the
RTX PRO before OCI qualification. The command surface lives in
`/home/hammad/projects/k2-horizon-inference/Taskfile.yml`:

```bash
cd /home/hammad/projects/k2-horizon-inference
task remote:bootstrap
task remote:test -- tests/v1/spec_decode/test_uno.py -q
task remote:restart
task remote:status
```

Bootstrap transfers a Git bundle of the local base commit and then overlays the
dirty worktree with checksum-based `rsync`. The remote clone is therefore
self-contained and does not depend on GitHub credentials or a linked-worktree
path from another machine. Subsequent no-op synchronization is sub-second on
the LAN. The retained vLLM venv supplies compiled extensions, while the remote
source overlay supplies Python and Triton code. Generated third-party sources
and compatible native binaries are reused from the existing remote build.

The server runs as the `vllm-sm120-dev.service` transient user unit, binds only
to loopback, and records the strict batch-invariant and deterministic SM120
flags in its managed environment. Start refuses to replace an unmanaged
listener. Restart stops only the managed unit, synchronizes source, waits for a
healthy endpoint, and fails with its journal if initialization exits. Docker,
image publication, and Posttrain submission remain separate release gates
after focused tests and direct warm benchmarks pass.

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
and replace the cache dtype with `auto` inside a spawned worker. The CarbonTeq
fork preserves the requested `turboquant_*` dtype for
`TQFullAttentionSpec` while retaining `auto` for genuinely skipped,
unquantized layers. The general serve and eval images still use the upstream
0.25.1 binary wheel, so the serve package's `vllm.general_plugins` entry point
applies the equivalent state-guarded quantization marker there. The guard is a
no-op when the CarbonTeq fork reports a native TurboQuant quantization mode.
The same entry point activates the CUDA toolkit view in controller,
engine-core, and worker processes.

The fork's source regression is complete. Release qualification still requires
the locked veRL image and real DAPO and SAMPO hybrid-Qwen runs. TurboQuant and
MTP must each pass independently and together before the combined profile is
described as supported.

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

The canonical managed online endpoint run is
`serve-online/lfm2.5-1.2b-thinking-95a0371d-a1` at clean revision `371a49c`.
The health/model probe passed and a streamed response stopped normally with
final content after 169 output tokens. It measured 168 ms TTFT and 953 ms
end-to-end latency in the first successful run; the refined canonical run
measured 42 ms TTFT and 884 ms end-to-end latency, with raw SSE events moved
from searchable trace metadata into a native-response artifact. Earlier capped attempts correctly exposed a product
distinction: endpoint health can succeed while a model response is truncated
inside reasoning, so the job now requires non-empty final content.
