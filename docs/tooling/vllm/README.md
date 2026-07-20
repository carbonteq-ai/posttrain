# vLLM

vLLM is an optional backend of `packages/serve` and an optional rollout dependency of `packages/train`.

```bash
uv sync --package serve --extra vllm --python 3.12
uv sync --package train --extra vllm --python 3.12
```

The vLLM dependency set pins PyTorch and the CUDA compiler components to the
same CUDA minor version. This is required because FlashInfer compiles kernels
locally: the runtime, headers, NVCC, NVVM, CRT, and CCCL cannot safely float to
different CUDA releases.

Both `serve[vllm]` and `train[vllm]` pin vLLM 0.25.1. Training resolves TRL
1.8.0 from the immutable CarbonTeq fork commit documented in
[ADR 0007](../../decisions/0007-trl-vllm-025-fork.md), which raises TRL's
validated vLLM ceiling without changing its trainer or weight-sync logic.
Future vLLM versions remain unsupported until both the fork tests and the lab's
GPU rollout smoke pass and the pins are advanced together.

NVIDIA's pip toolkit uses `lib` and versioned shared-object names, while CUDA JIT
builders commonly expect `CUDA_HOME/lib64` and linker names such as
`libcudart.so`. `packages/serve/src/serve/cuda.py` validates the toolkit against
the active PyTorch build and creates a cache-local conventional view. It does
not alter the installed wheels and does not disable FlashInfer.

Run the current offline benchmark with:

```bash
uv run --package serve --extra vllm --python 3.12 \
  serve-benchmark lfm2.5-1.2b-thinking
```

Plan the reusable workload matrix without loading a model:

```bash
uv run --package serve --extra vllm --python 3.12 \
  serve-benchmark-suite lfm2.5-1.2b-thinking --dry-run
```

The checked-in suite contains concurrency 1, 2, 4, and 8 for portability. On
this RTX 3070 Ti, execute only through concurrency 4:

```bash
uv run --package serve --extra vllm --python 3.12 \
  serve-benchmark-suite lfm2.5-1.2b-thinking \
  --concurrency 1 --concurrency 2 --concurrency 4
```

Each matrix cell is a separate `serving-benchmark` Trackio run carrying the
code-defined job, action, and suite-invocation IDs. The suite covers short interactive, decode-heavy, balanced,
and prefill-heavy shapes at 1K through 32K configured context. All 32K cells
resolve the model profile's `turboquant_k8v4` serve variant.

Model targets are selected through typed definitions in `profiles/models`;
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
