# Linear-attention kernels (fla-core, causal-conv1d)

Qwen3.5 mixes ordinary attention layers with Gated DeltaNet layers, a form of
linear attention: each such layer runs a short causal depthwise convolution
over its query/key/value projections and then a gated "delta rule" recurrence.
Transformers 5.14.1 (`models/qwen3_5/modeling_qwen3_5.py`) runs those layers
on its fast path only when both of these import:

- `fla` from **fla-core** (the kernel half of flash-linear-attention):
  `fla.ops.gated_delta_rule.chunk_gated_delta_rule`,
  `fused_recurrent_gated_delta_rule` and `fla.modules.FusedRMSNormGated`.
  They are Triton kernels, so the wheel is pure Python.
- `causal_conv1d` from **causal-conv1d**: `causal_conv1d_fn` and
  `causal_conv1d_update`. It is a compiled CUDA extension.

Otherwise the model logs "The fast path is not available because one of the
required library is not installed. Falling back to torch implementation" and
runs a Python loop over 64-token chunks. vLLM ships its own copy of the fla
kernels, so it was never affected; before this change the HF trainer and the
vLLM sampler ran different Gated DeltaNet implementations.

## Selection

| Package | Version | Source | Artifact SHA-256 |
| --- | --- | --- | --- |
| fla-core | 0.5.2 | PyPI (`fla_core-0.5.2-py3-none-any.whl`) | `5e830c85bad3d0d34677f98ac7074d08687a3756f0f0499d95ceb96eb6920761` |
| causal-conv1d | 1.7.0+cu130torch2.13 | CarbonTeq rebuild of upstream `v1.7.0` (`cd81f0413cad2fc1e6f17e785ac39f59aae690cd`) | `b69f39142ac88cac91cba5f954cb616c50bc49933cd84f349319420470a4947a` |

fla-core needs only `einops` (plus the image's torch and Triton 3.7.1). The
`flash-linear-attention` distribution adds `fla.layers`/`fla.models` and a
Transformers dependency that Posttrain does not use, so only fla-core is
selected. `posttrain-train[trl]` depends on `fla-core>=0.5.2,<0.6`, and
`profiles/supervised.txt` pins it, so the `supervised` and
`online-rl-trl-py312` kind images carry it.

causal-conv1d publishes source only on PyPI, and its GitHub release has no
wheel for PyTorch 2.13 (the newest is 2.10). The extension uses the PyTorch
C++ API, so it must be built against the exact PyTorch it runs with.

## Building causal-conv1d

`tools/kernel-wheels/causal-conv1d/build.sh WORK_DIR` clones the pinned tag,
checks the commit, and builds inside the published
`posttrain-kind-online-rl-trl-py312` kind image (digest pinned in the script),
not in a host environment. That image supplies CPython 3.13.12, torch
2.13.0+cu130 (C++11 ABI), the pip CUDA 13.0.88 toolkit (`nvcc`, headers,
`libcudart`) and GCC 12; the script makes a conventional `CUDA_HOME` view over
the pip toolkit and installs only `wheel==0.45.1` (hash-pinned) into a
throwaway directory. It sets `CAUSAL_CONV1D_FORCE_BUILD=TRUE` (never download
an upstream binary), `CAUSAL_CONV1D_LOCAL_VERSION=cu130torch2.13`,
`SOURCE_DATE_EPOCH` from the commit, and `MAX_JOBS=4`.

Upstream `setup.py` hard-codes `-gencode` flags for sm75, 80, 87, 90, 100, 103,
110, 120 and 121 with CUDA 13, which makes PyTorch ignore
`TORCH_CUDA_ARCH_LIST`. The script adds native sm86 (RTX 30-series, including
the local RTX 3070 Ti) and sm89 (Ada) through nvcc's `NVCC_APPEND_FLAGS`
without patching the source. The workstation's RTX PRO 6000 Blackwell uses the
sm120 code. A build takes about 8 minutes with 12 CPUs and yields a 250 MB
wheel (the `.so` is 315 MB).

The build is hermetic but not bit-for-bit reproducible: nvcc embeds
per-process `tmpxft_<pid>` identifiers, so a second build from the same inputs
produced a same-size wheel with a different hash
(`b1a7c3ad585618177f7d1bc7c6ee74a95ef2bd0d481e979a0820de5b74d22e0f`). The
retained wheel above is the identity; rebuilding does not reproduce its hash.

## Publication

causal-conv1d follows the retained-asset route in
[forks.md](../forks.md): the wheel is attached to an immutable GitHub Release
of `carbonteq-ai/causal-conv1d` (a mirror of upstream with a
`CARBONTEQ_FORK.md` stating that there is no source delta, only this rebuild),
then `.github/workflows/publish-causal-conv1d-internal.yml` downloads those
exact bytes on the `lan-release` runner, checks the hash and uploads them to
`https://pypi.lan/carbonteq/dev/`. The runner holds the index credentials
(`UV_PUBLISH_USERNAME`/`UV_PUBLISH_PASSWORD`); forks and workstations do not.

Status: **not yet published.** The mirror repository, its release
`carbonteq-v1.7.0+cu130torch2.13`, and the publisher workflow on `main` do not
exist yet. Until then causal-conv1d is not in `uv.lock` or any runtime lock,
and it was qualified only from a local wheelhouse. Once it is on
`carbonteq/dev`:

1. add `"causal-conv1d==1.7.0+cu130torch2.13; sys_platform == 'linux' and
   platform_machine == 'x86_64'"` (the only wheel is CPython 3.13 x86_64) to the
   `trl` extra in `packages/train/pyproject.toml` with
   `causal-conv1d = { index = "carbonteq-dev" }` under `[tool.uv.sources]`,
   and `causal-conv1d==1.7.0+cu130torch2.13` to
   `profiles/supervised.txt`;
2. run `uv lock` and `uv run posttrain-release lock-runtime-dependencies`;
3. extend the supervised and online-RL smoke stages to import `causal_conv1d`;
4. rebuild and publish the kind images and regenerate `published.toml`.

## Qualification evidence

Local RTX 3070 Ti (sm86), Qwen/Qwen3.5-0.8B at revision
`2fc06364715b967f1860aea9cf38778875588b17`. "Before" is the published
`posttrain-kind-online-rl-trl-py312@sha256:3a578d8b…`; "after" is the kind
image built from this change's locks plus the causal-conv1d wheel.

- Kernel checks against the Transformers torch reference, sequence 2,048,
  16 heads: chunked gated delta rule forward/backward relative error 5.6e-3
  (bf16), 7e-4 (fp16), 1.5e-3 (fp32); causal-conv1d forward/backward 3e-3
  (bf16), 4e-4 (fp16), 5e-8 (fp32); all gradients finite.
- The "fast path is not available" warning is gone and
  `is_fast_path_available` is true.
- LoRA actor step (rank 16, all linear layers, gradient checkpointing, chunked
  fp32 log-probs), median of five:

  | Tokens | bf16 before | bf16 after | fp16 before | fp16 after |
  | --- | --- | --- | --- | --- |
  | 2,048 | 2.09 s | 0.68 s | 2.02 s, NaN grads | 0.70 s |
  | 3,072 | 3.38 s | 1.00 s | 3.37 s, NaN grads | 1.04 s |
  | 4,096 | 5.00 s | 1.37 s | 4.83 s, NaN grads | 1.41 s |

  The locally built kind image with fla-core alone (the state the committed
  locks produce until causal-conv1d is published) takes 0.74 s, 1.08 s and
  1.47 s in bf16 (0.74/1.10/1.49 s in fp16, finite gradients): fla-core gives
  most of the gain and fixes fp16, while causal-conv1d removes the remaining
  7%. Transformers still prints the "fast path is not available" warning in
  that state because its check requires both packages.
  Peak memory is unchanged (3.0-3.1 GiB). With the torch fallback in fp16
  every step produced non-finite LoRA gradients; the first non-finite gradient
  appears inside layer 14's Gated DeltaNet backward.
- Per-token |log p(vLLM) - log p(HF)| on the retained 64-prompt set (48 GSM8K,
  16 AutomationBench transcripts; fp32 softmax on the trainer side) does not
  change: vLLM bf16 vs HF bf16 mean 0.0137 → 0.0134 (p99 0.124 both); vLLM
  fp16 vs HF fp16 mean 0.0019 → 0.0020 (p99 0.015 → 0.016). The sampler/trainer
  gap is bf16 rounding, not the Gated DeltaNet implementation.
- vLLM is unaffected: re-sampling the same 64 prompts with the same seeds in
  the new image gave identical completions and identical log-probs.

The scripts and raw results are recorded in
[the plan](../../plan/qwen35-fast-gdn-kernels.md).

## Remaining gates

causal-conv1d publication (above). The veRL kind (`verl-py313`, separate
backend lock) and the transform kind (llm-compressor calibration, separate
`tools/quantization` lock) also run HF Qwen3.5 forward passes and would
benefit, but have their own release gates and are not changed here. Eval and
serve images run Qwen3.5 only through vLLM.
