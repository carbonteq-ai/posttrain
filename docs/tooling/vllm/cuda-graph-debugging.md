# Debugging and qualifying vLLM CUDA graphs

Status: development guideline, 2026-09-19. This document changes no frozen
Posttrain product contract or dependency pin. It governs experiments in the
maintained vLLM fork; progress belongs in the
[inference optimization plan](../../plan/vllm-inference-optimization-platform.md).

## Objective and evidence rules

Find the first operation where replay differs from the eager reference, fix
that operation or its state ownership, and preserve the failure as a regression
test. A successful request, high acceptance rate, or configuration flag does
not prove that a CUDA graph executed correctly.

Distinguish four claims: graph requested, graph captured, graph replayed, and
replay qualified. Record each independently. Fallback success is useful
reliability evidence but is not a graph correctness or performance result.
Label observations, hypotheses, and validated causes explicitly. A failed
experiment must retain its artifacts and must not be described as eliminating
a cause unless the control actually isolated that cause.

## Start with upstream tools

vLLM recommends isolating the compilation stages. Use one change per control
and retain the resolved configuration; these settings diagnose a fault rather
than define the final optimized configuration.

| Control | What it establishes |
| --- | --- |
| `--enforce-eager` | Reference without torch.compile or CUDA graphs |
| `-cc.cudagraph_mode=NONE` | Compilation without CUDA replay |
| `-cc.mode=0` | No torch.compile; separately verify graph-mode eligibility |
| `-cc.backend=eager` | Compilation frontend without Inductor code generation |
| `TORCH_TRACE=<artifact-directory>` and `tlparse` | Compiler transformations and generated artifacts |
| `VLLM_LOGGING_LEVEL=DEBUG` | Includes explicit input-address checks in the inspected breakable wrapper |
| PyTorch Profiler | Operator shapes, stacks, and memory for a bounded diagnostic run |
| Nsight Systems `--cuda-graph-trace=node` | Which graph nodes and kernels actually execute |

Use `CUDA_LAUNCH_BLOCKING=1` for locating asynchronous CUDA errors, not for
performance measurement. Profiling traces locate execution and timing; they
do not replace numerical comparisons.

Upstream references, checked 2026-09-19:

- [Compilation debugging](https://docs.vllm.ai/en/latest/design/debug_vllm_compile/)
- [CUDA graph design](https://docs.vllm.ai/en/latest/design/cuda_graphs/)
- [Profiling](https://docs.vllm.ai/en/latest/contributing/profiling/)
- [Breakable graph implementation](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/breakable_cudagraph.py)

Moving upstream documentation must be checked against the exact fork revision
used in an experiment. Initial inspection used candidate HEAD
`4f18cc01bd14fb9cd4ebb85d9b8cf0993aac5b4d` plus uncommitted changes in
`/home/hammad/projects/vllm-sm120`; HEAD alone does not reproduce that source.

## Verify capture configuration before numerical debugging

For the inspected fork, `VLLM_USE_BREAKABLE_CUDAGRAPH` defaults to `0`.
`eager_break_during_capture` checks it when the decorated function is defined.
Constructing `BreakableCUDAGraphWrapper` does not enable those decorators.
Set the flag before Python starts when testing breakable attention capture;
changing it after imports is insufficient.

Assert the actual expected attention break points and graph segments in the
fixture. Record captures, replays, fallback reasons, descriptor keys, and
explicit and implicit tensor addresses. Do not infer attention exclusion
from the name `PIECEWISE`. Full-attention capture requires a separate backend
qualification. The missing flag in earlier launch commands is a hypothesis
to test, not an established cause of the Uno failure.

Reuse `tests/v1/cudagraph/test_breakable_cudagraph.py` in the fork. It already
tests eager/replay equivalence, eager-break counts, ordering, and multistream
attention. Run it in the GPU environment, from the fork root:

```bash
python -m pytest tests/v1/cudagraph/test_breakable_cudagraph.py -q
```

## Deterministic reproducer and first divergence

1. Save one failing real proposal batch and exact source/package/model/adapter
   identities. Include input IDs, positions, noise or RNG state, sequence lengths,
   block tables, slot mappings, row masks, LoRA slot layout, and policy version.
   Retain a patch/digest for dirty source and only allowlisted environment values.
2. Build an offline GPU fixture that loads once and replays this batch repeatedly.
   Record CUDA compiler/runtime, GPU, attention backend, graph mode, and launch
   command. Validate `nvcc`, `ninja`, and the intended Python executable before
   loading the model. Iterate over direct SSH without rebuilding job images.
3. Prove repeated eager execution is stable from the same starting state.
   Restore every mutable state touched by the forward, including KV and RNG.
   A KV snapshot must understand each cache layout and shared allocation; a
   generic dimension-zero copy is not sufficient evidence for every backend.
4. Test one layer with checkpoints at embedding/input, normalization, base
   projection, adapter contribution, rotary embedding, and attention. Identify
   the first mismatch before changing the full model. Debug instrumentation must
   copy into preallocated persistent buffers during capture/replay; Python hooks
   alone do not execute again during CUDA replay.
5. Compare eager with actual replay after capture, then mutate inputs A to B
   and repeat. Test tokens, positions, masks, sequence/KV metadata, and adapter
   routing independently. Restore A and verify again. Preserve graph outputs
   before another graph can reuse their storage. Weak capture outputs alone
   are not a reliable numerical oracle.
6. Record finite-value checks, absolute/relative errors, logits, chosen-token
   logprobs, and proposal tokens. Start with exact equality for an unchanged
   operation sequence. Any justified floating-point tolerance must be specified
   before promotion and supported by policy-level checks; do not relax it merely
   to pass a failing graph. Stochastic distribution correctness is separate from
   exact token equality under a deterministic fixture.

## Promotion and recovery

Progress from one operation to one layer to a complete model, then test changing
inputs, batch counts 1/2/3/4, request reordering, growing KV, cancellation,
adapter switches, and in-place weight updates with policy-version fencing.
Qualify proposer and attention graphs independently. Validate target tokens and
logprobs before trusting target-authoritative prefix emission.

Fallback tests must prove rejected descriptors remain eager on subsequent
calls, restore relevant state, and cannot recapture accidentally. A CUDA
illegal-address or device assertion may poison the process: restart it rather
than assuming an exception handler can safely continue. Test fixture restoration
itself before interpreting an output mismatch.

Only after correctness passes, run the matched warm benchmark with concurrency
4 and 16,384 output tokens per request, retaining exact prompt token counts,
warmup policy, prefix-cache hits, acceptance by position, preemptions, graph
replay counts, fallback counts, wall time, and aggregate output throughput.
Measure tool-call correctness separately: a forced-length test that removes
tool definitions cannot qualify tool use. Repeat qualified performance cells
to report variability. Short 128-token diagnostics are not comparable to the
retained 16K-output performance baseline.

## Immediate Uno experiment

The RTX PRO experiment verified import-time activation and observed 36 eager
attention breaks across 37 graph segments. The important failure was in the
oracle: vLLM padded each width-eight Uno request to nine executed rows, and the
qualifier treated the undefined padding row as a semantic output. Always retain
and refresh padding inputs, but compare model outputs, logits, tokens, and
logprobs only over `num_actual_tokens`.

With that correction, live eager-repeat, initial replay, and mutated-input
replay were bit-exact for request counts 1-4 under batch churn. Keep the saved
KV-block restore test and per-descriptor fallback, then optimize the remaining
37-segment/36-break execution overhead. Do not infer a model failure from a
padding-only mismatch, and do not weaken real-row token or logprob checks to
make a graph pass.
