# Native LFM BF16 and FP16 with FP32 masters

1 October2026. Native veRL now loads LFM2.5-1.2B-Thinking, accumulates two
microbatches and applies AdamW updates on the local8GB GPU. FSDP2's native
CPU-offload policy retains trainable parameters and optimizer moments in FP32;
model computation uses BF16 or FP16. This avoids casting trainable masters
to half precision merely to fit the larger model.

The checkpoint is immutable revision
`95053d21d8e0b7ca99421a2127ae39c64f685ff3`. Native veRL source is published
`8778c5d6e2ddd847d5098a24f4dc882f11ac57b4`, with the isolated Transformers
`d6c1e71bd717bf092f8293f0c3c9bd4a5ac5401a` runtime and Torch2.13.0+cu130.
No production runtime pin or framework recipe changed.

## Actual native checks

All arms use two supplied rows, q/v LoRA rank4/alpha8, LR1e-4,
microbatch1, accumulated population2, SAMPO bounds0.003/0.004 and beta0.01.
Alternating token credits include excluded response positions with credit99.
Behavior/reference scores remain frozen across two attempts. The8-token arm
has4 context/4 response tokens; the128-token arm has96 context/32 response
tokens. These are numerical fixtures, not rendered task conversations.

| Native CPU-offload arm | Applied updates | Independent loss/score checks | Peak Torch GPU allocation |
| --- | ---: | ---: | ---: |
| LFM BF16,8 tokens | 2 of2 | 4 of4 | 1.00GiB |
| LFM FP16 scale1024,8 tokens | 2 of2 | 4 of4 | 1.00GiB |
| LFM BF16,128 tokens | 2 of2 | 4 of4 | 1.00GiB |
| LFM FP16 scale1024,128 tokens | 2 of2 | 4 of4 | 1.00GiB |

All eight updates are applied without scaler skips. All16 independent
loss/score-gradient checks pass and context-score gradients are exactly zero.
The maximum loss error is2.88e-8 and score-gradient error1.70e-8. The
independent AdamW calculation, using already-unscaled/clipped native gradients,
matches parameter movement within2.13e-9. Masters reside on CPU in FP32;
the three final-runner arms explicitly record FP32 optimizer-moment dtypes.
CPU parameter/state storage, pinned memory, desktop, driver and CUDA allocator
reservation are additional to the reported Torch GPU allocation.

FSDP1's actor path deliberately disables native CPUOffload because of known
gradient-accumulation limitations. These runs select FSDP2's CPUOffloadPolicy,
not that FSDP1 path. Single-rank shard-local tensor references cover the entire
parameter here; they are not distributed optimizer qualification.

## Evidence and remaining limits

The external archive described in [README](README.md) retains the four JSON/log
receipts, executed runner versions and hash manifest in `results/native-lfm/`.
The first BF16 receipt's textual scope incorrectly says FSDP1; its structured
`engine_strategy=fsdp2` and retained executed source establish FSDP2. That
receipt is retained unchanged and the manifest records the correction.
The working probe is `working/native_verl_offload_run.py`; no runner or raw
receipt is committed.

These checks close the initially unexecuted native LFM engine path. They verify
score derivatives and optimizer arithmetic, not an independent derivative
through every model parameter or a proof of microbatch accumulation equivalence.
The subsequent [partition comparison](native-accumulation-results.md) verifies
accumulation against separately measured rows on the8-token fixture and exposes
precision sensitivity when those contributions cancel.
Still required: rendered LFM masks and unequal lengths, broader accumulated
LoRA-gradient checks, matched native TRL/veRL initial states, fresh AutomationBench
collection/learning, broader modules/contexts, distributed and fused paths,
and immutable production adoption.
