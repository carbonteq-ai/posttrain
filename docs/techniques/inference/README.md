# Inference optimization techniques

Techniques describe backend-neutral ideas; runtime profiles and implementation code describe how a particular engine realizes them.

| Technique | Typical measurements |
| --- | --- |
| Continuous batching | aggregate throughput, TPOT, queue latency |
| Prefix caching | cache hit rate, TTFT, memory |
| Chunked prefill | TTFT, TPOT, total throughput |
| Quantization | latency, throughput, VRAM, capability/task deltas |
| Speculative decoding (MTP, DFlash, EAGLE) | acceptance rate, per-request TPS, memory |
| Compilation and fused kernels | warmup cost, steady-state latency, compatibility |

A technique is not synonymous with vLLM, SGLang, or Triton. One technique may have several implementations and each implementation may expose different native metrics. `packages/serve` records a small common observation set for comparisons and preserves backend-native output as an artifact.

The first path-validation run uses LFM2.5-1.2B-Thinking. The next study is a
like-for-like Qwen3.5-2B baseline followed by native MTP versus non-speculative
inference on the RTX 3070 Ti. These are serve-profile variants executed by the
shared `serve.benchmark` operation, not a separate optimization subsystem.

Operational rules learned while implementing these techniques live in
[heuristics.md](./heuristics.md). They capture validated setup constraints
without turning machine-specific fixes into model profiles.
