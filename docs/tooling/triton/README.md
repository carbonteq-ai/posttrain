# Triton

`triton` may mean NVIDIA Triton Inference Server or custom kernels written with the Triton language. Every future implementation must state which one it uses.

Custom kernel implementation and its typed activation/compatibility profiles
belong in `packages/serve`. There is no separate inference-optimization
directory or normalized result store.

Triton work is deferred until the vLLM and SGLang profile/benchmark contract is proven.
