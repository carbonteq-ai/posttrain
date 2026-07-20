# Hardware

- GPU: RTX 3070 Ti **8 GB**
- CPU / RAM: i9-12900K / ~62 GB
- Constraint: QLoRA, short context, small GRPO groups
- The desktop display normally consumes roughly 1 GiB, leaving about 6.8 GiB free when idle.
- The NVIDIA driver is installed, but `/usr/local/cuda` and NVCC are not. Prefer prebuilt kernels and explicitly disable runtime JIT paths that require a CUDA toolkit.
- Qwen3.5-2B BF16 weights are about 4.55 GB and fit. Gemma 4 E2B is 2.3B effective parameters but 5.1B total with per-layer embeddings; its BF16 weights are about 10.25 GB and do not fit this GPU.

Model notes: [models.md](./models.md).
