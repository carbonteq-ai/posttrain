# Serving base configs

Reusable vLLM and SGLang settings and compatibility declarations owned by `packages/serve` belong here. The backend base contains only backend-wide defaults; model profiles set an explicit context and load policy, and runtime variants add TurboQuant, MTP, or custom kernels. MTP and TurboQuant are serving modes referenced by compatible model profiles; they are not additional model descendants unless weights change.
