"""Reusable inference-engine package.

vLLM, SGLang, TurboQuant, MTP, and custom-kernel integrations are implemented
behind this package boundary while their reusable settings live in profiles.

The package initializer intentionally avoids importing engine modules so CLI
subprocesses can start through ``python -m`` without preloading CUDA code.
"""

__all__: list[str] = []
