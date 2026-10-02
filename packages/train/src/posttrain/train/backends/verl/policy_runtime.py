"""Actual native actor identities and frozen base-reference score ownership."""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ...kl_reference import resolved_kl_reference
from ...update_records import InvalidPolicyUpdate
from ..policy_update_scoring import FrozenPopulationScores
from .contracts import VerlLaunchManifest
from .policy_updates import ResolvedVeRLPopulation

SCORE_CONTRACT = "posttrain.causal-text-tempered-logsoftmax-fp32@1"


def native_job_identity(manifest: VerlLaunchManifest, engine: Any) -> tuple[str, str]:
    """Bind selected meaning and effective native arithmetic, excluding output paths."""
    import torch

    tokenizer = engine.model_config.tokenizer
    if tokenizer is None or not tokenizer.chat_template:
        raise InvalidPolicyUpdate("native actor lacks the effective policy tokenizer template")
    template_payload = {"template": tokenizer.chat_template,
                        "renderer": manifest.payload.training.renderer.model_dump(mode="json")}
    template = hashlib.sha256(json.dumps(template_payload, sort_keys=True).encode()).hexdigest()
    root = Path(__file__).resolve().parents[2]
    sources = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted(root.rglob("*.py"))}
    for index, cls in enumerate(type(engine).__mro__):
        path = inspect.getsourcefile(cls) if cls is not object else None
        if path:
            sources[f"native-engine-{index}"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    payload = manifest.payload.model_dump(mode="json")
    payload.pop("resume_from")
    payload.pop("curriculum_from")
    payload["environment"].pop("bridge_snapshot")
    for name in ("python_executable", "working_directory"):
        payload["training"]["backend_options"].pop(name, None)

    def serialize(value: Any) -> Any:
        if is_dataclass(value) and not isinstance(value, type):
            return {item.name: getattr(value, item.name) for item in fields(value)}
        if isinstance(value, Mapping):
            return dict(value)
        raise TypeError(f"native runtime identity cannot serialize {type(value).__name__}")

    identity = {"schema": "posttrain.resolved-verl-job@1", "payload": payload,
        "backend_source_revision": manifest.backend_source_revision, "sources": sources, "template": template,
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "verl", "peft")},
        "engine_config": engine.engine_config, "optimizer_config": engine.optimizer_config,
        "model_config": engine.model_config.hf_config.to_dict(),
        "arithmetic": {"deterministic": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "fp16_reduced_gemm_accumulation": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            "bf16_reduced_gemm_accumulation": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            "fp16_gemm_accumulation": getattr(torch.backends.cuda.matmul, "allow_fp16_accumulation", None),
            "sdpa_flash": torch.backends.cuda.flash_sdp_enabled(),
            "sdpa_memory_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "sdpa_math": torch.backends.cuda.math_sdp_enabled(),
            "sdpa_cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
            "sdpa_math_reduced_precision": getattr(
                torch.backends.cuda, "fp16_bf16_reduction_math_sdp_allowed", lambda: None)()}}
    digest = hashlib.sha256(json.dumps(identity, default=serialize, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return f"resolved-verl-job/{digest}", f"renderer/{template}"


def base_reference_provider(manifest: VerlLaunchManifest, engine: Any):
    """Freeze all sampled actions using the native adapter-disabled base policy."""
    selected = manifest.payload.resolved_settings
    policy = manifest.payload.policy
    if selected is None or policy is None:
        raise InvalidPolicyUpdate("reference provider requires the selected native policy")
    settings = selected.settings
    if not settings.beta:
        return None
    reference = resolved_kl_reference(settings.beta, getattr(settings, "kl_reference", None), policy.form)
    if (reference != "base" or manifest.payload.training.update.kind != "lora"
            or manifest.payload.reference is not None or not callable(getattr(engine, "disable_adapter", None))):
        raise InvalidPolicyUpdate("resolved native composition requires qualified adapter-disabled base reference")

    def score(population: ResolvedVeRLPopulation) -> FrozenPopulationScores:
        snapshot = population.updates[0].population
        if snapshot.versions.reference is None:
            raise InvalidPolicyUpdate("native KL population lacks its frozen reference identity")
        with engine.disable_adapter():
            values = population._infer(engine, population._rows(tuple(item.action for item in snapshot.actions)))  # noqa: SLF001
        return FrozenPopulationScores(snapshot.digest, snapshot.versions.reference, population.score_contract,
            population.score_temperature, MappingProxyType({key: value.detach().clone() for key, value in values.items()}))

    return score
