"""Materialize derived weight variants without coupling to a quantization backend."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from posttrain.common import (
    ExecutionTarget,
    JsonValue,
    LocalArtifactRef,
    ModelVariant,
    ProducedArtifact,
    RunContext,
)

from .bindings import QuantizationPlan

type TransformContext = RunContext


@dataclass(frozen=True, slots=True)
class TransformRequest:
    model: ModelVariant
    plan: QuantizationPlan
    target: ExecutionTarget
    output_id: str

    def __post_init__(self) -> None:
        if self.plan.method == "qat":
            raise ValueError("QAT plans must be executed by a training operation")
        if self.model.form != "foundation" or self.model.quantization:
            raise ValueError("offline quantization currently requires an unquantized foundation model variant")


@dataclass(frozen=True, slots=True)
class TransformResult:
    source_model: ModelVariant
    model: ModelVariant
    artifact: ProducedArtifact


type TransformRunner = Callable[[TransformContext, TransformRequest, Path], Path]


def transform(
    context: TransformContext,
    request: TransformRequest,
    *,
    runner: TransformRunner,
) -> TransformResult:
    """Run a host-selected transformer and return its catalog-ready child variant."""

    attributes: dict[str, JsonValue] = {
        "source_model_variant_id": request.model.id,
        "quantization_plan_id": request.plan.id,
        "quantization_plan_revision": request.plan.revision,
        "quantization_recipe_digest": request.plan.recipe_digest,
        "quantization_backend": request.plan.backend,
        "dependency_lock_sha256": request.plan.dependency_lock_digest,
        "transform_method": request.plan.method,
        "target_id": request.target.id,
        "target_revision": request.target.revision,
    }
    if request.plan.calibration is not None:
        attributes.update(
            {
                "calibration_dataset_id": request.plan.calibration.dataset_id,
                "calibration_dataset_revision": request.plan.calibration.dataset_revision,
                "calibration_sample_count": request.plan.calibration.sample_count,
                "calibration_sequence_length": request.plan.calibration.sequence_length,
            }
        )
    context.event("model_transform_started", attributes)
    output_dir = context.workspace / "training" / "transform" / "weights"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    materialized = runner(context, request, output_dir).resolve()
    if not materialized.exists():
        raise FileNotFoundError(f"transform runner did not materialize output: {materialized}")
    if not materialized.is_relative_to(context.workspace):
        raise ValueError("transform outputs must be materialized inside the run workspace")
    runtime_versions: dict[str, JsonValue] = {}
    summary_path = materialized / "posttrain-quantization-summary.json" if materialized.is_dir() else None
    if summary_path is not None and summary_path.is_file():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        versions = payload.get("runtime_versions", {})
        if isinstance(versions, dict):
            runtime_versions = {str(name): value for name, value in versions.items() if isinstance(value, str)}
            attributes["runtime_versions"] = runtime_versions
    digest = _digest(materialized)
    reference = LocalArtifactRef(materialized, digest)
    artifact = ProducedArtifact(
        name=f"training/{request.output_id}/weights",
        kind="model-weights",
        reference=reference,
        metadata=attributes,
    )
    child = ModelVariant(
        id=request.output_id,
        artifact=reference,
        form="weight-quantized",
        weight_precision=request.plan.output_weight_precision,
        family=request.model.family,
        parameters=request.model.parameters,
        instruction_tuned=request.model.instruction_tuned,
        renderer=request.model.renderer,
        capabilities=request.model.capabilities,
        base=request.model.base,
        digest=digest,
        quantization={
            "method": request.plan.method,
            "weight_format": request.plan.weight_format,
            **request.plan.output_quantization,
        },
        parent=request.model.id,
        provenance={
            "operation": "transform",
            "quantization_plan_id": request.plan.id,
            "recipe_digest": request.plan.recipe_digest,
            "backend": request.plan.backend,
            "target_id": request.target.id,
            "runtime_versions": runtime_versions,
        },
    )
    context.artifact(artifact)
    context.event("model_transform_completed", {**attributes, "output_model_variant_id": child.id})
    return TransformResult(request.model, child, artifact)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.read_bytes())
        return digest.hexdigest()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        with child.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "TransformContext",
    "TransformRequest",
    "TransformResult",
    "TransformRunner",
    "transform",
]
