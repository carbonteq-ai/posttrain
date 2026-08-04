"""Filesystem-backed catalog composition for the qualification project."""

from __future__ import annotations

from posttrain.catalog import open_catalog
from posttrain.common import (
    Catalog,
    CatalogRef,
    ExecutionTarget,
    InferenceBinding,
    JsonValue,
    ModelVariant,
    Workload,
)
from posttrain.common.selections import Selection
from posttrain.eval import EnvironmentBinding
from posttrain.train import (
    DPOSettings,
    GRPOSettings,
    OnPolicyDistillationSettings,
    QuantizationPlan,
    SFTSettings,
    TrainingBinding,
    parameter_update_digest,
)

from .gemma4 import register_gemma4_renderer
from .project import ProjectLayout

register_gemma4_renderer()


def open_project_catalog(layout: ProjectLayout, *, scope: str | None = None) -> Catalog:
    """Open the framework base plus overlays selected by one project layout."""

    return open_catalog(
        scope=layout.project_id if scope is None else scope,
        overlays=layout.catalog_overlays,
        catalog_root=layout.base_catalog,
    )


def _selection[SelectionT: Selection](
    catalog: Catalog,
    ref: CatalogRef,
    expected: type[SelectionT],
) -> SelectionT:
    value = catalog.resolve(ref).value
    if not isinstance(value, expected):
        raise TypeError(f"catalog entry {ref.family}/{ref.id} has the wrong selection type")
    return value


_REFERENCE_CATALOG = open_catalog(scope="posttrain-lab")
AUTOMATIONBENCH_ZAPIER_GRPO = _selection(
    _REFERENCE_CATALOG,
    CatalogRef("environment", "automationbench-zapier-simple-grpo"),
    EnvironmentBinding,
)
QWEN_35_2B_BF16 = _selection(
    _REFERENCE_CATALOG,
    CatalogRef("model", "models/qwen3.5-2b@bf16"),
    ModelVariant,
)
QWEN_35_08B_BF16 = _selection(
    _REFERENCE_CATALOG,
    CatalogRef("model", "models/qwen3.5-0.8b@bf16"),
    ModelVariant,
)
LOCAL_CUDA_8GB = _selection(
    _REFERENCE_CATALOG,
    CatalogRef("target", "targets/local-cuda-8gb"),
    ExecutionTarget,
)
LOCAL_CUDA_ROLLOUT_8GB = _selection(
    _REFERENCE_CATALOG,
    CatalogRef("target", "targets/local-cuda-rollout-8gb"),
    ExecutionTarget,
)
LOCAL_CUDA_TEACHER_8GB = _selection(
    _REFERENCE_CATALOG,
    CatalogRef("target", "targets/local-cuda-teacher-8gb"),
    ExecutionTarget,
)
QWEN_SCREEN_VLLM = _selection(
    _REFERENCE_CATALOG,
    CatalogRef("inference", "inference/qwen3.5-2b-vllm-screen@1"),
    InferenceBinding,
)
QWEN_EVAL_VLLM = _selection(
    _REFERENCE_CATALOG,
    CatalogRef("inference", "inference/qwen3.5-2b-vllm-eval@2"),
    InferenceBinding,
)
QWEN_GRPO_VLLM = _selection(
    _REFERENCE_CATALOG,
    CatalogRef("inference", "inference/qwen3.5-2b-vllm-rollout@1"),
    InferenceBinding,
)
QWEN_GRPO_MTP_VLLM = _selection(
    _REFERENCE_CATALOG,
    CatalogRef("inference", "inference/qwen3.5-2b-vllm-rollout-mtp@1"),
    InferenceBinding,
)
QWEN_AUTOMATIONBENCH_GRPO_MTP_VLLM = _selection(
    _REFERENCE_CATALOG,
    CatalogRef("inference", "inference/qwen3.5-0.8b-vllm-automationbench-rollout-mtp@1"),
    InferenceBinding,
)
QWEN_DISTILL_ROLLOUT_VLLM = _selection(
    _REFERENCE_CATALOG,
    CatalogRef("inference", "inference/qwen3.5-0.8b-vllm-distill-rollout@1"),
    InferenceBinding,
)
QWEN_TEACHER_SCORE_VLLM = _selection(
    _REFERENCE_CATALOG,
    CatalogRef("inference", "inference/qwen3.5-2b-vllm-teacher-score@1"),
    InferenceBinding,
)
FOUNDATION_SMOKE = _selection(
    _REFERENCE_CATALOG,
    CatalogRef("workload", "workloads/foundation-smoke-v1@1"),
    Workload,
)
QWEN35_TRL_QLORA = _selection(
    _REFERENCE_CATALOG,
    CatalogRef("training", "training/qwen3.5-trl-qlora@1"),
    TrainingBinding,
)
QWEN35_TRL_LORA = _selection(
    _REFERENCE_CATALOG,
    CatalogRef("training", "training/qwen3.5-trl-lora@1"),
    TrainingBinding,
)
QWEN35_DISTILL_TRL_LORA = _selection(
    _REFERENCE_CATALOG,
    CatalogRef("training", "training/qwen3.5-0.8b-trl-distill-lora@1"),
    TrainingBinding,
)
QWEN35_AUTOMATIONBENCH_TRL_LORA_THINKING = _selection(
    _REFERENCE_CATALOG,
    CatalogRef("training", "training/qwen3.5-0.8b-trl-automationbench-lora-thinking@1"),
    TrainingBinding,
)
LFM25_TRL_QLORA = _selection(
    _REFERENCE_CATALOG,
    CatalogRef("training", "training/lfm2.5-trl-qlora@1"),
    TrainingBinding,
)
LFM25_TRL_LORA = _selection(
    _REFERENCE_CATALOG,
    CatalogRef("training", "training/lfm2.5-trl-lora@1"),
    TrainingBinding,
)
QWEN35_SFT = _selection(
    _REFERENCE_CATALOG,
    CatalogRef("training", "qwen3.5-2b/sft-smoke-v2"),
    SFTSettings,
)
QWEN35_SFT_PEFT_COMPARISON = _selection(
    _REFERENCE_CATALOG,
    CatalogRef("training", "qwen3.5-2b/sft-peft-comparison-v1"),
    SFTSettings,
)
QWEN35_SFT_VALIDATED_SMOKE = _selection(
    _REFERENCE_CATALOG,
    CatalogRef("training", "qwen3.5-2b/sft-validated-smoke-v1"),
    SFTSettings,
)
QWEN35_DPO = _selection(
    _REFERENCE_CATALOG,
    CatalogRef("training", "qwen3.5-2b/dpo-smoke-v2"),
    DPOSettings,
)
QWEN35_GRPO = _selection(
    _REFERENCE_CATALOG,
    CatalogRef("training", "qwen3.5-2b/grpo-smoke-v3"),
    GRPOSettings,
)
QWEN35_GRPO_MTP = _selection(
    _REFERENCE_CATALOG,
    CatalogRef("training", "qwen3.5-2b/grpo-mtp-smoke-v2"),
    GRPOSettings,
)
QWEN35_AUTOMATIONBENCH_GRPO_MTP = _selection(
    _REFERENCE_CATALOG,
    CatalogRef("training", "automationbench/qwen3.5-0.8b/grpo-mtp-smoke-v1"),
    GRPOSettings,
)
QWEN35_DISTILL = _selection(
    _REFERENCE_CATALOG,
    CatalogRef("training", "qwen3.5-0.8b/on-policy-distill-smoke-v1"),
    OnPolicyDistillationSettings,
)
LFM25_SFT = _selection(
    _REFERENCE_CATALOG,
    CatalogRef("training", "lfm2.5-1.2b/sft-smoke-v2"),
    SFTSettings,
)
LFM25_DPO = _selection(
    _REFERENCE_CATALOG,
    CatalogRef("training", "lfm2.5-1.2b/dpo-smoke-v2"),
    DPOSettings,
)
QWEN_35_2B_AWQ_4BIT = _selection(
    _REFERENCE_CATALOG,
    CatalogRef("quantization", "qwen3.5-2b/awq-4bit-v1"),
    QuantizationPlan,
)
QWEN_35_2B_RTN_W4A16 = _selection(
    _REFERENCE_CATALOG,
    CatalogRef("quantization", "qwen3.5-2b/rtn-w4a16-v3"),
    QuantizationPlan,
)


def resolved_snapshot(catalog: Catalog, *refs: CatalogRef) -> dict[str, JsonValue]:
    """Serialize resolved selections with provenance and reproducibility digests."""

    snapshot: dict[str, JsonValue] = {}
    for ref in refs:
        resolved = catalog.resolve(ref)
        value = resolved.value
        entry: dict[str, JsonValue] = {
            "ref": {"family": ref.family, "id": ref.id},
            "selection_id": value.id,
            "revision": getattr(value, "revision", None),
            "source_layer": resolved.source_layer,
            "overlay_id": resolved.overlay_id,
        }
        if isinstance(value, TrainingBinding):
            entry["parameter_update_kind"] = value.update.kind
            entry["parameter_update_digest"] = parameter_update_digest(value.update)
        if isinstance(value, QuantizationPlan):
            entry["recipe"] = value.recipe
            entry["recipe_digest"] = value.recipe_digest
        snapshot[f"{ref.family}/{ref.id}"] = entry
    return snapshot


__all__ = [
    "AUTOMATIONBENCH_ZAPIER_GRPO",
    "open_project_catalog",
    "FOUNDATION_SMOKE",
    "LFM25_TRL_QLORA",
    "LOCAL_CUDA_8GB",
    "LOCAL_CUDA_ROLLOUT_8GB",
    "LOCAL_CUDA_TEACHER_8GB",
    "QWEN35_DISTILL",
    "QWEN35_DISTILL_TRL_LORA",
    "QWEN35_AUTOMATIONBENCH_GRPO_MTP",
    "QWEN35_AUTOMATIONBENCH_TRL_LORA_THINKING",
    "QWEN35_SFT_VALIDATED_SMOKE",
    "QWEN35_TRL_QLORA",
    "QWEN_35_2B_AWQ_4BIT",
    "QWEN_35_2B_RTN_W4A16",
    "QWEN_35_2B_BF16",
    "QWEN_35_08B_BF16",
    "QWEN_DISTILL_ROLLOUT_VLLM",
    "QWEN_EVAL_VLLM",
    "QWEN_GRPO_MTP_VLLM",
    "QWEN_AUTOMATIONBENCH_GRPO_MTP_VLLM",
    "QWEN_GRPO_VLLM",
    "QWEN_SCREEN_VLLM",
    "QWEN_TEACHER_SCORE_VLLM",
    "open_catalog",
    "resolved_snapshot",
]
