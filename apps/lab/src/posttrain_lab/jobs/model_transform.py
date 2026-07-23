"""Host composition for model weight transformations."""

from posttrain.common import RunContext
from posttrain.train import TransformRequest, TransformResult, run_llm_compressor, transform


def run_quantization_transform(context: RunContext, request: TransformRequest) -> TransformResult:
    return transform(context, request, runner=run_llm_compressor)


__all__ = ["run_quantization_transform"]
