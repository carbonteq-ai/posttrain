"""Tests for vLLM inference-binding translation."""

from dataclasses import replace

import pytest
from posttrain.common import InferenceBinding, Workload
from posttrain.common.variants import NANBEIGE_42_3B
from posttrain.serve import ServeBenchmarkRequest
from posttrain.serve.backends.vllm.bindings import (
    benchmark_config,
    engine_config,
    frontend_args,
    resolve_binding_configuration,
)
from posttrain.serve.benchmarks import CORE_INFERENCE_V1
from posttrain.serve.profiles import VllmEngineConfig, VllmSpeculativeConfig


def test_qwen_screen_binding_captures_tested_8gb_constraints(qwen_screen_binding: InferenceBinding) -> None:
    binding = qwen_screen_binding
    kwargs = engine_config(binding).as_vllm_kwargs()
    assert kwargs["enforce_eager"] is True
    assert kwargs["gpu_memory_utilization"] == 0.75
    assert kwargs["limit_mm_per_prompt"] == {"image": 0, "video": 0, "audio": 0}
    assert kwargs["skip_mm_profiling"] is True
    assert kwargs["max_num_seqs"] == 4


def test_resolution_explains_explicit_and_default_settings(qwen_screen_binding: InferenceBinding) -> None:
    resolved = resolve_binding_configuration(qwen_screen_binding)
    origins = {origin.path: origin for origin in resolved.origins}

    assert origins["engine.enforce_eager"].kind == "explicit"
    assert origins["engine.enforce_eager"].value is True
    assert origins["sampling.ignore_eos"].kind == "default"
    assert origins["reasoning_mode"].kind == "default"
    assert origins["reasoning_mode"].value == qwen_screen_binding.model.default_reasoning_mode


def test_explicit_reasoning_mode_is_preserved_in_resolution(qwen_screen_binding: InferenceBinding) -> None:
    binding = replace(qwen_screen_binding, reasoning_mode="thinking")

    resolved = resolve_binding_configuration(binding)

    assert resolved.reasoning_mode == "thinking"
    assert next(origin for origin in resolved.origins if origin.path == "reasoning_mode").kind == "explicit"


def test_lfm_binding_uses_model_renderer_and_frontend_parsers(lfm_screen_binding: InferenceBinding) -> None:
    binding = lfm_screen_binding
    assert binding.renderer == binding.model.renderer.id
    assert frontend_args(binding) == (
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "lfm2",
        "--reasoning-parser",
        "lfm2",
    )


def test_qwen_tool_capability_uses_its_declared_xml_protocol(qwen_screen_binding: InferenceBinding) -> None:
    binding = replace(
        qwen_screen_binding,
        capabilities=("tool-calling",),
        engine={**qwen_screen_binding.engine, "reasoning_parser": "qwen3"},
    )

    assert binding.model.conversation.tool_calls is not None
    assert binding.model.conversation.tool_calls.id == "qwen3_xml"
    assert frontend_args(binding) == (
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_xml",
        "--reasoning-parser",
        "qwen3",
    )


def test_vllm_rejects_parser_override_that_conflicts_with_model_protocol(
    qwen_screen_binding: InferenceBinding,
) -> None:
    binding = replace(
        qwen_screen_binding,
        capabilities=("tool-calling",),
        engine={**qwen_screen_binding.engine, "tool_call_parser": "hermes"},
    )

    with pytest.raises(ValueError, match="conflicts with the selected model tool-call protocol"):
        frontend_args(binding)


def test_skip_mm_profiling_requires_text_only_mode() -> None:
    with pytest.raises(ValueError, match="text-only"):
        VllmEngineConfig(max_model_len=1_024, gpu_memory_utilization=0.75, skip_mm_profiling=True)


def test_tensor_parallel_size_is_validated_and_forwarded(qwen_screen_binding: InferenceBinding) -> None:
    binding = replace(
        qwen_screen_binding,
        engine={**qwen_screen_binding.engine, "tensor_parallel_size": 2},
    )

    engine = engine_config(binding)

    assert engine.tensor_parallel_size == 2
    assert engine.as_vllm_kwargs()["tensor_parallel_size"] == 2
    assert ("--tensor-parallel-size", "2") == engine.as_cli_args()[4:6]

    with pytest.raises(ValueError, match="tensor_parallel_size"):
        VllmEngineConfig(max_model_len=1_024, gpu_memory_utilization=0.75, tensor_parallel_size=0)


def test_mtp_requires_a_model_variant_that_declares_it(qwen_screen_binding: InferenceBinding) -> None:
    binding = replace(
        qwen_screen_binding,
        model=NANBEIGE_42_3B,
        renderer=NANBEIGE_42_3B.renderer.id,
        engine={
            **qwen_screen_binding.engine,
            "speculative_config": {"method": "mtp", "num_speculative_tokens": 2},
        },
    )

    with pytest.raises(ValueError, match="does not declare MTP capability"):
        engine_config(binding)


def test_local_matrix_stops_at_concurrency_four_and_requires_turboquant_at_32k() -> None:
    cells = CORE_INFERENCE_V1.cells(max_concurrency=4)
    assert {cell.concurrency for cell in cells} == {1, 2, 4}
    assert all(cell.required_variant == "turboquant" for cell in cells if cell.context_window == 32_768)


def test_representative_workload_resolves_and_verifies_packaged_corpus(
    qwen_screen_binding: InferenceBinding,
    representative_workload: Workload,
) -> None:
    config = benchmark_config(ServeBenchmarkRequest(qwen_screen_binding, representative_workload))

    assert config.cohort == "representative"
    assert config.cells[0].input_tokens is None
    assert config.corpus is not None
    assert config.corpus.manifest.id == "general-serving-v1"
    assert config.corpus.manifest.record_count == 128
    assert config.selection_seed == 17


def test_workload_concurrency_becomes_one_ordered_sweep(
    qwen_screen_binding: InferenceBinding,
    representative_workload: Workload,
) -> None:
    workload = Workload(
        id=representative_workload.id,
        revision=representative_workload.revision,
        requests=representative_workload.requests,
        concurrency=(1, 2, 4),
        warmup_repetitions=representative_workload.warmup_repetitions,
        measured_repetitions=representative_workload.measured_repetitions,
    )

    config = benchmark_config(ServeBenchmarkRequest(qwen_screen_binding, workload))

    assert tuple(cell.concurrency for cell in config.cells) == (1, 2, 4)


def test_dspark_binding_retains_immutable_draft_identity_and_uses_actual_variant(
    qwen_screen_binding: InferenceBinding,
    representative_workload: Workload,
) -> None:
    binding = replace(
        qwen_screen_binding,
        model=NANBEIGE_42_3B,
        backend="vllm@62f6de733d7ae63b759329993bc209e67afdf431",
        renderer=NANBEIGE_42_3B.renderer_contract,
        engine={
            **qwen_screen_binding.engine,
            "speculative_config": {
                "method": "dspark",
                "num_speculative_tokens": 7,
                "draft_model": {
                    "repo_id": "Nanbeige/Nanbeige4.2-3B-DSpark",
                    "revision": "a" * 40,
                    "path": "/models/nanbeige4.2-3b-dspark",
                },
            },
        },
    )

    config = benchmark_config(ServeBenchmarkRequest(binding, representative_workload))

    assert config.engine.speculative is not None
    assert config.engine.speculative.draft_model is not None
    assert config.engine.speculative.draft_model.revision == "a" * 40
    assert config.engine.speculative.as_vllm() == {
        "method": "dspark",
        "num_speculative_tokens": 7,
        "model": "/models/nanbeige4.2-3b-dspark",
    }
    assert all(cell.required_variant == "dspark" for cell in config.cells)


def test_dspark_and_turboquant_are_rejected_for_pinned_nanbeige_runtime(
    qwen_screen_binding: InferenceBinding,
    representative_workload: Workload,
) -> None:
    binding = replace(
        qwen_screen_binding,
        model=NANBEIGE_42_3B,
        backend="vllm@62f6de733d7ae63b759329993bc209e67afdf431",
        renderer=NANBEIGE_42_3B.renderer_contract,
        engine={
            **qwen_screen_binding.engine,
            "kv_cache_dtype": "turboquant_k8v4",
            "speculative_config": {
                "method": "dspark",
                "num_speculative_tokens": 7,
                "draft_model": {
                    "repo_id": "Nanbeige/Nanbeige4.2-3B-DSpark",
                    "revision": "5f12c792dabbfcdc4a0cf504e75f7216706d9590",
                },
            },
        },
    )

    with pytest.raises(ValueError, match="non-causal draft attention"):
        benchmark_config(ServeBenchmarkRequest(binding, representative_workload))


def test_dspark_turboquant_limit_does_not_prejudge_future_runtime(
    qwen_screen_binding: InferenceBinding,
    representative_workload: Workload,
) -> None:
    binding = replace(
        qwen_screen_binding,
        model=NANBEIGE_42_3B,
        backend=f"vllm@{'b' * 40}",
        renderer=NANBEIGE_42_3B.renderer_contract,
        engine={
            **qwen_screen_binding.engine,
            "kv_cache_dtype": "turboquant_k8v4",
            "speculative_config": {
                "method": "dspark",
                "num_speculative_tokens": 7,
                "draft_model": {
                    "repo_id": "Nanbeige/Nanbeige4.2-3B-DSpark",
                    "revision": "5f12c792dabbfcdc4a0cf504e75f7216706d9590",
                },
            },
        },
    )

    config = benchmark_config(ServeBenchmarkRequest(binding, representative_workload))

    assert all(cell.required_variant == "dspark-turboquant" for cell in config.cells)


def test_speculative_and_kv_cache_variants_are_composed_without_relabeling(
    qwen_screen_binding: InferenceBinding,
    representative_workload: Workload,
) -> None:
    binding = replace(
        qwen_screen_binding,
        engine={
            **qwen_screen_binding.engine,
            "kv_cache_dtype": "turboquant_k8v4",
            "speculative_config": {"method": "mtp", "num_speculative_tokens": 1},
        },
    )

    config = benchmark_config(ServeBenchmarkRequest(binding, representative_workload))

    assert all(cell.required_variant == "mtp-turboquant" for cell in config.cells)


def test_dspark_requires_a_host_materialized_immutable_draft() -> None:
    with pytest.raises(ValueError, match="immutable draft model"):
        VllmSpeculativeConfig(method="dspark", num_speculative_tokens=7)


def test_nanbeige_renderer_drives_its_vllm_parsers(qwen_screen_binding: InferenceBinding) -> None:
    binding = replace(
        qwen_screen_binding,
        model=NANBEIGE_42_3B,
        renderer=NANBEIGE_42_3B.renderer.id,
        capabilities=("tool-calling",),
        engine={**qwen_screen_binding.engine, "reasoning_parser": "nanbeige"},
    )

    assert frontend_args(binding) == (
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "nanbeige",
        "--reasoning-parser",
        "nanbeige",
    )
