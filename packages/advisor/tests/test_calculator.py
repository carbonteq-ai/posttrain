"""The settings calculator applies measured rules to hardware, model and task."""

import pytest
from posttrain.advisor.calculator import Architecture, Hardware, Task, suggest

# config.json excerpts of the pinned checkpoints (hybrid layouts preserved).
LFM25_26B = {
    "num_hidden_layers": 30,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "hidden_size": 2048,
    "max_position_embeddings": 131072,
    "layer_types": ["conv", "conv", "full_attention"] * 8 + ["conv"] * 6,
}
GEMMA4_12B = {
    "text_config": {
        "num_hidden_layers": 48,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "num_global_key_value_heads": 1,
        "head_dim": 256,
        "global_head_dim": 512,
        "hidden_size": 3840,
        "sliding_window": 1024,
        "max_position_embeddings": 262144,
        "layer_types": (["sliding_attention"] * 5 + ["full_attention"]) * 8,
    }
}


def test_architecture_counts_only_full_attention_kv_for_hybrids() -> None:
    lfm = Architecture.from_config(LFM25_26B, 2_600_000_000)
    assert (lfm.attention_layers, lfm.recurrent_layers) == (8, 22)
    # 8 layers x 8 KV heads x 64 dims x K,V x bf16: 4 GiB holds about 262k tokens.
    assert lfm.kv_bytes_per_token() == 16_384
    gemma = Architecture.from_config(GEMMA4_12B, 12_000_000_000)
    assert (gemma.attention_layers, gemma.sliding_window_layers) == (8, 40)
    assert gemma.kv_bytes_per_token() == 2 * 8 * 1 * 512 * 2
    assert gemma.windowed_kv_bytes_per_sequence(32_768) == 2 * 40 * 8 * 256 * 2 * 1024


def test_vortex_rollout_suggestion_matches_the_measured_profile() -> None:
    lfm = Architecture.from_config(LFM25_26B, 2_600_000_000)
    task = Task("rollout", 40, 20_480, 4_096, 0.8, colocated_trainer_gb=60.0, lora_rank=4)
    suggestion = suggest(Hardware(96, "RTXPRO6000"), lfm, task)
    engine = suggestion.engine
    assert engine["enforce_eager"] is False
    assert engine["enable_prefix_caching"] is True
    assert engine["flash_attn_version"] == 2
    assert engine["max_model_len"] == 24_576
    assert engine["max_num_seqs"] == 40
    assert engine["mode"] == "colocate" and engine["sleep_during_optimization"] is True
    assert engine["max_lora_rank"] == 8
    # The hand-tuned c40-4k@2 binding reserves 5 GiB of KV for this workload.
    assert 4.5 < engine["kv_cache_memory_bytes"] / 2**30 < 6.0
    assert suggestion.max_concurrency is not None and suggestion.max_concurrency > 40
    assert not any("ngram" in note for note in suggestion.notes)


def test_context_is_capped_by_the_model_and_reported() -> None:
    lfm = Architecture.from_config(LFM25_26B, 2_600_000_000)
    suggestion = suggest(Hardware(96, "RTXPRO6000"), lfm, Task("eval", 8, 150_000, 4_096))
    assert suggestion.engine["max_model_len"] == 131_072
    assert any("exceeds the model's 131072-token context" in note for note in suggestion.notes)


def test_small_colocated_gpu_falls_back_to_eager_and_says_why() -> None:
    lfm = Architecture.from_config(LFM25_26B, 2_600_000_000)
    task = Task("rollout", 4, 512, 128, colocated_trainer_gb=1.5)
    suggestion = suggest(Hardware(8, "RTX3070TI"), lfm, task)
    assert suggestion.engine["enforce_eager"] is True
    reason = next(item.reason for item in suggestion.settings if item.key == "enforce_eager")
    assert "VLLM_EAGER_DISABLES_CUDA_GRAPHS" in reason
    assert suggestion.memory_gb["cuda_graphs_estimate"] == 0.0


def test_pre_ampere_gpus_compute_in_float16_and_invariance_is_opt_in() -> None:
    lfm = Architecture.from_config(LFM25_26B, 2_600_000_000)
    turing = suggest(Hardware(16, supports_bf16=False), lfm, Task("eval", 4, 1024, 256))
    assert turing.engine["dtype"] == "float16"
    reproducible = suggest(Hardware(96, "RTXPRO6000"), lfm, Task("rollout", 8, 1024, 256, reproducible_logprobs=True))
    assert reproducible.environment == {"VLLM_BATCH_INVARIANT": "1"}


@pytest.mark.parametrize("model", ["RTXPRO6000", "rtx pro 6000", "RTX-PRO-6000"])
def test_accelerator_names_are_normalized(model: str) -> None:
    assert Hardware(96, model).facts is not None


def test_budget_with_no_room_for_kv_says_so_instead_of_suggesting_eager() -> None:
    lfm = Architecture.from_config(LFM25_26B, 2_600_000_000)
    suggestion = suggest(Hardware(8, "RTX3070TI"), lfm, Task("rollout", 4, 512, 128, colocated_trainer_gb=3.0))
    assert any("no room for KV cache" in note for note in suggestion.notes)
    assert suggestion.max_concurrency is None


QWEN3_NEXT_80B_A3B = {
    "num_hidden_layers": 48,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "hidden_size": 2048,
    "full_attention_interval": 4,
    "num_experts": 512,
    "num_experts_per_tok": 10,
    "moe_intermediate_size": 512,
}


def test_qwen3_next_interval_hybrid_and_moe_active_parameters() -> None:
    arch = Architecture.from_config(QWEN3_NEXT_80B_A3B, 80_000_000_000)
    assert (arch.attention_layers, arch.recurrent_layers) == (12, 36)
    assert arch.kv_bytes_per_token() == 2 * 12 * 2 * 256 * 2
    # 502 idle experts x 3 x 2048 x 512 x 48 layers are not read per token.
    assert arch.active_parameters == 80_000_000_000 - 502 * 3 * 2048 * 512 * 48
    assert arch.decode_parameters < arch.parameters


def test_dense_qwen_without_layer_types_is_all_full_attention() -> None:
    qwen3_4b = {
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "hidden_size": 2560,
    }
    arch = Architecture.from_config(qwen3_4b, 4_000_000_000)
    assert (arch.attention_layers, arch.recurrent_layers, arch.active_parameters) == (36, 0, None)
    assert arch.kv_bytes_per_token() == 147_456


def test_turboquant_keeps_float16_and_dedicated_engines_do_not_pin_kv() -> None:
    lfm = Architecture.from_config(LFM25_26B, 2_600_000_000)
    suggestion = suggest(Hardware(8, "RTX3070TI"), lfm, Task("eval", 4, 2048, 1024, kv_cache_dtype="turboquant_k8v4"))
    assert suggestion.engine["dtype"] == "float16"
    assert suggestion.engine["kv_cache_dtype"] == "turboquant_k8v4"
    assert "kv_cache_memory_bytes" not in suggestion.engine
    assert suggestion.engine["gpu_memory_utilization"] == 0.9


def test_draft_model_weights_and_kv_come_out_of_the_budget() -> None:
    gemma = Architecture(
        parameters=12_000_000_000,
        hidden_size=3_840,
        layers=48,
        attention_layers=8,
        kv_heads=8,
        head_dim=256,
        sliding_window_layers=40,
        sliding_window=1_024,
    )
    assistant = Architecture(
        parameters=420_000_000, hidden_size=1_024, layers=4, attention_layers=4, kv_heads=1, head_dim=256
    )
    hardware = Hardware(96, "RTXPRO6000")
    plain = suggest(hardware, gemma, Task("judge", 32, 30_000, 2_000, 0.0))
    drafted = suggest(hardware, gemma, Task("judge", 32, 30_000, 2_000, 0.0, speculative_tokens=2, draft=assistant))
    assert drafted.memory_gb["draft_weights"] > 0.7
    assert drafted.max_concurrency is not None and plain.max_concurrency is not None
    assert drafted.max_concurrency < plain.max_concurrency
    assert any("verifies 3 tokens" in note for note in drafted.notes)


def test_native_mtp_layers_add_kv_without_draft_weights() -> None:
    qwen = Architecture(
        parameters=2_000_000_000,
        hidden_size=2_048,
        layers=24,
        attention_layers=6,
        kv_heads=2,
        head_dim=256,
        recurrent_layers=18,
        mtp_layers=1,
    )
    hardware = Hardware(24, "RTXPRO4500")
    drafted = suggest(hardware, qwen, Task("eval", 16, 6_000, 2_000, 0.6, speculative_tokens=2))
    assert "draft_weights" not in drafted.memory_gb
    assert any("native MTP heads add" in note for note in drafted.notes)


def test_other_engines_on_the_device_shrink_the_share() -> None:
    lfm = Architecture(
        parameters=2_600_000_000,
        hidden_size=2_048,
        layers=30,
        attention_layers=8,
        kv_heads=8,
        head_dim=64,
        recurrent_layers=22,
    )
    alone = suggest(Hardware(96, "RTXPRO6000"), lfm, Task("eval", 16, 6_000, 2_000))
    shared = suggest(Hardware(96, "RTXPRO6000"), lfm, Task("eval", 16, 6_000, 2_000, co_tenant_gb=45.0))
    assert alone.engine["gpu_memory_utilization"] == 0.9
    assert shared.engine["gpu_memory_utilization"] < 0.5
    assert shared.memory_gb["other_engines"] == 45.0 and "kv_cache_memory_bytes" in shared.engine
