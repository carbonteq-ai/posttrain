"""Explicit ordinary-layout admission and effective numerical-profile validation."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from posttrain.train.backends.verl.contracts import VerlLoRAUpdate
from posttrain.train.backends.verl.policy_capabilities import (
    native_context_layout,
    native_execution_capabilities,
    validate_native_execution_profile,
)
from posttrain.train.backends.verl.policy_native import run_native, validate_native_selection
from posttrain.train.update_records import InvalidPolicyUpdate

from .test_update_verl_runtime import manifest


def dense_manifest(tmp_path, records=2, precision="bf16"):
    selected = manifest(tmp_path)
    resolved = selected.payload.resolved_settings
    assert resolved is not None
    settings = resolved.settings
    updates = settings.policy_updates
    assert updates is not None
    settings = replace(settings, policy_updates=replace(updates, execution=replace(updates.execution, records=records)))
    return selected.model_copy(
        update={
            "payload": selected.payload.model_copy(
                update={
                    "resolved_settings": resolved.model_copy(update={"settings": settings}),
                    "algorithm": selected.payload.algorithm.model_copy(
                        update={"policy_updates": settings.policy_updates}
                    ),
                    "training": selected.payload.training.model_copy(
                        update={
                            "backend_options": {
                                **selected.payload.training.backend_options,
                                "resolved_context_layout": "dense-population",
                                "training_precision": precision,
                            },
                            "update": VerlLoRAUpdate(
                                kind="lora", rank=8, alpha=16, dropout=0.0, target_modules="q_proj,v_proj"
                            ),
                        }
                    ),
                }
            )
        }
    )


def profile(selected):
    from verl.workers.config.engine import FSDPEngineConfig

    update = selected.payload.training.update
    model = {
        "lora_rank": update.rank,
        "lora_alpha": update.alpha,
        "lora_dropout": 0.0,
        "use_remove_padding": False,
        "override_config": {"attn_implementation": "sdpa"},
    }
    engine = FSDPEngineConfig(
        strategy="fsdp",
        use_orig_params=True,
        full_determinism=True,
        use_torch_compile=False,
        use_dynamic_bsz=False,
        use_fused_kernels=False,
        lora_fp32_compute=True,
        lora_rowwise_compute=True,
        contiguous_linear_output_gradients=True,
        full_precision_matmul=True,
        math_sdpa=True,
        mixed_precision={
            "param_dtype": selected.payload.training.backend_options["training_precision"],
            "reduce_dtype": "fp32",
            "buffer_dtype": "fp32",
        },
    )
    return model, engine


def test_legacy_layout_default_retains_one_record_admission(tmp_path):
    selected = manifest(tmp_path)
    validate_native_selection(selected)
    assert native_context_layout(selected) == "ragged"
    # The additional profile does not impose numeric settings on the old path.
    validate_native_execution_profile(selected, {}, {})


@pytest.mark.parametrize("records", [1, 2, 3])
@pytest.mark.parametrize("precision", ["bf16", "fp16"])
def test_explicit_dense_layout_admits_checked_capability_without_changing_schedule(tmp_path, records, precision):
    pytest.importorskip("verl")
    selected = dense_manifest(tmp_path, records, precision)
    validate_native_selection(selected)
    assert selected.payload.resolved_settings is not None
    before = selected.payload.resolved_settings.settings.policy_updates
    model, config = profile(selected)
    capabilities = native_execution_capabilities(selected, SimpleNamespace(model_config=model, engine_config=config))
    assert capabilities.context_layout == "dense-population"
    assert selected.payload.resolved_settings.settings.policy_updates == before
    assert capabilities.statistics == ("sampled-logp", "old-logp", "reference-logp")


@pytest.mark.parametrize("layout", [None, "ragged", "dense-pack", "unknown"])
def test_multi_record_selection_rejects_unqualified_or_implicit_layout(tmp_path, layout):
    selected = dense_manifest(tmp_path)
    options = selected.payload.training.backend_options
    if layout is None:
        options.pop("resolved_context_layout")
    else:
        options["resolved_context_layout"] = layout
    with pytest.raises(InvalidPolicyUpdate, match="layout"):
        validate_native_selection(selected)


@pytest.mark.parametrize(
    "field,value",
    [
        ("use_orig_params", False),
        ("full_determinism", False),
        ("lora_fp32_compute", False),
        ("lora_rowwise_compute", False),
        ("contiguous_linear_output_gradients", False),
        ("full_precision_matmul", False),
        ("math_sdpa", False),
        ("use_torch_compile", True),
        ("use_dynamic_bsz", True),
        ("use_fused_kernels", True),
        ("model_dtype", "bf16"),
        ("strategy", "fsdp2"),
        ("ulysses_sequence_parallel_size", 2),
        ("mixed_precision", {"param_dtype": "bf16", "reduce_dtype": "bf16", "buffer_dtype": "fp32"}),
        ("mixed_precision", {"param_dtype": "fp16", "reduce_dtype": "fp32", "buffer_dtype": "fp32"}),
    ],
)
def test_effective_engine_profile_rejects_each_unqualified_configuration(tmp_path, field, value):
    pytest.importorskip("verl")
    selected = dense_manifest(tmp_path)
    model, config = profile(selected)
    # Test the admission boundary independently of native dataclass validation.
    changed = {**dict(config), field: value}
    with pytest.raises(InvalidPolicyUpdate, match="qualified explicit"):
        validate_native_execution_profile(selected, model, changed)


@pytest.mark.parametrize(
    "field,value",
    [
        ("lora_dropout", 0.1),
        ("lora_rank", 0),
        ("use_remove_padding", True),
        ("use_fused_kernels", True),
        ("override_config", {"attn_implementation": "flash_attention_2"}),
        ("bitsandbytes", {"enable": True}),
    ],
)
def test_effective_model_profile_rejects_unqualified_configuration(tmp_path, field, value):
    pytest.importorskip("verl")
    selected = dense_manifest(tmp_path)
    model, config = profile(selected)
    with pytest.raises(InvalidPolicyUpdate, match="qualified explicit"):
        validate_native_execution_profile(selected, {**model, field: value}, config)


def test_invalid_dense_profile_rejects_before_ray_dispatch(tmp_path, monkeypatch):
    pytest.importorskip("verl")
    from hydra import compose, initialize_config_module

    selected = dense_manifest(tmp_path)
    assert selected.payload.resolved_settings is not None
    settings = selected.payload.resolved_settings.settings
    with initialize_config_module(config_module="verl.trainer.config", version_base=None):
        config = compose(
            config_name="ppo_trainer",
            overrides=[
                "algorithm.adv_estimator=grpo",
                f"trainer.total_training_steps={settings.loop.max_steps}",
                f"data.train_batch_size={settings.num_prompts_per_step}",
                f"actor_rollout_ref.rollout.n={settings.num_generations}",
                f"actor_rollout_ref.rollout.temperature={selected.payload.rollout.sampling.get('temperature', 1.0)}",
            ],
        )
    called = []
    monkeypatch.setattr("verl.trainer.main_ppo.run_ppo", lambda *args: called.append(args))
    with pytest.raises(InvalidPolicyUpdate, match="qualified explicit"):
        run_native(selected, config)
    assert not called


@pytest.mark.parametrize("precision", ["bf16", "fp16"])
def test_worker_composed_dense_profile_is_accepted_before_dispatch(tmp_path, precision):
    pytest.importorskip("verl")
    from hydra import compose, initialize_config_module
    from posttrain.train.backends.verl.worker import build_hydra_overrides

    selected = dense_manifest(tmp_path, precision=precision)
    options = selected.payload.training.backend_options
    options["attention_implementation"] = "sdpa"
    options["hydra_overrides"] = [
        "actor_rollout_ref.actor.fsdp_config.use_orig_params=true",
        "actor_rollout_ref.actor.fsdp_config.use_torch_compile=false",
        "actor_rollout_ref.actor.fsdp_config.full_determinism=true",
        *[
            f"++actor_rollout_ref.actor.fsdp_config.{field}=true"
            for field in (
                "lora_fp32_compute",
                "lora_rowwise_compute",
                "contiguous_linear_output_gradients",
                "full_precision_matmul",
                "math_sdpa",
            )
        ],
    ]
    overrides = build_hydra_overrides(selected, tmp_path / "data.parquet", tmp_path / "agents.yaml", tmp_path / "model")
    with initialize_config_module(config_module="verl.trainer.config", version_base=None):
        config = compose(config_name="ppo_trainer", overrides=overrides)
    validate_native_execution_profile(
        selected,
        config.actor_rollout_ref.model,
        config.actor_rollout_ref.actor.fsdp_config,
        actor_config=config.actor_rollout_ref.actor,
    )
    assert config.actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype == precision
