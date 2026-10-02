"""Effective native identity and recipe installation; not model GPU qualification."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from posttrain.common import RunContext
from posttrain.train.backends.verl.contracts import (
    VerlLaunchManifest,
    VerlResolvedGRPOSettings,
    VerlResolvedSAMPOSettings,
)
from posttrain.train.backends.verl.launcher import build_grpo_launch_plan
from posttrain.train.backends.verl.policy_native import run_native, validate_native_selection
from posttrain.train.backends.verl.policy_runtime import native_job_identity
from posttrain.train.profiles import ActiveGroupSampling, AdaptiveCurriculum
from posttrain.train.update_plan import PolicyExecutionBudget, PolicyUpdateSchedule, PolicyUpdateSettings
from posttrain.train.update_records import InvalidPolicyUpdate

from .test_verl_backend import _grpo_request, _sampo_request


def manifest(tmp_path):
    request = _grpo_request()
    selection = PolicyUpdateSettings(PolicyUpdateSchedule("episode", 1), PolicyExecutionBudget(1, 100, 1000))
    settings = replace(
        request.settings,
        loop=replace(request.settings.loop, gradient_accumulation_steps=1),
        beta=0.0,
        policy_updates=selection,
    )
    payload = build_grpo_launch_plan(request, tmp_path).model_dump()
    payload["payload"]["resolved_settings"] = {"kind": "grpo", "settings": settings}
    payload["payload"]["algorithm"]["policy_updates"] = selection
    payload["payload"]["algorithm"]["beta"] = 0.0
    payload["payload"]["training"]["loop"]["gradient_accumulation_steps"] = 1
    payload["payload"]["training"]["target"]["world_size"] = 1
    payload["payload"]["training"]["runtime"]["devices_per_node"] = 1
    payload["payload"]["rollout"]["sampling"].update({"top_p": 1.0, "top_k": 0})
    context = RunContext("project", "work", "run", "train.grpo", "job@1", tmp_path)
    payload["run_context"] = {**context.identity_attributes, "workspace": context.workspace}
    return VerlLaunchManifest.model_validate(payload)


def test_native_identity_binds_effective_engine_and_template_but_not_output_paths(tmp_path):
    pytest.importorskip("verl")
    from .test_update_verl import NativeOperatorEngine

    selected = manifest(tmp_path)
    engine = NativeOperatorEngine()
    engine.engine_config = {"dtype": "bfloat16"}
    engine.optimizer_config = {"lr": 0.0001}
    engine.model_config = SimpleNamespace(
        tokenizer=SimpleNamespace(chat_template="native-template"),
        hf_config=SimpleNamespace(to_dict=lambda: {"model_type": "fixture"}),
    )
    first = native_job_identity(selected, engine)
    assert first == native_job_identity(selected, engine)
    moved = selected.model_copy(
        update={"output_directory": tmp_path / "moved", "result_file": tmp_path / "moved/r.json"}
    )
    assert native_job_identity(moved, engine) == first
    engine.engine_config["dtype"] = "float16"
    assert native_job_identity(selected, engine)[0] != first[0]
    engine.model_config.tokenizer.chat_template = "changed-template"
    assert native_job_identity(selected, engine)[1] != first[1]


@pytest.mark.parametrize(
    "name",
    [
        "allow_fp16_reduced_precision_reduction",
        "allow_bf16_reduced_precision_reduction",
        "allow_fp16_accumulation",
        "flash_sdp_enabled",
        "mem_efficient_sdp_enabled",
        "math_sdp_enabled",
        "cudnn_sdp_enabled",
        "fp16_bf16_reduction_math_sdp_allowed",
    ],
)
def test_native_recovery_identity_binds_gemm_and_attention_arithmetic(tmp_path, monkeypatch, name):
    pytest.importorskip("verl")
    import torch

    from .test_update_verl import NativeOperatorEngine

    selected = manifest(tmp_path)
    engine = NativeOperatorEngine()
    engine.engine_config = {"dtype": "bfloat16"}
    engine.optimizer_config = {"lr": 0.0001}
    engine.model_config = SimpleNamespace(
        tokenizer=SimpleNamespace(chat_template="native-template"),
        hf_config=SimpleNamespace(to_dict=lambda: {"model_type": "fixture"}),
    )
    first = native_job_identity(selected, engine)
    owner = torch.backends.cuda.matmul if name.startswith("allow_") else torch.backends.cuda
    if not hasattr(owner, name):
        pytest.skip(f"Torch {torch.__version__} has no {name} arithmetic control")
    value = getattr(owner, name)
    if callable(value):
        monkeypatch.setattr(owner, name, lambda: not value())
    else:
        monkeypatch.setattr(owner, name, not value)
    changed = native_job_identity(selected, engine)
    assert changed[0] != first[0], "Changed kernel arithmetic must invalidate native checkpoint identity"
    assert changed[1] == first[1], "Kernel choices must not change tokenizer rendering identity"


@pytest.mark.parametrize("field", ["full_precision_matmul", "math_sdpa"])
def test_native_identity_binds_declared_scoped_arithmetic(tmp_path, field):
    pytest.importorskip("verl")
    from verl.workers.config.engine import FSDPEngineConfig

    from .test_update_verl import NativeOperatorEngine

    selected = manifest(tmp_path)
    engine = NativeOperatorEngine()
    engine.engine_config = FSDPEngineConfig()
    engine.optimizer_config = {"lr": 0.0001}
    engine.model_config = SimpleNamespace(
        tokenizer=SimpleNamespace(chat_template="native-template"),
        hf_config=SimpleNamespace(to_dict=lambda: {"model_type": "fixture"}),
    )
    before = native_job_identity(selected, engine)
    engine.engine_config = replace(engine.engine_config, **{field: True})
    after = native_job_identity(selected, engine)
    assert after[0] != before[0], "Native scoped arithmetic must invalidate checkpoint identity"
    assert after[1] == before[1]


def test_native_recipe_installs_actor_trainer_runner_before_ray_dispatch(tmp_path, monkeypatch):
    pytest.importorskip("verl")
    from hydra import compose, initialize_config_module

    selected = manifest(tmp_path)
    assert isinstance(selected.payload.resolved_settings, VerlResolvedGRPOSettings)
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
    installed = []
    monkeypatch.setattr("verl.trainer.main_ppo.run_ppo", lambda config, runner: installed.append((config, runner)))
    run_native(selected, config)
    assert len(installed) == 1 and installed[0][0] is config
    assert callable(installed[0][1].remote)
    from ray import cloudpickle

    # Ray exports the complete actor class, including captured trainer factory.
    # A RunContext created before dispatch contains a non-pickleable thread lock.
    cloudpickle.dumps(installed[0][1].__ray_metadata__.modified_class)


def test_native_selection_rejects_warped_sampler_before_dispatch(tmp_path):
    selected = manifest(tmp_path)
    validate_native_selection(selected)
    selected.payload.rollout.sampling["top_p"] = 0.9
    with pytest.raises(InvalidPolicyUpdate, match="unwarped"):
        validate_native_selection(selected)


def sampo_manifest(tmp_path, **changes):
    # Guard-level fixture: the launch payload is the checked GRPO plan, while
    # the resolved selection is SAMPO, which is what the native guard inspects.
    selected = manifest(tmp_path)
    sampo = _sampo_request().settings
    settings = replace(
        sampo,
        loop=replace(sampo.loop, gradient_accumulation_steps=1),
        policy_updates=selected.payload.algorithm.policy_updates,
        **changes,
    )
    payload = selected.payload.model_copy(update={"resolved_settings": VerlResolvedSAMPOSettings(settings=settings)})
    return selected.model_copy(update={"payload": payload})


def test_native_selection_admits_recorded_sampo_active_collection_only(tmp_path):
    # SAMPO active collection runs through the native buffer and its evidence
    # recorder; adaptive curriculum and GRPO-family filtering remain unqualified.
    validate_native_selection(sampo_manifest(tmp_path, active_sampling=ActiveGroupSampling(3)))
    with pytest.raises(InvalidPolicyUpdate, match="has not qualified"):
        validate_native_selection(sampo_manifest(tmp_path, adaptive_curriculum=AdaptiveCurriculum("domain")))
    selected = sampo_manifest(tmp_path, active_sampling=ActiveGroupSampling(3)).model_copy(update={"run_context": None})
    with pytest.raises(InvalidPolicyUpdate, match="actual host context"):
        validate_native_selection(selected)


def test_native_base_reference_uses_adapter_disable_and_remains_frozen(tmp_path):
    pytest.importorskip("verl")
    import copy
    from contextlib import contextmanager

    import torch
    from posttrain.train.backends.verl.contracts import VerlLoRAUpdate
    from posttrain.train.backends.verl.policy_runtime import base_reference_provider

    from .test_update_verl import NativeOperatorEngine, population

    class ReferenceEngine(NativeOperatorEngine):
        def __init__(self):
            super().__init__()
            self.base = copy.deepcopy(self.model)
            self.reference_calls = 0

        @contextmanager
        def disable_adapter(self):
            actor = self.model
            self.model = self.base
            self.reference_calls += 1
            try:
                yield
            finally:
                self.model = actor

    selected = manifest(tmp_path)
    assert isinstance(selected.payload.resolved_settings, VerlResolvedGRPOSettings)
    settings = replace(selected.payload.resolved_settings.settings, beta=0.01, kl_reference="base")
    selected = selected.model_copy(
        update={
            "payload": selected.payload.model_copy(
                update={
                    "resolved_settings": VerlResolvedGRPOSettings(settings=settings),
                    "training": selected.payload.training.model_copy(
                        update={
                            "update": VerlLoRAUpdate(
                                kind="lora", rank=8, alpha=16, dropout=0.0, target_modules="all-linear"
                            )
                        }
                    ),
                }
            )
        }
    )
    engine = ReferenceEngine()
    runtime = population()
    snapshot = runtime.updates[0].population
    snapshot = replace(snapshot, versions=replace(snapshot.versions, reference="frozen-base"))
    runtime.updates = tuple(replace(update, population=snapshot) for update in runtime.updates)
    provider = base_reference_provider(selected, engine)
    assert provider is not None
    before = provider(runtime)
    with torch.no_grad():
        for parameter in engine.model.parameters():
            parameter.add_(1.0)
    after = provider(runtime)
    assert engine.reference_calls == 2
    assert before.policy_version == "frozen-base"
    for key in before.values:
        torch.testing.assert_close(before.values[key], after.values[key], rtol=0, atol=0)
        assert not before.values[key].requires_grad


def test_worker_generated_overrides_compose_the_resolved_native_recipe(tmp_path, monkeypatch):
    pytest.importorskip("verl")
    from hydra import compose, initialize_config_module
    from posttrain.train.backends.verl.worker import build_hydra_overrides

    selected = manifest(tmp_path).model_copy(
        update={"backend_source_revision": "ef5aac6ff92d5a69f72cfe222f0a409af4220314"}
    )
    monkeypatch.setattr("posttrain.train.backends.verl.worker._model_path", lambda model: "/models/policy")
    overrides = build_hydra_overrides(
        selected, tmp_path / "data.parquet", tmp_path / "agent.json", tmp_path / "checkpoints"
    )
    with initialize_config_module(config_module="verl.trainer.config", version_base=None):
        config = compose(config_name="ppo_trainer", overrides=overrides)
    assert config.trainer.use_v1 and config.trainer.v1.trainer_mode == "sync"
    assert config.actor_rollout_ref.actor.strategy == "fsdp"
    assert config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu == 1
    assert config.actor_rollout_ref.actor.ppo_infer_micro_batch_size_per_gpu == 1
    installed = []
    monkeypatch.setattr("verl.trainer.main_ppo.run_ppo", lambda config, runner: installed.append(runner))
    run_native(selected, config)
    assert len(installed) == 1


def test_native_cli_runs_real_config_validation_before_dispatch(tmp_path, monkeypatch):
    pytest.importorskip("verl")
    from posttrain.train.backends.verl import policy_native
    from posttrain.train.backends.verl.worker import build_hydra_overrides

    selected = manifest(tmp_path).model_copy(
        update={"backend_source_revision": "ef5aac6ff92d5a69f72cfe222f0a409af4220314"}
    )
    path = tmp_path / "manifest.json"
    selected.write(path)
    monkeypatch.setattr("posttrain.train.backends.verl.worker._model_path", lambda model: "/models/policy")
    overrides = build_hydra_overrides(
        selected, tmp_path / "data.parquet", tmp_path / "agent.json", tmp_path / "checkpoints"
    )
    installed = []
    monkeypatch.setattr(policy_native, "run_native", lambda manifest, config: installed.append((manifest, config)))
    monkeypatch.setattr("sys.argv", ["policy_native", str(path), *overrides])
    policy_native.main()
    assert len(installed) == 1 and installed[0][0] == selected
