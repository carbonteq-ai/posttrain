"""veRL backend precision: FSDP mixed precision, rollout dtype, and loss-scaler skipped steps."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import pytest
from posttrain.train import LoRAUpdate
from posttrain.train.backends.verl.launcher import build_grpo_launch_plan
from posttrain.train.backends.verl.metrics import read_verl_metric_records
from posttrain.train.backends.verl.worker import _backend_hydra_overrides, build_hydra_overrides
from posttrain.train.grpo_observations import GRPOObservationFeatures, normalize_grpo_metrics
from posttrain.train.precision import resolve_precision, verl_mixed_precision_overrides, verl_rollout_dtype

from .test_verl_backend import _grpo_request

_POLICY = "{param_dtype:fp16,reduce_dtype:fp32,buffer_dtype:fp32}"


def _overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, options=None, engine=None) -> list[str]:
    request = _grpo_request(update=LoRAUpdate(rank=8, alpha=16))
    training = replace(request.training, backend_options={**request.training.backend_options, **(options or {})})
    inference = replace(request.inference, engine={**request.inference.engine, **(engine or {})})
    plan = build_grpo_launch_plan(replace(request, training=training, inference=inference), tmp_path)
    monkeypatch.setattr("posttrain.train.backends.verl.worker._model_path", lambda model: "/models/qwen35")
    return build_hydra_overrides(plan, tmp_path / "r.parquet", tmp_path / "a.json", tmp_path / "ckpt")


def test_bf16_default_adds_no_precision_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    default = _overrides(tmp_path, monkeypatch)
    explicit = _overrides(tmp_path, monkeypatch, options={"training_precision": "bf16"})
    assert default == explicit
    assert not any("mixed_precision" in value for value in default)
    assert "actor_rollout_ref.rollout.dtype=bfloat16" in default


def test_fp16_selects_fsdp_fp16_compute_for_actor_and_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    overrides = _overrides(tmp_path, monkeypatch, options={"training_precision": "fp16"}, engine={"dtype": "float16"})
    assert f"+actor_rollout_ref.actor.fsdp_config.mixed_precision={_POLICY}" in overrides
    assert f"+actor_rollout_ref.ref.fsdp_config.mixed_precision={_POLICY}" in overrides
    assert "actor_rollout_ref.rollout.dtype=float16" in overrides
    assert verl_mixed_precision_overrides({}) == []


def test_rollout_dtype_is_validated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert verl_rollout_dtype({}) == ("bfloat16", "backend-default")
    assert verl_rollout_dtype({"kv_cache_dtype": "turboquant_k8v4"}) == ("float16", "turboquant")
    assert verl_rollout_dtype({"dtype": "float32"}) == ("float32", "binding")
    with pytest.raises(ValueError, match="bfloat16, float16, float32"):
        _overrides(tmp_path, monkeypatch, engine={"dtype": "half"})
    with pytest.raises(ValueError, match="TurboQuant KV cache requires"):
        _overrides(tmp_path, monkeypatch, engine={"dtype": "bfloat16", "kv_cache_dtype": "turboquant_k8v4"})


def test_backend_overrides_cannot_replace_the_selected_precision() -> None:
    for value in (
        "actor_rollout_ref.rollout.dtype=float16",
        f"+actor_rollout_ref.actor.fsdp_config.mixed_precision={_POLICY}",
        "++actor_rollout_ref.ref.fsdp_config.mixed_precision.param_dtype=fp16",
    ):
        with pytest.raises(ValueError, match="cannot replace"):
            _backend_hydra_overrides({"hydra_overrides": [value]})


def test_resolved_verl_precision_describes_fsdp_master_weights() -> None:
    resolved = resolve_precision({"training_precision": "fp16"}, {"dtype": "float16"}, "bf16", backend="verl")
    assert resolved.summary() == (
        "trainer fp16 (FSDP float16 compute over float32 master weights, dynamic loss scaling from 1024); "
        "rollout vLLM float16 (binding)"
    )
    default = resolve_precision({}, {}, "bf16", backend="verl")
    assert (default.model_load_dtype, default.rollout_dtype, default.rollout_dtype_source) == (
        "float32",
        "bfloat16",
        "backend-default",
    )


def _sidecar(path: Path, rows: Sequence[Mapping[str, object]]) -> Path:
    path.write_text("".join(json.dumps({"step": i + 1, "data": row}) + "\n" for i, row in enumerate(rows)))
    return path


def test_loss_scaled_sidecar_records_skipped_steps(tmp_path: Path) -> None:
    path = _sidecar(
        tmp_path / "m.jsonl",
        [
            {"actor/grad_norm": math.inf, "actor/pg_loss": 0.1, "variance_proxy/x": math.nan},
            {"actor/grad_norm": 0.5, "actor/pg_loss": 0.2},
            {"training/global_step": 3},
        ],
    )
    with pytest.raises(ValueError, match="non-finite constant"):
        read_verl_metric_records(path)
    records = read_verl_metric_records(path, loss_scaling=True)
    assert dict(records[0].data) == {
        "actor/pg_loss": 0.1,
        "actor/optimizer_step_skipped": 1.0,
        "actor/optimizer_steps_skipped": 1.0,
    }
    assert records[1].data["actor/optimizer_step_skipped"] == 0.0
    assert records[1].data["actor/optimizer_steps_skipped"] == 1.0
    assert "actor/optimizer_step_skipped" not in records[2].data
    normalized = normalize_grpo_metrics(
        backend="verl", step=1, native=records[0].data, features=GRPOObservationFeatures()
    )
    assert normalized.metrics["train/optimizer_step_skipped"] == 1.0
    assert "train/grad_norm" not in normalized.metrics


def test_non_finite_values_without_a_skip_still_fail(tmp_path: Path) -> None:
    path = _sidecar(tmp_path / "m.jsonl", [{"actor/grad_norm": 0.5, "actor/pg_loss": math.nan}])
    with pytest.raises(ValueError, match="not a loss-scaler skip"):
        read_verl_metric_records(path, loss_scaling=True)


def test_verl_mismatch_metrics_normalize() -> None:
    native = {
        "training/rollout_probs_diff_mean": 0.001,
        "training/rollout_probs_diff_max": 0.2,
        "training/rollout_actor_probs_pearson_corr": 0.999,
    }
    metrics = normalize_grpo_metrics(backend="verl", step=1, native=native, features=GRPOObservationFeatures()).metrics
    assert metrics["train/rl/sampling_prob_delta_mean"] == 0.001
    assert metrics["train/rl/sampling_prob_delta_max"] == 0.2
    assert metrics["train/rl/sampling_prob_pearson_corr"] == 0.999


def test_post4_loss_scale_and_log_prob_gap_metrics_normalize(tmp_path: Path) -> None:
    rows = [
        {"actor/grad_norm": 0.4, "actor/loss_scale": 65536.0, "actor/optimizer_step_skipped": 0.0},
        {"actor/grad_norm": math.inf, "actor/loss_scale": 65536.0, "actor/optimizer_step_skipped": 0.5},
        {
            "actor/grad_norm": 0.3,
            "actor/loss_scale": 32768.0,
            "actor/optimizer_step_skipped": 0.0,
            "training/rollout_logp_diff_mean": 0.002,
            "training/rollout_logp_diff_p99": 0.02,
            "training/rollout_logp_diff_max": 0.1,
            "training/rollout_seq_logp_diff_abs_mean": 0.05,
            "training/rollout_probs_diff_mean": 0.001,
        },
    ]
    records = read_verl_metric_records(_sidecar(tmp_path / "m.jsonl", rows), loss_scaling=True)
    normalized = [
        normalize_grpo_metrics(backend="verl", step=r.step, native=r.data, features=GRPOObservationFeatures()).metrics
        for r in records
    ]
    assert [m["train/optimizer_step_skipped"] for m in normalized] == [0.0, 0.5, 0.0]
    assert [m["train/optimizer_steps_skipped"] for m in normalized] == [0.0, 1.0, 1.0]
    assert [m["train/loss_scale"] for m in normalized] == [65536.0, 65536.0, 32768.0]
    last = normalized[-1]
    assert last["train/rl/sampling_logp_delta_mean"] == 0.002
    assert last["train/rl/sampling_logp_delta_p99"] == 0.02
    assert last["train/rl/sampling_logp_delta_max"] == 0.1
    assert last["train/rl/sampling_sequence_logp_delta_abs_mean"] == 0.05
    assert last["train/rl/sampling_prob_delta_mean"] == 0.001


def test_fp16_initial_loss_scale_reaches_verls_sharded_grad_scaler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """veRL builds ShardedGradScaler(growth_interval=400) in its Ray actors; the setup hook starts it at 1024."""

    overrides = _overrides(tmp_path, monkeypatch, options={"training_precision": "fp16"}, engine={"dtype": "float16"})
    assert "++ray_kwargs.ray_init.runtime_env.env_vars.POSTTRAIN_FP16_INITIAL_LOSS_SCALE='1024'" in overrides
    assert (
        "++ray_kwargs.ray_init.runtime_env.worker_process_setup_hook="
        "posttrain.train.backends.verl.loss_scale_hook.configure_initial_loss_scale"
    ) in overrides
    selected = _overrides(
        tmp_path,
        monkeypatch,
        options={"training_precision": "fp16", "fp16_initial_loss_scale": 256},
        engine={"dtype": "float16"},
    )
    assert "++ray_kwargs.ray_init.runtime_env.env_vars.POSTTRAIN_FP16_INITIAL_LOSS_SCALE='256'" in selected
    assert not any("ray_kwargs" in value for value in _overrides(tmp_path, monkeypatch))

    # What the hook does in a Ray worker process, with veRL's own constructor call.
    sharded = pytest.importorskip("torch.distributed.fsdp.sharded_grad_scaler")
    from posttrain.train.backends.verl.loss_scale_hook import configure_initial_loss_scale

    monkeypatch.setattr(sharded.ShardedGradScaler, "__init__", sharded.ShardedGradScaler.__init__)
    monkeypatch.setenv("POSTTRAIN_FP16_INITIAL_LOSS_SCALE", "256")
    configure_initial_loss_scale()
    configure_initial_loss_scale()  # idempotent: wraps the original constructor once
    scaler = sharded.ShardedGradScaler(device="cpu", growth_interval=400)
    assert scaler.get_scale() == 256.0 and scaler._growth_interval == 400
    assert sharded.ShardedGradScaler(device="cpu", init_scale=8.0).get_scale() == 8.0
