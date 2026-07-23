"""Tests for the isolated AWQ backend boundary."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from posttrain.common import ExecutionTarget, NullObserver, RunContext
from posttrain.common.variants import QWEN_35_2B
from posttrain.train import CalibrationSelection, QuantizationPlan, TransformRequest
from posttrain.train.backends.quantization import run_llm_compressor

LOCAL_CUDA_8GB = ExecutionTarget("targets/local-cuda-8gb", "1", "nvidia-cuda", 8, {"world_size": 1})


def test_awq_backend_uses_explicit_isolated_interpreter_and_serialized_request(
    monkeypatch,
    tmp_path: Path,
) -> None:
    executable = tmp_path / "awq-python"
    executable.symlink_to(Path(sys.executable))
    monkeypatch.setenv("POSTTRAIN_QUANTIZATION_PYTHON", str(executable))
    observed: dict[str, object] = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        payload = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
        observed["payload"] = payload
        output = Path(payload["output_dir"])
        output.mkdir(parents=True)
        (output / "model.safetensors").write_bytes(b"awq")
        (output / "posttrain-quantization-summary.json").write_text(
            json.dumps(
                {
                    "metrics": {"runtime_seconds": 1.5, "peak_gpu_memory_gib": 2.0},
                    "runtime_versions": {"llmcompressor": "test"},
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="completed\n")

    monkeypatch.setattr("posttrain.train.backends.quantization.llm_compressor.subprocess.run", fake_run)
    context = RunContext(
        project_id="tests",
        work_package_id="train/awq",
        run_id="run-awq",
        job_kind="model.transform",
        job_definition_version="model/llm-compressor-awq@1",
        workspace=tmp_path,
        observer=NullObserver(),
    )
    plan = QuantizationPlan(
        "qwen3.5-2b/awq-test",
        "1",
        "awq",
        "awq-test",
        "a" * 64,
        "int4-group128",
        backend="llmcompressor@test",
        calibration=CalibrationSelection(
            "openai/gsm8k",
            "main",
            4,
            128,
            dataset_config="main",
            text_column="question",
        ),
        output_quantization={"scheme": "W4A16"},
    )
    request = TransformRequest(QWEN_35_2B, plan, LOCAL_CUDA_8GB, "models/qwen@awq-test")

    result = run_llm_compressor(context, request, tmp_path / "output")

    assert result == tmp_path / "output"
    command = observed["command"]
    assert isinstance(command, tuple)
    assert command == (str(executable), command[1], str(tmp_path / "quantization-request.json"))
    payload = observed["payload"]
    assert isinstance(payload, dict)
    assert payload["model_revision"] == QWEN_35_2B.base.revision
    assert payload["dataset_id"] == "openai/gsm8k"
    assert payload["scheme"] == "W4A16"
    assert payload["method"] == "awq"


def test_rtn_backend_does_not_require_calibration(monkeypatch, tmp_path: Path) -> None:
    executable = tmp_path / "quantization-python"
    executable.symlink_to(Path(sys.executable))
    monkeypatch.setenv("POSTTRAIN_QUANTIZATION_PYTHON", str(executable))
    observed: dict[str, object] = {}

    def fake_run(command, **kwargs):
        del kwargs
        payload = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
        observed.update(payload)
        output = Path(payload["output_dir"])
        output.mkdir(parents=True)
        (output / "model.safetensors").write_bytes(b"rtn")
        return SimpleNamespace(returncode=0, stdout="completed\n")

    monkeypatch.setattr("posttrain.train.backends.quantization.llm_compressor.subprocess.run", fake_run)
    context = RunContext(
        project_id="tests",
        work_package_id="train/rtn",
        run_id="run-rtn",
        job_kind="model.transform",
        job_definition_version="model/llm-compressor@1",
        workspace=tmp_path,
        observer=NullObserver(),
    )
    plan = QuantizationPlan(
        "qwen3.5-2b/rtn-test",
        "1",
        "rtn",
        "rtn-test",
        "b" * 64,
        "int4-group128",
        backend="llmcompressor@test",
        output_quantization={"scheme": "W4A16"},
    )

    run_llm_compressor(
        context,
        TransformRequest(QWEN_35_2B, plan, LOCAL_CUDA_8GB, "models/qwen@rtn-test"),
        tmp_path / "output",
    )

    assert observed["method"] == "rtn"
    assert "dataset_id" not in observed


def test_awq_backend_requires_configured_isolated_interpreter(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("POSTTRAIN_QUANTIZATION_PYTHON", raising=False)
    context = RunContext(
        project_id="tests",
        work_package_id="train/awq",
        run_id="run-awq",
        job_kind="model.transform",
        job_definition_version="model/llm-compressor-awq@1",
        workspace=tmp_path,
        observer=NullObserver(),
    )
    plan = QuantizationPlan(
        "qwen3.5-2b/awq-test",
        "1",
        "awq",
        "awq-test",
        "a" * 64,
        "int4-group128",
        backend="llmcompressor@test",
        calibration=CalibrationSelection("openai/gsm8k", "main", 1, 128),
    )

    with pytest.raises(RuntimeError, match="POSTTRAIN_QUANTIZATION_PYTHON"):
        run_llm_compressor(
            context,
            TransformRequest(QWEN_35_2B, plan, LOCAL_CUDA_8GB, "models/qwen@awq-test"),
            tmp_path / "output",
        )
