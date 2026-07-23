"""Isolated LLM Compressor adapter for materialized quantized variants."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from posttrain.common import RunContext

from ...transform import TransformRequest

_PYTHON_ENV = "POSTTRAIN_QUANTIZATION_PYTHON"


def run_llm_compressor(context: RunContext, request: TransformRequest, output_dir: Path) -> Path:
    """Run supported PTQ methods in a separately locked interpreter."""

    if request.plan.method not in {"awq", "rtn"}:
        raise ValueError(f"the LLM Compressor adapter cannot execute {request.plan.method!r}")
    calibration = request.plan.calibration
    if request.plan.method == "awq" and calibration is None:
        raise ValueError("AWQ requires an explicit calibration selection")
    if request.model.form == "weight-quantized" or request.model.quantization:
        raise ValueError("refusing to quantize an already quantized model variant")
    executable_value = os.environ.get(_PYTHON_ENV)
    if not executable_value:
        raise RuntimeError(
            f"quantization requires an isolated LLM Compressor interpreter; set {_PYTHON_ENV} to its Python executable"
        )
    # Do not resolve the venv Python symlink: its path is what activates the
    # environment's site-packages when the interpreter starts.
    executable = Path(os.path.abspath(Path(executable_value).expanduser()))
    if not executable.is_file():
        raise FileNotFoundError(f"isolated quantization Python does not exist: {executable}")
    worker = Path(__file__).with_name("worker.py")
    scheme = request.plan.output_quantization.get("scheme", "W4A16")
    if not isinstance(scheme, str):
        raise ValueError("output quantization scheme must be a string")
    payload = {
        "method": request.plan.method,
        "model_id": request.model.base.repo_id,
        "model_revision": request.model.base.revision,
        "scheme": scheme,
        "expected_bits": request.plan.output_quantization.get("bits"),
        "expected_group_size": request.plan.output_quantization.get("group_size"),
        "expected_zero_point": request.plan.output_quantization.get("zero_point"),
        "ignored_modules": ["lm_head", *request.plan.excluded_modules],
        "output_dir": str(output_dir),
    }
    if calibration is not None:
        payload.update(
            {
                "dataset_id": calibration.dataset_id,
                "dataset_revision": calibration.dataset_revision,
                "dataset_config": calibration.dataset_config,
                "split": calibration.split,
                "text_column": calibration.text_column,
                "sample_count": calibration.sample_count,
                "sequence_length": calibration.sequence_length,
                "batch_size": calibration.batch_size,
            }
        )
    payload_path = context.workspace / "quantization-request.json"
    payload_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    started = time.perf_counter()
    completed = subprocess.run(
        (str(executable), str(worker), str(payload_path)),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log = context.workspace / "quantization-worker.log"
    log.write_text(completed.stdout, encoding="utf-8")
    context.event(
        "quantization_worker_completed",
        {"returncode": completed.returncode, "runtime_seconds": time.perf_counter() - started},
    )
    if completed.returncode:
        raise RuntimeError(
            f"{request.plan.method.upper()} worker exited with code {completed.returncode}:\n{completed.stdout[-16_000:]}"
        )
    if not output_dir.is_dir() or not any(output_dir.iterdir()):
        raise RuntimeError("quantization worker completed without materializing model weights")
    summary_path = output_dir / "posttrain-quantization-summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics = summary.get("metrics", {})
        if isinstance(metrics, dict):
            context.metrics(
                {f"transform/{name}": float(value) for name, value in metrics.items() if isinstance(value, int | float)}
            )
    return output_dir


__all__ = ["run_llm_compressor"]
