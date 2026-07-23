"""Managed vLLM OpenAI server process."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import IO, Any, cast

import httpx
from posttrain.common import HubModelRef, LocalArtifactRef
from posttrain.common.cuda import TorchModule, cuda_environment

from ...online import Endpoint, ServeLaunchRequest, served_model_name
from .bindings import engine_config, frontend_args


def build_vllm_command(request: ServeLaunchRequest, chat_template_path: Path | None = None) -> tuple[str, ...]:
    executable = Path(sys.executable).with_name("vllm")
    model = request.inference.model
    engine = engine_config(request.inference)
    artifact = model.artifact
    if model.form in {"adapter", "peft-adapter"}:
        if not isinstance(artifact, LocalArtifactRef):
            raise ValueError("vLLM requires the host to materialize a PEFT adapter before launch")
        source = model.base.repo_id
        revision_args = ("--revision", model.base.revision)
        adapter_args = ("--enable-lora", "--lora-modules", f"{served_model_name(model)}={artifact.path}")
        base_served_name = model.base.repo_id
    elif isinstance(artifact, HubModelRef):
        source = artifact.repo_id
        revision_args = ("--revision", artifact.revision)
        adapter_args = ()
        base_served_name = served_model_name(model)
    elif isinstance(artifact, LocalArtifactRef):
        source = str(artifact.path)
        revision_args = ()
        adapter_args = ()
        base_served_name = served_model_name(model)
    else:
        raise ValueError("vLLM requires Hub-hosted or host-materialized model weights")
    values = [
        str(executable),
        "serve",
        source,
        *revision_args,
        "--served-model-name",
        base_served_name,
        "--host",
        request.host,
        "--port",
        str(request.port),
        *engine.as_cli_args(),
        *adapter_args,
        *frontend_args(request.inference),
    ]
    if chat_template_path is not None:
        values.extend(("--chat-template", str(chat_template_path)))
    return tuple(values)


def _server_environment() -> dict[str, str]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is not installed; install posttrain-serve[vllm]") from error
    return cuda_environment(cast(TorchModule, torch), environ=os.environ)


class VllmServer:
    def __init__(
        self,
        request: ServeLaunchRequest,
        log_path: Path,
        chat_template_path: Path | None = None,
    ) -> None:
        self.request = request
        self.endpoint = request.endpoint
        self.log_path = log_path
        self.command = build_vllm_command(request, chat_template_path)
        self._process: subprocess.Popen[str] | None = None
        self._log: IO[str] | None = None

    def start(self) -> Endpoint:
        if self._process is not None:
            raise RuntimeError("vLLM server is already started")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("w", encoding="utf-8")
        self._process = subprocess.Popen(
            self.command,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            text=True,
            env=_server_environment(),
        )
        deadline = time.monotonic() + self.request.startup_timeout_seconds
        while time.monotonic() < deadline:
            returncode = self._process.poll()
            if returncode is not None:
                self._close_log()
                tail = self.log_path.read_text(encoding="utf-8", errors="replace")[-16_000:]
                raise RuntimeError(f"vLLM server exited with code {returncode}:\n{tail}")
            try:
                response = httpx.get(self.endpoint.health_url, timeout=1)
                if response.is_success:
                    return self.endpoint
            except httpx.HTTPError:
                pass
            time.sleep(0.25)
        self.close()
        tail = self.log_path.read_text(encoding="utf-8", errors="replace")[-16_000:]
        raise TimeoutError(f"vLLM server did not become healthy before timeout:\n{tail}")

    def close(self) -> None:
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        self._process = None
        self._close_log()

    def _close_log(self) -> None:
        if self._log is not None:
            self._log.close()
            self._log = None

    def __enter__(self) -> Endpoint:
        return self.start()

    def __exit__(self, *_: Any) -> None:
        self.close()


__all__ = ["VllmServer", "build_vllm_command"]
