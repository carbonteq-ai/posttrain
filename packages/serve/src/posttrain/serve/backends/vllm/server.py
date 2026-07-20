"""Managed vLLM OpenAI server process."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import IO, Any, cast

import httpx

from ...cuda import TorchModule, resolve_cuda_home
from ...online import Endpoint, LaunchRequest


def build_vllm_command(request: LaunchRequest, chat_template_path: Path | None = None) -> tuple[str, ...]:
    executable = Path(sys.executable).with_name("vllm")
    values = [
        str(executable),
        "serve",
        request.model.artifact.repo_id,
        "--revision",
        request.model.artifact.revision,
        "--served-model-name",
        request.model.artifact.repo_id,
        "--host",
        request.host,
        "--port",
        str(request.port),
        *request.profile.engine.as_cli_args(request.model),
        *request.profile.frontend_args(),
    ]
    if chat_template_path is not None:
        values.extend(("--chat-template", str(chat_template_path)))
    return tuple(values)


def _server_environment() -> dict[str, str]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is not installed; install posttrain-serve[vllm]") from error
    cuda_home = resolve_cuda_home(cast(TorchModule, torch))
    environment = dict(os.environ)
    environment["CUDA_HOME"] = str(cuda_home)
    toolkit_bin = str(cuda_home / "bin")
    path_entries = [entry for entry in environment.get("PATH", "").split(":") if entry]
    if toolkit_bin not in path_entries:
        environment["PATH"] = ":".join([toolkit_bin, *path_entries])
    return environment


class VllmServer:
    def __init__(
        self,
        request: LaunchRequest,
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
