"""Read a checkpoint's architecture from the Hugging Face Hub at a pinned revision.

Plain HTTP (``config.json`` plus the model API's safetensors parameter count),
cached in memory, so it works in read-only containers such as Observatory's.
``HF_ENDPOINT`` and ``HF_TOKEN`` are honoured when set.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

import httpx

from .calculator import Architecture

_FAILURE_TTL_SECONDS = 300.0


class HubModelReader:
    """Callable ``(repo_id, revision) -> Architecture`` with a process-lifetime cache."""

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        token: str | None = None,
        timeout: float = 10.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._endpoint = (endpoint or os.getenv("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")
        token = token if token is not None else os.getenv("HF_TOKEN")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._client = client or httpx.Client(timeout=timeout, headers=headers, follow_redirects=True)
        self._cache: dict[tuple[str, str], Architecture] = {}
        self._failures: dict[tuple[str, str], tuple[float, Exception]] = {}
        self._lock = threading.Lock()

    def __call__(self, repo_id: str, revision: str | None) -> Architecture:
        key = (repo_id, revision or "main")
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            failed = self._failures.get(key)
            if failed is not None and time.monotonic() - failed[0] < _FAILURE_TTL_SECONDS:
                raise failed[1]
        try:
            architecture = self._read(*key)
        except Exception as error:
            with self._lock:
                self._failures[key] = (time.monotonic(), error)
            raise
        with self._lock:
            self._cache[key] = architecture
        return architecture

    def _read(self, repo_id: str, revision: str) -> Architecture:
        config = self._json(f"{self._endpoint}/{repo_id}/resolve/{revision}/config.json")
        info = self._json(f"{self._endpoint}/api/models/{repo_id}/revision/{revision}")
        safetensors = info.get("safetensors") if isinstance(info, dict) else None
        parameters = safetensors.get("total") if isinstance(safetensors, dict) else None
        if not isinstance(parameters, int) or parameters <= 0:
            raise LookupError(f"{repo_id}@{revision} does not publish a safetensors parameter count")
        return Architecture.from_config(config, parameters)

    def _json(self, url: str) -> Any:
        response = self._client.get(url)
        response.raise_for_status()
        return response.json()


__all__ = ["HubModelReader"]
