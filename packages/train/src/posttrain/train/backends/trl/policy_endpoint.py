"""Loopback token-generation endpoint for native Verifiers training clients."""

from __future__ import annotations

import asyncio
import importlib
import math
import secrets
from dataclasses import dataclass
from typing import Any, Protocol

from ...rollout_execution import CollectionKey


class _AsyncPolicySession(Protocol):
    async def open_policy(self, version: str) -> None: ...

    async def generate(self, request: Any) -> Any: ...

    async def abort(self, request_id: str) -> bool: ...

    async def stop_admission(self) -> None: ...


class _AdmissionClosed(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class _RequestRegistration:
    collection: CollectionKey
    session_id: str | None
    task: asyncio.Task[Any]


@dataclass(frozen=True, slots=True)
class _SessionGenerationRequest:
    request_id: str
    prompt_token_ids: tuple[int, ...]
    sampling_params: Any


class TrlPolicyEndpoint:
    """Own one loopback HTTP surface over an already-loaded async TRL session.

    The endpoint neither constructs model weights nor owns optimizer/reward
    behavior. It validates vLLM's token-in/token-out request schema and fences
    every admitted request to one fixed-policy collection.
    """

    def __init__(
        self,
        *,
        model_name: str,
        max_model_len: int,
        host: str = "127.0.0.1",
        port: int = 0,
        max_request_bytes: int = 8 * 1024 * 1024,
    ) -> None:
        if not model_name.strip():
            raise ValueError("policy endpoint model name cannot be empty")
        if max_model_len < 1:
            raise ValueError("policy endpoint max_model_len must be positive")
        if host not in {"127.0.0.1", "::1", "localhost"}:
            raise ValueError("TRL policy endpoint must bind to loopback")
        if port < 0 or port > 65535:
            raise ValueError("policy endpoint port is invalid")
        if max_request_bytes < 1:
            raise ValueError("policy endpoint request limit must be positive")
        self._model_name = model_name
        self._max_model_len = max_model_len
        self._host = host
        self._port = port
        self._max_request_bytes = max_request_bytes
        self._session: _AsyncPolicySession | None = None
        self._collection: CollectionKey | None = None
        self._requests: dict[str, _RequestRegistration] = {}
        self._runner: Any | None = None
        self._site: Any | None = None
        self._base_url: str | None = None
        self._fatal_error: BaseException | None = None
        self._lock = asyncio.Lock()

    @property
    def base_url(self) -> str:
        if self._base_url is None:
            raise RuntimeError("TRL policy endpoint has not started")
        return self._base_url

    @property
    def active_request_ids(self) -> frozenset[str]:
        return frozenset(self._requests)

    @property
    def fatal_error(self) -> BaseException | None:
        """First engine/response failure that makes the collection unusable."""
        return self._fatal_error

    async def start(self, session: _AsyncPolicySession) -> None:
        """Start the loopback server without opening collection admission."""
        async with self._lock:
            if self._runner is not None:
                raise RuntimeError("TRL policy endpoint is already started")
            self._session = session
        try:
            from aiohttp import web
        except ImportError as error:
            raise RuntimeError("install posttrain-train with the trl-vllm extra") from error

        app = web.Application(client_max_size=self._max_request_bytes)
        app.router.add_get("/v1/models", self._handle_models)
        app.router.add_post("/inference/v1/generate", self._handle_generate)
        app.router.add_post("/inference/v1/abort", self._handle_abort)
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, self._host, self._port)
        try:
            await site.start()
        except BaseException:
            await runner.cleanup()
            async with self._lock:
                self._session = None
            raise
        server = getattr(site, "_server", None)
        sockets = () if server is None else tuple(server.sockets or ())
        if len(sockets) != 1:
            await runner.cleanup()
            async with self._lock:
                self._session = None
            raise RuntimeError("TRL policy endpoint did not bind exactly one loopback socket")
        bound_port = int(sockets[0].getsockname()[1])
        url_host = "[::1]" if self._host == "::1" else self._host
        async with self._lock:
            self._runner = runner
            self._site = site
            self._base_url = f"http://{url_host}:{bound_port}/v1"

    async def open_admission(self, collection: CollectionKey) -> None:
        """Fence new requests to ``collection`` after its policy is ready."""
        async with self._lock:
            session = self._require_started()
            if self._collection is not None:
                raise RuntimeError("TRL policy endpoint already has an open collection")
            if self._requests:
                raise RuntimeError("cannot open a collection while prior requests remain active")
        await session.open_policy(collection.policy_version)
        async with self._lock:
            if self._session is not session or self._runner is None:
                raise RuntimeError("TRL policy endpoint closed while opening admission")
            if self._collection is not None:
                raise RuntimeError("TRL policy endpoint collection changed while opening admission")
            self._fatal_error = None
            self._collection = collection

    async def stop_admission(self, collection: CollectionKey) -> None:
        """Stop new requests for the exact active collection."""
        async with self._lock:
            session = self._require_started()
            if self._collection != collection:
                raise RuntimeError("TRL policy endpoint collection identity does not match")
            self._collection = None
        await session.stop_admission()

    async def abort(self, request_id: str) -> bool:
        """Abort one endpoint request and acknowledge its terminalization."""
        async with self._lock:
            session = self._require_started()
            registration = self._requests.get(request_id)
        if registration is None:
            return False
        cancelled = await session.abort(request_id)
        if cancelled and not registration.task.done():
            registration.task.cancel()
        return cancelled

    async def aclose(self) -> None:
        """Fence admission, abort active requests, and stop the HTTP server."""
        async with self._lock:
            if self._runner is None:
                return
            session = self._require_started()
            collection = self._collection
            self._collection = None
            request_ids = tuple(self._requests)
            runner = self._runner
        try:
            if collection is not None:
                await session.stop_admission()
            for request_id in request_ids:
                await self.abort(request_id)
            async with self._lock:
                active = tuple(registration.task for registration in self._requests.values())
            if active:
                await asyncio.gather(*active, return_exceptions=True)
        finally:
            await runner.cleanup()
            async with self._lock:
                self._runner = None
                self._site = None
                self._base_url = None
                self._session = None

    async def _handle_models(self, _request: Any) -> Any:
        from aiohttp import web

        return web.json_response(
            {
                "object": "list",
                "data": [
                    {
                        "id": self._model_name,
                        "object": "model",
                        "owned_by": "posttrain",
                        "max_model_len": self._max_model_len,
                    }
                ],
            }
        )

    async def _handle_generate(self, request: Any) -> Any:
        from aiohttp import web

        try:
            payload = await request.json()
            request_id, native_request = self._validate_generate_payload(payload)
            task = asyncio.current_task()
            if task is None:  # pragma: no cover - aiohttp handlers always have a task
                raise RuntimeError("policy request is not running in an asyncio task")
            async with self._lock:
                session = self._require_started()
                collection = self._collection
                if collection is None:
                    raise _AdmissionClosed("TRL policy endpoint is not admitting a collection")
                if request_id in self._requests:
                    raise ValueError(f"duplicate active policy request id {request_id!r}")
                self._requests[request_id] = _RequestRegistration(
                    collection=collection,
                    session_id=request.headers.get("X-Session-ID"),
                    task=task,
                )
            try:
                output = await session.generate(native_request)
                return web.json_response(
                    self._response_payload(request_id, native_request.prompt_token_ids, output)
                )
            finally:
                async with self._lock:
                    self._requests.pop(request_id, None)
        except asyncio.CancelledError:
            raise
        except ValueError as error:
            return web.json_response({"error": str(error)}, status=400)
        except _AdmissionClosed as error:
            return web.json_response({"error": str(error)}, status=409)
        except Exception as error:
            async with self._lock:
                if self._fatal_error is None:
                    self._fatal_error = error
            return web.json_response(
                {"error": f"policy engine request failed: {type(error).__name__}"},
                status=500,
            )

    async def _handle_abort(self, request: Any) -> Any:
        from aiohttp import web

        try:
            payload = await request.json()
        except ValueError as error:
            return web.json_response({"error": str(error)}, status=400)
        request_id = payload.get("request_id") if isinstance(payload, dict) else None
        if not isinstance(request_id, str) or not request_id.strip():
            return web.json_response({"error": "abort request_id must be a non-empty string"}, status=400)
        return web.json_response({"cancelled": await self.abort(request_id)})

    def _validate_generate_payload(self, payload: Any) -> tuple[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("generate request body must be an object")
        supported = {"request_id", "token_ids", "sampling_params", "model", "stream", "cache_salt", "priority"}
        unsupported = set(payload).difference(supported)
        if unsupported:
            raise ValueError(f"TRL policy endpoint does not support: {', '.join(sorted(unsupported))}")
        protocol = importlib.import_module("vllm.entrypoints.scale_out.token_in_token_out.protocol")
        try:
            parsed = protocol.GenerateRequest.model_validate(payload)
        except Exception as error:
            raise ValueError(f"invalid token generation request: {error}") from error
        if parsed.stream:
            raise ValueError("TRL policy endpoint does not support streaming responses")
        if parsed.cache_salt is not None:
            raise ValueError("TRL policy endpoint does not support cache_salt")
        if parsed.priority != 0:
            raise ValueError("TRL policy endpoint does not support priority scheduling")
        if parsed.model is not None and parsed.model != self._model_name:
            raise ValueError("generate request model does not match the loaded policy")
        if parsed.sampling_params.n != 1:
            raise ValueError("TRL policy endpoint requires exactly one completion per request")
        if parsed.sampling_params.logprobs is None:
            raise ValueError("TRL policy endpoint requires sampled-token logprobs")
        max_tokens = parsed.sampling_params.max_tokens
        if max_tokens is None or max_tokens < 1:
            raise ValueError("TRL policy endpoint requires a positive max_tokens value")
        if len(parsed.token_ids) + max_tokens > self._max_model_len:
            raise ValueError(
                "token generation request exceeds the loaded policy context: "
                f"{len(parsed.token_ids)} prompt + {max_tokens} completion > {self._max_model_len}"
            )
        request_id = parsed.request_id or secrets.token_hex(16)
        native_request = _SessionGenerationRequest(
            request_id=request_id,
            prompt_token_ids=tuple(parsed.token_ids),
            sampling_params=parsed.sampling_params,
        )
        return request_id, native_request

    def _response_payload(
        self,
        request_id: str,
        expected_prompt_ids: tuple[int, ...],
        output: Any,
    ) -> dict[str, Any]:
        if not getattr(output, "finished", False):
            raise RuntimeError("policy engine returned a nonterminal output")
        if getattr(output, "request_id", request_id) != request_id:
            raise RuntimeError("policy engine response request identity does not match")
        choices = getattr(output, "outputs", None)
        if not isinstance(choices, list) or len(choices) != 1:
            raise RuntimeError("policy engine must return exactly one completion")
        completion = choices[0]
        token_ids = [int(value) for value in completion.token_ids]
        logprobs = completion.logprobs
        if not token_ids or logprobs is None or len(logprobs) != len(token_ids):
            raise RuntimeError("policy engine returned incomplete sampled-token evidence")
        content = []
        for index, (token_id, step) in enumerate(zip(token_ids, logprobs, strict=True)):
            selected = step.get(token_id) if step is not None else None
            value = getattr(selected, "logprob", None)
            if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
                raise RuntimeError(f"policy engine logprob {index} is missing or non-finite")
            content.append(
                {
                    "token": f"token_id:{token_id}",
                    "logprob": max(float(value), -9999.0),
                    "top_logprobs": [],
                }
            )
        prompt_ids = [int(value) for value in (getattr(output, "prompt_token_ids", None) or [])]
        if prompt_ids != list(expected_prompt_ids):
            raise RuntimeError("policy engine response prompt token identity does not match")
        return {
            "request_id": request_id,
            "model": self._model_name,
            "prompt_token_ids": prompt_ids,
            "choices": [
                {
                    "index": int(completion.index),
                    "token_ids": token_ids,
                    "finish_reason": completion.finish_reason or "stop",
                    "logprobs": {"content": content},
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(token_ids),
                "total_tokens": len(prompt_ids) + len(token_ids),
            },
        }

    def _require_started(self) -> _AsyncPolicySession:
        if self._runner is None or self._session is None:
            raise RuntimeError("TRL policy endpoint has not started")
        return self._session


__all__ = ["TrlPolicyEndpoint"]
