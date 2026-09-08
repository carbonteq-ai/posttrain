"""Run-scoped request admission in front of TRL's native async vLLM server."""

from __future__ import annotations

import asyncio
import secrets
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from ...online_rl import BehaviorPolicySpan
from ...rollout_execution import CollectionExecutionError, EpisodeKey


@dataclass(frozen=True, slots=True)
class _ActiveRequest:
    task: asyncio.Task[Any]
    session_id: str


class TrlAsyncPolicyGateway:
    """Gate Verifiers model turns around native async-TRL weight publication.

    The gateway forwards the native token endpoint without interpreting
    prompts, rewards, or algorithms. A tool-running episode remains alive while
    admission is closed; its next HTTP turn waits until the new policy is live.
    """

    def __init__(
        self,
        *,
        upstream_base_url: str,
        host: str = "127.0.0.1",
        port: int = 0,
        request_timeout_s: float = 300.0,
        max_request_bytes: int = 8 * 1024 * 1024,
    ) -> None:
        parsed = urlparse(upstream_base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("async policy upstream must be an absolute HTTP URL")
        if host not in {"127.0.0.1", "::1", "localhost"}:
            raise ValueError("async policy gateway must bind to loopback")
        if not 0 <= port <= 65535:
            raise ValueError("async policy gateway port is invalid")
        if request_timeout_s <= 0 or max_request_bytes < 1:
            raise ValueError("async policy gateway bounds must be positive")
        self._upstream_root = upstream_base_url.rstrip("/").removesuffix("/v1")
        self._host = host
        self._port = port
        self._request_timeout_s = request_timeout_s
        self._max_request_bytes = max_request_bytes
        self._runner: Any | None = None
        self._client: Any | None = None
        self._base_url: str | None = None
        self._active_version = 0
        self._pending_version: int | None = None
        self._admission_open = False
        self._requests: dict[str, _ActiveRequest] = {}
        self._session_spans: dict[str, BehaviorPolicySpan] = {}
        self._fatal_error: BaseException | None = None
        self._condition = asyncio.Condition()

    @property
    def base_url(self) -> str:
        if self._base_url is None:
            raise RuntimeError("async policy gateway has not started")
        return self._base_url

    @property
    def fatal_error(self) -> BaseException | None:
        return self._fatal_error

    async def start(self, initial_model_version: int) -> None:
        if initial_model_version < 0:
            raise ValueError("initial async policy version cannot be negative")
        if self._runner is not None:
            raise RuntimeError("async policy gateway is already started")
        from aiohttp import ClientSession, ClientTimeout, web

        self._active_version = initial_model_version
        self._pending_version = None
        self._fatal_error = None
        self._admission_open = True
        client = ClientSession(timeout=ClientTimeout(total=self._request_timeout_s))
        app = web.Application(client_max_size=self._max_request_bytes)
        app.router.add_get("/v1/models", self._handle_models)
        app.router.add_post("/inference/v1/generate", self._handle_generate)
        app.router.add_post("/inference/v1/abort", self._handle_abort)
        runner = web.AppRunner(app, access_log=None)
        try:
            await runner.setup()
            site = web.TCPSite(runner, self._host, self._port)
            await site.start()
            server = getattr(site, "_server", None)
            sockets = tuple(getattr(server, "sockets", ()) or ())
            if len(sockets) != 1:
                raise RuntimeError("async policy gateway did not bind exactly one socket")
        except BaseException:
            try:
                await runner.cleanup()
            finally:
                await client.close()
            raise
        bound_port = int(sockets[0].getsockname()[1])
        url_host = "[::1]" if self._host == "::1" else self._host
        self._client = client
        self._runner = runner
        self._base_url = f"http://{url_host}:{bound_port}/v1"

    async def prepare_model_update(self, model_version: int) -> None:
        async with self._condition:
            self._raise_if_fatal()
            if model_version <= self._active_version or self._pending_version is not None:
                raise ValueError("next async policy version must exceed the active version")
            self._pending_version = model_version
            self._admission_open = False
            while self._requests:
                await self._condition.wait()
                self._raise_if_fatal()

    async def activate_model_version(self, model_version: int) -> None:
        async with self._condition:
            self._raise_if_fatal()
            if self._pending_version != model_version or self._requests:
                raise RuntimeError("async policy activation does not match a completed request drain")
            self._active_version = model_version
            self._pending_version = None
            self._admission_open = True
            self._condition.notify_all()

    async def behavior_policy_for_episode(self, key: EpisodeKey, episode: Any) -> BehaviorPolicySpan:
        traces = [trace for trace in episode.traces if trace.agent.trainable]
        if len(traces) != 1:
            async with self._condition:
                for trace in episode.traces:
                    self._session_spans.pop(str(trace.id), None)
            raise CollectionExecutionError("cannot resolve policy span without one trainable native trace")
        session_id = str(traces[0].id)
        async with self._condition:
            span = self._session_spans.pop(session_id, None)
        if span is None:
            raise CollectionExecutionError("native episode has no served policy-version evidence")
        try:
            dispatched_version = int(key.collection.policy_version)
        except ValueError as error:
            raise CollectionExecutionError("async collection policy version is not numeric") from error
        if span.start != dispatched_version:
            raise CollectionExecutionError("native episode started on a different policy than its collection")
        return span

    async def aclose(self) -> None:
        runner, client = self._runner, self._client
        if runner is None:
            return
        async with self._condition:
            self._admission_open = False
            active = tuple(self._requests.items())
        cleanup_errors: list[BaseException] = []
        for request_id, registration in active:
            try:
                if not await self._abort_upstream(request_id):
                    raise CollectionExecutionError(
                        f"active async policy request {request_id} did not acknowledge abort"
                    )
            except BaseException as error:
                cleanup_errors.append(error)
            registration.task.cancel()
        if active:
            await asyncio.gather(*(item.task for _, item in active), return_exceptions=True)
        try:
            await runner.cleanup()
        except BaseException as error:
            cleanup_errors.append(error)
        try:
            if client is not None:
                await client.close()
        except BaseException as error:
            cleanup_errors.append(error)
        finally:
            async with self._condition:
                self._runner = None
                self._client = None
                self._base_url = None
                self._requests.clear()
                self._session_spans.clear()
                self._condition.notify_all()
        if cleanup_errors:
            raise CollectionExecutionError(
                "async policy gateway shutdown could not prove request drainage: "
                + "; ".join(f"{type(error).__name__}: {error}" for error in cleanup_errors)
            ) from cleanup_errors[0]

    async def _handle_models(self, request: Any) -> Any:
        return await self._forward(request, "/v1/models")

    async def _handle_generate(self, request: Any) -> Any:
        from aiohttp import web

        try:
            payload = await request.json()
        except ValueError as error:
            return web.json_response({"error": str(error)}, status=400)
        if not isinstance(payload, dict):
            return web.json_response({"error": "generate request body must be an object"}, status=400)
        request_id = payload.get("request_id") or secrets.token_hex(16)
        session_id = request.headers.get("X-Session-ID")
        if not isinstance(request_id, str) or not request_id.strip():
            return web.json_response({"error": "request_id must be a non-empty string"}, status=400)
        if not session_id:
            return web.json_response({"error": "X-Session-ID is required for policy provenance"}, status=400)
        payload["request_id"] = request_id
        current = asyncio.current_task()
        assert current is not None
        async with self._condition:
            while not self._admission_open:
                self._raise_if_fatal()
                await self._condition.wait()
            self._raise_if_fatal()
            if request_id in self._requests:
                return web.json_response({"error": "duplicate active request_id"}, status=409)
            version = self._active_version
            self._requests[request_id] = _ActiveRequest(current, session_id)
            observed = BehaviorPolicySpan(version, version)
            prior = self._session_spans.get(session_id)
            self._session_spans[session_id] = observed if prior is None else prior.merge(observed)
        try:
            return await self._forward(request, "/inference/v1/generate", payload=payload)
        except asyncio.CancelledError:
            await self._abort_upstream(request_id)
            raise
        finally:
            async with self._condition:
                self._requests.pop(request_id, None)
                self._condition.notify_all()

    async def _handle_abort(self, request: Any) -> Any:
        from aiohttp import web

        try:
            payload = await request.json()
        except ValueError as error:
            return web.json_response({"error": str(error)}, status=400)
        request_id = payload.get("request_id") if isinstance(payload, dict) else None
        if not isinstance(request_id, str) or not request_id.strip():
            return web.json_response({"error": "abort request_id must be a non-empty string"}, status=400)
        cancelled = await self._abort_upstream(request_id)
        async with self._condition:
            registration = self._requests.get(request_id)
        if cancelled and registration is not None:
            registration.task.cancel()
        return web.json_response({"cancelled": cancelled})

    async def _forward(self, request: Any, path: str, *, payload: Any | None = None) -> Any:
        from aiohttp import web

        client = self._client
        if client is None:
            raise RuntimeError("async policy gateway has not started")
        headers = {
            name: request.headers[name]
            for name in ("Authorization", "X-Session-ID")
            if name in request.headers
        }
        try:
            async with client.request(
                request.method,
                f"{self._upstream_root}{path}",
                json=payload,
                headers=headers,
            ) as response:
                body = await response.read()
                if response.status >= 500:
                    error = CollectionExecutionError(
                        f"async policy upstream returned HTTP {response.status}"
                    )
                    self._record_fatal(error)
                return web.Response(
                    body=body,
                    status=response.status,
                    content_type=response.content_type,
                )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            failure = CollectionExecutionError(
                f"async policy upstream request failed: {type(error).__name__}: {error}"
            )
            self._record_fatal(failure)
            raise failure from error

    async def _abort_upstream(self, request_id: str) -> bool:
        client = self._client
        if client is None:
            return False
        try:
            async with client.post(
                f"{self._upstream_root}/inference/v1/abort",
                json={"request_id": request_id},
            ) as response:
                if response.status == 404:
                    return False
                response.raise_for_status()
                payload = await response.json()
                return bool(payload.get("cancelled"))
        except Exception as error:
            failure = CollectionExecutionError(
                f"async policy upstream abort failed: {type(error).__name__}: {error}"
            )
            self._record_fatal(failure)
            raise failure from error

    def _record_fatal(self, error: BaseException) -> None:
        if self._fatal_error is None:
            self._fatal_error = error

    def _raise_if_fatal(self) -> None:
        if self._fatal_error is not None:
            raise CollectionExecutionError(f"async policy gateway failed: {self._fatal_error}") from self._fatal_error


__all__ = ["TrlAsyncPolicyGateway"]
