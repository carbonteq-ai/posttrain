"""Run-local hard cost enforcement for paid OpenAI-compatible services."""

from __future__ import annotations

import json
import secrets
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx
from posttrain.common import JsonValue
from posttrain.serve import Endpoint

_USD_MICROS = Decimal(1_000_000)
_TOKEN_OVERHEAD = 4_096


class InferenceCostLimitError(RuntimeError):
    """A projected or reserved paid request would exceed the run ceiling."""


@dataclass(frozen=True, slots=True)
class TokenPrices:
    prompt: Decimal
    completion: Decimal

    def __post_init__(self) -> None:
        if not self.prompt.is_finite() or not self.completion.is_finite():
            raise ValueError("paid inference prices must be finite")
        if self.prompt < 0 or self.completion < 0:
            raise ValueError("paid inference prices cannot be negative")

    def cost(self, input_tokens: int, output_tokens: int) -> Decimal:
        return self.prompt * input_tokens + self.completion * output_tokens


class CostLedger:
    """Concurrency-safe reservations and reconciled usage for one run service."""

    def __init__(self, limit_usd_micros: int, prices: TokenPrices, *, initial_cost: Decimal = Decimal(0)) -> None:
        self.limit = Decimal(limit_usd_micros) / _USD_MICROS
        self.prices = prices
        self._spent = initial_cost
        self._reserved = Decimal(0)
        self._requests = 0
        self._lock = threading.Lock()
        if initial_cost > self.limit:
            raise InferenceCostLimitError("judge readiness cost already exceeds the selected run cost ceiling")

    def reserve(self, body: bytes, payload: Mapping[str, Any], maximum_output_tokens: int) -> Decimal:
        requested = payload.get("max_tokens", payload.get("max_completion_tokens"))
        if isinstance(requested, bool) or not isinstance(requested, int) or requested < 1:
            raise ValueError("paid inference requests must declare a positive output-token limit")
        if requested > maximum_output_tokens:
            raise ValueError("paid inference request exceeds the selected output-token limit")
        # BPE token counts cannot exceed the byte representation of their input.
        # The fixed allowance covers protocol-added role/control tokens.
        input_upper_bound = len(body) + _TOKEN_OVERHEAD
        reservation = self.prices.cost(input_upper_bound, requested)
        with self._lock:
            if self._spent + self._reserved + reservation > self.limit:
                raise InferenceCostLimitError(
                    "judge request rejected before dispatch because its reservation would exceed the run cost ceiling"
                )
            self._reserved += reservation
            self._requests += 1
        return reservation

    def settle(self, reservation: Decimal, usage: object) -> None:
        actual = _usage_cost(usage, self.prices)
        # Missing or malformed usage is never interpreted as free inference.
        charge = reservation if actual is None else actual
        if charge > reservation:
            charge = reservation
        with self._lock:
            self._reserved -= reservation
            self._spent += charge

    def snapshot(self) -> dict[str, JsonValue]:
        with self._lock:
            return {
                "limit_usd": _decimal_text(self.limit),
                "spent_usd": _decimal_text(self._spent),
                "reserved_usd": _decimal_text(self._reserved),
                "requests_dispatched": self._requests,
            }


def projected_cost(input_tokens: int, output_tokens: int, prices: TokenPrices) -> Decimal:
    return prices.cost(input_tokens, output_tokens)


def usage_cost(usage: object, prices: TokenPrices) -> Decimal:
    cost = _usage_cost(usage, prices)
    if cost is None:
        raise InferenceCostLimitError("paid judge response omitted valid token usage")
    return cost


@contextmanager
def metered_openai_gateway(
    *,
    upstream: httpx.Client,
    upstream_base_url: str,
    model: str,
    maximum_output_tokens: int,
    ledger: CostLedger,
    request_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> Iterator[Endpoint]:
    """Expose a loopback endpoint that reserves spend before forwarding calls."""

    local_key = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        server_version = "PosttrainCostGuard/1"

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if self.path.rstrip("/") != "/v1/chat/completions":
                self._json_error(404, "unsupported endpoint", "not_found")
                return
            if self.headers.get("Authorization") != f"Bearer {local_key}":
                self._json_error(401, "invalid local service credential", "unauthorized")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length)
                payload = json.loads(body)
                if not isinstance(payload, dict):
                    raise ValueError("request body must be an object")
                if payload.get("model") != model:
                    raise ValueError("request model differs from the selected hosted model")
                if payload.get("stream") is True:
                    raise ValueError("streaming paid judge calls are not supported by the run cost guard")
                reservation = ledger.reserve(body, payload, maximum_output_tokens)
            except InferenceCostLimitError as error:
                self._json_error(402, str(error), "posttrain_judge_cost_limit")
                return
            except (ValueError, json.JSONDecodeError) as error:
                self._json_error(400, str(error), "invalid_request")
                return
            response: httpx.Response | None = None
            try:
                upstream_payload = request_transform(payload) if request_transform is not None else payload
                response = upstream.post(
                    f"{upstream_base_url.rstrip('/')}/chat/completions",
                    content=json.dumps(upstream_payload, separators=(",", ":")).encode(),
                    headers={"Content-Type": "application/json"},
                )
                try:
                    response_payload = response.json()
                except ValueError:
                    response_payload = None
                usage = response_payload.get("usage") if isinstance(response_payload, dict) else None
                ledger.settle(reservation, usage)
                content_type = response.headers.get("content-type", "application/json")
                self.send_response(response.status_code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(response.content)))
                self.end_headers()
                self.wfile.write(response.content)
            except httpx.HTTPError as error:
                if response is None:
                    ledger.settle(reservation, None)
                self._json_error(
                    502, f"upstream paid inference request failed: {type(error).__name__}", "upstream_error"
                )

        def log_message(self, format: str, *args: object) -> None:
            del format, args

        def _json_error(self, status: int, message: str, code: str) -> None:
            content = json.dumps({"error": {"message": message, "type": code, "code": code}}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, name="posttrain-judge-cost-guard", daemon=True)
    thread.start()
    try:
        host = str(server.server_address[0])
        port = int(server.server_address[1])
        # Endpoint.base_url is the OpenAI API root.  Keeping `/v1` here makes
        # the guarded endpoint interchangeable with managed vLLM endpoints.
        yield Endpoint(f"http://{host}:{port}/v1", model, local_key)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _usage_cost(usage: object, prices: TokenPrices) -> Decimal | None:
    if not isinstance(usage, Mapping):
        return None
    input_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
    output_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
    values = (input_tokens, output_tokens)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        return None
    return prices.cost(input_tokens, output_tokens)


def _decimal_text(value: Decimal) -> str:
    micros = (value * _USD_MICROS).to_integral_value(rounding=ROUND_CEILING)
    return f"{micros / _USD_MICROS:f}"


__all__ = [
    "CostLedger",
    "InferenceCostLimitError",
    "TokenPrices",
    "metered_openai_gateway",
    "projected_cost",
    "usage_cost",
]
