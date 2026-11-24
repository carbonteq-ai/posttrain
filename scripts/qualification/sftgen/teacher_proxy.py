"""Teacher proxy: a local OpenAI-compatible endpoint between the evaluation loop and the teacher provider (plan v4 P4).

The evaluation loop (verifiers run_eval, the same environment config as the 4B's runs) sends each turn here exactly
as it sends it to the 4B's local endpoint. The proxy forwards it to the provider with the settings that make the
data what plan v4 says, whatever the loop sent:
  - model and provider pinned (OpenRouter `provider.order` with `allow_fallbacks: false`); a response served by any
    other provider is refused (HTTP 409 to the loop) and recorded;
  - reasoning on or off (`reasoning.enabled`), the model card's sampling, `max_tokens` = the reply limit;
  - `usage.include` so every response carries its own cost;
  - a hard spend cap: once the recorded cost reaches the cap, every further request gets HTTP 402 and a STOP file is
    written (requests already in flight may add at most their own cost);
  - retries only for transport errors and 5xx from the provider (`retries`), and for provider rate limits (429) with
    longer waits (`retries_429`, default 10; Retry-After honoured; at most `max_backoff_s`, default 120 s); every other
    status goes back as is.
Every request writes one line to `ledger.jsonl` (index, time, status, provider, model, response id, usage, cost,
finish reason, latency, attempts, sha256 of the first user message) and the response body to `responses.jsonl.gz`;
the request bodies (messages and tools exactly as the loop sent them) go to `requests.jsonl.gz`. The API key is read
from the environment variable named by --api-key-var and is never written anywhere.

usage: <python with aiohttp> teacher_proxy.py --config CONFIG.json --out DIR [--port 0]
CONFIG: {"upstream": "https://openrouter.ai/api/v1", "model": "qwen/qwen3.5-397b-a17b", "provider": "alibaba",
         "reasoning": {"enabled": true}, "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0},
         "max_tokens": 32768, "cap_usd": 30.0, "retries": 3, "api_key_var": "OPENROUTER_API_KEY"}
The chosen port is written to DIR/port once listening.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import os
import time
from pathlib import Path

from aiohttp import ClientError, ClientSession, ClientTimeout, web

RETRY_STATUSES = {429, 500, 502, 503, 504}
REQUIRED = ("upstream", "model", "provider", "reasoning", "sampling", "max_tokens", "cap_usd", "retries", "api_key_var")


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    missing = [key for key in REQUIRED if key not in config]
    if missing:
        raise ValueError(f"proxy config misses {missing}")
    if not 0 < float(config["cap_usd"]):
        raise ValueError("cap_usd must be positive")
    return config


def upstream_body(body: dict, config: dict) -> dict:
    """The provider request: the loop's messages and tools, our pinned model, provider, reasoning and sampling."""
    if body.get("stream"):
        raise ValueError("streaming requests are not supported by the teacher proxy")
    out = {key: value for key, value in body.items()
           if key not in ("model", "max_tokens", "max_completion_tokens", "temperature", "top_p", "top_k", "min_p",
                          "presence_penalty", "repetition_penalty", "provider", "reasoning", "usage", "stream")}
    out["model"] = config["model"]
    out.update(config["sampling"])
    out["max_tokens"] = int(config["max_tokens"])
    out["reasoning"] = dict(config["reasoning"])
    out["provider"] = {"order": [config["provider"]], "allow_fallbacks": False}
    out["usage"] = {"include": True}
    return out


def same_provider(served: str | None, pinned: str) -> bool:
    norm = lambda text: "".join(ch for ch in (text or "").lower() if ch.isalnum())  # noqa: E731
    return bool(served) and norm(served) == norm(pinned)


def first_user_sha(body: dict) -> str | None:
    for message in body.get("messages") or []:
        if message.get("role") == "user":
            content = message.get("content")
            text = content if isinstance(content, str) else json.dumps(content, sort_keys=True)
            return hashlib.sha256(text.encode()).hexdigest()
    return None


class Ledger:
    def __init__(self, out: Path, cap: float):
        self.out, self.cap = out, cap
        self.spent, self.index, self.stopped = 0.0, 0, (out / "STOP").exists()
        self.lock = asyncio.Lock()
        if (out / "ledger.jsonl").exists():  # resume: carry the recorded spend forward
            for line in (out / "ledger.jsonl").read_text().splitlines():
                record = json.loads(line)
                self.spent += float(record.get("cost") or 0.0)
                self.index = max(self.index, int(record["index"]) + 1)
        if self.spent >= cap:
            self.stopped = True

    async def next_index(self) -> int:
        async with self.lock:
            index, self.index = self.index, self.index + 1
            return index

    async def record(self, entry: dict, request: dict, response_text: str) -> None:
        async with self.lock:
            self.spent += float(entry.get("cost") or 0.0)
            entry["spent_after"] = round(self.spent, 6)
            with (self.out / "ledger.jsonl").open("a") as stream:
                stream.write(json.dumps(entry) + "\n")
            with gzip.open(self.out / "requests.jsonl.gz", "at") as stream:
                stream.write(json.dumps({"index": entry["index"], "body": request}) + "\n")
            with gzip.open(self.out / "responses.jsonl.gz", "at") as stream:
                stream.write(json.dumps({"index": entry["index"], "status": entry["status"], "text": response_text}) + "\n")
            if self.spent >= self.cap and not self.stopped:
                self.stopped = True
                (self.out / "STOP").write_text(json.dumps({"spent": self.spent, "cap": self.cap, "at": time.time()}))


def make_app(config: dict, out: Path) -> web.Application:
    ledger = Ledger(out, float(config["cap_usd"]))
    key = os.environ.get(config["api_key_var"])
    if not key:
        raise RuntimeError(f"environment variable {config['api_key_var']} is not set")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    url = config["upstream"].rstrip("/") + "/chat/completions"

    async def handle(request: web.Request) -> web.Response:
        body = await request.json()
        index = await ledger.next_index()
        if ledger.stopped:
            return web.json_response({"error": {"message": "teacher proxy spend cap reached", "type": "cap"}},
                                     status=402)
        try:
            forward = upstream_body(body, config)
        except ValueError as error:
            return web.json_response({"error": {"message": str(error)}}, status=400)
        started, attempts, rate_limited, status, text = time.monotonic(), 0, 0, 0, ""
        while True:
            attempts += 1
            retry_after = None
            try:
                async with request.app["session"].post(url, json=forward, headers=headers) as response:
                    status, text = response.status, await response.text()
                    retry_after = response.headers.get("Retry-After")
            except (ClientError, asyncio.TimeoutError) as error:
                status, text = 599, json.dumps({"error": {"message": f"transport: {type(error).__name__}"}})
            if status == 429 and rate_limited < int(config.get("retries_429", 10)):
                # provider rate limit: no generation happened; wait (Retry-After if given) and ask again
                rate_limited += 1
                wait = float(retry_after) if (retry_after or "").replace(".", "", 1).isdigit() else 5 * 2 ** rate_limited
                await asyncio.sleep(min(float(config.get("max_backoff_s", 120)), wait))
                continue
            if (status in RETRY_STATUSES - {429} or status == 599) and attempts - rate_limited <= int(config["retries"]):
                await asyncio.sleep(min(60, 2 ** (attempts - rate_limited)))
                continue
            break
        entry = {"index": index, "time": time.time(), "status": status, "attempts": attempts, "rate_limited": rate_limited,
                 "latency_s": round(time.monotonic() - started, 3), "first_user_sha256": first_user_sha(body),
                 "n_messages": len(body.get("messages") or []), "pinned_provider": config["provider"]}
        data = None
        if status == 200:
            try:
                data = json.loads(text)
            except ValueError:
                status = 502
        if data is not None:
            usage = data.get("usage") or {}
            choice = (data.get("choices") or [{}])[0]
            entry |= {"id": data.get("id"), "provider": data.get("provider"), "model": data.get("model"),
                      "finish_reason": choice.get("finish_reason"), "usage": usage, "cost": usage.get("cost")}
            if usage.get("cost") is None:
                entry["cost_missing"] = True
            if not same_provider(data.get("provider"), config["provider"]):
                entry["refused"] = "served by another provider"
                status = 409
        await ledger.record(entry, forward, text)
        if status != 200:
            return web.Response(text=text or json.dumps({"error": {"message": f"status {status}"}}), status=status,
                                content_type="application/json")
        return web.json_response(data)

    async def on_startup(app: web.Application) -> None:
        app["session"] = ClientSession(timeout=ClientTimeout(total=None, sock_connect=30))

    async def on_cleanup(app: web.Application) -> None:
        await app["session"].close()

    app = web.Application(client_max_size=256 * 1024 * 1024)
    app.router.add_post("/v1/chat/completions", handle)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app["ledger"] = ledger
    return app


async def serve(config: dict, out: Path, port: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    runner = web.AppRunner(make_app(config, out))
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    bound = site._server.sockets[0].getsockname()[1]  # noqa: SLF001
    (out / "port").write_text(str(bound))
    while True:
        await asyncio.sleep(3600)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args(argv)
    asyncio.run(serve(load_config(args.config), args.out, args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
