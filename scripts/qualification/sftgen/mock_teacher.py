"""Mock OpenRouter chat endpoint for testing the teacher pipeline without any paid call (plan v4 P4).

It answers each request from a script: turn k (the number of assistant messages already in the request) gets
script[k], returned in OpenRouter's response shape (message.reasoning, tool_calls with JSON-string arguments,
`provider`, `usage.cost`). Behaviours for tests, selected by the script entry's "mode":
  "ok" (default), "error500_once" (a 500 before the real answer), "wrong_provider", "no_cost".
It records every request body to requests.jsonl, so tests can check what the proxy forwarded.

usage: python mock_teacher.py --script SCRIPT.json --out DIR [--port 0]   (port written to DIR/port)
SCRIPT: {"provider": "Alibaba", "turns": [{"reasoning": "...", "content": "...",
          "calls": [{"name": "...", "arguments": {...}}], "mode": "ok"}, ...]}
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from aiohttp import web


def make_app(script: dict, out: Path) -> web.Application:
    state = {"failed_once": set()}

    async def handle(request: web.Request) -> web.Response:
        body = await request.json()
        with (out / "requests.jsonl").open("a") as stream:
            stream.write(json.dumps(body) + "\n")
        turn = sum(1 for m in body.get("messages") or [] if m.get("role") == "assistant")
        entry = script["turns"][min(turn, len(script["turns"]) - 1)]
        mode = entry.get("mode", "ok")
        if mode == "error500_once" and turn not in state["failed_once"]:
            state["failed_once"].add(turn)
            return web.json_response({"error": {"message": "mock 500"}}, status=500)
        if mode.startswith("rate429x"):  # rate-limit the first N requests of this turn
            n = int(mode[len("rate429x"):])
            state.setdefault("rate", {}).setdefault(turn, 0)
            if state["rate"][turn] < n:
                state["rate"][turn] += 1
                return web.json_response({"error": {"message": "rate-limited", "code": 429}}, status=429,
                                         headers={"Retry-After": "0"})
        calls = [{"id": f"call_{turn}_{i}", "type": "function",
                  "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                 for i, c in enumerate(entry.get("calls") or [])]
        message = {"role": "assistant", "content": entry.get("content") or "", "reasoning": entry.get("reasoning")}
        if calls:
            message["tool_calls"] = calls
        usage = {"prompt_tokens": 1000 + 10 * turn, "completion_tokens": 100, "total_tokens": 1100 + 10 * turn,
                 "completion_tokens_details": {"reasoning_tokens": 40}}
        if mode != "no_cost":
            usage["cost"] = 0.001
        return web.json_response({
            "id": f"gen-mock-{turn}", "model": body.get("model"),
            "provider": "OtherHost" if mode == "wrong_provider" else script.get("provider", "Alibaba"),
            "object": "chat.completion", "created": 0,
            "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if calls else "stop"}],
            "usage": usage})

    app = web.Application(client_max_size=256 * 1024 * 1024)
    app.router.add_post("/chat/completions", handle)
    app.router.add_post("/v1/chat/completions", handle)
    return app


async def serve(script: dict, out: Path, port: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    runner = web.AppRunner(make_app(script, out))
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    (out / "port").write_text(str(site._server.sockets[0].getsockname()[1]))  # noqa: SLF001
    while True:
        await asyncio.sleep(3600)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args(argv)
    asyncio.run(serve(json.loads(args.script.read_text()), args.out, args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
