"""teacher_proxy.py against mock_teacher.py, in process, no network: pinning, forwarded settings, retries, refusals,
the spend cap, and that the key never reaches any written file."""

import asyncio
import gzip
import json
from pathlib import Path

import pytest
from aiohttp import ClientSession, web

import mock_teacher
import teacher_proxy

KEY = "sk-test-not-a-real-key-123"


async def start(app):
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    return runner, site._server.sockets[0].getsockname()[1]  # noqa: SLF001


def config(upstream_port: int, cap: float = 1.0) -> dict:
    return {"upstream": f"http://127.0.0.1:{upstream_port}", "model": "qwen/qwen3.5-397b-a17b", "provider": "alibaba",
            "reasoning": {"enabled": True}, "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0},
            "max_tokens": 32768, "cap_usd": cap, "retries": 2, "api_key_var": "TEST_PROXY_KEY"}


def request(n_assistant: int = 0, **extra) -> dict:
    messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    messages += [{"role": "assistant", "content": "a"}] * n_assistant
    return {"model": "teacher", "messages": messages, "temperature": 1.0, "max_tokens": 10, **extra}


async def run_case(tmp_path: Path, turns: list[dict], requests: list[dict], cap: float = 1.0):
    mock_out, proxy_out = tmp_path / "mock", tmp_path / "proxy"
    mock_out.mkdir()
    mock_runner, mock_port = await start(mock_teacher.make_app({"provider": "Alibaba", "turns": turns}, mock_out))
    proxy_out.mkdir()
    proxy_runner, proxy_port = await start(teacher_proxy.make_app(config(mock_port, cap), proxy_out))
    results = []
    try:
        async with ClientSession() as session:
            for body in requests:
                async with session.post(f"http://127.0.0.1:{proxy_port}/v1/chat/completions", json=body) as resp:
                    results.append((resp.status, await resp.json()))
    finally:
        await proxy_runner.cleanup()
        await mock_runner.cleanup()
    forwarded = [json.loads(line) for line in (mock_out / "requests.jsonl").read_text().splitlines()] \
        if (mock_out / "requests.jsonl").exists() else []
    ledger = [json.loads(line) for line in (proxy_out / "ledger.jsonl").read_text().splitlines()] \
        if (proxy_out / "ledger.jsonl").exists() else []
    return results, forwarded, ledger, proxy_out


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv("TEST_PROXY_KEY", KEY)


def test_forwards_pinned_settings_and_records_cost(tmp_path):
    turns = [{"reasoning": "r", "content": "c", "calls": [{"name": "t", "arguments": {"b": 1, "a": 2}}]}]
    results, forwarded, ledger, out = asyncio.run(run_case(tmp_path, turns, [request()]))
    assert results[0][0] == 200
    body = forwarded[0]
    assert body["model"] == "qwen/qwen3.5-397b-a17b" and body["max_tokens"] == 32768
    assert (body["temperature"], body["top_p"], body["top_k"], body["min_p"]) == (0.6, 0.95, 20, 0.0)
    assert body["provider"] == {"order": ["alibaba"], "allow_fallbacks": False}
    assert body["reasoning"] == {"enabled": True} and body["usage"] == {"include": True}
    assert ledger[0]["cost"] == 0.001 and ledger[0]["provider"] == "Alibaba"
    message = results[0][1]["choices"][0]["message"]
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {"b": 1, "a": 2}
    assert list(json.loads(message["tool_calls"][0]["function"]["arguments"])) == ["b", "a"]  # key order kept


def test_retries_a_500_then_succeeds(tmp_path):
    turns = [{"content": "c", "mode": "error500_once"}]
    results, forwarded, ledger, _ = asyncio.run(run_case(tmp_path, turns, [request()]))
    assert results[0][0] == 200 and ledger[0]["attempts"] == 2 and len(forwarded) == 2


def test_refuses_another_provider(tmp_path):
    results, _, ledger, _ = asyncio.run(run_case(tmp_path, [{"content": "c", "mode": "wrong_provider"}], [request()]))
    assert results[0][0] == 409 and ledger[0]["refused"] == "served by another provider"


def test_flags_a_response_without_cost(tmp_path):
    results, _, ledger, _ = asyncio.run(run_case(tmp_path, [{"content": "c", "mode": "no_cost"}], [request()]))
    assert results[0][0] == 200 and ledger[0]["cost_missing"] is True


def test_refuses_streaming(tmp_path):
    results, forwarded, _, _ = asyncio.run(run_case(tmp_path, [{"content": "c"}], [request(stream=True)]))
    assert results[0][0] == 400 and forwarded == []


def test_spend_cap_stops_further_requests(tmp_path):
    turns = [{"content": "c"}]
    results, forwarded, ledger, out = asyncio.run(run_case(tmp_path, turns, [request(), request(1), request(2)],
                                                           cap=0.002))
    assert [status for status, _ in results] == [200, 200, 402]
    assert len(forwarded) == 2 and (out / "STOP").exists()
    assert ledger[-1]["spent_after"] == pytest.approx(0.002)


def test_resume_carries_spend_and_stop(tmp_path):
    out = tmp_path / "proxy"
    out.mkdir()
    (out / "ledger.jsonl").write_text(json.dumps({"index": 4, "cost": 0.0025}) + "\n")
    ledger = teacher_proxy.Ledger(out, 0.002)
    assert ledger.stopped and ledger.index == 5 and ledger.spent == pytest.approx(0.0025)


def test_key_never_written(tmp_path):
    turns = [{"reasoning": "r", "content": "c"}]
    _, _, _, out = asyncio.run(run_case(tmp_path, turns, [request(), request(1)]))
    for path in out.iterdir():
        data = gzip.open(path, "rt").read() if path.suffix == ".gz" else path.read_text()
        assert KEY not in data, path


def test_rate_limits_wait_and_retry_beyond_the_5xx_budget(tmp_path):
    results, forwarded, ledger, _ = asyncio.run(run_case(tmp_path, [{"content": "c", "mode": "rate429x5"}], [request()]))
    assert results[0][0] == 200 and ledger[0]["rate_limited"] == 5 and len(forwarded) == 6
