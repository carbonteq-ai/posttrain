import asyncio
from types import SimpleNamespace

import pytest
from posttrain.train.backends.trl.async_policy_gateway import TrlAsyncPolicyGateway
from posttrain.train.online_rl import BehaviorPolicySpan
from posttrain.train.rollout_execution import CollectionExecutionError, CollectionKey, EpisodeKey


def episode_key(version="0"):
    return EpisodeKey(
        collection=CollectionKey("run", "group", version),
        example_id="task",
        group_id="group",
        occurrence_id="rollout",
        seed=1,
    )


@pytest.mark.asyncio
async def test_drains_requests_gates_next_turn_and_preserves_session_policy_span():
    from aiohttp import ClientSession, web

    entered = asyncio.Queue()
    releases = {"first": asyncio.Event(), "second": asyncio.Event()}

    async def generate(request):
        payload = await request.json()
        request_id = payload["request_id"]
        await entered.put(request_id)
        await releases[request_id].wait()
        return web.json_response({"request_id": request_id, "choices": []})

    async def abort(_request):
        return web.json_response({"cancelled": True})

    upstream_app = web.Application()
    upstream_app.router.add_post("/inference/v1/generate", generate)
    upstream_app.router.add_post("/inference/v1/abort", abort)
    upstream_runner = web.AppRunner(upstream_app)
    await upstream_runner.setup()
    upstream_site = web.TCPSite(upstream_runner, "127.0.0.1", 0)
    await upstream_site.start()
    sockets = tuple(getattr(getattr(upstream_site, "_server", None), "sockets", ()) or ())
    assert len(sockets) == 1
    socket = sockets[0]
    upstream_url = f"http://127.0.0.1:{socket.getsockname()[1]}"

    gateway = TrlAsyncPolicyGateway(upstream_base_url=upstream_url)
    await gateway.start(0)
    try:
        async with ClientSession() as client:
            first = asyncio.create_task(
                client.post(
                    f"{gateway.base_url.removesuffix('/v1')}/inference/v1/generate",
                    json={"request_id": "first", "token_ids": [1], "sampling_params": {}},
                    headers={"X-Session-ID": "trace-1"},
                )
            )
            assert await asyncio.wait_for(entered.get(), timeout=1) == "first"
            preparing = asyncio.create_task(gateway.prepare_model_update(1))
            await asyncio.sleep(0)
            assert not preparing.done()

            releases["first"].set()
            first_response = await first
            assert first_response.status == 200
            await preparing

            second = asyncio.create_task(
                client.post(
                    f"{gateway.base_url.removesuffix('/v1')}/inference/v1/generate",
                    json={"request_id": "second", "token_ids": [1], "sampling_params": {}},
                    headers={"X-Session-ID": "trace-1"},
                )
            )
            await asyncio.sleep(0)
            assert entered.empty()

            await gateway.activate_model_version(1)
            assert await asyncio.wait_for(entered.get(), timeout=1) == "second"
            releases["second"].set()
            second_response = await second
            assert second_response.status == 200

        episode = SimpleNamespace(traces=[SimpleNamespace(id="trace-1", agent=SimpleNamespace(trainable=True))])
        span = await gateway.behavior_policy_for_episode(episode_key(), episode)
        assert span == BehaviorPolicySpan(0, 1)
    finally:
        await gateway.aclose()
        await upstream_runner.cleanup()


@pytest.mark.asyncio
async def test_shared_upstream_failure_poisoning_is_not_a_low_reward():
    from aiohttp import ClientSession, web

    async def fail(_request):
        return web.json_response({"error": "engine failed"}, status=503)

    upstream_app = web.Application()
    upstream_app.router.add_post("/inference/v1/generate", fail)
    upstream_app.router.add_post("/inference/v1/abort", fail)
    upstream_runner = web.AppRunner(upstream_app)
    await upstream_runner.setup()
    upstream_site = web.TCPSite(upstream_runner, "127.0.0.1", 0)
    await upstream_site.start()
    sockets = tuple(getattr(getattr(upstream_site, "_server", None), "sockets", ()) or ())
    assert len(sockets) == 1
    socket = sockets[0]
    gateway = TrlAsyncPolicyGateway(upstream_base_url=f"http://127.0.0.1:{socket.getsockname()[1]}")
    await gateway.start(0)
    try:
        async with ClientSession() as client:
            response = await client.post(
                f"{gateway.base_url.removesuffix('/v1')}/inference/v1/generate",
                json={"request_id": "failed", "token_ids": [1], "sampling_params": {}},
                headers={"X-Session-ID": "trace-1"},
            )
            assert response.status == 503
        assert isinstance(gateway.fatal_error, CollectionExecutionError)
        with pytest.raises(CollectionExecutionError, match="HTTP 503"):
            await gateway.prepare_model_update(1)
    finally:
        await gateway.aclose()
        await upstream_runner.cleanup()


@pytest.mark.asyncio
async def test_shutdown_requires_upstream_abort_acknowledgement_and_still_cleans_up():
    from aiohttp import ClientSession, web

    entered = asyncio.Event()
    release = asyncio.Event()

    async def generate(_request):
        entered.set()
        await release.wait()
        return web.json_response({"request_id": "active", "choices": []})

    async def refuse_abort(_request):
        release.set()
        return web.json_response({"cancelled": False})

    upstream_app = web.Application()
    upstream_app.router.add_post("/inference/v1/generate", generate)
    upstream_app.router.add_post("/inference/v1/abort", refuse_abort)
    upstream_runner = web.AppRunner(upstream_app)
    await upstream_runner.setup()
    upstream_site = web.TCPSite(upstream_runner, "127.0.0.1", 0)
    await upstream_site.start()
    sockets = tuple(getattr(getattr(upstream_site, "_server", None), "sockets", ()) or ())
    assert len(sockets) == 1
    socket = sockets[0]
    gateway = TrlAsyncPolicyGateway(upstream_base_url=f"http://127.0.0.1:{socket.getsockname()[1]}")
    await gateway.start(0)
    client = ClientSession()
    request = asyncio.create_task(
        client.post(
            f"{gateway.base_url.removesuffix('/v1')}/inference/v1/generate",
            json={"request_id": "active", "token_ids": [1], "sampling_params": {}},
            headers={"X-Session-ID": "trace-1"},
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)

        with pytest.raises(CollectionExecutionError, match="did not acknowledge abort"):
            await gateway.aclose()

        with pytest.raises(RuntimeError, match="has not started"):
            _ = gateway.base_url
        await asyncio.gather(request, return_exceptions=True)
    finally:
        await client.close()
        await upstream_runner.cleanup()
