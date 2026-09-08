import asyncio
from types import SimpleNamespace

import httpx
import pytest
from posttrain.train.backends.trl.policy_endpoint import TrlPolicyEndpoint
from posttrain.train.rollout_execution import CollectionKey


def _output(request_id, prompt_ids=(1, 2), completion_ids=(3, 4), logprobs=(-0.25, -0.5)):
    steps = [
        {token_id: SimpleNamespace(logprob=logprob)}
        for token_id, logprob in zip(completion_ids, logprobs, strict=True)
    ]
    completion = SimpleNamespace(
        index=0,
        token_ids=completion_ids,
        logprobs=steps,
        finish_reason="length",
    )
    return SimpleNamespace(
        request_id=request_id,
        prompt_token_ids=prompt_ids,
        outputs=[completion],
        finished=True,
    )


class FakeSession:
    def __init__(self):
        self.opened = []
        self.stops = 0
        self.requests = []
        self.tasks = {}

    async def open_policy(self, version):
        self.opened.append(version)

    async def generate(self, request):
        self.requests.append(request)
        if request.request_id == "slow":
            self.tasks[request.request_id] = asyncio.current_task()
            await asyncio.Event().wait()
        return _output(request.request_id, prompt_ids=request.prompt_token_ids)

    async def abort(self, request_id):
        task = self.tasks.get(request_id)
        if task is None or task.done():
            return False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return True

    async def stop_admission(self):
        self.stops += 1


def _body(request_id="request-1"):
    return {
        "request_id": request_id,
        "model": "policy-model",
        "token_ids": [1, 2],
        "sampling_params": {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 2,
            "logprobs": 1,
            "skip_special_tokens": False,
        },
    }


@pytest.mark.asyncio
async def test_loopback_endpoint_preserves_native_token_response_and_collection_fence():
    from renderers.client import _parse_completion_logprobs
    from vllm.entrypoints.scale_out.token_in_token_out.protocol import GenerateResponse

    session = FakeSession()
    endpoint = TrlPolicyEndpoint(model_name="policy-model", max_model_len=4096)
    collection = CollectionKey("run", "collection-1", "policy-7")
    await endpoint.start(session)
    try:
        await endpoint.open_admission(collection)
        async with httpx.AsyncClient() as client:
            models = (await client.get(f"{endpoint.base_url}/models")).json()
            assert models["data"][0]["max_model_len"] == 4096
            response = await client.post(
                f"{endpoint.base_url.removesuffix('/v1')}/inference/v1/generate",
                json=_body(),
                headers={"X-Session-ID": "episode-1"},
            )
            assert response.status_code == 200
            payload = response.json()
            GenerateResponse.model_validate(payload)
            assert payload["prompt_token_ids"] == [1, 2]
            assert payload["choices"][0]["token_ids"] == [3, 4]
            assert _parse_completion_logprobs(payload["choices"][0], [3, 4]) == [-0.25, -0.5]

            await endpoint.stop_admission(collection)
            rejected = await client.post(
                f"{endpoint.base_url.removesuffix('/v1')}/inference/v1/generate",
                json=_body("late"),
            )
            assert rejected.status_code == 409
        assert session.opened == ["policy-7"]
        assert session.stops == 1
        assert endpoint.active_request_ids == frozenset()
    finally:
        await endpoint.aclose()


@pytest.mark.asyncio
async def test_explicit_abort_cancels_the_registered_http_request():
    session = FakeSession()
    endpoint = TrlPolicyEndpoint(model_name="policy-model", max_model_len=4096)
    collection = CollectionKey("run", "collection-1", "policy-7")
    await endpoint.start(session)
    await endpoint.open_admission(collection)
    root = endpoint.base_url.removesuffix("/v1")
    try:
        async with httpx.AsyncClient() as client:
            generation = asyncio.create_task(
                client.post(f"{root}/inference/v1/generate", json=_body("slow"))
            )
            async with asyncio.timeout(5):
                while endpoint.active_request_ids != frozenset({"slow"}):
                    await asyncio.sleep(0)
            aborted = await client.post(f"{root}/inference/v1/abort", json={"request_id": "slow"})
            assert aborted.status_code == 200
            assert aborted.json() == {"cancelled": True}
            with pytest.raises(httpx.HTTPError):
                await generation
            assert endpoint.active_request_ids == frozenset()
    finally:
        await endpoint.aclose()


@pytest.mark.asyncio
async def test_endpoint_rejects_context_overflow_without_calling_engine():
    session = FakeSession()
    endpoint = TrlPolicyEndpoint(model_name="policy-model", max_model_len=3)
    collection = CollectionKey("run", "collection-1", "policy-7")
    await endpoint.start(session)
    await endpoint.open_admission(collection)
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{endpoint.base_url.removesuffix('/v1')}/inference/v1/generate",
                json=_body(),
            )
        assert response.status_code == 400
        assert "2 prompt + 2 completion > 3" in response.json()["error"]
        assert session.requests == []
        assert endpoint.fatal_error is None
    finally:
        await endpoint.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("cache_salt", "private-prefix", "does not support cache_salt"),
        ("priority", -1, "does not support priority scheduling"),
    ],
)
async def test_endpoint_rejects_scheduling_fields_it_cannot_preserve(field, value, message):
    session = FakeSession()
    endpoint = TrlPolicyEndpoint(model_name="policy-model", max_model_len=4096)
    collection = CollectionKey("run", "collection-1", "policy-7")
    await endpoint.start(session)
    await endpoint.open_admission(collection)
    body = _body()
    body[field] = value
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{endpoint.base_url.removesuffix('/v1')}/inference/v1/generate",
                json=body,
            )
        assert response.status_code == 400
        assert message in response.json()["error"]
        assert session.requests == []
        assert endpoint.fatal_error is None
    finally:
        await endpoint.aclose()


@pytest.mark.asyncio
async def test_response_identity_mismatch_is_collection_fatal():
    class WrongIdentitySession(FakeSession):
        async def generate(self, request):
            return _output("wrong-request")

    session = WrongIdentitySession()
    endpoint = TrlPolicyEndpoint(model_name="policy-model", max_model_len=4096)
    collection = CollectionKey("run", "collection-1", "policy-7")
    await endpoint.start(session)
    await endpoint.open_admission(collection)
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{endpoint.base_url.removesuffix('/v1')}/inference/v1/generate",
                json=_body(),
            )
        assert response.status_code == 500
        assert isinstance(endpoint.fatal_error, RuntimeError)
        assert "request identity" in str(endpoint.fatal_error)
    finally:
        await endpoint.aclose()


@pytest.mark.asyncio
async def test_response_prompt_identity_mismatch_is_collection_fatal():
    class WrongPromptSession(FakeSession):
        async def generate(self, request):
            return _output(request.request_id, prompt_ids=(9, 9))

    session = WrongPromptSession()
    endpoint = TrlPolicyEndpoint(model_name="policy-model", max_model_len=4096)
    collection = CollectionKey("run", "collection-1", "policy-7")
    await endpoint.start(session)
    await endpoint.open_admission(collection)
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{endpoint.base_url.removesuffix('/v1')}/inference/v1/generate",
                json=_body(),
            )
        assert response.status_code == 500
        assert isinstance(endpoint.fatal_error, RuntimeError)
        assert "prompt token identity" in str(endpoint.fatal_error)
    finally:
        await endpoint.aclose()
