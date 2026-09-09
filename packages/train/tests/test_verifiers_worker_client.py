from types import SimpleNamespace
from typing import Any, cast

import pytest
from posttrain.common.variants import LFM_25_12B_THINKING
from posttrain.train.backends.trl.policy_endpoint import TrlPolicyEndpoint
from posttrain.train.integrations.verifiers_workers import create_verifiers_train_client_config
from posttrain.train.profiles import LFM25_RENDERER
from posttrain.train.rendering import create_renderer
from posttrain.train.rollout_execution import CollectionKey

# Native worker tests exercise the optional Verifiers client and renderer
# stack, not the backend-neutral train package.  Do not make the default
# workspace suite depend on that integration extra being installed.
pytest.importorskip(
    "verifiers.v1.clients.train",
    reason="native worker client tests require the optional Verifiers integration",
)
AutoTokenizer = pytest.importorskip("transformers").AutoTokenizer
pytest.importorskip("renderers")


def test_native_worker_uses_exact_lfm_template_and_tokens(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    model_name = LFM_25_12B_THINKING.base.repo_id
    revision = LFM_25_12B_THINKING.base.revision
    try:
        direct_tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            local_files_only=True,
        )
    except OSError as error:
        pytest.skip(f"native worker parity requires the selected LFM tokenizer in the local cache: {error}")
    direct = create_renderer(direct_tokenizer, LFM_25_12B_THINKING, LFM25_RENDERER)
    config = create_verifiers_train_client_config(
        base_url="http://127.0.0.1:8123/v1",
        renderer_model_name=model_name,
        model=LFM_25_12B_THINKING,
        renderer=LFM25_RENDERER,
        multiplex=8,
    )

    worker_tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        local_files_only=True,
    )
    worker_tokenizer.chat_template = config.chat_template
    from renderers import create_renderer as create_native_renderer

    worker = create_native_renderer(worker_tokenizer, config.renderer)
    messages = [
        {"role": "user", "content": "Update the contact."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "contact_update",
                        "arguments": {"id": "003001", "phone": "+1-555-0101"},
                    },
                }
            ],
        },
        {"role": "tool", "name": "contact_update", "content": '{"success":true}'},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "contact_update",
                "description": "Update one contact.",
                "parameters": {"type": "object"},
            },
        }
    ]

    direct_tokens = direct.render(messages, tools=tools, add_generation_prompt=True)
    worker_tokens = worker.render(messages, tools=cast(Any, tools), add_generation_prompt=True)

    assert config.chat_template == LFM_25_12B_THINKING.conversation.chat_template.text()
    assert config.renderer_model_name == model_name
    assert worker_tokens.token_ids == direct_tokens.token_ids
    assert worker_tokens.message_indices == direct_tokens.message_indices
    assert worker_tokens.is_content == direct_tokens.is_content


@pytest.mark.asyncio
async def test_native_train_client_round_trips_exact_lfm_tokens_over_loopback(monkeypatch):
    from verifiers.v1.clients.train import TrainClient
    from verifiers.v1.dialects import ChatDialect
    from verifiers.v1.types import SamplingConfig

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    model = LFM_25_12B_THINKING
    model_name = model.base.repo_id
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=model.base.revision,
            local_files_only=True,
        )
    except OSError as error:
        pytest.skip(f"native worker round-trip requires the selected LFM tokenizer in the local cache: {error}")
    completion_ids = tokenizer.encode("Done.", add_special_tokens=False)

    class Session:
        def __init__(self):
            self.requests = []

        async def open_policy(self, version: str) -> None:
            del version
            return None

        async def generate(self, request):
            self.requests.append(request)
            completion = SimpleNamespace(
                index=0,
                token_ids=completion_ids,
                logprobs=[
                    {token_id: SimpleNamespace(logprob=-0.125)}
                    for token_id in completion_ids
                ],
                finish_reason="stop",
            )
            return SimpleNamespace(
                request_id=request.request_id,
                prompt_token_ids=request.prompt_token_ids,
                outputs=[completion],
                finished=True,
            )

        async def abort(self, request_id: str) -> bool:
            del request_id
            return False

        async def stop_admission(self):
            return None

    session = Session()
    endpoint = TrlPolicyEndpoint(model_name=model_name, max_model_len=4096)
    await endpoint.start(session)
    await endpoint.open_admission(CollectionKey("run", "collection-1", "policy-7"))
    config = create_verifiers_train_client_config(
        base_url=endpoint.base_url,
        renderer_model_name=model_name,
        model=model,
        renderer=LFM25_RENDERER,
        multiplex=8,
    )
    client = TrainClient(config)
    try:
        response = await client.get_response(
            ChatDialect(),
            {
                "model": model_name,
                "messages": [{"role": "user", "content": "Finish the task."}],
            },
            SamplingConfig(temperature=0.0, max_tokens=len(completion_ids)),
            session_id="episode-1",
        )
        assert response.message.content == "Done."
        assert response.tokens is not None
        assert response.tokens.completion_ids == completion_ids
        assert response.tokens.completion_logprobs == [-0.125] * len(completion_ids)
        assert tuple(response.tokens.prompt_ids) == session.requests[0].prompt_token_ids
    finally:
        await client.close()
        await endpoint.aclose()
