from typing import Any, cast

from posttrain.common.variants import LFM_25_12B_THINKING
from posttrain.train.integrations.verifiers_workers import create_verifiers_train_client_config
from posttrain.train.profiles import LFM25_RENDERER
from posttrain.train.rendering import create_renderer
from transformers import AutoTokenizer


def test_native_worker_uses_exact_lfm_template_and_tokens(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    model_name = LFM_25_12B_THINKING.base.repo_id
    revision = LFM_25_12B_THINKING.base.revision
    direct_tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, local_files_only=True)
    direct = create_renderer(direct_tokenizer, LFM_25_12B_THINKING, LFM25_RENDERER)
    config = create_verifiers_train_client_config(
        base_url="http://127.0.0.1:8123/v1",
        renderer_model_name=model_name,
        model=LFM_25_12B_THINKING,
        renderer=LFM25_RENDERER,
        multiplex=8,
    )

    worker_tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, local_files_only=True)
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
