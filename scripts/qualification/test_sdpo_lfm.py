"""Reject cross-family/profile mistakes and special-token destruction."""

import json
from types import SimpleNamespace

import pytest
from sdpo_lfm_config import profile
from sdpo_lfm_runtime import prompt_ids, split_native


def settings(**overrides):
    return SimpleNamespace(**(dict(run_profile="lfm-pool124-20", model_family="lfm2.5",
        steps=20, batch_size=32, online_rollouts=2, lr=1e-5, lr_end_ratio=.1,
        lr_schedule="hold10-decay20-flat", context_limit=16384, reply_limit=2048,
        thinking=True, qualify_capacity=True, shared_budget=None,
        teacher_feedback="solution-or-feedback", tool_parser_mode="lfm2-native",
        selection_mode="failure-frontier", alpha=1., topk=20, teacher_rate=.01) | overrides))


def test_lfm_profile_never_inherits_qwen_parser_or_old_budget():
    assert profile(settings()) == {"batch": 32, "evaluate_every": 10, "save_every": 1, "time_limit_s": 7100}
    for changed in ({"model_family": "qwen3.5"}, {"tool_parser_mode": "schema-bound-v1"},
                    {"thinking": False}, {"steps": 50}, {"batch_size": 16}, {"qualify_capacity": False}):
        with pytest.raises(ValueError):
            profile(settings(**changed))


def test_lfm_native_decode_keeps_tool_delimiters_and_excludes_unclosed_reasoning():
    class Tokenizer:
        eos_token_id = 99

        def decode(self, ids, skip_special_tokens):
            assert skip_special_tokens is False and 99 not in ids
            return "Check.</think><|tool_call_start|>[lookup_code(key='readiness')]<|tool_call_end|>"

    raw, reasoning, visible, complete = split_native(Tokenizer(), [1, 2, 99], True)
    assert complete and reasoning == "Check." and "<|tool_call_start|>" in visible
    assert "</think>" in raw and not split_native(Tokenizer(), [1, 2], False)[3]


def test_lfm_render_converts_json_arguments_without_mutating_student_history():
    class Tokens:
        def __getitem__(self, key):
            return self

        def tolist(self):
            return [1, 2]

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            args = messages[0]["tool_calls"][0]["function"]["arguments"]
            assert args == {"id": "001", "text": "a'b"}
            assert kwargs["preserve_thinking"] is False and "enable_thinking" not in kwargs
            return Tokens()

    args = json.dumps({"id": "001", "text": "a'b"})
    history = [{"role": "assistant", "tool_calls": [{"function": {"arguments": args}}]}]
    assert prompt_ids(Tokenizer(), history, []) == [1, 2]
    assert history[0]["tool_calls"][0]["function"]["arguments"] == args
