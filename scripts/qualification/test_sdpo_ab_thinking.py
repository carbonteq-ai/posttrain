"""D97: the `--thinking on|off` flag of sdpo_ab_runner (default off, byte-identical to every earlier run)."""
import os

import pytest

import sdpo_ab_runner as runner

needs_tokenizer = pytest.mark.skipif(not os.environ.get("SDPO_AB_TOKENIZER"), reason="needs SDPO_AB_TOKENIZER")


class _Tokenizer:
    """Chat template stand-in that records the enable_thinking it was called with and renders it into the text."""
    eos_token = "<E>"
    eos_token_id = 1

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=True,
                            enable_thinking=False, return_dict=False):
        self.calls.append(enable_thinking)
        text = "".join(f"<{m['role']}>{m.get('content') or ''}{self.eos_token}" for m in messages)
        text += ("<assistant><think>" if enable_thinking else "<assistant><nothink>") if add_generation_prompt else ""
        if not tokenize:
            return text
        ids = self.encode(text)
        return {"input_ids": ids} if return_dict else ids

    def encode(self, text, add_special_tokens=False):
        out, i = [], 0
        while i < len(text):
            if text.startswith(self.eos_token, i):
                out.append(self.eos_token_id)
                i += len(self.eos_token)
            else:
                out.append(ord(text[i]))
                i += 1
        return out


MESSAGES = [{"role": "system", "content": "s"}, {"role": "user", "content": "do the task"}]
TOOL_RESULT = [{"role": "tool", "content": "ok", "tool_call_id": "t1"}]


@pytest.fixture(autouse=True)
def _thinking_off():
    runner.THINKING = False
    yield
    runner.THINKING = False


def test_default_renders_with_thinking_off_as_before():
    tok = _Tokenizer()
    first, delta = runner.first_prompt(tok, MESSAGES, []), runner.render_delta(tok, TOOL_RESULT)
    assert tok.calls == [False, False]
    assert first == runner.first_prompt(tok, MESSAGES, [], thinking=False)
    assert delta == runner.render_delta(tok, TOOL_RESULT, thinking=False)
    assert tok.encode("<nothink>")[-9:] == first[-9:] == delta[-9:]


def test_flag_on_renders_every_prompt_with_thinking():
    tok = _Tokenizer()
    runner.THINKING = True
    first, delta = runner.first_prompt(tok, MESSAGES, []), runner.render_delta(tok, TOOL_RESULT)
    assert tok.calls == [True, True]
    assert tok.encode("<think>")[-7:] == first[-7:] == delta[-7:]
    # an explicit argument still wins (training-side callers that pin their own rendering)
    assert runner.first_prompt(tok, MESSAGES, [], thinking=False)[-9:] == tok.encode("<nothink>")[-9:]
    assert tok.calls[-1] is False


def test_visible_reply_splits_reasoning_from_the_shown_text():
    assert runner.visible_reply("I think.\n</think>\n\nHello <tool_call>x</tool_call>") == (
        "I think.\n", "Hello <tool_call>x</tool_call>", True)
    assert runner.visible_reply("no reasoning at all") == ("", "no reasoning at all", False)
    assert runner.visible_reply("cut inside the reasoning ...") == ("", "cut inside the reasoning ...", False)
    assert runner.visible_reply("</think>\n\n") == ("", "", True)


def test_flag_defaults_off_and_is_refused_outside_collect_ours():
    base = ["--phase", "collect", "--model", "m", "--output", "o", "--protocol", "ours"]
    args = runner.build_parser().parse_args(base)
    assert args.thinking == "off"
    runner.check_collect_options(args)
    on = runner.build_parser().parse_args(base + ["--thinking", "on", "--sampling", "card"])
    assert on.thinking == "on"
    runner.check_collect_options(on)
    for bad in (["--phase", "smoke", "--model", "m", "--output", "o", "--thinking", "on"],
                base[:-2] + ["--protocol", "handover", "--thinking", "on"]):
        with pytest.raises(ValueError):
            runner.check_collect_options(runner.build_parser().parse_args(bad))
    with pytest.raises(SystemExit):
        runner.build_parser().parse_args(base + ["--thinking", "yes"])


def _real_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(os.environ["SDPO_AB_TOKENIZER"])


TOOLS = [{"type": "function", "function": {"name": "send", "description": "send", "parameters": {
    "type": "object", "properties": {"to": {"type": "string"}}, "required": ["to"]}}}]


@needs_tokenizer
def test_qwen_template_default_is_the_old_hard_coded_rendering_and_on_opens_think():
    tok = _real_tokenizer()
    old_first = tok.apply_chat_template(MESSAGES, tools=TOOLS, tokenize=True, add_generation_prompt=True,
                                        enable_thinking=False, return_dict=True)["input_ids"]
    assert runner.first_prompt(tok, MESSAGES, TOOLS) == list(old_first)
    off_tail = tok.encode("<think>\n\n</think>\n\n", add_special_tokens=False)
    on_tail = tok.encode("<think>\n", add_special_tokens=False)
    assert runner.first_prompt(tok, MESSAGES, TOOLS)[-len(off_tail):] == off_tail
    assert runner.render_delta(tok, TOOL_RESULT)[-len(off_tail):] == off_tail
    runner.THINKING = True
    on_first, on_delta = runner.first_prompt(tok, MESSAGES, TOOLS), runner.render_delta(tok, TOOL_RESULT)
    assert on_first[-len(on_tail):] == on_tail and on_delta[-len(on_tail):] == on_tail
    assert on_first[:-len(on_tail)] == list(old_first)[:-len(off_tail)]  # only the generation prompt differs
    assert tok.convert_tokens_to_ids("</think>") not in on_first[len(old_first) - len(off_tail):]
