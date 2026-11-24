"""Plan v4 (12 Nov 2026): episode limits and thinking-card sampling are settings; defaults keep D10 behaviour."""
import pytest

import sdpo_ab_replay
import sdpo_ab_runner as runner
from sdpo_frozen_environment import COLLECT_SAMPLINGS


def parse(*extra):
    return runner.build_parser().parse_args(["--phase", "collect", "--model", "m", "--output", "o", *extra])


def test_defaults_are_d10():
    assert runner.episode_limits(parse()) == (50, 65536, 4096, 3600)
    env = sdpo_ab_replay.environment(["finance.ap_turnover_analysis"])
    assert env["agent"]["max_turns"] == 50 and env["agent"]["max_output_tokens"] == 65536
    assert env["agent"]["timeout"]["rollout"] == 3600


def test_plan_v4_limits_flow():
    args = parse("--reply-limit", "32768", "--context-limit", "131072", "--rollout-timeout", "10800")
    turns, context, reply, timeout = runner.episode_limits(args)
    assert (turns, context, reply, timeout) == (50, 131072, 32768, 10800)
    env = sdpo_ab_replay.environment(["finance.ap_turnover_analysis"], turns, max_output_tokens=context, rollout_timeout=timeout)
    assert env["agent"]["max_output_tokens"] == 131072 and env["agent"]["timeout"]["rollout"] == 10800


def test_reply_above_context_refused():
    with pytest.raises(ValueError):
        runner.episode_limits(parse("--reply-limit", "70000", "--context-limit", "65536"))


def test_card_thinking_sampling():
    args = parse("--sampling", "card-thinking", "--thinking", "on")
    assert runner.collect_sampling(args.sampling) == {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                                                      "presence_penalty": 1.5, "repetition_penalty": 1.0}
    assert set(COLLECT_SAMPLINGS) == {"protocol", "card", "card-thinking"}
    runner.check_collect_options(args)


def test_adapter_effect_requires_the_engine_to_change():
    before = {"rollout": [-1.0, -2.0], "max_abs_delta": 0.01}
    moved = {"rollout": [-1.2, -2.0], "max_abs_delta": 0.02}
    assert runner.adapter_effect(before, moved)["rollout_logprob_max_abs_change"] == pytest.approx(0.2)
    with pytest.raises(ValueError, match="unchanged"):
        runner.adapter_effect(before, {"rollout": [-1.0, -2.0], "max_abs_delta": 0.3})


def test_runaway_is_thinking_aware():
    long_thinking = {"raw": "x" * 20000, "visible_chars": 300, "thinking_closed": True, "limit_reason": None}
    long_visible = {"raw": "y" * 9000, "visible_chars": 9000, "thinking_closed": True, "limit_reason": None}
    unclosed = {"raw": "z" * 100, "visible_chars": 0, "thinking_closed": False, "limit_reason": "reply_limit"}
    thinking_off = {"raw": "w" * 8500, "limit_reason": None}
    assert not runner.runaway([long_thinking]) and runner.legacy_runaway([long_thinking])
    assert runner.runaway([long_visible]) and runner.runaway([unclosed]) and runner.runaway([thinking_off])
    assert runner.reply_limit_cut([unclosed]) and not runner.reply_limit_cut([long_visible])
