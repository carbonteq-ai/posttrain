import asyncio
import hashlib
import json

import pytest
from sdpo_native_prefix_gate import NativePrefixMismatch, qualify


class Rollout:
    def __init__(self, jitter=False):
        self.engine = self
        self.jitter = jitter
        self.calls = 0

    async def reset_prefix_cache(self):
        return True

    async def generate(self, ids, max_tokens):
        self.calls += 1
        await asyncio.sleep(0)
        delta = 1e-7 if self.jitter and self.calls > 4 else 0.
        return [ids[-1], 9], True, {"sampled_logprobs": [-.1 + delta, -.2], "num_cached_tokens": 0}


def fixture():
    cases = []
    for i in range(4):
        ids = [i] * 2000
        cases.append({"task": f"task{i}", "prefix_ids": ids,
                      "prefix_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest()})
    return {"cases": cases}


def test_real_forty_request_dispatch_and_all_cache_states_are_checked():
    result = asyncio.run(qualify(Rollout(), fixture()))
    assert result["qualified"] and result["peak_in_flight"] == 40
    assert len(result["records"]) == 88


def test_tiny_probability_difference_fails_instead_of_weakening_exact_threshold():
    with pytest.raises(NativePrefixMismatch) as caught:
        asyncio.run(qualify(Rollout(jitter=True), fixture()))
    assert not caught.value.receipt["qualified"]


def test_changed_native_fixture_tokens_are_rejected_before_generation():
    values = fixture()
    values["cases"][0]["prefix_ids"][0] = 77
    with pytest.raises(ValueError, match="fixture identity"):
        asyncio.run(qualify(Rollout(), values))


def test_probe_forces_greedy_and_restores_online_temperature_on_success_or_failure():
    class OnlineRollout(Rollout):
        temperature = 1.

        async def generate(self, ids, max_tokens):
            assert self.temperature == 0.
            return await super().generate(ids, max_tokens)

    for jitter in (False, True):
        rollout = OnlineRollout(jitter=jitter)
        try:
            asyncio.run(qualify(rollout, fixture()))
        except NativePrefixMismatch:
            assert jitter
        assert rollout.temperature == 1.
