import asyncio
import hashlib
import json

import pytest
from sdpo_native_mixed_gate import MixedPrefixMismatch, qualify


class Rollout:
    temperature = 1.

    def __init__(self, mismatch=False):
        self.calls = 0
        self.mismatch = mismatch

    async def generate(self, prefix, max_tokens):
        assert self.temperature == 0.
        assert max_tokens in (64, 8192)
        self.calls += 1
        await asyncio.sleep(0)
        delta = .000001 if self.mismatch and self.calls > 4 else 0.
        return [prefix[0], 2], True, {"sampled_logprobs": [-.1 + delta, -.2], "num_cached_tokens": 0}


def fixture():
    def case(i):
        ids = [i] * (100+i)
        return {"prefix_ids": ids, "prefix_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                "budget": 8192}
    return {"cases": [case(i) for i in range(4)], "backgrounds": [case(i+4) for i in range(16)]}


def test_full_budgets_mixed40_orders_and_temperature_restore():
    rollout = Rollout()
    result = asyncio.run(qualify(rollout, fixture()))
    assert result["qualified"] and result["peak_in_flight"] == 40
    assert len(result["records"]) == 16 and rollout.temperature == 1.


def test_mixed_native_probability_mismatch_is_not_tolerated():
    rollout = Rollout(mismatch=True)
    with pytest.raises(MixedPrefixMismatch) as caught:
        asyncio.run(qualify(rollout, fixture()))
    assert not caught.value.receipt["qualified"] and rollout.temperature == 1.
