"""Check explicit configuration and real requested probe populations."""

import pytest
from sdpo_speculation import batch_orders, speculative_config, validate_qualification_request


def test_requested_40_is_actually_probed():
    orders = batch_orders(40)
    assert [len(x) for x in orders] == [1, 2, 4, 8, 16, 32, 40, 1]
    assert all(0 in x for x in orders)
    assert orders[2] == [1, 0, 1, 1]  # Retain the previous reordered-batch witness.


def test_small_limit_is_respected():
    assert max(map(len, batch_orders(3))) == 3
    with pytest.raises(ValueError):
        batch_orders(0)


def test_no_silent_speculation_fallback():
    assert speculative_config("none") is None
    assert speculative_config("mtp") == {"method": "mtp", "num_speculative_tokens": 2}
    assert speculative_config("mtp", 1) == {"method": "mtp", "num_speculative_tokens": 1}
    with pytest.raises(ValueError):
        speculative_config("mtp", 0)
    with pytest.raises(ValueError):
        speculative_config("unknown")


@pytest.mark.parametrize("field,value", [
    ("speculative_config", {"method": "mtp", "num_speculative_tokens": 2}),
    ("concurrency", 32), ("context", 32768), ("prefix_caching", True),
    ("chunked_prefill", True), ("enforce_eager", False), ("optimizer_updates", 1),
])
def test_one_token_receipts_cannot_reuse_other_operating_points(field, value):
    request = {"method": "mtp", "speculative_config": speculative_config("mtp", 1),
               "concurrency": 40, "context": 65536, "optimizer_updates": 0,
               "prefix_caching": False, "chunked_prefill": False, "enforce_eager": True}
    options = dict(prefix_caching=False, chunked_prefill=False, enforce_eager=True)
    validate_qualification_request(request, 1, **options)
    request[field] = value
    with pytest.raises(ValueError, match="operating point differs"):
        validate_qualification_request(request, 1, **options)


def test_plain_decoding_requires_actual_non_speculative_request_evidence():
    request = {"method": "none", "speculative_config": None, "concurrency": 40,
               "context": 65536, "optimizer_updates": 0, "prefix_caching": False,
               "chunked_prefill": False, "enforce_eager": False}
    options = dict(prefix_caching=False, chunked_prefill=False, enforce_eager=False, method="none")
    validate_qualification_request(request, 2, **options)
    request["speculative_config"] = speculative_config("mtp", 2)
    with pytest.raises(ValueError, match="operating point differs"):
        validate_qualification_request(request, 2, **options)
