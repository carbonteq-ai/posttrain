"""Paid inference reservations remain below the selected run ceiling."""

from decimal import Decimal

import httpx
import pytest
from posttrain.jobs.providers.cost_control import (
    CostLedger,
    InferenceCostLimitError,
    TokenPrices,
    metered_openai_gateway,
)


def test_concurrent_reservations_cannot_cross_limit() -> None:
    ledger = CostLedger(5_000, TokenPrices(Decimal("0.000001"), Decimal("0.000001")))
    payload = {"max_tokens": 100}
    first = ledger.reserve(b"{}", payload, 100)
    with pytest.raises(InferenceCostLimitError, match="before dispatch"):
        ledger.reserve(b"{}", payload, 100)
    ledger.settle(first, {"prompt_tokens": 2, "completion_tokens": 2})


def test_gateway_uses_local_credential_and_blocks_before_upstream_dispatch() -> None:
    upstream_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        upstream_requests.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 2},
            },
        )

    with httpx.Client(
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer upstream-secret"},
    ) as upstream:
        ledger = CostLedger(300, TokenPrices(Decimal("0.00000005"), Decimal("0.00000016")))
        with metered_openai_gateway(
            upstream=upstream,
            upstream_base_url="https://openrouter.ai/api/v1",
            model="provider/model",
            maximum_output_tokens=1_000,
            ledger=ledger,
        ) as endpoint:
            accepted = httpx.post(
                f"{endpoint.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {endpoint.api_key}"},
                json={"model": endpoint.model, "messages": [], "max_tokens": 10},
            )
            assert accepted.status_code == 200
            assert upstream_requests[0].headers["Authorization"] == "Bearer upstream-secret"
            assert endpoint.api_key != "upstream-secret"

            denied = httpx.post(
                f"{endpoint.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {endpoint.api_key}"},
                json={"model": endpoint.model, "messages": [], "max_tokens": 1_000},
            )
            assert denied.status_code == 402
            assert denied.json()["error"]["code"] == "posttrain_judge_cost_limit"
    assert len(upstream_requests) == 1
