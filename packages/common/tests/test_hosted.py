"""Hosted model and provider-endpoint ownership boundaries."""

from typing import Any, cast

import pytest
from posttrain.common import HostedModel, ProviderEndpointProfile


def test_hosted_model_rejects_service_transport_capabilities() -> None:
    with pytest.raises(ValueError, match="provider endpoint profile"):
        HostedModel(
            "hosted-models/test@1",
            "1",
            "provider/model",
            4096,
            {"reasoning": True, "structured-output": "json-object"},
        )


def test_provider_endpoint_profile_rejects_unknown_transport() -> None:
    with pytest.raises(ValueError, match="unsupported provider structured-output transport"):
        ProviderEndpointProfile(cast(Any, "yaml-mode"))
