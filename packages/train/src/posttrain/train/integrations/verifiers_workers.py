"""Native Verifiers worker configuration for environment-driven training."""

from __future__ import annotations

from typing import Any

from posttrain.common import ModelVariant

from ..profiles import TrainingRenderer
from ..rendering import create_renderer_config


def create_verifiers_train_client_config(
    *,
    base_url: str,
    renderer_model_name: str,
    model: ModelVariant,
    renderer: TrainingRenderer,
    api_key_var: str = "POSTTRAIN_INPROCESS_POLICY",
    multiplex: int = 256,
) -> Any:
    """Build the serializable exact-token client used by native env workers.

    ``renderer_model_name`` is a resolved immutable artifact or local snapshot
    path supplied by composition. It must not be inferred from a mutable model
    alias here.
    """

    try:
        from verifiers.v1.configs.client import TrainClientConfig
    except ImportError as error:
        raise RuntimeError("install the Verifiers integration dependencies") from error
    if "chat_template" not in TrainClientConfig.model_fields:
        raise RuntimeError(
            "native rollout workers require a Verifiers TrainClientConfig with exact chat-template support"
        )
    values: dict[str, Any] = {
        "base_url": base_url,
        "api_key_var": api_key_var,
        "renderer": create_renderer_config(model, renderer),
        "renderer_model_name": renderer_model_name,
        "chat_template": model.conversation.chat_template.text(),
        "multiplex": multiplex,
    }
    return TrainClientConfig(**values)


__all__ = ["create_verifiers_train_client_config"]
