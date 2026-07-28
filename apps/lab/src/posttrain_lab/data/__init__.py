"""Job-owned dataset composition built on reusable engine contracts."""

from .gsm8k import GSM8K_REVISION, GSM8KSupervisedSource
from .halcyon_graphql import HALCYON_GRAPHQL_REVISION, HalcyonGraphQLSupervisedSource
from .smol_smoltalk import SMOL_SMOLTALK_REVISION, SmolSmolTalkSupervisedSource

__all__ = [
    "GSM8K_REVISION",
    "GSM8KSupervisedSource",
    "HALCYON_GRAPHQL_REVISION",
    "HalcyonGraphQLSupervisedSource",
    "SMOL_SMOLTALK_REVISION",
    "SmolSmolTalkSupervisedSource",
]
