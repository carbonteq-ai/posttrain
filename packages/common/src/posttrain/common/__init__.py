"""Framework-neutral contracts for the post-training platform."""

from .artifacts import (
    ArtifactRef,
    HubModelRef,
    JsonValue,
    LocalArtifactRef,
    ProducedArtifact,
    PublishedArtifact,
    StoredArtifactRef,
    TrackioArtifactRef,
)
from .catalog import Catalog, CatalogLayer, CatalogRef, Resolved
from .errors import ContractError, OperationCancelled, PostTrainError
from .execution import (
    CancellationToken,
    EventObservation,
    MetricBatchObservation,
    MetricObservation,
    NullObserver,
    Observer,
    RunContext,
    TraceObservation,
)
from .models import (
    ChatTemplate,
    ConversationProfile,
    ModelCapabilities,
    ModelVariant,
    ReasoningMode,
    RendererContract,
    ToolCallProtocol,
)
from .selections import ExecutionTarget, InferenceBinding, Workload

__all__ = [
    "ArtifactRef",
    "CancellationToken",
    "Catalog",
    "CatalogLayer",
    "CatalogRef",
    "ChatTemplate",
    "ContractError",
    "EventObservation",
    "ExecutionTarget",
    "HubModelRef",
    "InferenceBinding",
    "JsonValue",
    "LocalArtifactRef",
    "MetricObservation",
    "MetricBatchObservation",
    "ModelCapabilities",
    "ModelVariant",
    "ConversationProfile",
    "ReasoningMode",
    "RendererContract",
    "ToolCallProtocol",
    "NullObserver",
    "Observer",
    "OperationCancelled",
    "PostTrainError",
    "ProducedArtifact",
    "PublishedArtifact",
    "Resolved",
    "RunContext",
    "StoredArtifactRef",
    "TraceObservation",
    "TrackioArtifactRef",
    "Workload",
]
