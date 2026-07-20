"""Framework-neutral contracts for the post-training platform."""

from .artifacts import (
    ArtifactRef,
    HubModelRef,
    JsonValue,
    LocalArtifactRef,
    ProducedArtifact,
    TrackioArtifactRef,
)
from .errors import ContractError, OperationCancelled, PostTrainError
from .execution import (
    CancellationToken,
    EventObservation,
    ExecutionContext,
    MetricBatchObservation,
    MetricObservation,
    NullObserver,
    Observer,
    TraceObservation,
)
from .jobs import Invocation, Job, JobAction, RunAttempt
from .models import ModelCapabilities, ModelProfile

__all__ = [
    "ArtifactRef",
    "CancellationToken",
    "ContractError",
    "EventObservation",
    "ExecutionContext",
    "HubModelRef",
    "Invocation",
    "Job",
    "JobAction",
    "JsonValue",
    "LocalArtifactRef",
    "MetricObservation",
    "MetricBatchObservation",
    "ModelCapabilities",
    "ModelProfile",
    "NullObserver",
    "Observer",
    "OperationCancelled",
    "PostTrainError",
    "ProducedArtifact",
    "RunAttempt",
    "TraceObservation",
    "TrackioArtifactRef",
]
