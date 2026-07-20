from __future__ import annotations

import tempfile
import unittest
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from posttrain.common import (
    ContractError,
    EventObservation,
    ExecutionContext,
    HubModelRef,
    Invocation,
    Job,
    JobAction,
    MetricBatchObservation,
    MetricObservation,
    OperationCancelled,
    ProducedArtifact,
    RunAttempt,
    TraceObservation,
)
from posttrain.common.profiles import FOUNDATION_PROFILES


@dataclass
class RecordingObserver:
    events: list[EventObservation] = field(default_factory=list)
    metric_observations: list[MetricObservation] = field(default_factory=list)

    def event(self, observation: EventObservation) -> None:
        self.events.append(observation)

    def metric(self, observation: MetricObservation) -> None:
        self.metric_observations.append(observation)

    def metrics(self, observation: MetricBatchObservation) -> None:
        for name, value in observation.values.items():
            self.metric_observations.append(MetricObservation(name, value, observation.step, observation.attributes))

    def trace(self, observation: TraceObservation) -> None:
        del observation

    def artifact(self, artifact: ProducedArtifact) -> None:
        del artifact


class IdentityContractTests(unittest.TestCase):
    def test_hub_model_requires_an_immutable_commit(self) -> None:
        with self.assertRaisesRegex(ContractError, "commit SHAs"):
            HubModelRef("Qwen/Qwen3.5-2B", "main")

    def test_foundation_profiles_are_pinned_and_distinct(self) -> None:
        self.assertEqual(set(FOUNDATION_PROFILES), {"qwen3.5-2b", "lfm2.5-1.2b-thinking"})
        for profile in FOUNDATION_PROFILES.values():
            self.assertEqual(len(profile.artifact.revision), 40)
            self.assertTrue(profile.instruction_tuned)
            self.assertIn(profile.default_reasoning_mode, profile.capabilities.reasoning_modes)


class ExecutionContextTests(unittest.TestCase):
    def test_emits_without_a_platform_specific_observer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            context = self._context(Path(directory))
            context.event("operation.started")
            context.metric("train/loss", 1, step=0)

    def test_records_identity_scoped_observations(self) -> None:
        observer = RecordingObserver()
        with tempfile.TemporaryDirectory() as directory:
            context = self._context(Path(directory), observer=observer)
            context.event("operation.started", {"backend": "test"})
            context.metric("train/loss", 0.5, step=1)

        self.assertEqual(observer.events[0].name, "operation.started")
        self.assertEqual(observer.metric_observations[0].step, 1)
        self.assertEqual(observer.metric_observations[0].value, 0.5)

    def test_cancellation_stops_future_observation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            context = self._context(Path(directory))
            context.cancellation.cancel()
            with self.assertRaises(OperationCancelled):
                context.event("operation.started")

    @staticmethod
    def _context(workspace: Path, observer: RecordingObserver | None = None) -> ExecutionContext:
        commit = "a" * 40
        job = Job("gsm8k-posttraining", commit, "GSM8K post-training")
        action = JobAction(job.id, "sft", "training-sft")
        return ExecutionContext(
            job=job,
            action=action,
            invocation=Invocation(str(uuid.UUID(int=1))),
            attempt=RunAttempt(str(uuid.UUID(int=2)), 1),
            workspace=workspace.resolve(),
            observer=observer or RecordingObserver(),
            clock=lambda: datetime(2026, 7, 20, tzinfo=UTC),
        )


if __name__ == "__main__":
    unittest.main()
