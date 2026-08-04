"""Tests for shared framework contracts."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from posttrain.common import (
    Catalog,
    CatalogRef,
    ContractError,
    EventObservation,
    ExecutionTarget,
    HubModelRef,
    InferenceBinding,
    MetricBatchObservation,
    MetricObservation,
    OperationCancelled,
    ProducedArtifact,
    RunContext,
    TraceObservation,
    Workload,
)
from posttrain.common.variants import FOUNDATION_VARIANTS


@dataclass
class RecordingObserver:
    events: list[EventObservation] = field(default_factory=list)
    metric_observations: list[MetricObservation] = field(default_factory=list)
    traces: list[TraceObservation] = field(default_factory=list)

    def event(self, observation: EventObservation) -> None:
        self.events.append(observation)

    def metric(self, observation: MetricObservation) -> None:
        self.metric_observations.append(observation)

    def metrics(self, observation: MetricBatchObservation) -> None:
        for name, value in observation.values.items():
            self.metric_observations.append(MetricObservation(name, value, observation.step, observation.attributes))

    def trace(self, observation: TraceObservation) -> None:
        self.traces.append(observation)

    def artifact(self, artifact: ProducedArtifact) -> None:
        del artifact


class IdentityContractTests(unittest.TestCase):
    def test_hub_model_requires_an_immutable_commit(self) -> None:
        with self.assertRaisesRegex(ContractError, "commit SHAs"):
            HubModelRef("Qwen/Qwen3.5-2B", "main")

    def test_foundation_profiles_are_pinned_and_distinct(self) -> None:
        self.assertEqual(
            set(FOUNDATION_VARIANTS),
            {
                "qwen3.5-0.8b",
                "qwen3.5-2b",
                "lfm2.5-1.2b-thinking",
                "gemma4-12b-it",
                "gemma4-e4b-it",
            },
        )
        for variant in FOUNDATION_VARIANTS.values():
            self.assertEqual(len(variant.base.revision), 40)
            self.assertTrue(variant.instruction_tuned)
            self.assertEqual(
                variant.conversation.reasoning_mode(variant.default_reasoning_mode).id,
                variant.default_reasoning_mode,
            )
            self.assertIn(variant.conversation.chat_template.source, {"tokenizer", "package"})

    def test_model_variants_carry_explicit_renderer_and_base_contracts(self) -> None:
        foundation = FOUNDATION_VARIANTS["qwen3.5-2b"]
        self.assertEqual(foundation.artifact, foundation.base)
        self.assertEqual(foundation.form, "foundation")
        self.assertEqual(foundation.revision, foundation.base.revision)
        self.assertEqual(foundation.renderer_contract, "qwen3.5-tools@1")
        self.assertEqual(foundation.artifact_uri, foundation.base.uri)

        with self.assertRaisesRegex(ContractError, "pinned base artifact"):
            replace(foundation, artifact=HubModelRef("Qwen/Qwen3.5-2B", "b" * 40))

    def test_workload_requires_an_ordered_unique_concurrency_sweep(self) -> None:
        workload = Workload("workloads/capacity", "1", {"context_window": 32_768}, (1, 2, 4, 8))

        self.assertEqual(workload.concurrency, (1, 2, 4, 8))
        with self.assertRaisesRegex(ContractError, "unique"):
            replace(workload, concurrency=(1, 2, 2))
        with self.assertRaisesRegex(ContractError, "strictly increasing"):
            replace(workload, concurrency=(1, 4, 2))


class CanonicalRunContextTests(unittest.TestCase):
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

    def test_runtime_phase_records_paired_boundaries_and_failures(self) -> None:
        observer = RecordingObserver()
        failed_id: str | None = None
        timestamps = iter(datetime(2026, 7, 20, tzinfo=UTC) + timedelta(seconds=offset) for offset in range(4))
        with tempfile.TemporaryDirectory() as directory:
            context = RunContext(
                project_id="tests",
                work_package_id="train/context",
                run_id="00000000-0000-0000-0000-000000000002",
                job_kind="train.sft",
                job_definition_version="1",
                workspace=Path(directory).resolve(),
                observer=observer,
                clock=lambda: next(timestamps),
            )
            with context.phase("initialization") as completed_id:
                pass
            with self.assertRaisesRegex(RuntimeError, "phase failed"):
                with context.phase("actor_update") as failed_id:
                    raise RuntimeError("phase failed")

        self.assertEqual(
            [event.name for event in observer.events],
            [
                "runtime_phase_started",
                "runtime_phase_completed",
                "runtime_phase_started",
                "runtime_phase_failed",
            ],
        )
        self.assertEqual(observer.events[0].attributes["phase_id"], completed_id)
        self.assertEqual(observer.events[1].attributes["phase_id"], completed_id)
        self.assertIsNotNone(failed_id)
        self.assertEqual(observer.events[2].attributes["phase_id"], failed_id)
        self.assertEqual(observer.events[3].attributes["error_type"], "RuntimeError")

    @staticmethod
    def _context(workspace: Path, observer: RecordingObserver | None = None) -> RunContext:
        return RunContext(
            project_id="tests",
            work_package_id="train/context",
            run_id="00000000-0000-0000-0000-000000000002",
            job_kind="train.sft",
            job_definition_version="1",
            workspace=workspace.resolve(),
            observer=observer or RecordingObserver(),
            clock=lambda: datetime(2026, 7, 20, tzinfo=UTC),
        )


class RunContextTests(unittest.TestCase):
    def test_emits_events_metrics_and_traces_through_the_observer(self) -> None:
        observer = RecordingObserver()
        with tempfile.TemporaryDirectory() as directory:
            context = RunContext(
                project_id="memory-agent",
                work_package_id="screen/foundations",
                run_id="00000000-0000-0000-0000-000000000001",
                job_kind="serve.benchmark",
                job_definition_version="posttrain-serve@0.1.0",
                workspace=Path(directory).resolve(),
                observer=observer,
                clock=lambda: datetime(2026, 7, 21, tzinfo=UTC),
            )
            context.event("operation.started")
            context.metric("serve/ttft_ms", 12.5, step=1)
            context.trace(TraceObservation("serve.request", "request-1", {"status": "ok"}))

        self.assertEqual(observer.events[0].occurred_at, datetime(2026, 7, 21, tzinfo=UTC))
        self.assertEqual(observer.metric_observations[0].name, "serve/ttft_ms")
        self.assertEqual(observer.traces[0].external_id, "request-1")

    def test_null_observer_supports_the_same_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            context = RunContext(
                project_id="memory-agent",
                work_package_id="screen/foundations",
                run_id="00000000-0000-0000-0000-000000000001",
                job_kind="serve.benchmark",
                job_definition_version="posttrain-serve@0.1.0",
                workspace=Path(directory).resolve(),
            )
            context.event("operation.started")
            context.metric("serve/ttft_ms", 12.5)
            context.trace(TraceObservation("serve.request", "request-1", {"status": "ok"}))


class CatalogTests(unittest.TestCase):
    def test_overlay_wins_and_records_its_source(self) -> None:
        model = FOUNDATION_VARIANTS["qwen3.5-2b"]
        target = ExecutionTarget("targets/rtx3070ti-8gb", "1", "cuda", 8)
        base_binding = InferenceBinding(
            "inference/qwen-screen",
            "1",
            model,
            "vllm@0.10.2",
            model.renderer_contract,
            {"gpu_memory_utilization": 0.8},
            {"temperature": 0.0},
            target,
            ("screen",),
        )
        overlay_binding = InferenceBinding(
            "inference/qwen-screen",
            "2",
            model,
            "vllm@0.10.2",
            model.renderer_contract,
            {"gpu_memory_utilization": 0.7},
            {"temperature": 0.0},
            target,
            ("screen",),
        )
        catalog = Catalog.open(
            {
                "model": {model.id: model},
                "target": {target.id: target},
                "inference": {base_binding.id: base_binding},
            },
            overlays=[{"layer_id": "memory-agent", "inference": {overlay_binding.id: overlay_binding}}],
            scope="memory-agent",
        )
        self.assertEqual(catalog.base_id, "base")
        self.assertEqual(catalog.overlay_ids, ("memory-agent",))

        resolved = catalog.resolve(CatalogRef("inference", base_binding.id))

        self.assertEqual(resolved.value, overlay_binding)
        self.assertEqual(resolved.source_layer, "overlay")
        self.assertEqual(resolved.overlay_id, "memory-agent")
        self.assertEqual(
            catalog.resolve(CatalogRef("model", model.id)).source_layer,
            "base",
        )
        self.assertTrue(catalog.contains(CatalogRef("target", target.id)))
        self.assertEqual(len(catalog.list("inference")), 1)
        self.assertEqual(
            catalog.transitive_refs((CatalogRef("inference", overlay_binding.id),)),
            (
                CatalogRef("inference", overlay_binding.id),
                CatalogRef("model", model.id),
                CatalogRef("target", target.id),
            ),
        )
        self.assertEqual(catalog.refs_for_values((target,)), (CatalogRef("target", target.id),))

    def test_json_loader_builds_model_target_and_inference_selections(self) -> None:
        revision = "a" * 40
        payload = {
            "layer_id": "framework-v1",
            "inference": {
                "inference/qwen-screen": {
                    "revision": "1",
                    "model": "models/qwen-2b@bf16",
                    "backend": "vllm@0.10.2",
                    "renderer": "qwen-tools@1",
                    "engine": {"gpu_memory_utilization": 0.8},
                    "sampling": {"temperature": 0.0},
                    "target": "targets/rtx3070ti-8gb",
                    "purpose": ["screen"],
                }
            },
            "target": {
                "targets/rtx3070ti-8gb": {
                    "revision": "1",
                    "device_class": "cuda",
                    "memory_gb": 8,
                }
            },
            "model": {
                "models/qwen-2b@bf16": {
                    "artifact": {
                        "kind": "hub",
                        "repo_id": "Qwen/Qwen3.5-2B",
                        "revision": revision,
                    },
                    "form": "foundation",
                    "weight_precision": "bf16",
                    "family": "qwen3.5",
                    "parameters": 2_000_000_000,
                    "instruction_tuned": True,
                    "renderer_contract": "qwen3.5-tools@1",
                    "capabilities": {
                        "modalities": ["text"],
                        "native_context_window": 32768,
                    },
                }
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            catalog = Catalog.open(path, scope="memory-agent")

        resolved = catalog.resolve(CatalogRef("inference", "inference/qwen-screen"))
        self.assertIsInstance(resolved.value, InferenceBinding)
        assert isinstance(resolved.value, InferenceBinding)
        self.assertEqual(resolved.value.model.id, "models/qwen-2b@bf16")
        self.assertEqual(resolved.value.target.memory_gb, 8)
        self.assertEqual(resolved.source_layer, "base")

    def test_catalog_schema_rejects_unknown_external_fields(self) -> None:
        with self.assertRaisesRegex(ContractError, "invalid catalog entry"):
            Catalog.open(
                {
                    "target": {
                        "targets/local-cuda-8gb": {
                            "revision": "1",
                            "device_class": "nvidia-cuda",
                            "memory_gb": 8,
                            "gpu_memroy_typo": "8gb",
                        }
                    }
                },
                scope="memory-agent",
            )


if __name__ == "__main__":
    unittest.main()
