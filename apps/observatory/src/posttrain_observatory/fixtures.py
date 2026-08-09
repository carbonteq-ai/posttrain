"""Deterministic product fixtures used by local preview and end-to-end tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from posttrain.tracking import (
    ArtifactIntegrityResult,
    ArtifactLink,
    ArtifactSet,
    EventRecord,
    MetricPoint,
    MetricSeries,
    RunDataSource,
    RunDetail,
    RunQuery,
    RunSummary,
    StoredArtifact,
    TracePage,
    TraceQuery,
    TraceRecord,
    TrackingCapabilities,
)

NOW = datetime(2026, 7, 22, 4, 0, tzinfo=UTC)


def _series(
    name: str,
    values: tuple[float, ...],
    *,
    observed_start: datetime | None = None,
) -> MetricSeries:
    return MetricSeries(
        name=name,
        points=tuple(
            MetricPoint(
                value=value,
                step=index * 10,
                observed_at=(observed_start + timedelta(minutes=index * 2 - 1) if observed_start is not None else None),
            )
            for index, value in enumerate(values, 1)
        ),
    )


def _summary(run_id: str, job_kind: str, stage: str, index: int, status: str = "succeeded") -> RunSummary:
    started = NOW - timedelta(minutes=index * 17)
    work_package_id = {
        "train": "train/reward-v2",
        "screen": "screen/serving-capacity-v1",
        "qualify": "qualify/reward-v2",
    }[stage]
    return RunSummary(
        provider="fixture",
        provider_run_id=f"fixture-{run_id}",
        run_id=run_id,
        display_name=run_id.rsplit("/", 1)[-1].replace("-", " ").title(),
        project_id="projects/automation-agent",
        work_package_id=work_package_id,
        stage=stage,  # type: ignore[arg-type]
        job_kind=job_kind,
        job_definition_version=f"{job_kind}@1",
        status=status,  # type: ignore[arg-type]
        started_at=started,
        finished_at=started + timedelta(minutes=12, seconds=index * 4),
    )


def _runtime_phase_events(summary: RunSummary) -> tuple[EventRecord, ...]:
    finished_at = summary.finished_at
    if finished_at is None:
        raise AssertionError("fixture runs are complete")
    start = summary.started_at
    boundaries = [
        ("runtime_phase_started", "operation", "operation-1", start),
    ]
    if summary.job_kind.startswith("train."):
        boundaries.extend(
            [
                ("runtime_phase_started", "model_loading", "model-loading-1", start),
                (
                    "runtime_phase_completed",
                    "model_loading",
                    "model-loading-1",
                    start + timedelta(minutes=2),
                ),
                (
                    "runtime_phase_started",
                    "actor_update",
                    "actor-update-1",
                    start + timedelta(minutes=2),
                ),
                (
                    "runtime_phase_completed",
                    "actor_update",
                    "actor-update-1",
                    start + timedelta(minutes=10),
                ),
                (
                    "runtime_phase_started",
                    "artifact_export",
                    "artifact-export-1",
                    start + timedelta(minutes=10),
                ),
                (
                    "runtime_phase_completed",
                    "artifact_export",
                    "artifact-export-1",
                    finished_at,
                ),
            ]
        )
        if summary.job_kind == "train.grpo":
            boundaries.extend(
                [
                    (
                        "runtime_phase_started",
                        "rollout",
                        "rollout-1",
                        start + timedelta(minutes=3, seconds=30),
                    ),
                    (
                        "runtime_phase_completed",
                        "rollout",
                        "rollout-1",
                        start + timedelta(minutes=5, seconds=30),
                    ),
                    (
                        "runtime_phase_started",
                        "rollout",
                        "rollout-2",
                        start + timedelta(minutes=7, seconds=30),
                    ),
                    (
                        "runtime_phase_completed",
                        "rollout",
                        "rollout-2",
                        start + timedelta(minutes=9, seconds=30),
                    ),
                ]
            )
    elif summary.job_kind.startswith("eval."):
        boundaries.extend(
            [
                ("runtime_phase_started", "evaluation", "evaluation-1", start),
                (
                    "runtime_phase_completed",
                    "evaluation",
                    "evaluation-1",
                    finished_at,
                ),
            ]
        )
    else:
        boundaries.extend(
            [
                (
                    "runtime_phase_started",
                    "backend_execution",
                    "backend-execution-1",
                    start,
                ),
                (
                    "runtime_phase_completed",
                    "backend_execution",
                    "backend-execution-1",
                    finished_at,
                ),
            ]
        )
    boundaries.append(("runtime_phase_completed", "operation", "operation-1", finished_at))
    return tuple(
        EventRecord(
            name=name,
            occurred_at=occurred_at,
            attributes={"phase": phase, "phase_id": phase_id},
        )
        for name, phase, phase_id, occurred_at in sorted(boundaries, key=lambda item: item[3])
    )


class FixtureRunDataSource(RunDataSource):
    def __init__(self) -> None:
        specs = (
            ("runs/sft-calm-harbor", "train.sft", "train"),
            ("runs/dpo-amber-field", "train.dpo", "train"),
            ("runs/grpo-silver-pine", "train.grpo", "train"),
            ("runs/eval-violet-river", "eval.domain", "qualify"),
            ("runs/serve-cedar-point", "serve.benchmark", "screen"),
            ("runs/custom-orbit", "custom.team_job", "train"),
        )
        self._details: dict[str, RunDetail] = {}
        self._metrics: dict[str, dict[str, MetricSeries]] = {}
        self._traces: dict[str, tuple[TraceRecord, ...]] = {}
        self._artifacts: dict[str, ArtifactSet] = {}
        metric_values = {
            "train.sft": {
                "train/loss": (1.42, 1.08, 0.82, 0.61, 0.47, 0.39),
                "train/final_loss": (0.39,),
                "train/learning_rate": (0.00002, 0.00008, 0.0001, 0.00008, 0.00004, 0.00001),
                "train/grad_norm": (1.8, 1.5, 1.3, 1.1, 1.0, 0.94),
                "train/mean_token_accuracy": (0.64, 0.71, 0.76, 0.79, 0.82, 0.84),
                "train/samples_per_second": (0.37,),
                "train/peak_gpu_memory_gib": (3.6,),
            },
            "train.dpo": {
                "train/loss": (0.91, 0.77, 0.64, 0.56, 0.49),
                "train/rewards/margins": (-0.05, 0.08, 0.17, 0.23, 0.31),
                "train/rewards/chosen": (0.22, 0.29, 0.36, 0.42, 0.48),
                "train/rewards/rejected": (0.27, 0.21, 0.19, 0.19, 0.17),
                "train/rewards/accuracies": (0.0, 0.5, 1.0, 1.0, 1.0),
            },
            "train.grpo": {
                "train/rl/reward_mean": (0.31, 0.38, 0.47, 0.54, 0.62, 0.68),
                "train/rl/reward_std": (0.29, 0.27, 0.24, 0.23, 0.21, 0.20),
                "train/rl/group_zero_variance_fraction": (0.24, 0.20, 0.17, 0.14, 0.12, 0.10),
                "train/rl/policy_loss": (-0.008, -0.012, -0.017, -0.021, -0.025, -0.027),
                "train/rl/kl": (0.01, 0.018, 0.024, 0.029, 0.033, 0.036),
                "train/rl/entropy": (1.9, 1.84, 1.78, 1.71, 1.68, 1.64),
                "train/rl/clip_fraction": (0.04, 0.05, 0.07, 0.08, 0.09, 0.08),
                "train/grad_norm": (1.42, 1.31, 1.18, 1.09, 1.02, 0.97),
                "train/learning_rate": (0.00002, 0.00006, 0.0001, 0.00008, 0.00005, 0.00002),
                "train/step_time_seconds": (22.8, 22.1, 21.7, 21.4, 21.0, 20.8),
                "train/rl/rollouts_attempted": (16, 16, 16, 16, 16, 16),
                "train/rl/rollouts_completed": (16, 16, 16, 16, 16, 16),
                "train/rl/rollouts_failed": (0, 0, 0, 0, 0, 0),
                "train/rl/rollouts_truncated": (1, 1, 0, 0, 0, 0),
                "train/rl/rollouts_unscorable": (0, 0, 0, 0, 0, 0),
                "train/rl/completion_tokens_mean": (188, 194, 201, 208, 214, 219),
                "train/rl/completion_tokens_max": (384, 402, 417, 431, 445, 452),
                "train/rl/completion_truncation_rate": (0.0625, 0.0625, 0, 0, 0, 0),
                "train/rl/rollout_tokens_per_second": (143, 149, 154, 160, 166, 171),
                "train/rl/sampling_logp_delta_mean": (0.014, 0.012, 0.010, 0.009, 0.008, 0.008),
                "train/rl/sampling_logp_delta_max": (0.081, 0.074, 0.068, 0.061, 0.058, 0.055),
                "train/rl/importance_sampling_ratio_mean": (1.01, 1.01, 1.00, 1.00, 1.00, 1.00),
                "train/rl/importance_sampling_ratio_min": (0.88, 0.89, 0.90, 0.91, 0.92, 0.92),
                "train/rl/importance_sampling_ratio_max": (1.14, 1.13, 1.11, 1.10, 1.09, 1.08),
                "train/rl/time/rollout_seconds": (15.6, 15.1, 14.8, 14.5, 14.2, 14.0),
                "train/rl/time/reward_seconds": (1.8, 1.8, 1.7, 1.7, 1.7, 1.6),
                "train/rl/time/actor_update_seconds": (4.7, 4.6, 4.5, 4.5, 4.4, 4.4),
                "train/rl/time/weight_sync_seconds": (0.7, 0.6, 0.6, 0.5, 0.5, 0.5),
            },
            "eval.domain": {
                "eval/run/rollouts_complete": (24.0,),
                "eval/run/rollouts_failed": (1.0,),
                "eval/run/rollouts_truncated": (2.0,),
                "eval/trace_sync_complete": (1.0,),
            },
            "serve.benchmark": {
                "serve/requests": (4,),
                "serve/output_tokens": (232,),
                "serve/elapsed_seconds": (4,),
                "serve/output_token_throughput": (58,),
                "serve/p50_ttft": (0.42,),
                "serve/p95_ttft": (0.70,),
                "serve/p50_tpot": (0.018,),
                "serve/p95_tpot": (0.024,),
                "serve/context_window": (32_768,),
                "serve/concurrency": (4,),
                "serve/peak_gpu_memory_gib": (7.2,),
                "serve/corpus_records_measured": (4,),
                "serve/input_tokens_mean": (612,),
                "serve/input_tokens_p95": (944,),
                "serve/backend/kv_cache_peak_usage_ratio": (0.78,),
            },
            "custom.team_job": {
                "custom/quality": (0.41, 0.49, 0.53, 0.59),
                "custom/latency_ms": (940, 870, 815, 790),
                "system/gpu_utilization": (62, 74, 81, 78),
            },
        }
        system_values = {
            "system/gpu_utilization": (52, 68, 79, 84, 81, 77),
            "system/gpu_vram_used_bytes": (
                12_100_000_000,
                13_800_000_000,
                14_900_000_000,
                15_300_000_000,
                15_100_000_000,
                14_700_000_000,
            ),
            "system/cpu_percent": (31, 38, 44, 47, 42, 39),
            "system/process_rss_bytes": (
                2_900_000_000,
                3_200_000_000,
                3_500_000_000,
                3_650_000_000,
                3_620_000_000,
                3_580_000_000,
            ),
            "system/wall_time_s": (120, 240, 360, 480, 600, 720),
        }
        for index, (run_id, job_kind, stage) in enumerate(specs, 1):
            summary = _summary(run_id, job_kind, stage, index)
            values = {**system_values, **metric_values[job_kind]}
            if job_kind in {"eval.domain", "train.grpo"}:
                values = {
                    **values,
                    "tracking/traces_written": (2, 4, 6, 8, 10, 12),
                    "tracking/traces_dropped": (0, 0, 0, 0, 0, 0),
                }
            metrics = {
                name: _series(
                    name,
                    points,
                    observed_start=summary.started_at if name.startswith("system/") else None,
                )
                for name, points in values.items()
            }
            self._metrics[run_id] = metrics
            traces = (
                self._evaluation_traces()
                if job_kind in {"eval.domain", "train.grpo"}
                else self._serving_traces()
                if job_kind == "serve.benchmark"
                else ()
            )
            self._traces[run_id] = traces
            self._details[run_id] = RunDetail(
                summary=summary,
                resolved_inputs={
                    "model": {"id": "models/qwen3.5-2b@bf16"},
                    "training": {"learning_rate": 0.0001, "api_key": "fixture-secret"},
                    "execution_targets": {
                        "schema_version": 1,
                        "targets": [
                            {
                                "selection_id": (
                                    "targets/fixture-cuda-8gb"
                                    if job_kind == "serve.benchmark"
                                    else "targets/fixture-cuda-24gb"
                                ),
                                "revision": "1",
                                "roles": ["screen_inference"] if job_kind == "serve.benchmark" else ["training"],
                                "device_class": "nvidia-cuda",
                                "memory_gb": 8 if job_kind == "serve.benchmark" else 24,
                                "placement": {"world_size": 1},
                                "host_constraints": {},
                            }
                        ],
                    },
                    **(
                        {
                            "settings": {"beta": 0.1, "num_generations": 4},
                            "inference": {
                                "backend": "vllm@1",
                                "engine": {"mode": "colocate", "kv_cache_dtype": "auto"},
                            },
                        }
                        if job_kind == "train.grpo"
                        else {}
                    ),
                    **(
                        {
                            "screen_inference": {
                                "selection_id": "inference/qwen3.5-0.8b-vllm@1",
                                "revision": "1",
                                "resolved": {
                                    "backend": "vllm@0.25.1",
                                    "renderer": "qwen3.5-tools@1",
                                    "engine": {
                                        "dtype": "float16",
                                        "max_model_len": 32_768,
                                        "gpu_memory_utilization": 0.85,
                                        "max_num_seqs": 8,
                                        "max_num_batched_tokens": 4_096,
                                        "kv_cache_dtype": "fp8",
                                        "enable_chunked_prefill": True,
                                        "enforce_eager": False,
                                        "experimental_scheduler": "async",
                                    },
                                },
                            },
                            "workload": {
                                "selection_id": "workloads/representative-serving@1",
                                "revision": "1",
                                "resolved": {
                                    "requests": {
                                        "suite_id": "general-serving-v1",
                                        "shape_id": "representative-128out",
                                        "context_window": 32_768,
                                        "output_tokens": 58,
                                        "cohort": "representative",
                                        "corpus": {
                                            "id": "general-serving-v1",
                                            "revision": "1",
                                            "digest": "sha256:fixture-corpus",
                                        },
                                    },
                                    "concurrency": [1, 2, 4],
                                    "saturation_state": "saturated",
                                },
                            },
                            "target": {
                                "selection_id": "targets/fixture-cuda-8gb",
                                "revision": "1",
                                "resolved": {
                                    "device_class": "nvidia-cuda",
                                    "memory_gb": 8,
                                    "placement": {"world_size": 1},
                                },
                            },
                            "project_brief": {
                                "digest": "sha256:fixture-requirements",
                                "schema_version": 1,
                                "serving": {
                                    "required_context_tokens": 32_768,
                                    "min_sustained_output_tokens_per_second": 50,
                                    "max_p95_ttft_ms": 1_000,
                                    "max_p95_tpot_ms": 30,
                                    "max_failure_rate": 0.01,
                                },
                            },
                        }
                        if job_kind == "serve.benchmark"
                        else {}
                    ),
                    "work_package": {
                        "work_package_id": summary.work_package_id,
                        "description": (
                            "Train and compare candidate model-improvement techniques on the bounded automation task population."
                            if stage == "train"
                            else "Qualify the selected descendant against held-out automation behavior."
                        ),
                    },
                    "job_definition": {
                        "id": summary.job_definition_version,
                        "kind": job_kind,
                        "description": {
                            "train.sft": "Fit supervised demonstrations with the selected SFT training binding.",
                            "train.dpo": "Optimize preference ordering over selected chosen and rejected completions.",
                            "train.grpo": "Generate grouped rollouts and optimize the policy from verifier rewards.",
                            "eval.domain": "Measure held-out domain behavior through the selected Verifiers environment.",
                            "serve.benchmark": "Measure a bounded serving workload on the selected execution target.",
                            "custom.team_job": "Run the team-owned custom evidence probe.",
                        }[job_kind],
                    },
                },
                source_metadata={"commit": "a" * 40, "dirty": False},
                metric_names=tuple(sorted(metrics)),
                events=tuple(
                    sorted(
                        (
                            EventRecord(name="run_started", occurred_at=summary.started_at),
                            *_runtime_phase_events(summary),
                            EventRecord(
                                name="artifact_published",
                                occurred_at=summary.finished_at or summary.started_at,
                                attributes={"kind": "model" if stage == "train" else "evaluation"},
                            ),
                        ),
                        key=lambda event: event.occurred_at,
                    )
                ),
                trace_count=len(traces),
            )
            kind = "model-weights" if stage == "train" else "verifiers-evaluation"
            self._artifacts[run_id] = ArtifactSet(
                items=(
                    ArtifactLink(
                        direction="output",
                        logical_name="trained-model" if stage == "train" else "evaluation-bundle",
                        kind=kind,
                        artifact=StoredArtifact(
                            provider="fixture",
                            namespace="observatory",
                            name=run_id.replace("/", "-"),
                            version="v1",
                            digest=f"sha256:{index:064x}",
                        ),
                    ),
                )
            )

    @property
    def capabilities(self) -> TrackingCapabilities:
        return TrackingCapabilities(
            provider="fixture",
            live_metrics=True,
            live_traces=True,
            artifacts=True,
            artifact_lineage=True,
        )

    @staticmethod
    def _evaluation_traces() -> tuple[TraceRecord, ...]:
        return tuple(
            TraceRecord(
                trace_type="verifiers",
                external_id=f"rollout-{index:02d}",
                payload={
                    "task": "calendar" if index % 3 else "crm",
                    "reward": round(0.42 + index * 0.025, 3),
                    "success": index not in {4, 9},
                    "truncated": index == 7,
                    "num_tool_calls": 2 + index % 4,
                    "latency_ms": 11_600 + index * 740,
                    "tokens": 2_400 + index * 317,
                    "reward_components": {
                        "correctness": round(0.35 + index * 0.02, 3),
                        "efficiency": round(0.5 + index * 0.01, 3),
                    },
                    "messages": [
                        {"role": "user", "content": f"Complete automation task {index}."},
                        {"role": "assistant", "content": "I will inspect the tools and complete it."},
                    ],
                },
                attributes={"split": "held-out"},
            )
            for index in range(1, 13)
        )

    @staticmethod
    def _serving_traces() -> tuple[TraceRecord, ...]:
        ttft = (0.32, 0.42, 0.55, 0.70)
        tpot = (0.015, 0.018, 0.021, 0.024)
        return tuple(
            TraceRecord(
                trace_type="inference",
                external_id=f"request-{index}",
                payload={
                    "record_id": f"fixture-{index}",
                    "sweep_index": 2,
                    "concurrency": 4,
                    "warmup": False,
                    "input_tokens": 500 + index * 75,
                    "output_tokens": 58,
                    "ttft_seconds": ttft[index],
                    "tpot_seconds": tpot[index],
                    "queue_seconds": 0.001,
                    "prefill_seconds": ttft[index] - 0.001,
                    "decode_seconds": tpot[index] * 57,
                    "engine_e2e_seconds": ttft[index] + tpot[index] * 57,
                    "error_class": None,
                },
                attributes={"cohort": "representative"},
            )
            for index in range(4)
        )

    async def list_runs(self, query: RunQuery) -> tuple[RunSummary, ...]:
        values = [detail.summary for detail in self._details.values()]
        values = [
            value
            for value in values
            if (query.project_id is None or value.project_id == query.project_id)
            and (query.work_package_id is None or value.work_package_id == query.work_package_id)
            and (not query.job_kinds or value.job_kind in query.job_kinds)
            and (not query.statuses or value.status in query.statuses)
        ]
        values.sort(key=lambda value: value.started_at, reverse=True)
        return tuple(values[: query.limit])

    async def get_run(self, run_id: str) -> RunDetail:
        try:
            return self._details[run_id]
        except KeyError as error:
            raise LookupError(f"fixture run {run_id!r} does not exist") from error

    async def metric_series(self, run_id: str, names: tuple[str, ...]) -> tuple[MetricSeries, ...]:
        values = self._metrics[run_id]
        return tuple(values.get(name, MetricSeries(name=name)) for name in names)

    async def traces(self, run_id: str, query: TraceQuery) -> TracePage:
        values = self._traces[run_id]
        if query.trace_type is not None:
            values = tuple(value for value in values if value.trace_type == query.trace_type)
        offset = int(query.cursor or 0)
        page = values[offset : offset + query.limit]
        next_cursor = str(offset + query.limit) if offset + query.limit < len(values) else None
        return TracePage(items=page, next_cursor=next_cursor, live=True)

    async def artifacts(self, run_id: str) -> ArtifactSet:
        return self._artifacts[run_id]

    async def verify_artifact(self, reference, *, deep: bool = False) -> ArtifactIntegrityResult:
        del reference
        return ArtifactIntegrityResult("unsupported", deep=deep)


__all__ = ["FixtureRunDataSource"]
