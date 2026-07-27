"""Run one tracked Qwen 3.5 0.8B serving-capacity concurrency sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from posttrain.catalog import discover_project, open_catalog
from posttrain.common import CatalogRef, Workload
from posttrain.jobs import build_job_runtime
from posttrain.serve.prompts import load_prompt_corpus
from posttrain.serve.results import BenchmarkSweepResult
from posttrain.work import (
    ProjectExecutionRequest,
    Recipe,
    RecipeJob,
    WorkPackage,
    load_project_brief,
    run_work_package_job,
)

VARIANTS = {
    "standard": ("inference/qwen3.5-0.8b-vllm-screen-standard@1", 4),
    "mtp": ("inference/qwen3.5-0.8b-vllm-screen-mtp@1", 4),
    "turboquant": ("inference/qwen3.5-0.8b-vllm-screen-turboquant@1", 4),
    "mtp-turboquant": ("inference/qwen3.5-0.8b-vllm-screen-mtp-turboquant@1", 4),
    "mtp-capacity64": ("inference/qwen3.5-0.8b-vllm-screen-mtp-capacity64@1", 64),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument(
        "--concurrency",
        dest="concurrencies",
        action="append",
        type=int,
        choices=(1, 2, 4, 8, 12, 16, 24, 32, 48, 64),
        help="Concurrency point to measure; repeat to override the variant's default sweep.",
    )
    parser.add_argument(
        "--cohort",
        choices=("representative", "controlled"),
        default="representative",
        help="Use the audited corpus by default; controlled is only for exact-token diagnostics.",
    )
    args = parser.parse_args()
    inference_binding_id, max_concurrency = VARIANTS[args.variant]
    default_sweep = (4, 8, 12, 16, 24, 32, 48, 64) if max_concurrency == 64 else (1, 2, 4)
    concurrencies = tuple(sorted(set(args.concurrencies or default_sweep)))
    if concurrencies[-1] > max_concurrency:
        parser.error(f"{args.variant} supports concurrency up to {max_concurrency}")

    root = Path(__file__).resolve().parents[1]
    layout = discover_project(root, explicit_root=root)
    catalog = open_catalog(scope=layout.project_id, overlays=layout.catalog_overlays)
    brief = load_project_brief(layout.project_brief) if layout.project_brief is not None else None
    if args.cohort == "representative":
        corpus = load_prompt_corpus("general-serving-v1")
        requests = {
            "suite_id": "qwen08b-capacity-representative-v1",
            "shape_id": "general-serving-v1-128out",
            "context_window": 32_768,
            "output_tokens": 128,
            "cohort": "representative",
            "corpus": {
                "id": corpus.manifest.id,
                "revision": corpus.manifest.revision,
                "digest": corpus.manifest.digest,
            },
            "selection_seed": 17,
            "record_count": corpus.manifest.record_count,
        }
        measured_repetitions = 1
        workload_id = "workloads/qwen08b-representative-32k-sweep@1"
    else:
        requests = {
            "suite_id": "qwen08b-capacity-diagnostic-v1",
            "shape_id": "controlled-512in-128out",
            "context_window": 32_768,
            "input_tokens": 512,
            "output_tokens": 128,
            "cohort": "controlled",
        }
        measured_repetitions = 3
        workload_id = "workloads/qwen08b-controlled-32k-sweep@1"
    workload = Workload(
        id=workload_id,
        revision="1",
        requests=requests,
        concurrency=concurrencies,
        warmup_repetitions=1,
        measured_repetitions=measured_repetitions,
        required_measures=(
            "serve/run/requests_attempted",
            "serve/run/requests_measured",
            "serve/run/output_tokens_measured",
            "serve/run/measurement_duration_s",
            "serve/request/ttft_seconds",
            "serve/request/tpot_seconds",
            "serve/backend/peak_vram_bytes",
        ),
    )
    package = WorkPackage(
        project_id=layout.project_id,
        work_package_id="screen/qwen3.5-0.8b-serving-capacity",
        stage="screen",
        description="Compare Qwen 3.5 0.8B vLLM, MTP, and TurboQuant capacity on the local 8 GiB GPU.",
        recipe=Recipe(
            id="recipes/qwen08b-serving-capacity-diagnostic@1",
            revision="1",
            stage="screen",
            seats={
                "model": "model",
                "screen_inference": "inference",
                "workload": "workload",
                "target": "target",
            },
            jobs=(
                RecipeJob(
                    id="benchmark",
                    kind="serve.benchmark",
                    definition="serve/vllm-benchmark@1",
                ),
            ),
            expected_artifacts=("serving benchmark result",),
        ),
        bindings={
            "model": CatalogRef("model", "models/qwen3.5-0.8b@bf16"),
            "screen_inference": CatalogRef("inference", inference_binding_id),
            "workload": workload,
            "target": CatalogRef("target", "targets/local-cuda-8gb"),
        },
        metadata={
            "cohort": args.cohort,
            "variant": args.variant,
            "concurrency": list(concurrencies),
        },
    )
    runtime = build_job_runtime(
        ProjectExecutionRequest(
            project_id=layout.project_id,
            project_root=layout.root,
            state_dir=layout.state,
            work_package_path=Path(__file__).resolve(),
            catalog=catalog,
            project_brief=brief,
        ),
        tracking=layout.tracking,
    )
    result = run_work_package_job(runtime, package, "benchmark")
    job = result.jobs[0]
    value = job.value
    if not isinstance(value, BenchmarkSweepResult):
        raise RuntimeError(f"serve.benchmark returned an unexpected value: {type(value).__name__}")
    payload = value.as_json()
    print(json.dumps({"run_id": job.run_id, "variant": args.variant, **payload}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
