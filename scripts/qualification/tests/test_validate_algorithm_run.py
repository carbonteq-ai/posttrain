from __future__ import annotations

import json
from pathlib import Path

from posttrain_lab.qualification.scenarios import scenario_by_id

from scripts.qualification.validate_algorithm_run import (
    LocalAlgorithmEvidence,
    RemoteAlgorithmEvidence,
    acceptance_failures,
    collect_local_evidence,
)


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_local_validator_reads_updates_rollouts_and_model_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "training" / "grpo"
    _write_jsonl(
        root / "trainer" / "verl-metrics.jsonl",
        [
            {
                "step": step,
                "data": {
                    "training/global_step": step,
                    "actor/grad_norm": 0.2 if step == 1 else 0.0,
                    "perf/time_per_step": 2.0,
                },
            }
            for step in range(1, 16)
        ],
    )
    traces = []
    for step in range(1, 16):
        for example in range(2):
            for generation, reward in enumerate((0.0, 1.0, 1.0, 1.0)):
                traces.append(
                    {
                        "id": f"{step}-{example}-{generation}",
                        "run": {"step": step},
                        "info": {"example_id": f"math/{example}"},
                        "rewards": {"correct": reward},
                        "is_completed": True,
                        "nodes": [
                            {
                                "sampled": True,
                                "message": {
                                    "role": "assistant",
                                    "tool_calls": [{"name": "solve"}],
                                },
                            },
                            {
                                "sampled": True,
                                "message": {"role": "assistant", "content": "4"},
                            },
                        ],
                    }
                )
    _write_jsonl(root / "verifiers-traces.jsonl", traces)
    adapter = root / "trainer" / "model" / "lora_adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"changed")
    checkpoint = root / "trainer" / "checkpoints" / "global_step_15"
    checkpoint.mkdir(parents=True)
    (checkpoint / "state.bin").write_bytes(b"checkpoint")

    evidence = collect_local_evidence(
        scenario_by_id("gsm8k-qwen35-08b-grpo-15"),
        tmp_path,
    )

    assert evidence.optimizer_updates == 15
    assert evidence.trace_count == 120
    assert evidence.completed_trace_count == 120
    assert evidence.reward_variant_group_observed is True
    assert evidence.continued_after_tool_call_observed is True
    assert evidence.nonzero_gradient_observed is True
    assert evidence.adapter_digest is not None
    assert evidence.checkpoint_digest is not None
    assert evidence.runtime_seconds == 30


def test_acceptance_reports_remote_observation_gaps() -> None:
    scenario = scenario_by_id("automationbench-qwen35-08b-grpo-10")
    local = LocalAlgorithmEvidence(10, 40, 40, True, True, True, "a", "b", 10.0)
    remote = RemoteAlgorithmEvidence(
        status="succeeded",
        trace_count=0,
        optimizer_updates=10,
        gradient_points=10,
        reward_std_points=0,
        rollout_population_points=0,
        nonzero_gradient_observed=True,
        reward_variance_observed=False,
        model_artifact_observed=True,
        observatory_mode="job",
        observatory_complete=False,
        observatory_research_ready=False,
        tool_evidence_state="missing",
        provider_run_id="provider-run",
    )

    failures = acceptance_failures(scenario, local, remote)

    assert "remote live trace count is below the acceptance minimum" in failures
    assert "remote Observatory is not complete and research-ready" in failures
