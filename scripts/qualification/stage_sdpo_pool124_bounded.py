"""Seal broad-corpus preparations only; never submit a GPU job or start its clock."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def stage(template, output, slot, reference_path):
    if slot not in (9, 10) or output.exists():
        raise ValueError("new preparation identity and remaining slot required")
    cohort = json.loads((template / "cohorts.json").read_text())
    if len(set(cohort["train"])) != 124 or cohort["development"] or cohort["reserve"]:
        raise ValueError("full124training pool required; old internal reserves retired")
    for line in (template / "source-sha256.txt").read_text().splitlines():
        expected, name = line.split("  ", 1)
        if hashlib.sha256((template / name).read_bytes()).hexdigest() != expected:
            raise ValueError("template source changed")
    output.mkdir()
    shutil.copytree(template / "driver", output / "driver")
    for path in (output / "driver").iterdir():
        current = Path(__file__).parent / path.name
        if current.is_file():
            shutil.copyfile(current, path)
    extra = ["sdpo_pool124_bounded_config.py", "run_sdpo_agent_ablation.sh",
             "analyze_sdpo_pool124.py", "stage_sdpo_pool124_bounded.py"]
    for name in extra:
        shutil.copyfile(Path(__file__).parent / name, output / "driver" / name)
    (output / "driver/run_sdpo_agent_ablation.sh").chmod(0o755)
    for name in ("cohorts.json", "train-pool.json", "heldout62.json"):
        shutil.copyfile(template / name, output / name)
    references = json.loads(reference_path.read_text())
    if (not references["current_rubric_state_checks_complete"] or len(references["references"]) != 5
            or not set(references["references"]) <= set(cohort["train"])):
        raise ValueError("passing training-only offline reference identities required")
    cohort["reference_diagnostic"] = sorted(references["references"])
    (output / "cohorts.json").write_text(json.dumps(cohort, indent=2) + "\n")
    if slot == 9:
        shutil.copyfile(reference_path, output / "luna-reviewed.json")
    root = f"/var/lib/posttrain/qualifications/sdpo-agent{slot:02d}-20261018-r5"
    kind = "luna-paired" if slot == 9 else "thinking"
    config = {"budget_slot": slot, "run_name": f"sdpo-qwen35-4b-agent{slot:02d}-pool124-{kind}-20261018-r5",
        "capsule": root, "status": "CPU_prepared_NOT_GPU_submitted", "training_tasks": 124,
        "run_profile": "pool124-bounded", "batch_size": 32, "steps": 12 if slot == 9 else 6,
        "wall_time_cap_seconds": 7200, "gpu_timeout_seconds": 7100,
        "cpu_preflight_budget_seconds": 60, "branch_wall_time_seconds": 3500,
        "workflow": "paired-luna" if slot == 9 else "single",
        "thinking": slot == 10, "qualify_capacity": True,
        "shared_slot10_budget": slot == 10, "teacher_feedback": "solution-or-feedback",
        "lr": 1e-5, "lr_schedule": "cosine-decay", "lr_end_ratio": .1,
        "alpha": 1., "topk": 20, "teacher_rate": .01, "online_rollouts": 2,
        "selection_mode": "failure-frontier", "tool_parser_mode": "schema-bound-v1",
        "context_limit": 32768, "reply_limit": 4096, "episode_output_limit": 12288,
        "turn_limit": 8, "evaluate_every": 10,
        "heldout": {"tasks": 62, "replicas": 5, "context": 65536,
                    "reply": 8192, "episode": 32768, "turns": 20},
        "hypothesis": ("Teacher-only verified offline Luna observations improve over a fresh matched "
                       "no-Luna6step control; only5reference-covered tasks,119native-feedback fallbacks."
                       if slot == 9 else "Student+teacher native thinking improves over its own fresh "
                       "thinking base and matched6step nonthinking control, with all124training tasks."),
        "falsifier": "No native gain, added preservation violations, harmed successful-task retention, "
                     "incomplete coverage, token/channel/gradient/admission failure or timeout rejects qualification.",
        "matched_control": "Same32K/4K/batch32/full124round-robin/6steps/cosine/parser/objective. "
                           "Luna compares independent fresh branches; thinking has a fresh thinking-mode base.",
        "launch_order": "After accepted pool124-50 run terminates and is audited; inspect exactowned/GPU first.",
        "heldout_caveat": "Repeated monitoring and selection, not pristine final validation; no reserve phase.",
        "external_model_calls": 0, "planned_native_episodes": 2048 if slot == 9 else 1024,
        "training_sentinel_diagnostics": False,
        "reference_training_diagnostics": {"tasks": cohort["reference_diagnostic"], "replicas": 2,
            "steps": [0, 6], "scope": "Training-only acquisition/retention diagnostics, not validation. "
            "Thinking/control use same task names without offline Luna context."},
        "budget_rationale": "Measured full62x5evaluation about680–701sec per population; "
                            "startup/gates452sec and nativeupdates210–230sec. "
                            "6updates plus2full evaluations forecast3180sec plus32K backward-gate overhead "
                            "per branch;3500sec phase cap/7100sec combined cap remain hard, no guaranteed fit.",
        "power_limit": "Short6update horizon,192selected actions over124tasks; "
                       "no gain cannot establish failure of longer training. No smaller task/evaluation population."}
    (output / "experiment.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "evidence").mkdir()
    files = sorted(p for p in output.rglob("*") if p.is_file())
    manifest = "".join(hashlib.sha256(p.read_bytes()).hexdigest() + "  "
                       + str(p.relative_to(output)) + "\n" for p in files)
    (output / "source-sha256.txt").write_text(manifest)
    return {"slot": slot, "run_name": config["run_name"], "capsule": root,
            "source_manifest_sha256": hashlib.sha256(manifest.encode()).hexdigest(),
            "GPU_submitted": False, "deadline_started": False, "training_tasks": 124,
            "batch_size": 32, "planned_native_episodes": config["planned_native_episodes"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--slot", type=int, choices=(9, 10), required=True)
    parser.add_argument("--references", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(stage(args.template, args.output, args.slot, args.references)))
