"""Submit exactly one immutable user-authorized LFM20 capsule on the idle GPU."""

import argparse
import fcntl
import hashlib
import json
from pathlib import Path

from run_sdpo_pool124 import IMAGE, PYTHONPATH, RUNTIME, command, state
from sdpo_lfm_config import MODEL_REVISION


def launch(root):
    config = json.loads((root / "experiment.json").read_text())
    if (config["steps"] != 20 or config["training_tasks"] != 124 or config["batch_size"] != 32
            or config["model_revision"] != MODEL_REVISION or config["wall_time_cap_seconds"] != 7200):
        raise ValueError("authorization mismatch")
    lock = (root / "evidence/submission.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    name = config["run_name"]
    if state(name) is not None or (root / "evidence/train").exists() or (root / "evidence/submission.json").exists():
        raise ValueError("retained run: monitor exact identity, never duplicate")
    command(["sha256sum", "-c", "source-sha256.txt"], cwd=root)
    command(["sha256sum", "-c", "source-sha256.txt"], cwd=RUNTIME)
    preflight = json.loads((root / "evidence/cpu-preflight.json").read_text())
    digest = hashlib.sha256((root / "source-sha256.txt").read_bytes()).hexdigest()
    if not preflight["qualified"] or preflight["source_manifest_sha256"] != digest:
        raise ValueError("exact CPU preflight missing")
    if command(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"]):
        raise ValueError("GPU busy; no competing submission")
    args = ["docker", "run", "-d", "--name", name, "--gpus", "device=0", "--network", "none",
            "--shm-size", "16g", "--entrypoint", "/usr/bin/timeout"]
    for key, value in {"PYTHONPATH": PYTHONPATH, "VLLM_BATCH_INVARIANT": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "HF_HOME": "/hf", "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1", "WANDB_DISABLED": "true", "DO_NOT_TRACK": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false",
            "TRITON_CACHE_DIR": "/triton", "VLLM_CACHE_ROOT": "/vllm-cache"}.items():
        args += ["-e", f"{key}={value}"]
    for mount in (f"{root}:/adaptive:ro", f"{root}/evidence:/evidence", f"{RUNTIME}:/workspace:ro",
            "/var/lib/posttrain/cache/huggingface:/hf:ro", "/var/lib/posttrain/cache/triton:/triton",
            "/var/lib/posttrain/cache/vllm:/vllm-cache"):
        args += ["-v", mount]
    args += [IMAGE, "--signal=TERM", "--kill-after=30", "7170", "/opt/posttrain/venv/bin/python",
        "/adaptive/driver/sdpo_agent_ablation.py", "--model",
        f"/hf/hub/models--LiquidAI--LFM2.5-2.6B/snapshots/{MODEL_REVISION}",
        "--model-family", "lfm2.5", "--cohorts", "/adaptive/cohorts.json",
        "--heldout", "/adaptive/heldout62.json", "--training-pool", "/adaptive/train-pool.json",
        "--output", "/evidence/train", "--run-name", name, "--steps", "20",
        "--run-profile", "lfm-pool124-20", "--batch-size", "32", "--online-rollouts", "2",
        "--lr", "1e-5", "--lr-schedule", "hold10-decay20-flat", "--lr-end-ratio", ".1",
        "--alpha", "1", "--topk", "20", "--teacher-rate", ".01",
        "--teacher-feedback", "solution-or-feedback", "--selection-mode", "failure-frontier",
        "--tool-parser-mode", "lfm2-native", "--context-limit", "16384", "--reply-limit", "2048",
        "--thinking", "--qualify-capacity"]
    identifier = command(args, timeout=120)
    receipt = {"run_name": name, "container_id": identifier, "source_manifest_sha256": digest,
               "docker_state": state(name), "actual_command": args, "planned_updates": 20}
    (root / "evidence/submission.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capsule", type=Path)
    args = parser.parse_args()
    print(json.dumps(launch(args.capsule)))
