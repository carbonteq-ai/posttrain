"""Offline immutable pool124 qualification, followed by the authorized50step run."""

import argparse
import fcntl
import hashlib
import json
import subprocess
import time
from pathlib import Path

IMAGE = "registry.lan/carbonteq/posttrain-lab/posttrain-lab/posttrain-job@sha256:e976e727786135f851f4cc96356ebff1f57c75ca5f2d36a2c6aacbf6b2822337"
RUNTIME = "/var/lib/posttrain/qualifications/self-campaign-2048-20261017-r5"
MODEL = "/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
PYTHONPATH = "/adaptive/driver:/workspace/driver:/workspace/trl:/workspace/kernel-deps:/workspace/training-deps"


def command(args, **kwargs):
    return subprocess.check_output(args, text=True, timeout=kwargs.pop("timeout", 30), **kwargs).strip()


def state(name):
    result = subprocess.run(["docker", "inspect", "--format", "{{json .State}}", name],
                            capture_output=True, text=True, timeout=15)
    return json.loads(result.stdout) if result.returncode == 0 else None


def launch(root, mode):
    config = json.loads((root / "campaign.json").read_text())
    if config["training_tasks"] != 124 or config["batch_size"] != 32 or config["initial_updates"] != 50:
        raise ValueError("authorization/config mismatch")
    command(["sha256sum", "-c", "source-sha256.txt"], cwd=root)
    command(["sha256sum", "-c", "source-sha256.txt"], cwd=RUNTIME)
    name = config["qualification_run_name"] if mode == "qualify" else config["run_name"]
    if state(name) is not None or (root / f"evidence/{mode}").exists():
        raise ValueError("retained owned run/evidence exists; monitor, never duplicate")
    if command(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"]):
        raise ValueError("GPU busy; no competing submission")
    if mode == "train":
        gate = json.loads((root / "evidence/qualification-gate.json").read_text())
        before = state(config["qualification_run_name"])
        if (not gate["qualified"] or gate["actual_updates"] != 4 or gate["distinct_training_tasks"] != 124
                or before is None or before["Running"] or before["ExitCode"] != 0
                or gate["source_manifest_sha256"] != hashlib.sha256((root / "source-sha256.txt").read_bytes()).hexdigest()
                or gate["qualification_receipt_sha256"] != hashlib.sha256((root / "evidence/qualify/receipt.json").read_bytes()).hexdigest()):
            raise ValueError("full-pool qualification not accepted")
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
    args += [IMAGE, "--signal=TERM", "--kill-after=30", "7100" if mode == "qualify" else "86370",
             "/opt/posttrain/venv/bin/python", "/adaptive/driver/sdpo_agent_ablation.py",
             "--model", MODEL, "--cohorts", "/adaptive/cohorts.json", "--heldout", "/adaptive/heldout62.json",
             "--training-pool", "/adaptive/train-pool.json", "--output", f"/evidence/{mode}",
             "--run-name", name, "--steps", "4" if mode == "qualify" else "50", "--run-profile",
             "pool124-qualify" if mode == "qualify" else "pool124-50", "--batch-size", "32",
             "--online-rollouts", "2", "--lr", "1e-5", "--lr-schedule", "hold10-decay20-flat",
             "--lr-end-ratio", ".1", "--alpha", "1", "--topk", "20", "--teacher-rate", ".01",
             "--teacher-feedback", "solution-or-feedback", "--selection-mode", "failure-frontier",
             "--tool-parser-mode", "schema-bound-v1", "--context-limit", "16384", "--reply-limit", "2048"]
    identifier = command(args, timeout=120)
    receipt = {"mode": mode, "run_name": name, "container_id": identifier,
               "source_manifest_sha256": hashlib.sha256((root / "source-sha256.txt").read_bytes()).hexdigest()}
    (root / f"evidence/{mode}-submission.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def queue(root):
    lock = (root / "evidence/queue.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = json.loads((root / "campaign.json").read_text())
    if state(config["run_name"]) is not None:
        print("Training already retained; monitor only", flush=True)
        return
    name = config["qualification_run_name"]
    if state(name) is None:
        print(json.dumps(launch(root, "qualify")), flush=True)
    deadline = time.monotonic() + 7400
    while time.monotonic() < deadline:
        current = state(name)
        if current is None:
            raise ValueError("owned qualification disappeared; no automatic replacement")
        if not current["Running"]:
            if current["ExitCode"] != 0:
                raise ValueError(f"qualification failed exit{current['ExitCode']}; preserve, no train")
            break
        time.sleep(30)
    else:
        raise TimeoutError("qualification observation expired; inspect same run")
    audit = ["docker", "run", "--rm", "--network", "none", "--cpus", "2", "--memory", "4g",
             "--entrypoint", "/usr/bin/timeout", "-e", f"PYTHONPATH={PYTHONPATH}",
             "-v", f"{root}:/adaptive:ro", "-v", f"{root}/evidence:/adaptive/evidence",
             "-v", f"{RUNTIME}:/workspace:ro", IMAGE, "180", "/opt/posttrain/venv/bin/python",
             "/adaptive/driver/qualify_sdpo_pool124.py", "/adaptive"]
    with (root / "evidence/qualification-cpu-audit.log").open("w") as log:
        subprocess.run(audit, stdout=log, stderr=subprocess.STDOUT, timeout=210, check=True)
    print(json.dumps(launch(root, "train")), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("qualify", "train", "queue"))
    parser.add_argument("--capsule", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    if args.mode == "queue":
        queue(args.capsule)
    else:
        print(json.dumps(launch(args.capsule, args.mode)))
