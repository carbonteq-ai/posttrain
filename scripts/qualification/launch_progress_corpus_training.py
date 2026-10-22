"""Launch exactly one approved ten-step full124 run on the local workstation."""

import argparse
import fcntl
import hashlib
import json
import subprocess
import time
from pathlib import Path

from progress_corpus_training import gate

IMAGE = "registry.lan/carbonteq/posttrain-lab/posttrain-lab/posttrain-job@sha256:e976e727786135f851f4cc96356ebff1f57c75ca5f2d36a2c6aacbf6b2822337"
RUNTIME = "/var/lib/posttrain/qualifications/self-campaign-2048-20261017-r5"
PRIOR = "/var/lib/posttrain/qualifications/sdpo-guidance-five-20261019-experiment03-plain-r3"
MODEL = "/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def run(command, **kwargs):
    return subprocess.check_output(command, text=True, timeout=30, **kwargs).strip()


def launch(root):
    config = json.loads((root / "experiment.json").read_text())
    qualification = Path(config["qualification_capsule"])
    plan = json.loads((root / "ten-step-training-corpus-plan.json").read_text())
    heldout = json.loads((root / "heldout62.json").read_text())
    gate(qualification, plan, heldout)
    for line in (root / "source-sha256.txt").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("training capsule source changed")
    remaining = int(config["deadline_unix"] - time.time() - 30)
    if (config["deadline_unix"] - config["started_unix"] > 7200
            or remaining <= 120 or config["steps"] != 10 or config["batch_distinct_tasks"] != 32):
        raise ValueError("training authorization/time budget mismatch")
    lock = Path("/tmp/sdpo-guidance-five-launch.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    name = config["run_name"]
    retained = subprocess.run(["docker", "inspect", name], capture_output=True, timeout=15)
    if retained.returncode == 0 or (root / "evidence/train").exists():
        raise ValueError("owned job/evidence retained: monitor exact run, never duplicate")
    if run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"]):
        raise ValueError("GPU busy; no competing submission")
    cmd = ["docker", "run", "-d", "--name", name, "--gpus", "device=0", "--network", "none",
           "--shm-size", "16g", "--entrypoint", "/usr/bin/timeout"]
    environment = {"PYTHONPATH": "/local-environment/src:/training/driver:/prior/driver:/workspace/driver:/workspace/trl:/workspace/kernel-deps:/workspace/training-deps",
        "VLLM_BATCH_INVARIANT": "1", "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "VLLM_GDN_DECODE_KERNEL": "triton",
        "POSTTRAIN_QUALIFICATION_TRACE_SECONDS": "0", "HF_HOME": "/hf", "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1", "WANDB_DISABLED": "true", "DO_NOT_TRACK": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false",
        "TRITON_CACHE_DIR": "/triton", "VLLM_CACHE_ROOT": "/vllm-cache"}
    for key, value in environment.items():
        cmd += ["-e", f"{key}={value}"]
    for binding in [f"{root}:/training:ro", f"{root}/evidence:/evidence",
                    f"{qualification}:{qualification}:ro", f"{qualification}/environment:/local-environment:ro",
                    f"{RUNTIME}:/workspace:ro", f"{PRIOR}:/prior:ro",
                    "/var/lib/posttrain/cache/huggingface:/hf:ro", "/var/lib/posttrain/cache/triton:/triton",
                    "/var/lib/posttrain/cache/vllm:/vllm-cache"]:
        cmd += ["-v", binding]
    cmd += [IMAGE, "--signal=TERM", "--kill-after=30", str(remaining), "/opt/posttrain/venv/bin/python",
            "/training/driver/progress_corpus_training.py", "--qualification", str(qualification),
            "--plan", "/training/ten-step-training-corpus-plan.json", "--heldout", "/training/heldout62.json",
            "--model", MODEL, "--run-name", name, "--output", "/evidence/train"]
    identifier = run(cmd)
    result = {"run_name": name, "container_id": identifier, "deadline_unix": config["deadline_unix"],
              "source_manifest_sha256": hashlib.sha256((root / "source-sha256.txt").read_bytes()).hexdigest(),
              "qualification_capsule": str(qualification), "optimizer_updates_confirmed": 0}
    (root / "evidence/launch-receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(launch(args.root)))
