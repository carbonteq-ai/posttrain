"""Launch ONE of five adaptive generation experiments, with immutable receipts."""

import argparse
import hashlib
import json
import shlex
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from luna_ssr_sft import exclusions
from stage_luna_solver_experiment import FILES, HOST, IMAGE, SSH, remote


def run(args):
    project = Path(__file__).resolve().parents[2]
    root = project / "docs/research/evidence/luna-ssr-sft-20261018/adaptive-next-five"
    ledger = json.loads((root / "ledger.json").read_text())
    if len(ledger["experiments"]) != args.number - 1:
        raise ValueError("each prior experiment must be recorded; no duplicate or blanket launch")
    if ledger["experiments"] and ledger["experiments"][-1].get("status") != "reviewed":
        raise ValueError("review prior evidence before choosing the next experiment")
    local = root / f"experiment{args.number:02d}"
    local.mkdir(exist_ok=False)
    capsule = f"/var/lib/posttrain/qualifications/luna-generation-opt{args.number:02d}-20261019-r1"
    name = f"luna-generation-qwen35-{args.model}-opt{args.number:02d}-20261019-r1"
    prior = project / "docs/research/evidence/luna-ssr-sft-20261018/sft-batch01-20261019-r2"
    inputs = prior / "inputs.json" if args.inputs is None else args.inputs
    names = [row["task"] for row in json.loads(inputs.read_text())["tasks"]]
    protected, _ = exclusions(project)
    if len(names) != len(set(names)) or set(names) & set(protected):
        raise ValueError("duplicate or protected task in generation experiment")
    (local / "inputs.json").write_bytes(inputs.read_bytes())
    config = {"number": args.number, "run_id": name, "model": args.model, "variant": args.variant,
              "hypothesis": args.hypothesis, "falsifier": args.falsifier, "status": "recorded_before_launch",
              "hard_timeout_seconds": 3600, "optimizer_updates": 0,
              "control": ("fresh same-model control" if args.with_control else
                          "fixed batch01-r2 base9B control; no pooling or new independent-control claim"),
              "recorded_at_utc": datetime.now(UTC).isoformat()}
    (local / "experiment.json").write_text(json.dumps(config, indent=2) + "\n")
    ledger["experiments"].append(config)
    (root / "ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")
    remote(f'test ! -e {capsule} && ! docker inspect {name} >/dev/null 2>&1 && '
           f'test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" && '
           f'mkdir -p {capsule}/driver {capsule}/evidence')
    subprocess.run(["scp", *SSH, *[str(project / "scripts/qualification" / f) for f in FILES],
                    f"{HOST}:{capsule}/driver/"], check=True)
    subprocess.run(["scp", *SSH, str(local / "experiment.json"), str(local / "inputs.json"),
                    f"{HOST}:{capsule}/"], check=True)
    source = remote(f"cd {capsule} && sha256sum driver/* inputs.json experiment.json > source-sha256.txt && cat source-sha256.txt")
    (local / "source-sha256.txt").write_text(source)
    runtime = "/var/lib/posttrain/qualifications/self-campaign-2048-20261017-r5"
    mounts = [capsule + ":/pilot:ro", capsule + "/evidence:/evidence", runtime + ":/workspace:ro",
              "/var/lib/posttrain/cache/huggingface:/hf:ro",
              "/var/lib/posttrain/cache/triton:/triton", "/var/lib/posttrain/cache/vllm:/vllm-cache"]
    env = ["PYTHONPATH=/pilot/driver:/workspace/driver:/workspace/kernel-deps:/workspace/training-deps",
           "VLLM_BATCH_INVARIANT=1", "VLLM_WORKER_MULTIPROC_METHOD=spawn", "HF_HOME=/hf",
           "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1", "WANDB_DISABLED=true", "DO_NOT_TRACK=1",
           "HF_HUB_DISABLE_TELEMETRY=1", "TOKENIZERS_PARALLELISM=false",
           "TRITON_CACHE_DIR=/triton", "VLLM_CACHE_ROOT=/vllm-cache"]
    revision = {"4b": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", "9b": "c202236235762e1c871ad0ccb60c8ee5ba337b9a"}[args.model]
    model = f"/hf/hub/models--Qwen--Qwen3.5-{args.model.upper()}/snapshots/{revision}"
    base = ["docker", "run", "--network", "none"]
    for value in env:
        base += ["-e", value]
    for value in mounts:
        base += ["-v", value]
    driver = ["/pilot/driver/luna_solver_pilot.py", "--model", model, "--inputs", "/pilot/inputs.json",
              "--run-name", name, "--instruction-variant", args.variant]
    if not args.with_control:
        driver.append("--guided-only")
    preflight = base + ["--rm", "--entrypoint", "/opt/posttrain/venv/bin/python", IMAGE,
                        *driver, "--output", "/evidence/preflight", "--preflight-only"]
    print(remote(shlex.join(preflight), timeout=100), flush=True)
    remote(f'! docker inspect {name} >/dev/null 2>&1 && '
           f'test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" && '
           f'cd {capsule} && sha256sum -c source-sha256.txt >/dev/null')
    cmd = base + ["-d", "--name", name, "--gpus", "device=0", "--shm-size", "16g",
                  "--entrypoint", "/usr/bin/timeout", IMAGE, "--signal=TERM", "--kill-after=30", "3570",
                  "/opt/posttrain/venv/bin/python", *driver, "--output", f"/evidence/{args.model}"]
    container = remote(shlex.join(cmd)).strip()
    receipt = {"container_id": container, "capsule": capsule, "run_id": name, "image": IMAGE,
               "source_manifest_sha256": hashlib.sha256(source.encode()).hexdigest()}
    (local / "launch.json").write_text(json.dumps(receipt, indent=2) + "\n")
    ledger["experiments"][-1].update(status="accepted", **receipt)
    (root / "ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("number", type=int, choices=range(1, 6))
    parser.add_argument("variant")
    parser.add_argument("--model", choices=["4b", "9b"], default="9b")
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--falsifier", required=True)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--with-control", action="store_true")
    run(parser.parse_args())
