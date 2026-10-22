"""Launch the authorized 27B comparison against the sealed 4B opt05 recipe."""

import argparse
import hashlib
import json
import shlex
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from stage_luna_solver_experiment import HOST, IMAGE, SSH, remote


def main(repair=False):
    project = Path(__file__).resolve().parents[2]
    root = project / "docs/research/evidence/luna-ssr-sft-20261018"
    previous = root / "adaptive-next-five/experiment05"
    suffix = "r2" if repair else "r1"
    local = root / f"solver-27b-comparison-20261019-{suffix}"
    capsule = f"/var/lib/posttrain/qualifications/luna-solver-27b-comparison-20261019-{suffix}"
    original = "/var/lib/posttrain/qualifications/luna-generation-opt05-20261019-r1"
    name = f"luna-solver-qwen35-27b-comparison-20261019-{suffix}"
    model = "/hf/hub/models--Qwen--Qwen3.5-27B/snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654"
    config = {
        "run_id": name, "model": "27b", "model_path": model,
        "variant": "scope-repair-natural-provenance-v5", "optimizer_updates": 0,
        "arms": ["control", "guided"], "expected_episodes": 6,
        "comparison": "sealed 4B opt05 fresh control and guided; same three reused training tasks",
        "pipeline_source": original, "hard_timeout_seconds": 3600,
        "public_query_gate_repair": repair,
        "repair_scope": "Only in:inbox plus at least two alphabetic public-request words is exempt from copied-value rejection" if repair else None,
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "interpretation": "One rollout/task/arm; end-to-end model recipe comparison, not unseen benchmark or SFT qualification",
    }
    if local.exists():
        raise ValueError("Retained comparison exists; inspect it instead of resubmitting")
    remote(f'test ! -e {capsule} && ! docker inspect {name} >/dev/null 2>&1 && '
           f'test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" && '
           f'cd {original} && sha256sum -c source-sha256.txt >/dev/null && '
           f'test -f /var/lib/posttrain/cache/huggingface/hub/models--Qwen--Qwen3.5-27B/snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654/config.json')
    local.mkdir()
    (local / "experiment.json").write_text(json.dumps(config, indent=2) + "\n")
    (local / "inputs.json").write_bytes((previous / "inputs.json").read_bytes())
    remote(f"mkdir -p {capsule}/evidence && cp -a {original}/driver {capsule}/driver")
    subprocess.run(["scp", *SSH, str(local / "experiment.json"), str(local / "inputs.json"),
                    f"{HOST}:{capsule}/"], check=True)
    if repair:
        subprocess.run(["scp", *SSH, str(project / "scripts/qualification/luna_solver_prompts.py"),
                        f"{HOST}:{capsule}/driver/"], check=True)
    source = remote(f"cd {capsule} && sha256sum driver/* inputs.json experiment.json > source-sha256.txt && cat source-sha256.txt")
    (local / "source-sha256.txt").write_text(source)
    prior_hashes = {line.split(maxsplit=1)[1]: line.split()[0]
                    for line in (previous / "source-sha256.txt").read_text().splitlines()}
    for line in source.splitlines():
        digest, path = line.split(maxsplit=1)
        if repair and path == "driver/luna_solver_prompts.py":
            assert digest == hashlib.sha256((project / "scripts/qualification/luna_solver_prompts.py").read_bytes()).hexdigest()
            continue
        if path.startswith("driver/") or path == "inputs.json":
            assert prior_hashes[path] == digest, path
    runtime = "/var/lib/posttrain/qualifications/self-campaign-2048-20261017-r5"
    mounts = [capsule + ":/pilot:ro", capsule + "/evidence:/evidence", runtime + ":/workspace:ro",
              "/var/lib/posttrain/cache/huggingface:/hf:ro", "/var/lib/posttrain/cache/triton:/triton",
              "/var/lib/posttrain/cache/vllm:/vllm-cache"]
    env = ["PYTHONPATH=/pilot/driver:/workspace/driver:/workspace/kernel-deps:/workspace/training-deps",
           "VLLM_BATCH_INVARIANT=1", "VLLM_WORKER_MULTIPROC_METHOD=spawn", "HF_HOME=/hf",
           "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1", "WANDB_DISABLED=true", "DO_NOT_TRACK=1",
           "HF_HUB_DISABLE_TELEMETRY=1", "TOKENIZERS_PARALLELISM=false", "TRITON_CACHE_DIR=/triton",
           "VLLM_CACHE_ROOT=/vllm-cache"]
    base = ["docker", "run", "--network", "none"]
    for value in env:
        base += ["-e", value]
    for value in mounts:
        base += ["-v", value]
    driver = ["/pilot/driver/luna_solver_pilot.py", "--model", model, "--inputs", "/pilot/inputs.json",
              "--run-name", name, "--instruction-variant", config["variant"]]
    preflight = base + ["--rm", "--entrypoint", "/opt/posttrain/venv/bin/python", IMAGE,
                        *driver, "--output", "/evidence/preflight", "--preflight-only"]
    print(remote(shlex.join(preflight), timeout=100), flush=True)
    remote(f'! docker inspect {name} >/dev/null 2>&1 && '
           f'test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" && '
           f'cd {capsule} && sha256sum -c source-sha256.txt >/dev/null')
    command = base + ["-d", "--name", name, "--gpus", "device=0", "--shm-size", "16g",
                      "--entrypoint", "/usr/bin/timeout", IMAGE, "--signal=TERM", "--kill-after=30", "3570",
                      "/opt/posttrain/venv/bin/python", *driver, "--output", "/evidence/27b"]
    container = remote(shlex.join(command)).strip()
    receipt = {"container_id": container, "capsule": capsule, "run_id": name, "image": IMAGE,
               "source_manifest_sha256": hashlib.sha256(source.encode()).hexdigest(),
               "driver_and_inputs_identical_to_4b_opt05": not repair,
               "sole_changed_driver_from_4b": "luna_solver_prompts.py public query gate" if repair else None}
    (local / "launch.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-query-repair", action="store_true")
    main(parser.parse_args().public_query_repair)
