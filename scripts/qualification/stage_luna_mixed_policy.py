"""Stage one immutable 27B-critic / 4B-solver policy-retrieval diagnostic."""

import argparse
import hashlib
import json
import shlex
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from luna_ssr_sft import exclusions
from stage_luna_solver_experiment import FILES, HOST, IMAGE, SSH, remote

CAPSULE = "/var/lib/posttrain/qualifications/luna-mixed-policy-20261019-r1"
ROOT = "docs/research/evidence/luna-ssr-sft-20261018/mixed-4b-27b-policy-20261019-r1"
VARIANT = "scope-repair-natural-policy-provenance-v5"


def main(phase):
    project = Path(__file__).resolve().parents[2]
    local = project / ROOT
    donor = project / "docs/research/evidence/luna-ssr-sft-20261018/adaptive-next-five/experiment05"
    size = "27b" if phase == "critic" else "4b"
    name = f"luna-policy-{phase}{size}-20261019-r1"
    remote(f'! docker inspect {name} >/dev/null 2>&1 && '
           f'test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"')
    if phase == "critic":
        if local.exists():
            raise ValueError("retained staging exists; inspect instead of resubmitting")
        names = [r["task"] for r in json.loads((donor / "inputs.json").read_text())["tasks"]]
        protected, _ = exclusions(project)
        assert len(names) == len(set(names)) == 3 and not set(names) & set(protected)
        local.mkdir()
        config = {"run_id": "luna-mixed-policy-20261019-r1", "variant": VARIANT,
                  "critic": "Qwen3.5-27B@fc05daec18b0a78c049392ed2e771dde82bdf654",
                  "solver": "Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
                  "proposal_source": str(donor.relative_to(project) / "4b/instruction-stages.jsonl"),
                  "recorded_at": datetime.now(UTC).isoformat(), "tasks": names,
                  "optimizer_updates": 0, "actor_arms": ["fresh4b control", "4b with27b reviewed/repaired policy guidance"],
                  "expected_native_episodes": 6, "phase_caps_seconds": {"critic": 3600, "solver": 3600},
                  "changes": "27B critic and bounded requirements/assembly repairs of saved4B drafts; typed policy-title retrieval and sanitized critic-to-repair context; unchanged model/native budgets"}
        (local / "experiment.json").write_text(json.dumps(config, indent=2) + "\n")
        (local / "inputs.json").write_bytes((donor / "inputs.json").read_bytes())
        (local / "proposal-stages.jsonl").write_bytes((donor / "4b/instruction-stages.jsonl").read_bytes())
        remote(f"test ! -e {CAPSULE} && mkdir -p {CAPSULE}/driver {CAPSULE}/evidence")
        subprocess.run(["scp", *SSH, *[str(project / "scripts/qualification" / f) for f in FILES],
                        f"{HOST}:{CAPSULE}/driver/"], check=True)
        subprocess.run(["scp", *SSH, *[str(local / f) for f in ("inputs.json", "experiment.json", "proposal-stages.jsonl")],
                        f"{HOST}:{CAPSULE}/"], check=True)
        source = remote(f"cd {CAPSULE} && sha256sum driver/* inputs.json experiment.json proposal-stages.jsonl > source-sha256.txt && cat source-sha256.txt")
        (local / "source-sha256.txt").write_text(source)
    else:
        receipt = json.loads(remote(f"cat {CAPSULE}/evidence/critic27b/receipt.json"))
        state = json.loads(remote("docker inspect -f '{{json .State}}' luna-policy-critic27b-20261019-r1"))
        assert state["Status"] == "exited" and state["ExitCode"] == 0
        assert receipt["preparation_only"] and receipt["all_instructions_admitted"]
        assert len(receipt["native_diagnostic_eligible_tasks"]) == 3
    model = {"27b": "/hf/hub/models--Qwen--Qwen3.5-27B/snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654",
             "4b": "/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"}[size]
    runtime = "/var/lib/posttrain/qualifications/self-campaign-2048-20261017-r5"
    base = ["docker", "run", "--network", "none"]
    for value in ["PYTHONPATH=/pilot/driver:/workspace/driver:/workspace/kernel-deps:/workspace/training-deps",
                  "VLLM_BATCH_INVARIANT=1", "VLLM_WORKER_MULTIPROC_METHOD=spawn", "HF_HOME=/hf",
                  "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1", "WANDB_DISABLED=true", "DO_NOT_TRACK=1",
                  "HF_HUB_DISABLE_TELEMETRY=1", "TOKENIZERS_PARALLELISM=false",
                  "TRITON_CACHE_DIR=/triton", "VLLM_CACHE_ROOT=/vllm-cache"]:
        base += ["-e", value]
    for value in [CAPSULE + ":/pilot:ro", CAPSULE + "/evidence:/evidence", runtime + ":/workspace:ro",
                  "/var/lib/posttrain/cache/huggingface:/hf:ro", "/var/lib/posttrain/cache/triton:/triton",
                  "/var/lib/posttrain/cache/vllm:/vllm-cache"]:
        base += ["-v", value]
    driver = ["/pilot/driver/luna_solver_pilot.py", "--model", model, "--inputs", "/pilot/inputs.json",
              "--run-name", name, "--instruction-variant", VARIANT]
    if phase == "critic":
        driver += ["--preparation-only", "--proposal-stages", "/pilot/proposal-stages.jsonl"]
        print(remote(shlex.join(base + ["--rm", "--entrypoint", "/opt/posttrain/venv/bin/python", IMAGE,
                                       *driver, "--output", "/evidence/preflight", "--preflight-only"]), timeout=100), flush=True)
    else:
        driver += ["--prepared-guidance", "/evidence/critic27b/compressed-hints.json"]
    remote(f'! docker inspect {name} >/dev/null 2>&1 && '
           f'test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" && '
           f'cd {CAPSULE} && sha256sum -c source-sha256.txt >/dev/null')
    command = base + ["-d", "--name", name, "--gpus", "device=0", "--shm-size", "16g",
                      "--entrypoint", "/usr/bin/timeout", IMAGE, "--signal=TERM", "--kill-after=30", "3570",
                      "/opt/posttrain/venv/bin/python", *driver,
                      "--output", "/evidence/critic27b" if phase == "critic" else "/evidence/4b"]
    container = remote(shlex.join(command)).strip()
    receipt = {"run_id": name, "container_id": container, "capsule": CAPSULE, "phase": phase,
               "image": IMAGE, "source_manifest_sha256": hashlib.sha256((local / "source-sha256.txt").read_bytes()).hexdigest()}
    (local / f"{phase}-launch.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["critic", "solver"])
    main(parser.parse_args().phase)
