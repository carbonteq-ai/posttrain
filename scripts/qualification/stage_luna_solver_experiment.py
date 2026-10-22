"""Stage one immutable, bounded two-model experiment; never select the next arm."""

import argparse
import hashlib
import json
import shlex
import subprocess
from pathlib import Path

SSH = [
    "-F",
    "/dev/null",
    "-o",
    "BatchMode=yes",
    "-o",
    "IdentitiesOnly=yes",
    "-o",
    "IdentityAgent=none",
    "-o",
    "GSSAPIAuthentication=no",
    "-o",
    "ConnectTimeout=8",
    "-i",
    "/home/hammad/projects/ai-infra/.state/ssh/dstack_worker",
]
HOST = "dstack@192.168.110.12"
IMAGE = "registry.lan/carbonteq/posttrain-lab/posttrain-lab/posttrain-job@sha256:e976e727786135f851f4cc96356ebff1f57c75ca5f2d36a2c6aacbf6b2822337"
FILES = [
    "luna_policy_guidance.py",
    "luna_effect_scope.py",
    "luna_solver_schema.py",
    "luna_solver_pilot.py",
    "luna_solver_prompts.py",
    "luna_solver_stages.py",
    "luna_solver_separation.py",
    "generate_luna_ssr_pilot.py",
    "luna_ssr_quality_stages.py",
    "luna_ssr_sft.py",
    "run_luna_solver_pilot.sh",
]


def remote(command, timeout=30):
    return subprocess.run(
        ["ssh", *SSH, HOST, command], check=True, text=True, capture_output=True, timeout=timeout
    ).stdout


def run(args):
    project = Path(__file__).resolve().parents[2]
    local = project / "docs/research/evidence/luna-ssr-sft-20261018/adaptive-five" / f"experiment{args.number:02d}"
    local.mkdir(parents=True, exist_ok=False)
    tag = f"20261019-r{10 + args.number}"
    capsule = f"/var/lib/posttrain/qualifications/luna-solver-adaptive{args.number:02d}-20261019-r1"
    old = project / "docs/research/evidence/luna-ssr-sft-20261018/solver-pilot2-20261019-r1"
    config = json.loads((old / "experiment.json").read_text())
    config.update(
        run_tag=tag,
        instruction_variant=args.variant,
        adaptive_experiment=args.number,
        hypothesis=args.hypothesis,
        falsifier=args.falsifier,
        run_ids=[f"luna-solver-qwen35-{s}-two-task-{tag}" for s in ("4b", "9b")],
    )
    (local / "experiment.json").write_text(json.dumps(config, indent=2) + "\n")
    (local / "inputs.json").write_bytes((old.parent / "solver-pilot2-r2/inputs.json").read_bytes())
    remote(
        f'test ! -e {capsule} && test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" && mkdir -p {capsule}/driver {capsule}/evidence'
    )
    subprocess.run(
        ["scp", *SSH, *[str(project / "scripts/qualification" / name) for name in FILES], f"{HOST}:{capsule}/driver/"],
        check=True,
    )
    subprocess.run(
        ["scp", *SSH, str(local / "experiment.json"), str(local / "inputs.json"), f"{HOST}:{capsule}/"], check=True
    )
    source = remote(
        f"cd {capsule} && sha256sum driver/* experiment.json inputs.json > source-sha256.txt && cat source-sha256.txt"
    )
    (local / "source-sha256.txt").write_text(source)
    runtime = "/var/lib/posttrain/qualifications/self-campaign-2048-20261017-r5"
    cmd = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--entrypoint",
        "/opt/posttrain/venv/bin/python",
        "-e",
        "PYTHONPATH=/pilot/driver:/workspace/driver:/workspace/kernel-deps:/workspace/training-deps",
        "-e",
        "HF_HOME=/hf",
        "-e",
        "HF_HUB_OFFLINE=1",
        "-e",
        "TRANSFORMERS_OFFLINE=1",
        "-v",
        capsule + ":/pilot:ro",
        "-v",
        capsule + "/evidence:/evidence",
        "-v",
        runtime + ":/workspace:ro",
        "-v",
        "/var/lib/posttrain/cache/huggingface:/hf:ro",
        IMAGE,
        "/pilot/driver/luna_solver_pilot.py",
        "--preflight-only",
        "--model",
        "/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "--inputs",
        "/pilot/inputs.json",
        "--output",
        "/evidence/preflight",
        "--run-name",
        "preflight",
        "--instruction-variant",
        args.variant,
    ]
    print(remote(shlex.join(cmd), timeout=100), flush=True)
    launch_code = (
        "import pathlib,subprocess,json;"
        f"p=pathlib.Path({capsule!r});"
        "assert not (p/'evidence/queue-lock').exists();"
        "log=(p/'evidence/queue.log').open('w');"
        "v=subprocess.Popen(['bash','driver/run_luna_solver_pilot.sh'],cwd=p,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True);"
        "print(json.dumps({'queue_pid':v.pid}))"
    )
    receipt = json.loads(remote("python3 -c " + shlex.quote(launch_code)))
    receipt.update(
        capsule=capsule, source_manifest_sha256=hashlib.sha256(source.encode()).hexdigest(), run_ids=config["run_ids"]
    )
    (local / "launch.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("number", type=int, choices=range(1, 6))
    parser.add_argument("variant")
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--falsifier", required=True)
    run(parser.parse_args())
