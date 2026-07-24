"""Qualify one immutable Posttrain GitHub release on a remote GPU host."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

REPOSITORY = "carbonteq-ai/posttrain"
PROJECT_FIXTURE = Path(__file__).resolve().parents[1] / "examples" / "gpu-qualification"


def _run(*command: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Install a tagged wheelhouse on a remote NVIDIA GPU and retain tracked evidence.",
    )
    parser.add_argument("--host", required=True, help="SSH host or configured alias")
    parser.add_argument("--release", required=True, help="immutable GitHub release tag")
    parser.add_argument("--output", type=Path, help="local JSON evidence path")
    return parser


def _remote_script(remote_root: str, release: str) -> str:
    root = shlex.quote(remote_root)
    archive = shlex.quote(f"posttrain-wheelhouse-{release}.tar.gz")
    return f"""
set -euo pipefail
cd {root}
sha256sum --check release-SHA256SUMS
mkdir wheelhouse
tar -xzf {archive} -C wheelhouse
(cd wheelhouse && sha256sum --check SHA256SUMS)
command -v uv >/dev/null
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader > gpu.csv
uv venv .venv --python 3.12
uv pip install \
  --python .venv/bin/python \
  --constraint wheelhouse/github-constraints.txt \
  --find-links wheelhouse \
  posttrain 'posttrain-lab[gpu-eval]' posttrain-observatory
export TRACKIO_DIR="$PWD/project/.posttrain/state/trackio"
export POSTTRAIN_TRACKIO_PROJECT=gpu-qualification
export POSTTRAIN_PROJECT_REVISION={shlex.quote(release)}
.venv/bin/posttrain --project-root project doctor
.venv/bin/posttrain --project-root project work-package validate \
  foundation_screen.yaml
CUDA_VISIBLE_DEVICES=0 .venv/bin/posttrain --json --project-root project \
  work-package run foundation_screen.yaml --job benchmark > execution.json
run_id=$(.venv/bin/python -c \
  'import json; print(json.load(open("execution.json"))["jobs"][0]["run_id"])')
.venv/bin/posttrain-observatory run trackio-local "$run_id" > observatory.json
.venv/bin/python - <<'PY'
import json
from pathlib import Path

execution = json.loads(Path("execution.json").read_text())
observatory = json.loads(Path("observatory.json").read_text())
job = execution["jobs"][0]
view = observatory["view"]
if execution["status"] != "succeeded" or job["status"] != "succeeded":
    raise SystemExit("remote work package did not succeed")
if view["run"]["run_id"] != job["run_id"]:
    raise SystemExit("Observatory readback run id does not match execution")
result = {{
    "schema_version": 1,
    "release": {release!r},
    "gpu": Path("gpu.csv").read_text().strip(),
    "project_id": execution["project_id"],
    "work_package_id": execution["work_package_id"],
    "job_kind": job["kind"],
    "job_definition": job["definition"],
    "run_id": job["run_id"],
    "status": job["status"],
    "observatory_mode": observatory["resolved_mode"],
    "observatory_run_id": view["run"]["run_id"],
}}
print(json.dumps(result, sort_keys=True))
PY
"""


def main() -> int:
    args = _parser().parse_args()
    for executable in ("gh", "ssh", "scp", "sha256sum"):
        if shutil.which(executable) is None:
            raise SystemExit(f"required executable is unavailable: {executable}")
    if not PROJECT_FIXTURE.is_dir():
        raise SystemExit(f"GPU qualification fixture is missing: {PROJECT_FIXTURE}")

    with tempfile.TemporaryDirectory(prefix="posttrain-gpu-release-") as directory:
        staging = Path(directory)
        _run(
            "gh",
            "release",
            "download",
            args.release,
            "--repo",
            REPOSITORY,
            "--pattern",
            "posttrain-wheelhouse-*.tar.gz",
            "--pattern",
            "release-SHA256SUMS",
            cwd=staging,
        )
        _run("sha256sum", "--check", "release-SHA256SUMS", cwd=staging)
        remote_root = _run(
            "ssh",
            "-o",
            "BatchMode=yes",
            args.host,
            "mktemp -d /tmp/posttrain-gpu-qualification.XXXXXX",
        ).stdout.strip()
        if not remote_root.startswith("/tmp/posttrain-gpu-qualification."):
            raise SystemExit(f"remote host returned an unsafe qualification directory: {remote_root!r}")
        _run(
            "scp",
            str(staging / f"posttrain-wheelhouse-{args.release}.tar.gz"),
            str(staging / "release-SHA256SUMS"),
            f"{args.host}:{remote_root}/",
        )
        _run("scp", "-r", str(PROJECT_FIXTURE), f"{args.host}:{remote_root}/project")
        completed = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", args.host, "bash", "-s"],
            input=_remote_script(remote_root, args.release),
            check=True,
            capture_output=True,
            text=True,
        )
        evidence = json.loads(completed.stdout.splitlines()[-1])

    output = args.output or (Path(".posttrain") / "state" / "qualification" / f"remote-gpu-{args.release}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"evidence": str(output.resolve()), **evidence}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
