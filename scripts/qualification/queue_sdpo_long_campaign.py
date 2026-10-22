"""Idempotent host queue: preserve08, qualify batch32, then submit one fresh run."""

import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

IMAGE = ("registry.lan/carbonteq/posttrain-lab/posttrain-lab/posttrain-job@sha256:"
         "e976e727786135f851f4cc96356ebff1f57c75ca5f2d36a2c6aacbf6b2822337")
PREDECESSOR = "sdpo-qwen35-4b-agent08-cosine20-20261018-r1"
RUNTIME = "/var/lib/posttrain/qualifications/self-campaign-2048-20261017-r5"


def command(args, timeout=30):
    return subprocess.check_output(args, text=True, timeout=timeout).strip()


def state(name):
    result = subprocess.run(["docker", "inspect", "--format", "{{json .State}}", name],
                            text=True, capture_output=True, timeout=15, check=False)
    if result.returncode:
        if "No such" in result.stderr or "no such" in result.stderr:
            return None
        raise RuntimeError(result.stderr)
    return json.loads(result.stdout)


def record(capsule, stage, **details):
    output = capsule / "evidence/queue-state.json"
    value = {"stage": stage, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **details}
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(output)
    print(json.dumps(value), flush=True)


def wait_exit(name, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = state(name)
        if current is None:
            raise ValueError("expected accepted container missing")
        if current["Status"] == "exited":
            if current["ExitCode"] != 0:
                raise ValueError(f"preserve failed {name}, exit{current['ExitCode']}")
            return current
        if current["Status"] not in ("running", "created"):
            raise ValueError(f"unexpected retained state {current['Status']}")
        time.sleep(30)
    raise TimeoutError("bounded queue wait expired")


def wait_gpu():
    deadline = time.monotonic() + 21600
    while time.monotonic() < deadline:
        if not command(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"]):
            return
        time.sleep(30)
    raise TimeoutError("GPU did not become free; no workload interrupted")


def main(capsule):
    os.chdir(capsule)
    lock = (capsule / "evidence/queue.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    command(["sha256sum", "-c", "source-sha256.txt"], timeout=30)  # cwd is sealed capsule
    config = json.loads((capsule / "campaign.json").read_text())
    train_name, qualify_name = config["run_name"], config["qualification_run_name"]
    if state(train_name) is not None:
        record(capsule, "train_already_retained_monitor_only", run_name=train_name)
        return
    record(capsule, "waiting_for_experiment08", predecessor=PREDECESSOR)
    wait_exit(PREDECESSOR, 10800)
    launcher = str(capsule / "driver/run_sdpo_long_campaign.sh")
    if state(qualify_name) is None:
        wait_gpu()
        identifier = command(["bash", launcher, "qualify"], timeout=120).splitlines()[-1]
        record(capsule, "qualification_accepted", run_name=qualify_name, container_id=identifier)
    wait_exit(qualify_name, 7400)
    record(capsule, "qualification_cpu_audit", run_name=qualify_name)
    gate_command = ["docker", "run", "--rm", "--network", "none", "--entrypoint", "/usr/bin/timeout",
        "-e", "PYTHONPATH=/adaptive/driver:/workspace/driver:/workspace/trl:/workspace/kernel-deps:/workspace/training-deps",
        "-v", f"{capsule}:/adaptive:ro", "-v", f"{capsule}/evidence:/adaptive/evidence",
        "-v", f"{RUNTIME}:/workspace:ro", IMAGE, "120", "/opt/posttrain/venv/bin/python",
        "/adaptive/driver/qualify_sdpo_long_campaign.py", "/adaptive"]
    with (capsule / "evidence/qualification-cpu-audit.log").open("w") as log:
        subprocess.run(gate_command, stdout=log, stderr=subprocess.STDOUT, timeout=150, check=True)
    gate = json.loads((capsule / "evidence/qualification-gate.json").read_text())
    if not gate["qualified"]:
        raise ValueError("qualification rejected")
    wait_gpu()
    # Launcher repeats retained-container/source/GPU guards immediately before submission.
    identifier = command(["bash", launcher, "train"], timeout=120).splitlines()[-1]
    record(capsule, "training_accepted", run_name=train_name, container_id=identifier,
           source_manifest_sha256=hashlib.sha256((capsule / "source-sha256.txt").read_bytes()).hexdigest())


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    try:
        main(root)
    except BlockingIOError:
        print("Existing queue owns this capsule; no duplicate", flush=True)
        sys.exit(0)
    except Exception as exc:
        record(root, "failed_preserve_and_report", error=f"{type(exc).__name__}: {exc}")
        sys.exit(1)
