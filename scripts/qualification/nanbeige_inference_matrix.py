"""Qualify Nanbeige serving variants sequentially on one isolated GPU.

This probes inference/runtime behavior only. It does not calibrate judge quality
and must not be used as evidence that a scorer is suitable for training.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

TARGET_REPO = "Nanbeige/Nanbeige4.2-3B"
TARGET_REVISION = "3384e426066d1a49c3aea90a7190b81260a6533f"
DRAFT_REPO = "Nanbeige/Nanbeige4.2-3B-DSpark"
DRAFT_REVISION = "5f12c792dabbfcdc4a0cf504e75f7216706d9590"
VLLM_REVISION = "62f6de733d7ae63b759329993bc209e67afdf431"
EXPECTED_GPU_NAME = "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"


@dataclass(frozen=True, slots=True)
class Profile:
    name: str
    turboquant: bool = False
    dspark: bool = False
    supported: bool = True
    unsupported_reason: str | None = None


PROFILES = (
    Profile("standard"),
    Profile("turboquant", turboquant=True),
    Profile("dspark", dspark=True),
    Profile(
        "dspark-turboquant",
        turboquant=True,
        dspark=True,
        supported=False,
        unsupported_reason=(
            "Nanbeige vLLM 62f6de733 cannot apply TurboQuant KV cache to "
            "DSpark's non-causal draft attention"
        ),
    ),
)


def _request_json(url: str, payload: dict[str, Any] | None = None, *, timeout: float = 10) -> Any:
    body = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed loopback URL
        return json.loads(response.read())


def _read_text(url: str, *, timeout: float = 5) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - fixed loopback URL
        return response.read().decode(errors="replace")


def _gpu_snapshot() -> dict[str, str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    fields = (
        "name",
        "uuid",
        "driver_version",
        "memory_total_mib",
        "memory_used_mib",
        "memory_free_mib",
        "utilization_pct",
    )
    values = [value.strip() for value in result.stdout.strip().split(",")]
    return dict(zip(fields, values, strict=True))


def _command(profile: Profile, port: int) -> list[str]:
    command = [
        "vllm",
        "serve",
        TARGET_REPO,
        "--revision",
        TARGET_REVISION,
        "--served-model-name",
        "nanbeige-judge",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--max-model-len",
        "32768",
        "--gpu-memory-utilization",
        "0.80",
        "--dtype",
        "bfloat16",
        "--enforce-eager",
        "--enable-chunked-prefill",
        "--max-num-seqs",
        "2",
        "--max-num-batched-tokens",
        "8192",
        "--reasoning-parser",
        "nanbeige",
    ]
    if profile.turboquant:
        command.extend(("--kv-cache-dtype", "turboquant_k8v4"))
    if profile.dspark:
        command.extend(
            (
                "--speculative-config",
                json.dumps(
                    {
                        "method": "dspark",
                        "num_speculative_tokens": 7,
                        "model": DRAFT_REPO,
                        "revision": DRAFT_REVISION,
                    },
                    separators=(",", ":"),
                ),
            )
        )
    return command


def _wait_ready(process: subprocess.Popen[str], base_url: str, log_path: Path, timeout: float) -> float:
    started = time.monotonic()
    deadline = started + timeout
    while time.monotonic() < deadline:
        returncode = process.poll()
        if returncode is not None:
            tail = log_path.read_text(errors="replace")[-20_000:]
            raise RuntimeError(f"server exited with code {returncode}:\n{tail}")
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=1):  # noqa: S310 - fixed loopback URL
                return time.monotonic() - started
        except (OSError, urllib.error.URLError):
            time.sleep(0.5)
    raise TimeoutError(f"server did not become healthy within {timeout} seconds")


def _generate(base_url: str, *, enable_thinking: bool, max_tokens: int) -> dict[str, Any]:
    prompt = (
        "Rate whether the claim is correct. Claim: 15:00 in Asia/Karachi "
        "(UTC+05:00) is 10:00 UTC. Return a concise answer."
        if enable_thinking
        else "Reply with exactly READY and no other text."
    )
    payload = {
        "model": "nanbeige-judge",
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    started = time.monotonic()
    response = _request_json(f"{base_url}/v1/chat/completions", payload, timeout=180)
    elapsed = time.monotonic() - started
    choice = response["choices"][0]
    message = choice["message"]
    return {
        "elapsed_seconds": elapsed,
        "finish_reason": choice.get("finish_reason"),
        "message": message,
        "content": message.get("content"),
        "reasoning": message.get("reasoning"),
        "reasoning_content": message.get("reasoning_content"),
        "usage": response.get("usage"),
    }


def _stop(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=20)


def _run_profile(profile: Profile, output: Path, port: int, startup_timeout: float) -> dict[str, Any]:
    profile_root = output / profile.name
    profile_root.mkdir(parents=True, exist_ok=False)
    log_path = profile_root / "server.log"
    command = _command(profile, port)
    before = _gpu_snapshot()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(  # noqa: S603 - frozen internal command
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            base_url = f"http://127.0.0.1:{port}"
            startup_seconds = _wait_ready(process, base_url, log_path, startup_timeout)
            ready = _gpu_snapshot()
            direct = _generate(base_url, enable_thinking=False, max_tokens=128)
            reasoning = _generate(base_url, enable_thinking=True, max_tokens=4_096)
            metrics = _read_text(f"{base_url}/metrics")
            (profile_root / "metrics.prom").write_text(metrics)
            result = {
                "profile": asdict(profile),
                "command": command,
                "startup_seconds": startup_seconds,
                "gpu_before": before,
                "gpu_ready": ready,
                "direct": direct,
                "reasoning": reasoning,
                "passed": all(
                    probe["finish_reason"] == "stop" and bool(probe["content"])
                    for probe in (direct, reasoning)
                ),
            }
        finally:
            _stop(process)
    result["gpu_after"] = _gpu_snapshot()
    (profile_root / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--runtime-image",
        required=True,
        help="Immutable OCI image reference used for this run (must contain @sha256:).",
    )
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--startup-timeout", type=float, default=1_200)
    args = parser.parse_args()
    if "@sha256:" not in args.runtime_image:
        parser.error("--runtime-image must be an immutable digest reference containing @sha256:")
    args.output.mkdir(parents=True, exist_ok=False)
    gpu_initial = _gpu_snapshot()
    if gpu_initial["name"] != EXPECTED_GPU_NAME:
        raise RuntimeError(
            f"qualification requires {EXPECTED_GPU_NAME!r}, found {gpu_initial['name']!r}"
        )
    manifest = {
        "scope": "inference runtime qualification; not judge-quality calibration",
        "target": {"repo_id": TARGET_REPO, "revision": TARGET_REVISION},
        "draft": {"repo_id": DRAFT_REPO, "revision": DRAFT_REVISION},
        "runtime": {
            "repository": "https://github.com/Nanbeige/vllm",
            "revision": VLLM_REVISION,
            "image": args.runtime_image,
        },
        "profiles": [asdict(profile) for profile in PROFILES],
        "gpu_initial": gpu_initial,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    results: list[dict[str, Any]] = []
    for profile in PROFILES:
        if not profile.supported:
            result = {
                "profile": asdict(profile),
                "passed": False,
                "status": "unsupported",
                "reason": profile.unsupported_reason,
                "evidence": (
                    "matrix-1 failed during draft initialization because all available "
                    "attention backends reject turboquant_k8v4 for non-causal attention"
                ),
            }
            profile_root = args.output / profile.name
            profile_root.mkdir(parents=True, exist_ok=False)
            (profile_root / "result.json").write_text(
                json.dumps(result, indent=2, sort_keys=True)
            )
            results.append(result)
            print(
                json.dumps({"profile": profile.name, "status": result["status"]}),
                flush=True,
            )
            continue
        try:
            result = _run_profile(profile, args.output, args.port, args.startup_timeout)
        except Exception as error:
            result = {
                "profile": asdict(profile),
                "passed": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
            (args.output / profile.name / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
        results.append(result)
        print(json.dumps({"profile": profile.name, "passed": result["passed"]}), flush=True)
    supported_results = [result for result in results if result.get("status") != "unsupported"]
    summary = {
        "passed": all(result["passed"] for result in supported_results),
        "unsupported_profiles": [
            result["profile"]["name"]
            for result in results
            if result.get("status") == "unsupported"
        ],
        "results": results,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
