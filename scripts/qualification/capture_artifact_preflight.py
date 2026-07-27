#!/usr/bin/env python3
"""Capture a redacted artifact-lifecycle preflight without changing services."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from posttrain.work import GIB, StorageCapacity, StorageRequirement, assess_storage

REPO = Path(__file__).resolve().parents[2]
DEFAULT_AI_INFRA = Path("/home/hammad/projects/ai-infra")
FULL_COMMIT = re.compile(r"(?<![0-9a-f])([0-9a-f]{40})(?![0-9a-f])")
IMAGE_LINE = re.compile(r"^\s*image:\s*(?P<image>\S+)\s*$", re.MULTILINE)
IMAGE_REVISION = re.compile(r":(?P<revision>[0-9a-f]{7,40})$")
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _run(command: list[str], *, cwd: Path, timeout: int = 30) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"ok": False, "returncode": None, "output": type(error).__name__}
    output = ANSI_ESCAPE.sub("", completed.stdout.strip())
    if completed.returncode != 0:
        output = ANSI_ESCAPE.sub("", completed.stderr.strip()) or output
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "output": output[-8000:],
    }


def _git_snapshot(repo: Path) -> dict[str, Any]:
    revision = _run(["git", "rev-parse", "HEAD"], cwd=repo)
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo)
    status = _run(["git", "status", "--porcelain=v1"], cwd=repo)
    dirty_paths = []
    if status["ok"]:
        dirty_paths = [line[3:] for line in str(status["output"]).splitlines() if len(line) >= 4]
    return {
        "revision": revision["output"] if revision["ok"] else None,
        "branch": branch["output"] if branch["ok"] else None,
        "dirty_paths": dirty_paths,
    }


def _trackio_pin() -> str | None:
    manifest = (REPO / "packages/tracking-trackio/pyproject.toml").read_text(encoding="utf-8")
    for line in manifest.splitlines():
        if "carbonteq-trackio @" not in line:
            continue
        match = FULL_COMMIT.search(line)
        if match:
            return match.group(1)
    return None


def _declared_trackio_image(ai_infra: Path) -> str | None:
    compose = ai_infra / "ansible/roles/control/files/compose.yml"
    if not compose.is_file():
        return None
    contents = compose.read_text(encoding="utf-8")
    trackio_block = contents.split("\n  trackio:\n", maxsplit=1)
    if len(trackio_block) != 2:
        return None
    match = IMAGE_LINE.search(trackio_block[1])
    return match.group("image") if match else None


def _revision_alignment(expected: str | None, image: str | None) -> dict[str, Any]:
    match = IMAGE_REVISION.search(image or "")
    observed = match.group("revision") if match else None
    aligned = bool(expected and observed and expected.startswith(observed))
    return {
        "expected_revision": expected,
        "declared_image": image,
        "declared_revision": observed,
        "aligned": aligned,
    }


def _trackio_reachability(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url.rstrip("/") + "/", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            content_type = response.headers.get("content-type")
            return {
                "reachable": 200 <= response.status < 400,
                "status_code": response.status,
                "content_type": content_type,
            }
    except (urllib.error.URLError, TimeoutError) as error:
        return {
            "reachable": False,
            "status_code": None,
            "error_type": type(error).__name__,
        }


def _base_models() -> list[dict[str, str]]:
    catalog_path = REPO / "packages/catalog/src/posttrain/catalog/base/models.yaml"
    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    selected: list[dict[str, str]] = []
    for model_id, value in catalog.get("model", {}).items():
        artifact = value.get("artifact", {})
        repo_id = artifact.get("repo_id")
        if repo_id not in {"Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B"}:
            continue
        selected.append(
            {
                "selection_id": model_id,
                "repo_id": repo_id,
                "revision": artifact["revision"],
            }
        )
    return sorted(selected, key=lambda item: item["repo_id"])


def _local_gpu() -> dict[str, Any]:
    result = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ],
        cwd=REPO,
    )
    if not result["ok"]:
        return {"available": False, "diagnostic": result["output"]}
    fields = [field.strip() for field in str(result["output"]).splitlines()[0].split(",")]
    if len(fields) != 4:
        return {"available": False, "diagnostic": "unexpected nvidia-smi output"}
    return {
        "available": True,
        "name": fields[0],
        "memory_total_mib": int(fields[1]),
        "memory_free_mib": int(fields[2]),
        "driver_version": fields[3],
    }


def _dstack_snapshot(ai_infra: Path) -> dict[str, Any]:
    wrapper = ai_infra / "scripts/dstack"
    if not wrapper.is_file():
        return {"available": False, "diagnostic": "dstack wrapper is missing"}
    version = _run([str(wrapper), "--version"], cwd=ai_infra)
    fleet = _run([str(wrapper), "fleet", "list"], cwd=ai_infra)
    fleet_text = str(fleet["output"])
    node_match = re.search(r"local-gpu-workers\s+(?P<nodes>\d+)", fleet_text)
    return {
        "available": bool(version["ok"] and fleet["ok"]),
        "version": version["output"] if version["ok"] else None,
        "configured_nodes": int(node_match.group("nodes")) if node_match else None,
        "idle_nodes": len(re.findall(r"\bidle\b", fleet_text)),
        "fleet_output": fleet_text,
    }


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trackio-url", default="https://trackio.lan")
    parser.add_argument("--ai-infra-dir", type=Path, default=DEFAULT_AI_INFRA)
    parser.add_argument("--storage-path", type=Path, default=REPO)
    parser.add_argument("--download-gib", type=float, default=0)
    parser.add_argument("--workspace-gib", type=float, default=0.0625)
    parser.add_argument("--retained-output-gib", type=float, default=0.0625)
    parser.add_argument("--safety-margin-ratio", type=float, default=0.15)
    parser.add_argument("--minimum-free-gib", type=float, default=30)
    return parser


def main() -> int:
    args = _parser().parse_args()
    disk = shutil.disk_usage(args.storage_path)
    requirement = StorageRequirement(
        download_bytes=int(args.download_gib * GIB),
        peak_workspace_bytes=int(args.workspace_gib * GIB),
        retained_output_bytes=int(args.retained_output_gib * GIB),
        safety_margin_ratio=args.safety_margin_ratio,
        minimum_free_bytes=int(args.minimum_free_gib * GIB),
    )
    admission = assess_storage(
        StorageCapacity(total_bytes=disk.total, available_bytes=disk.free),
        requirement,
    )
    pin = _trackio_pin()
    image = _declared_trackio_image(args.ai_infra_dir)
    alignment = _revision_alignment(pin, image)
    trackio = _trackio_reachability(args.trackio_url)
    dstack = _dstack_snapshot(args.ai_infra_dir)

    blockers: list[str] = []
    if not admission.accepted:
        blockers.append("storage_admission")
    if not alignment["aligned"]:
        blockers.append("trackio_revision_drift")
    if not trackio["reachable"]:
        blockers.append("trackio_unreachable")
    if not dstack["available"] or dstack["configured_nodes"] != 2:
        blockers.append("dstack_fleet")

    payload: dict[str, Any] = {
        "schema_version": 1,
        "captured_at": datetime.now(UTC).isoformat(),
        "status": "pass" if not blockers else "blocked",
        "blockers": blockers,
        "framework": _git_snapshot(REPO),
        "trackio": {
            "url": args.trackio_url,
            "reachability": trackio,
            "revision_alignment": alignment,
            "write_token_present": bool(os.environ.get("TRACKIO_WRITE_TOKEN")),
        },
        "dstack": dstack,
        "storage": {
            "path": str(args.storage_path.resolve()),
            "total_bytes": disk.total,
            "available_bytes": disk.free,
            "requirement": asdict(requirement),
            "admission": asdict(admission),
        },
        "local_gpu": _local_gpu(),
        "base_models": _base_models(),
        "credentials": {
            "trackio_write_token_present": bool(os.environ.get("TRACKIO_WRITE_TOKEN")),
            "hugging_face_token_present": bool(
                os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            ),
        },
    }
    _write_atomic(args.output, payload)
    print(f"preflight: {payload['status']}")
    print(f"blockers: {','.join(blockers) if blockers else 'none'}")
    print(f"dstack_workers: {dstack['configured_nodes']}")
    print(f"trackio_reachable: {trackio['reachable']}")
    print(f"storage_admission: {admission.status}")
    print(f"evidence: {args.output.resolve()}")
    return 0 if not blockers else 2


if __name__ == "__main__":
    sys.exit(main())
