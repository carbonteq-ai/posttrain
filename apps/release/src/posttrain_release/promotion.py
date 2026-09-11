"""Receipt binding a qualified development artifact to a stable release tag."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path


def create_promotion_receipt(
    candidate_receipt: Path,
    *,
    final_receipt: Path,
    target_version: str,
    candidate_run_id: str,
    candidate_source_sha: str,
    candidate_source_tree: str,
    merged_sha: str,
    merged_tree: str,
) -> dict[str, object]:
    """Bind an already qualified distribution receipt to one merged release tree."""

    source = candidate_receipt.read_bytes()
    distribution = json.loads(source)
    if not isinstance(distribution, dict) or distribution.get("schema") != "posttrain.python-release-receipt.v1":
        raise ValueError("promotion requires a valid Python distribution receipt")
    candidate_version = distribution.get("version")
    if not isinstance(candidate_version, str) or not candidate_version:
        raise ValueError("distribution receipt is missing version")
    if re.fullmatch(rf"{re.escape(target_version)}rc[1-9][0-9]*", candidate_version) is None:
        raise ValueError(f"candidate version {candidate_version!r} is not an RC of target {target_version!r}")
    final_source = final_receipt.read_bytes()
    final_distribution = json.loads(final_source)
    if not isinstance(final_distribution, dict) or final_distribution.get("schema") != distribution.get("schema"):
        raise ValueError("promotion requires a valid final Python distribution receipt")
    if final_distribution.get("version") != target_version:
        raise ValueError("final distribution receipt does not match the target version")
    if final_distribution.get("packages") != distribution.get("packages"):
        raise ValueError("candidate and final distribution package sets differ")
    if final_distribution.get("image_manifest_sha256") != distribution.get("image_manifest_sha256"):
        raise ValueError("candidate and final runtime image manifests differ")
    values = {
        "candidate_source_sha": candidate_source_sha,
        "candidate_source_tree": candidate_source_tree,
        "merged_sha": merged_sha,
        "merged_tree": merged_tree,
    }
    for name, value in values.items():
        if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"promotion {name} must be a full lowercase Git SHA")
    if not candidate_run_id:
        raise ValueError("promotion requires candidate run id")
    return {
        "schema": "posttrain.release-promotion.v2",
        "version": target_version,
        "candidate_version": candidate_version,
        "candidate_run_id": candidate_run_id,
        **values,
        "candidate_receipt_sha256": hashlib.sha256(source).hexdigest(),
        "final_receipt_sha256": hashlib.sha256(final_source).hexdigest(),
        "created_at": datetime.now(UTC).isoformat(),
    }


def write_promotion_receipt(receipt: dict[str, object], destination: Path) -> None:
    destination.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
