"""Tests for the package-owned representative serving corpus builder."""

from __future__ import annotations

import hashlib
import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
from posttrain.common import Workload
from posttrain.serve import materialize_workload, verify_workload
from posttrain.serve.benchmarks.general_serving import GENERAL_SERVING_V1
from posttrain.serve.benchmarks.general_serving.build import _first_party_records

CORPUS_ROOT = Path(__file__).parents[1] / "src/posttrain/serve/benchmarks/general_serving/resources"


def test_definition_declares_inert_builder_and_expected_population() -> None:
    assert GENERAL_SERVING_V1.builder.endswith(":build")
    assert GENERAL_SERVING_V1.record_count == 128
    assert dict(GENERAL_SERVING_V1.category_counts) == {
        "chat": 8,
        "code": 32,
        "extraction": 8,
        "reasoning": 64,
        "structured-output": 8,
        "tool-use": 8,
    }
    assert GENERAL_SERVING_V1.expected_content_sha256 == (
        "9a9467fd8a5e744968d09a4d8fd6f4d92a089c50a84e1e6e7e5c5520a9f4e50e"
    )


def test_definition_import_does_not_import_builder() -> None:
    command = (
        "import sys; "
        "from posttrain.serve.benchmarks.general_serving import GENERAL_SERVING_V1; "
        "assert GENERAL_SERVING_V1.id == 'general-serving-v1'; "
        "assert 'posttrain.serve.benchmarks.general_serving.build' not in sys.modules"
    )
    result = subprocess.run([sys.executable, "-c", command], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_first_party_prompts_are_reviewable_package_input() -> None:
    records = _first_party_records()
    assert len(records) == 32
    assert {str(record["category"]) for record in records} == {
        "chat",
        "extraction",
        "structured-output",
        "tool-use",
    }
    assert sum(bool(record.get("tools")) for record in records) == 8


def test_checked_in_corpus_matches_definition_digest() -> None:
    records = CORPUS_ROOT.joinpath("general-serving-v1.jsonl").read_bytes()
    assert len(records.splitlines()) == GENERAL_SERVING_V1.record_count
    assert hashlib.sha256(records).hexdigest() == GENERAL_SERVING_V1.expected_content_sha256


def test_serve_owned_operation_materializes_and_verifies_workload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records_text = CORPUS_ROOT.joinpath("general-serving-v1.jsonl").read_text(encoding="utf-8")
    manifest_text = CORPUS_ROOT.joinpath("general-serving-v1.manifest.json").read_text(encoding="utf-8")
    build_module = importlib.import_module("posttrain.serve.benchmarks.general_serving.build")
    monkeypatch.setattr(build_module, "build", lambda: (records_text, manifest_text))
    workload = Workload(
        id="workloads/general-serving-32k-sweep@1",
        revision="1",
        requests={
            "corpus": {
                "id": "general-serving-v1",
                "revision": "1",
                "digest": GENERAL_SERVING_V1.expected_content_sha256,
            }
        },
    )

    materialized = materialize_workload(workload, output=tmp_path / "population")
    verified = verify_workload(workload)

    assert Path(materialized.path).read_text(encoding="utf-8") == records_text
    assert Path(materialized.manifest).read_text(encoding="utf-8") == manifest_text
    assert materialized.materialized is True
    assert verified.materialized is False
    assert verified.content_sha256 == materialized.content_sha256
    assert verified.record_count == 128


@pytest.mark.network
def test_builder_reproduces_checked_in_corpus_with_pinned_sources() -> None:
    if os.environ.get("POSTTRAIN_SERVING_CORPUS_NETWORK") != "1":
        pytest.skip("set POSTTRAIN_SERVING_CORPUS_NETWORK=1 to run the pinned-source parity gate")
    from posttrain.serve.benchmarks.general_serving.build import build

    records_text, manifest_text = build()
    assert records_text == CORPUS_ROOT.joinpath("general-serving-v1.jsonl").read_text(encoding="utf-8")
    assert manifest_text == CORPUS_ROOT.joinpath("general-serving-v1.manifest.json").read_text(encoding="utf-8")
