from __future__ import annotations

from pathlib import Path

import pytest
from posttrain.common import ContractError
from posttrain.execution import BundleRef, build_bundle, verify_bundle


def test_bundle_is_explicit_deterministic_and_verifiable(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "job.py").write_text("print('ok')\n")
    (project / ".env").write_text("must-not-be-implicitly-copied\n")
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text('{"id":"one"}\n')

    first = build_bundle(
        {"src": project / "job.py", "data/train.jsonl": dataset},
        (tmp_path / "first").resolve(),
    )
    second = build_bundle(
        {"data/train.jsonl": dataset, "src": project / "job.py"},
        (tmp_path / "second").resolve(),
    )

    assert first.digest == second.digest
    assert not (first.path / ".env").exists()
    verify_bundle(first)


def test_bundle_rejects_tampering_path_escape_and_symlinks(tmp_path: Path) -> None:
    payload = tmp_path / "payload.txt"
    payload.write_text("original")
    bundle = build_bundle({"payload.txt": payload}, (tmp_path / "bundle").resolve())
    (bundle.path / "payload.txt").write_text("changed")

    with pytest.raises(ContractError, match="manifest does not match"):
        verify_bundle(bundle)
    with pytest.raises(ContractError, match="normalized relative"):
        build_bundle({"../escape": payload}, (tmp_path / "escape").resolve())

    link = tmp_path / "link"
    link.symlink_to(payload)
    with pytest.raises(ContractError, match="symlink"):
        build_bundle({"link": link}, (tmp_path / "linked").resolve())


def test_bundle_ref_digest_must_match_manifest(tmp_path: Path) -> None:
    payload = tmp_path / "payload.txt"
    payload.write_text("original")
    bundle = build_bundle({"payload.txt": payload}, (tmp_path / "bundle").resolve())

    with pytest.raises(ContractError, match="digest does not match"):
        verify_bundle(BundleRef(bundle.path, "0" * 64))


@pytest.mark.parametrize("fixture", ("runtime_smoke_job.py", "queue_probe_job.py"))
def test_provider_qualification_payloads_are_explicit_verifiable_bundle_inputs(tmp_path: Path, fixture: str) -> None:
    payload = Path(__file__).parent / "fixtures" / fixture

    bundle = build_bundle({"job.py": payload}, (tmp_path / fixture).resolve())

    assert (bundle.path / "job.py").read_bytes() == payload.read_bytes()
    verify_bundle(bundle)
