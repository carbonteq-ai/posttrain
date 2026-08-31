from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "src" / "posttrain" / "runtime_images"
PROFILE_ROOT = ROOT / "containers" / "posttrain-job-kinds" / "verl-py313"
MODULE_PATH = PROFILE_ROOT / "validate_shared_fallback.py"
SPEC = importlib.util.spec_from_file_location("posttrain_validate_shared_fallback", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
fallback = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = fallback
SPEC.loader.exec_module(fallback)


def _install_distribution(site: Path, *, name: str, version: str) -> None:
    metadata = site / f"{name.replace('-', '_')}-{version}.dist-info"
    metadata.mkdir(parents=True, exist_ok=True)
    (metadata / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
        encoding="utf-8",
    )
    (metadata / "RECORD").write_text("", encoding="utf-8")


def _fixture(tmp_path: Path, *, backend_copy: bool = False, locked_version: str = "1.0") -> dict[str, object]:
    control = tmp_path / "control"
    backend = tmp_path / "backend"
    control.mkdir()
    backend.mkdir()
    _install_distribution(control, name="torch", version="1.0")
    if backend_copy:
        _install_distribution(backend, name="torch", version="1.0")
    lock = tmp_path / "uv.lock"
    lock.write_text(f'[[package]]\nname = "torch"\nversion = "{locked_version}"\n', encoding="utf-8")
    pth = backend / "posttrain-control-fallback.pth"
    pth.write_text(f"{control.resolve()}\n", encoding="utf-8")
    return {
        "control_site": control,
        "backend_site": backend,
        "backend_lock": lock,
        "policy": fallback.SharingPolicy(distributions=("torch",)),
        "fallback_file": pth,
    }


def test_partial_sync_fallback_is_valid_and_deterministic(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)

    first = fallback.validate_fallback(**inputs)
    second = fallback.validate_fallback(**inputs)

    assert first == second
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first == {
        "packages": [{"name": "torch", "version": "1.0"}],
        "schema_version": 1,
        "strategy": "uv-partial-sync-pth-fallback",
    }


def test_backend_copy_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(fallback.FallbackError, match="unexpectedly installed"):
        fallback.validate_fallback(**_fixture(tmp_path, backend_copy=True))


def test_lock_version_mismatch_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(fallback.FallbackError, match="does not match backend lock"):
        fallback.validate_fallback(**_fixture(tmp_path, locked_version="2.0"))


def test_fallback_file_cannot_execute_code_or_add_another_path(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    fallback_file = inputs["fallback_file"]
    assert isinstance(fallback_file, Path)
    fallback_file.write_text("import sys; sys.path.insert(0, '/tmp')\n", encoding="utf-8")

    with pytest.raises(fallback.FallbackError, match="must contain only"):
        fallback.validate_fallback(**inputs)


def test_policy_is_normalized_sorted_and_version_free() -> None:
    policy = fallback.load_policy(PROFILE_ROOT / "shared-heavy.toml")

    assert len(policy.distributions) == 16
    assert policy.distributions == tuple(sorted(policy.distributions))
    assert "torch" in policy.distributions
    assert "nvidia-cuda-runtime" not in policy.distributions
    assert "==" not in (PROFILE_ROOT / "shared-heavy.toml").read_text(encoding="utf-8")
