"""Release tooling, and the boundary that keeps it away from consumers."""

from __future__ import annotations

import json
import os
import subprocess
import tomllib
import zipfile
from pathlib import Path, PurePosixPath
from shutil import which

import posttrain_release.versioning as versioning
import pytest
from posttrain.runtime_images import (
    RUNTIME_VARIANTS,
    VERL_BACKEND_LOCK,
    constraint_lock,
    lock_digest,
)
from posttrain.runtime_images.manifest import ManifestError, PublishedImage, load_manifest
from posttrain_release.artifacts import create_distribution_receipt, verify_distribution_receipt
from posttrain_release.candidate import next_candidate_version
from posttrain_release.cli import main
from posttrain_release.manifest_render import render_manifest
from posttrain_release.repository_audit import evaluate_repository, inspect_repository
from posttrain_release.runtime_lock import materialize_runtime_lock
from posttrain_release.versioning import (
    check_release,
    lock_dependencies,
    prepare_release,
    stage_release,
)

_REPOSITORY_ROOT_DEPTH = 3


def _fake_dstack(tmp_path: Path, responses: list[str]) -> tuple[Path, Path]:
    response_root = tmp_path / "dstack-responses"
    response_root.mkdir()
    for index, response in enumerate(responses, start=1):
        (response_root / str(index)).write_text(response, encoding="utf-8")
    (response_root / "last").write_text(responses[-1], encoding="utf-8")
    state = tmp_path / "dstack-attempts"
    executable = tmp_path / "dstack"
    executable.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
state="${FAKE_DSTACK_STATE:?}"
responses="${FAKE_DSTACK_RESPONSES:?}"
attempt=0
if [[ -f "${state}" ]]; then
  attempt="$(<"${state}")"
fi
attempt="$((attempt + 1))"
printf '%s' "${attempt}" >"${state}"
response="${responses}/${attempt}"
if [[ ! -f "${response}" ]]; then
  response="${responses}/last"
fi
command cat "${response}"
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable, state


def _dstack_fleet(*, status: str = "idle") -> str:
    return json.dumps(
        {
            "id": "fleet-1",
            "name": "local-gpu-workers",
            "status": "active",
            "instances": [
                {
                    "id": "worker-1",
                    "instance_num": 1,
                    "hostname": "carbonteq-ai-workstation.lan",
                    "status": status,
                    "health_status": "healthy",
                    "unreachable": False,
                    "total_blocks": 1,
                    "instance_type": {
                        "resources": {
                            "gpus": [{"name": "RTXPRO6000", "vendor": "nvidia", "memory_mib": 98304}],
                            "cpus": 32,
                            "memory_mib": 60856,
                            "disk": {"size_mib": 1200803},
                        }
                    },
                }
            ],
        }
    )


def _version_repository(root: Path, version: str = "0.2.5") -> None:
    (root / "release").mkdir(parents=True)
    (root / "release" / "manifest.toml").write_text(
        f'schema_version = 1\nversion = "{version}"\n',
        encoding="utf-8",
    )
    (root / "pyproject.toml").write_text(
        """[dependency-groups]
dev = []

[tool.uv.workspace]
members = ["packages/*"]

[tool.uv]
package = false
""",
        encoding="utf-8",
    )
    train = root / "packages/train"
    train.mkdir(parents=True)
    train.joinpath("pyproject.toml").write_text(
        """[project]
name = "posttrain-train"
version = "0.0.0"
dependencies = ["posttrain-common"]

[project.optional-dependencies]
trl = ["trl==1.9.2.post1"]

[tool.posttrain.trl]
version = "1.9.2.post1"
release-tag = "carbonteq-v1.9.2.post1"
source-revision = "91b0ce707631d503fbed337b42444a9d3fac3acb"
wheel-sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
sdist-sha256 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
""",
        encoding="utf-8",
    )
    (root / "uv.lock").write_text(
        """version = 1
revision = 3
requires-python = ">=3.13"

[[package]]
name = "posttrain-train"
version = "0.0.0"
source = { editable = "packages/train" }
dependencies = [
    { name = "posttrain-common" },
]
""",
        encoding="utf-8",
    )
    digest = __import__("hashlib").sha256((root / "uv.lock").read_bytes()).hexdigest()
    catalog = root / "packages/catalog/src/posttrain/catalog/base"
    catalog.mkdir(parents=True)
    (catalog / "training.yaml").write_text(
        "training:\n  example:\n    backend_options:\n      dependency_lock: trl-fork@current\n",
        encoding="utf-8",
    )
    (catalog / "locks.toml").write_text(
        f"""schema_version = 1

[locks."trl-fork@current"]
source = "uv.lock"
source_revision = "91b0ce707631d503fbed337b42444a9d3fac3acb"
dependency_lock_sha256 = "{digest}"
""",
        encoding="utf-8",
    )


def _image(name: str, variant: str, digest_char: str) -> PublishedImage:
    lock = constraint_lock(variant)
    backend_lock = VERL_BACKEND_LOCK if variant == "online-rl-verl-py313" else None
    return PublishedImage(
        name=name,
        repository=f"posttrain-kind-{variant}",
        digest="sha256:" + digest_char * 64,
        lock_digest=lock_digest(lock),
        constraint_lock=lock,
        backend_constraint_lock=backend_lock,
        backend_lock_digest=lock_digest(backend_lock) if backend_lock is not None else None,
    )


def _render_all() -> str:
    return render_manifest(
        framework_version="9.9.9",
        default_prefix="registry.lan/carbonteq",
        base=PublishedImage(
            name="base",
            repository="posttrain-base",
            digest="sha256:" + "a" * 64,
            lock_digest=lock_digest(),
            constraint_lock=PurePosixPath("containers/posttrain-job-kinds/locks/workspace.lock.txt"),
        ),
        kinds={
            variant: _image(f"kinds.{variant}", variant, str(index)) for index, variant in enumerate(RUNTIME_VARIANTS)
        },
    )


def test_rendered_manifest_parses_and_round_trips() -> None:
    document = tomllib.loads(_render_all())
    assert document["schema_version"] == 1
    assert document["framework_version"] == "9.9.9"
    assert document["default_prefix"] == "registry.lan/carbonteq"
    assert set(document["kinds"]) == set(RUNTIME_VARIANTS)


def test_release_check_uses_the_manifest_as_version_authority(tmp_path: Path) -> None:
    _version_repository(tmp_path)

    result = check_release(tmp_path)

    assert result.version == "0.2.5"
    assert result.package_count == 1
    assert result.internal_pin_count == 1


def test_runtime_lock_materializes_published_internal_wheel_receipts(tmp_path: Path) -> None:
    _version_repository(tmp_path)
    runtime = tmp_path / "packages/runtime-images/src/posttrain/runtime_images/containers/posttrain-job-kinds"
    (runtime / "profiles").mkdir(parents=True)
    (runtime / "profiles/supervised.txt").write_text("trl==1.9.2.post1\n", encoding="utf-8")
    (runtime / "locks").mkdir()
    lock = runtime / "locks/workspace.lock.txt"
    lock.write_text(
        "--index-url https://pypi.org/simple\n\n"
        "trl @ git+https://github.com/carbonteq-ai/trl.git@" + "a" * 40 + "\n"
        "    # via posttrain-train\n"
        "typer==0.25.1 \\\n"
        "    --hash=sha256:" + "b" * 64 + "\n",
        encoding="utf-8",
    )
    with (tmp_path / "uv.lock").open("a", encoding="utf-8") as handle:
        handle.write(
            "\n[[package]]\n"
            'name = "trl"\n'
            'version = "1.9.2.post1"\n'
            'source = { registry = "https://pypi.lan/carbonteq/stable/+simple/" }\n'
            'wheels = [{ url = "https://pypi.lan/carbonteq/stable/+f/abc/trl-1.9.2.post1-py3-none-any.whl", '
            'hash = "sha256:' + "c" * 64 + '" }]\n'
        )
    lock_dependencies(tmp_path)

    pending = check_release(tmp_path, allow_pending_runtime_lock=True)
    assert pending.runtime_lock_pending is True
    with pytest.raises(ValueError, match="lock-runtime-dependencies"):
        check_release(tmp_path)

    result = materialize_runtime_lock(tmp_path)

    assert result.changed is True
    assert result.packages == ("trl",)
    assert (
        "trl @ https://pypi.lan/carbonteq/stable/+f/abc/trl-1.9.2.post1-py3-none-any.whl#sha256=" + "c" * 64
    ) in lock.read_text(encoding="utf-8")
    assert check_release(tmp_path).runtime_lock_pending is False


def test_pending_runtime_lock_allows_old_image_manifest_until_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _version_repository(tmp_path)
    runtime = tmp_path / "packages/runtime-images/src/posttrain/runtime_images/containers/posttrain-job-kinds"
    (runtime / "profiles").mkdir(parents=True)
    (runtime / "profiles/supervised.txt").write_text(
        "carbonteq-trackio==0.31.5.post12\ntrl==1.9.2.post1\n", encoding="utf-8"
    )
    (runtime / "locks").mkdir()
    (runtime / "locks/workspace.lock.txt").write_text(
        "carbonteq-trackio @ git+https://example.invalid/trackio@" + "a" * 40 + "\n", encoding="utf-8"
    )
    with (tmp_path / "uv.lock").open("a", encoding="utf-8") as handle:
        handle.write(
            "\n[[package]]\n"
            'name = "carbonteq-trackio"\n'
            'version = "0.31.5.post12"\n'
            'source = { registry = "https://pypi.lan/carbonteq/stable/+simple/" }\n'
            'wheels = [{ url = "https://pypi.lan/carbonteq/stable/+f/abc/trackio.whl", '
            'hash = "sha256:' + "c" * 64 + '" }]\n'
        )
    lock_dependencies(tmp_path)
    versioning_lock_error = (
        "base: published image records constraint lock digest old, but the shipped lock hashes to new"
    )

    def fake_load_manifest(*, verify_locks: bool = True, verify_variants: bool = True) -> object:
        del verify_variants
        if verify_locks:
            raise ManifestError(versioning_lock_error)
        return object()

    monkeypatch.setattr(versioning, "load_manifest", fake_load_manifest)
    pending = check_release(tmp_path, allow_pending_runtime_lock=True)
    assert pending.runtime_lock_pending is True


def test_release_check_ignores_a_virtual_root_but_checks_publishable_members(tmp_path: Path) -> None:
    _version_repository(tmp_path)
    member = tmp_path / "packages/train/pyproject.toml"
    member.write_text(
        member.read_text(encoding="utf-8").replace('version = "0.0.0"', 'version = "9.9.9"', 1),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"packages/train/pyproject[.]toml: source project[.]version is '9[.]9[.]9'",
    ):
        check_release(tmp_path)


def test_repository_audit_reports_legacy_root_ignore_and_documentation_findings(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text(".agents/\n", encoding="utf-8")
    (tmp_path / "README.md").write_text(
        "[maintained](docs/exists.md) [missing](docs/missing.md) [outside](../outside.md)\n```markdown\n[example](docs/example.md)\n```\n",
        encoding="utf-8",
    )
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "exists.md").write_text("present\n", encoding="utf-8")
    (tmp_path / ".agents").mkdir()
    (tmp_path / ".agents" / "old-plan.md").write_text("legacy\n", encoding="utf-8")

    report = evaluate_repository(
        repository_root=tmp_path,
        root_entries=("README.md", "docs", ".agents"),
        tracked_ignored_paths=(Path(".agents/old-plan.md"),),
        markdown_documents={
            Path("README.md"): (tmp_path / "README.md").read_text(encoding="utf-8"),
        },
    )

    assert report.unreviewed_root_entries == (".agents",)
    assert report.tracked_ignored_paths == (Path(".agents/old-plan.md"),)
    assert [(link.source, link.target) for link in report.broken_markdown_links] == [
        (Path("README.md"), "../outside.md"),
        (Path("README.md"), "docs/missing.md"),
    ]
    assert "repository ownership audit (report-only)" in report.render()


def test_repository_audit_uses_git_ignore_rules_for_tracked_files(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text(".agents/\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("clean\n", encoding="utf-8")
    (tmp_path / ".agents").mkdir()
    (tmp_path / ".agents" / "old-plan.md").write_text("legacy\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "-f", ".agents/old-plan.md"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", ".gitignore", "README.md"], check=True)

    report = inspect_repository(tmp_path)

    assert report.tracked_ignored_paths == (Path(".agents/old-plan.md"),)
    assert report.unreviewed_root_entries == (".agents",)


def test_repository_check_command_is_explicitly_report_only(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "README.md").write_text("[missing](missing.md)\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "README.md"], check=True)

    assert main(["repository-check", "--repository-root", str(tmp_path), "--report-only"]) == 0

    output = capsys.readouterr().out
    assert "repository ownership audit (report-only)" in output
    assert "broken local Markdown links: 1" in output


def test_release_check_names_a_drifted_internal_pin(tmp_path: Path) -> None:
    _version_repository(tmp_path)
    pyproject = tmp_path / "packages/train/pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8").replace("posttrain-common", "posttrain-common==0.2.4", 1),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"source dependency 'posttrain-common==0[.]2[.]4' contains a release pin"):
        check_release(tmp_path)


def test_prepare_changes_only_the_authored_manifest(tmp_path: Path) -> None:
    _version_repository(tmp_path)
    source_metadata = (tmp_path / "packages/train/pyproject.toml").read_text(encoding="utf-8")

    result = prepare_release(tmp_path, "0.3.0")

    assert result.version == "0.3.0"
    assert (tmp_path / "packages/train/pyproject.toml").read_text(encoding="utf-8") == source_metadata


def test_stage_expands_static_wheel_metadata_without_touching_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "staged"
    _version_repository(source, version="0.3.0")

    result = stage_release(source, destination)

    source_text = (source / "packages/train/pyproject.toml").read_text(encoding="utf-8")
    staged_text = (destination / "packages/train/pyproject.toml").read_text(encoding="utf-8")
    assert result.package_count == 1
    assert 'version = "0.0.0"' in source_text
    assert 'version = "0.3.0"' in staged_text
    assert '"posttrain-common==0.3.0"' in staged_text
    assert 'version = "0.0.0"' in (source / "uv.lock").read_text(encoding="utf-8")
    assert 'version = "0.3.0"' in (destination / "uv.lock").read_text(encoding="utf-8")


def test_stage_uses_committed_source_and_excludes_runner_state(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "staged"
    _version_repository(source, version="0.3.0")
    (source / ".posttrain/state").mkdir(parents=True)
    (source / ".posttrain/state" / "stale.json").write_text("stale", encoding="utf-8")
    (source / ".venv").mkdir()
    (source / ".venv" / "generated.txt").write_text("generated", encoding="utf-8")
    subprocess.run(["git", "-C", str(source), "init", "--quiet"], check=True)
    subprocess.run(["git", "-C", str(source), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(source), "config", "user.name", "Release Test"], check=True)
    subprocess.run(["git", "-C", str(source), "add", "release", "pyproject.toml", "packages", "uv.lock"], check=True)
    subprocess.run(["git", "-C", str(source), "commit", "--quiet", "-m", "fixture"], check=True)

    stage_release(source, destination)

    assert not (destination / ".posttrain/state/stale.json").exists()
    assert not (destination / ".venv/generated.txt").exists()
    assert (destination / "packages/train/pyproject.toml").is_file()


def test_distribution_builder_overlays_generated_runtime_manifest() -> None:
    repository_root = Path(__file__).resolve().parents[_REPOSITORY_ROOT_DEPTH]
    builder = (repository_root / "scripts/release/build-python-distributions").read_text(encoding="utf-8")
    assert "generated_image_manifest=" in builder
    assert "staged_image_manifest=" in builder
    assert 'cp "${generated_image_manifest}" "${staged_image_manifest}"' in builder


def test_dstack_capacity_retries_a_malformed_response_and_retains_valid_receipt(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[_REPOSITORY_ROOT_DEPTH]
    executable, state = _fake_dstack(tmp_path, ["temporary upstream response\n", _dstack_fleet()])
    receipt = tmp_path / "capacity.json"

    result = subprocess.run(
        [str(repository_root / "scripts/release/verify-dstack-capacity"), str(receipt)],
        env={
            **os.environ,
            "POSTTRAIN_DSTACK_BIN": str(executable),
            "POSTTRAIN_DSTACK_CAPACITY_ATTEMPTS": "2",
            "POSTTRAIN_DSTACK_CAPACITY_RETRY_SECONDS": "0",
            "FAKE_DSTACK_STATE": str(state),
            "FAKE_DSTACK_RESPONSES": str(tmp_path / "dstack-responses"),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert state.read_text(encoding="utf-8") == "2"
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["status"] == "accepted"
    assert payload["target_host"] == "carbonteq-ai-workstation.lan"
    assert payload["instance"]["gpu"]["memory_mib"] == 98304


def test_dstack_capacity_persists_sanitized_evidence_for_invalid_json(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[_REPOSITORY_ROOT_DEPTH]
    executable, state = _fake_dstack(tmp_path, ["not-json\n"])
    receipt = tmp_path / "capacity.json"

    result = subprocess.run(
        [str(repository_root / "scripts/release/verify-dstack-capacity"), str(receipt)],
        env={
            **os.environ,
            "POSTTRAIN_DSTACK_BIN": str(executable),
            "POSTTRAIN_DSTACK_CAPACITY_ATTEMPTS": "2",
            "POSTTRAIN_DSTACK_CAPACITY_RETRY_SECONDS": "0",
            "FAKE_DSTACK_STATE": str(state),
            "FAKE_DSTACK_RESPONSES": str(tmp_path / "dstack-responses"),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 3
    assert state.read_text(encoding="utf-8") == "2"
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["status"] == "rejected"
    assert payload["failure_kind"] == "invalid_json"
    assert payload["attempts"] == 2
    assert payload["response"]["stdout_bytes"] == len("not-json\n")
    assert len(payload["response"]["stdout_sha256"]) == 64
    assert "not-json" not in receipt.read_text(encoding="utf-8")


def test_dstack_capacity_distinguishes_a_valid_but_busy_target(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[_REPOSITORY_ROOT_DEPTH]
    executable, state = _fake_dstack(tmp_path, [_dstack_fleet(status="busy")])
    receipt = tmp_path / "capacity.json"

    result = subprocess.run(
        [str(repository_root / "scripts/release/verify-dstack-capacity"), str(receipt)],
        env={
            **os.environ,
            "POSTTRAIN_DSTACK_BIN": str(executable),
            "POSTTRAIN_DSTACK_CAPACITY_ATTEMPTS": "1",
            "POSTTRAIN_DSTACK_CAPACITY_RETRY_SECONDS": "0",
            "FAKE_DSTACK_STATE": str(state),
            "FAKE_DSTACK_RESPONSES": str(tmp_path / "dstack-responses"),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 3
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["failure_kind"] == "target_not_ready"
    assert '"status": "busy"' in result.stderr


def test_stage_can_render_an_rc_without_changing_the_authored_target(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "staged"
    _version_repository(source, version="0.3.1")

    result = stage_release(source, destination, version="0.3.1rc2")

    assert result.version == "0.3.1rc2"
    assert 'version = "0.3.1"' in (source / "release/manifest.toml").read_text(encoding="utf-8")
    assert 'version = "0.3.1rc2"' in (destination / "packages/train/pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "0.3.1rc2"' in (destination / "uv.lock").read_text(encoding="utf-8")


def test_stage_rejects_an_override_outside_the_authored_release_line(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _version_repository(source, version="0.3.1")

    with pytest.raises(ValueError, match="must be a release candidate of 0.3.1"):
        stage_release(source, tmp_path / "staged", version="0.3.2rc1")


def test_next_candidate_version_skips_every_immutable_rc_already_on_the_index() -> None:
    assert (
        next_candidate_version(
            "0.3.1",
            (
                "posttrain-0.3.1rc1-py3-none-any.whl",
                "posttrain-0.3.1rc3.tar.gz",
                "unrelated-0.3.1rc2-py3-none-any.whl",
            ),
        )
        == "0.3.1rc2"
    )


def test_distribution_receipt_binds_wheel_sdist_lock_and_image_manifest(tmp_path: Path) -> None:
    distributions = tmp_path / "dist"
    distributions.mkdir()
    wheel = distributions / "posttrain_widget-0.3.1rc2-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "posttrain_widget-0.3.1rc2.dist-info/METADATA",
            "Metadata-Version: 2.3\nName: posttrain-widget\nVersion: 0.3.1rc2\n",
        )
    (distributions / "posttrain_widget-0.3.1rc2.tar.gz").write_bytes(b"sdist")
    uv_lock = tmp_path / "uv.lock"
    uv_lock.write_text("lock\n", encoding="utf-8")
    image_manifest = tmp_path / "published.toml"
    image_manifest.write_text("schema_version = 1\n", encoding="utf-8")

    receipt = create_distribution_receipt(
        distributions,
        version="0.3.1rc2",
        revision="a" * 40,
        uv_lock=uv_lock,
        image_manifest=image_manifest,
    )
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(__import__("json").dumps(receipt), encoding="utf-8")

    verified = verify_distribution_receipt(receipt_path, distributions)
    assert verified["packages"] == ["posttrain-widget"]
    artifacts = verified["artifacts"]
    assert isinstance(artifacts, list)
    assert len(artifacts) == 2


def test_tag_workflow_builds_the_versioned_stage_not_the_source_workspace() -> None:
    repository_root = Path(__file__).resolve().parents[_REPOSITORY_ROOT_DEPTH]
    workflow = (repository_root / ".github/workflows/release.yml").read_text(encoding="utf-8")

    assert "scripts/release/build-python-distributions" in workflow
    builder = (repository_root / "scripts/release/build-python-distributions").read_text(encoding="utf-8")
    assert "uv build" in builder
    assert "--all-packages" in builder
    assert "--no-sources" in builder
    assert "--python 3.13" in builder
    assert 'build_cache_dir="${UV_CACHE_DIR:-$build_cache_dir}"' in builder
    assert 'UV_CACHE_DIR="$build_cache_dir"' in builder
    assert "uv build environments/" not in workflow
    assert 'generated_runtime_lock="${repository_root}/packages/runtime-images' in builder
    assert 'cp "${generated_runtime_lock}" "${staged_runtime_lock}"' in builder


def test_protected_release_workflows_keep_the_build_and_qualification_boundaries() -> None:
    repository_root = Path(__file__).resolve().parents[_REPOSITORY_ROOT_DEPTH]
    candidate = (repository_root / ".github/workflows/release-candidate.yml").read_text(encoding="utf-8")
    final = (repository_root / ".github/workflows/release.yml").read_text(encoding="utf-8")

    for workflow in (candidate, final):
        assert "runs-on: [self-hosted, linux, x64, lan-release]" in workflow
        assert "uv pip install" in workflow
        assert '--index-url "${PYPI_DEV_SIMPLE}"' in workflow
        assert "runtime images verify" in workflow
        assert 'XDG_CONFIG_HOME="${image_verify_config}"' in workflow
        assert "printf 'POSTTRAIN_REGISTRY=%s\\n'" in workflow
        assert "trap 'rm -f \"${image_verify_env}\"' EXIT" in workflow
        assert "Prepare release evidence directory" in workflow
        assert "posttrain[dstack,trackio]==" in workflow
        assert "posttrain-lab==" in workflow
        assert "if-no-files-found: error" in workflow
        assert "include-hidden-files: true" in workflow
        assert ".release/consumer-venv/bin/posttrain --project-root apps/lab job run" in workflow
        assert "run wait" in workflow
        assert 'run reconcile \\\n            "release-' in workflow
        assert 'run cleanup \\\n            "release-' in workflow

    assert 'framework_wheelhouse="$(realpath .release/wheelhouse)"' in candidate
    assert 'consumer-venv/bin/python - "${candidate_version}" "${framework_wheelhouse}"' in candidate
    assert "framework wheelhouse contains non-candidate wheels" in candidate
    assert '--framework-wheelhouse "${framework_wheelhouse}"' in candidate

    assert "candidate-version --simple-url" in candidate
    assert "posttrain-release check --allow-pending-runtime-lock" in candidate
    assert "posttrain-release lock-runtime-dependencies" in candidate
    assert ".release/workspace.lock.txt" in candidate
    assert 'authored_framework_version="$(sed -n' in candidate
    assert 'published_framework_version="$(sed -n' in candidate
    assert '"${published_framework_version}" = "${authored_framework_version}"' in candidate
    assert '--framework-version "${authored_framework_version}"' in candidate
    assert candidate.index("Build changed OCI inputs and retain the generated manifest") < candidate.index(
        "Build and hash the Python wheelhouse"
    )
    assert "for attempt in $(seq 1 120)" in candidate
    assert 'select(.headSha == $sha and (.event == "push" or .event == "pull_request" or .event == "workflow_dispatch"))' in candidate
    assert 'git ls-remote --exit-code origin "refs/tags/v${POSTTRAIN_RELEASE_VERSION}"' in final
    assert '"${DEVPI_CLIENT}" push -y' in final
    assert "exact final bytes are already present in the stable index" in final
    assert final.index("exact final bytes are already present in the stable index") < final.index(
        '"${DEVPI_CLIENT}" push -y'
    )
    assert final.index("Capture bounded cache evidence") < final.index("Tag and create the GitHub release last")
    assert final.index("Retain final receipt and cache evidence") < final.index(
        "Tag and create the GitHub release last"
    )
    assert "for attempt in $(seq 1 120)" in final
    assert 'select(.headSha == $sha and .event == "push")' in final
    assert "REQUESTS_CA_BUNDLE: /etc/ssl/certs/ca-certificates.crt" in final
    assert "exact final bytes are already present in the development index" in final
    assert "development index already contains" in final
    assert 'grep -Eqi -- "${normalized}-${POSTTRAIN_RELEASE_VERSION}([.]|$)"' in final
    assert 'grep -Fq "${POSTTRAIN_RELEASE_VERSION}"' not in final
    assert "candidate_run_id:" in final
    assert "Restore the accepted candidate runtime image manifest" in final
    assert 'git merge-base --is-ancestor "${candidate_sha}" "${RELEASE_SOURCE_SHA}"' in final
    assert 'test "${candidate_version}" = "${release_version}"' in final
    assert 'cp "${candidate_manifest}" packages/runtime-images/src/posttrain/runtime_images/published.toml' in final
    assert "resume_from_run_id" in final
    assert 'gh run view "${RESUME_FROM_RUN_ID}"' in final
    assert "workflowName // empty" in final
    assert "conclusion // empty" in final
    assert "gh run download" in final
    assert 'git merge-base --is-ancestor "${source_sha}"' in final
    assert 'git tag -a "v${POSTTRAIN_RELEASE_VERSION}" "${RELEASE_SOURCE_SHA}"' in final
    assert "Materialize and verify the retained release wheelhouse" in final
    assert "receipt-check .release/python-release-receipt.json" in final


@pytest.mark.skipif(which("uv") is None, reason="requires uv to validate the staged workspace lock")
def test_staged_workspace_syncs_with_the_projected_lock(tmp_path: Path) -> None:
    """Staging must not require a second dependency resolution to be installable."""

    source = tmp_path / "source"
    destination = tmp_path / "staged"
    (source / "release").mkdir(parents=True)
    (source / "release" / "manifest.toml").write_text('schema_version = 1\nversion = "0.3.0"\n', encoding="utf-8")
    (source / "pyproject.toml").write_text(
        """[dependency-groups]
dev = []

[tool.uv.workspace]
members = ["packages/*"]

[tool.uv]
package = false
""",
        encoding="utf-8",
    )
    package = source / "packages" / "widget"
    package.mkdir(parents=True)
    (package / "pyproject.toml").write_text(
        """[project]
name = "posttrain-widget"
version = "0.0.0"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/posttrain_widget"]
""",
        encoding="utf-8",
    )
    source_package = package / "src" / "posttrain_widget"
    source_package.mkdir(parents=True)
    (source_package / "__init__.py").write_text("", encoding="utf-8")

    subprocess.run(["uv", "lock", "--offline"], cwd=source, check=True, capture_output=True, text=True)
    source_lock = (source / "uv.lock").read_bytes()

    stage_release(source, destination)

    subprocess.run(
        ["uv", "sync", "--locked", "--offline", "--no-install-project"],
        cwd=destination,
        check=True,
        capture_output=True,
        text=True,
    )
    assert (source / "uv.lock").read_bytes() == source_lock
    assert 'name = "posttrain-widget"\nversion = "0.3.0"' in (destination / "uv.lock").read_text(encoding="utf-8")


def test_dependency_lock_generation_has_one_record(tmp_path: Path) -> None:
    _version_repository(tmp_path)
    (tmp_path / "uv.lock").write_text("updated lock\n", encoding="utf-8")

    digest = lock_dependencies(tmp_path)

    document = tomllib.loads(
        (tmp_path / "packages/catalog/src/posttrain/catalog/base/locks.toml").read_text(encoding="utf-8")
    )
    assert set(document["locks"]) == {"trl-fork@current"}
    assert document["locks"]["trl-fork@current"]["dependency_lock_sha256"] == digest


def test_rendered_lock_digests_come_from_the_shipped_locks() -> None:
    """The manifest can only ever claim what the shipped locks actually hash to."""
    document = tomllib.loads(_render_all())
    for variant in RUNTIME_VARIANTS:
        assert document["kinds"][variant]["lock_digest"] == lock_digest(constraint_lock(variant))


def test_transform_keeps_its_own_constraint_lock() -> None:
    document = tomllib.loads(_render_all())
    assert document["kinds"]["transform"]["constraint_lock"].endswith("transform.lock.txt")
    assert document["kinds"]["supervised"]["constraint_lock"].endswith("workspace.lock.txt")
    assert document["kinds"]["transform"]["lock_digest"] != document["kinds"]["supervised"]["lock_digest"]


def test_provided_packages_survive_rendering() -> None:
    lock = constraint_lock("eval")
    rendered = render_manifest(
        framework_version="1.0.0",
        default_prefix="registry.lan/carbonteq",
        base=PublishedImage(
            name="base",
            repository="posttrain-base",
            digest="sha256:" + "a" * 64,
            lock_digest=lock_digest(),
            constraint_lock=constraint_lock("supervised"),
        ),
        kinds={
            "eval": PublishedImage(
                name="kinds.eval",
                repository="posttrain-kind-eval",
                digest="sha256:" + "b" * 64,
                lock_digest=lock_digest(lock),
                constraint_lock=lock,
                provided_packages=("verifiers",),
            )
        },
    )
    document = tomllib.loads(rendered)
    assert document["kinds"]["eval"]["provided_packages"] == ["verifiers"]


def test_backend_constraints_survive_rendering() -> None:
    image = _image("kinds.online-rl-verl-py313", "online-rl-verl-py313", "b")
    rendered = render_manifest(
        framework_version="1.0.0",
        default_prefix="registry.lan/carbonteq",
        base=PublishedImage(
            name="base",
            repository="posttrain-base",
            digest="sha256:" + "a" * 64,
            lock_digest=lock_digest(),
            constraint_lock=constraint_lock("supervised"),
        ),
        kinds={
            "online-rl-verl-py313": PublishedImage(
                name=image.name,
                repository=image.repository,
                digest=image.digest,
                lock_digest=image.lock_digest,
                constraint_lock=image.constraint_lock,
                backend_constraint_lock=VERL_BACKEND_LOCK,
                backend_lock_digest=lock_digest(VERL_BACKEND_LOCK),
                backend_provided_packages=("verl",),
            )
        },
    )
    document = tomllib.loads(rendered)
    published = document["kinds"]["online-rl-verl-py313"]
    assert published["backend_constraint_lock"] == VERL_BACKEND_LOCK.as_posix()
    assert published["backend_lock_digest"] == lock_digest(VERL_BACKEND_LOCK)
    assert published["backend_provided_packages"] == ["verl"]


def test_an_empty_release_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one job-kind image"):
        render_manifest(
            framework_version="1.0.0",
            default_prefix="registry.lan/carbonteq",
            base=PublishedImage(
                name="base",
                repository="posttrain-base",
                digest="sha256:" + "a" * 64,
                lock_digest=lock_digest(),
                constraint_lock=constraint_lock("supervised"),
            ),
            kinds={},
        )


def test_the_shipped_manifest_matches_what_the_renderer_would_produce() -> None:
    """The committed manifest must be generated output, not hand-edited.

    Digests differ because the shipped manifest records a real publication, so
    only the generated structure is compared.
    """
    shipped = load_manifest()
    rendered = tomllib.loads(_render_all())
    assert set(rendered["kinds"]) == set(shipped.kinds)
    for variant, image in shipped.kinds.items():
        assert rendered["kinds"][variant]["repository"] == image.repository
        assert rendered["kinds"][variant]["constraint_lock"] == image.constraint_lock.as_posix()
        assert rendered["kinds"][variant]["lock_digest"] == image.lock_digest
        assert rendered["kinds"][variant].get("backend_constraint_lock") == (
            image.backend_constraint_lock.as_posix() if image.backend_constraint_lock else None
        )
        assert rendered["kinds"][variant].get("backend_lock_digest") == image.backend_lock_digest


def test_unknown_variant_is_rejected() -> None:
    from posttrain_release.publish import _normalize_variants

    with pytest.raises(ValueError, match="unknown runtime variant"):
        _normalize_variants(["nope"])


def test_variant_subset_preserves_canonical_order() -> None:
    from posttrain_release.publish import _normalize_variants

    assert _normalize_variants(["transform", "eval"]) == ("eval", "transform")


def test_public_ci_trackio_mirror_matches_locked_distribution() -> None:
    root = Path(__file__).resolve().parents[_REPOSITORY_ROOT_DEPTH]
    lock = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8"))
    trackio = next(package for package in lock["package"] if package["name"] == "carbonteq-trackio")
    wheel = next(item for item in trackio["wheels"] if item["url"].endswith(".whl"))
    wheel_sha256 = wheel["hash"].removeprefix("sha256:")
    version = trackio["version"]
    filename = f"carbonteq_trackio-{version}-py3-none-any.whl"
    workflow = (root / ".github/workflows/quality.yml").read_text(encoding="utf-8")

    assert (
        "CARBONTEQ_TRACKIO_WHEEL_URL: "
        f"https://github.com/carbonteq-ai/trackio/releases/download/carbonteq-v{version}/{filename}"
    ) in workflow
    assert f"CARBONTEQ_TRACKIO_WHEEL_SHA256: {wheel_sha256}" in workflow
    assert f"POSTTRAIN_CONSUMER_EXTRA_WHEELS: /tmp/{filename}" in workflow
    assert "uv lock --check" not in workflow
    assert workflow.count("uv sync --frozen") == 4
    assert workflow.count("--no-install-package carbonteq-trackio") == 4


def test_final_release_restores_candidate_runtime_lock_with_manifest() -> None:
    root = Path(__file__).resolve().parents[_REPOSITORY_ROOT_DEPTH]
    workflow = (root / ".github/workflows/release.yml").read_text(encoding="utf-8")

    assert 'candidate_workspace_lock="$(find .release/candidate' in workflow
    assert 'test -n "${candidate_workspace_lock}"' in workflow
    assert (
        'cp "${candidate_workspace_lock}" \\\n'
        "            packages/runtime-images/src/posttrain/runtime_images/containers/"
        "posttrain-job-kinds/locks/workspace.lock.txt"
    ) in workflow
    assert 'cp "${candidate_workspace_lock}" .release/workspace.lock.txt' in workflow


def test_release_can_read_prior_manifest_while_adding_a_variant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import posttrain.runtime_images.manifest as manifest_module

    original_variants = manifest_module.RUNTIME_VARIANTS
    monkeypatch.setattr(
        manifest_module,
        "RUNTIME_VARIANTS",
        (*original_variants, "new-runtime"),
    )
    manifest_module.load_manifest.cache_clear()
    try:
        with pytest.raises(ManifestError, match="new-runtime"):
            manifest_module.load_manifest(verify_locks=False)

        prior = manifest_module.load_manifest(
            verify_locks=False,
            verify_variants=False,
        )
        assert prior.kinds
        assert "new-runtime" not in prior.kinds
    finally:
        manifest_module.load_manifest.cache_clear()

    """Publishing must be unreachable from a consumer environment.

    `posttrain` is what a project developer installs. If it depended on this
    package, release authority would land in every consumer's environment.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[_REPOSITORY_ROOT_DEPTH]
    consumer = tomllib.loads((root / "apps/cli/pyproject.toml").read_text(encoding="utf-8"))
    declared = consumer["project"]["dependencies"]
    optional = [
        dependency for extra in consumer["project"].get("optional-dependencies", {}).values() for dependency in extra
    ]
    assert not any("posttrain-release" in item for item in [*declared, *optional])
