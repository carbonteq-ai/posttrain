"""Release tooling, and the boundary that keeps it away from consumers."""

from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path, PurePosixPath

import pytest
from posttrain.runtime_images import RUNTIME_VARIANTS, constraint_lock, lock_digest
from posttrain.runtime_images.manifest import ManifestError, PublishedImage, load_manifest
from posttrain_release.cli import main
from posttrain_release.manifest_render import render_manifest
from posttrain_release.repository_audit import evaluate_repository, inspect_repository
from posttrain_release.versioning import (
    check_release,
    lock_dependencies,
    prepare_release,
    stage_release,
)

_REPOSITORY_ROOT_DEPTH = 3


def _version_repository(root: Path, version: str = "0.2.5") -> None:
    (root / "release").mkdir(parents=True)
    (root / "release" / "manifest.toml").write_text(
        f'schema_version = 1\nversion = "{version}"\n',
        encoding="utf-8",
    )
    (root / "pyproject.toml").write_text(
        '[project]\nname = "lab"\nversion = "0.0.0"\ndependencies = ["posttrain-common"]\n',
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
trl = ["trl @ git+https://github.com/carbonteq-ai/trl.git@6e7739b8ec741d21ecd79c0c212694cd15ff20d8"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
""",
        encoding="utf-8",
    )
    (root / "uv.lock").write_text("lock\n", encoding="utf-8")
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
source_revision = "6e7739b8ec741d21ecd79c0c212694cd15ff20d8"
dependency_lock_sha256 = "{digest}"
""",
        encoding="utf-8",
    )


def _image(name: str, variant: str, digest_char: str) -> PublishedImage:
    lock = constraint_lock(variant)
    return PublishedImage(
        name=name,
        repository=f"posttrain-kind-{variant}",
        digest="sha256:" + digest_char * 64,
        lock_digest=lock_digest(lock),
        constraint_lock=lock,
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


def test_unknown_variant_is_rejected() -> None:
    from posttrain_release.publish import _normalize_variants

    with pytest.raises(ValueError, match="unknown runtime variant"):
        _normalize_variants(["nope"])


def test_variant_subset_preserves_canonical_order() -> None:
    from posttrain_release.publish import _normalize_variants

    assert _normalize_variants(["transform", "eval"]) == ("eval", "transform")


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
