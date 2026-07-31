"""Release tooling, and the boundary that keeps it away from consumers."""

from __future__ import annotations

import tomllib
from pathlib import Path, PurePosixPath

import pytest
from posttrain.runtime_images import RUNTIME_VARIANTS, constraint_lock, lock_digest
from posttrain.runtime_images.manifest import ManifestError, PublishedImage, load_manifest
from posttrain_release.manifest_render import render_manifest
from posttrain_release.versioning import check_release, prepare_release

_REPOSITORY_ROOT_DEPTH = 3


def _version_repository(root: Path, version: str = "0.2.5") -> None:
    (root / "release").mkdir(parents=True)
    (root / "release" / "manifest.toml").write_text(
        f'schema_version = 1\nversion = "{version}"\n',
        encoding="utf-8",
    )
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "posttrain"\nversion = "{version}"\ndependencies = ["posttrain-common=={version}"]\n',
        encoding="utf-8",
    )
    (root / "uv.lock").write_text("lock\n", encoding="utf-8")
    digest = __import__("hashlib").sha256((root / "uv.lock").read_bytes()).hexdigest()
    catalog = root / "packages/catalog/src/posttrain/catalog/base"
    catalog.mkdir(parents=True)
    (catalog / "training.yaml").write_text(
        f"training:\n  example:\n    dependency_lock_sha256: {digest}\n",
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


def test_release_check_names_a_drifted_internal_pin(tmp_path: Path) -> None:
    _version_repository(tmp_path)
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8").replace("posttrain-common==0.2.5", "posttrain-common==0.2.4"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"pyproject[.]toml: posttrain-common is pinned to '0[.]2[.]4'.*'0[.]2[.]5'"):
        check_release(tmp_path)


def test_prepare_expands_versions_refreshes_lock_and_rechecks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _version_repository(tmp_path)

    def fake_run(*args, **kwargs):
        del args, kwargs
        (tmp_path / "uv.lock").write_text("new lock\n", encoding="utf-8")
        return __import__("subprocess").CompletedProcess(["uv", "lock"], 0, "", "")

    monkeypatch.setattr("posttrain_release.versioning.subprocess.run", fake_run)

    result = prepare_release(tmp_path, "0.3.0")

    assert result.version == "0.3.0"
    assert 'version = "0.3.0"' in (tmp_path / "pyproject.toml").read_text(encoding="utf-8")
    assert "posttrain-common==0.3.0" in (tmp_path / "pyproject.toml").read_text(encoding="utf-8")


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
