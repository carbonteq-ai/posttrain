from __future__ import annotations

from pathlib import Path
from typing import cast

from posttrain.catalog import load_project_layout
from posttrain.execution import JobPackageManifest, RuntimeImageRef
from posttrain.execution_pack import ImagePublicationSpec, PackageMaterializationRecord, PublishedJobImage
from posttrain_cli.execution_planning import PlannedJobPackage
from posttrain_cli.package_history import retained_packages


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    control = root / ".posttrain"
    control.mkdir(parents=True)
    (control / "project.toml").write_text('schema_version = 1\nproject_id = "history-test"\n', encoding="utf-8")
    return root


def test_history_reads_compact_record_without_retained_context(tmp_path: Path) -> None:
    root = _project(tmp_path)
    manifest = JobPackageManifest(
        project_id="project",
        work_package_id="work",
        job_id="job",
        job_definition_id="definition",
        job_kind="train.sft",
        resolved_inputs_digest="a" * 64,
        framework_source_digest="a" * 64,
        project_source_digest="a" * 64,
        runtime_dependencies_digest="a" * 64,
        code_requirements_digest="a" * 64,
        resolved_config_digest="a" * 64,
        project_config_digest="a" * 64,
        universal_image=RuntimeImageRef(f"registry.example/base@sha256:{'b' * 64}"),
        kind_image=RuntimeImageRef(f"registry.example/kind@sha256:{'c' * 64}"),
        runtime_variant="supervised",
    )
    record = PackageMaterializationRecord(
        package_key=manifest.package_key,
        context_digest="d" * 64,
        publication_key="e" * 64,
        manifest=manifest,
    )
    records = root / ".posttrain" / "state" / "packages" / "materializations"
    records.mkdir(parents=True)
    record_path = records / f"{manifest.package_key}.json"
    record_path.write_bytes(record.to_bytes())

    packages = retained_packages(load_project_layout(root))

    assert len(packages) == 1
    assert packages[0].package_key == manifest.package_key
    assert packages[0].root == record_path
    assert packages[0].payload["schema"] == "posttrain.job-package.v1"
    assert packages[0].work_package_id == "work"
    assert packages[0].job_id == "job"


def test_published_pack_reuses_record_without_materializing_context(tmp_path: Path) -> None:
    root = _project(tmp_path)
    layout = load_project_layout(root)
    manifest = JobPackageManifest(
        project_id="project",
        work_package_id="work",
        job_id="job",
        job_definition_id="definition",
        job_kind="train.sft",
        resolved_inputs_digest="a" * 64,
        framework_source_digest="a" * 64,
        project_source_digest="a" * 64,
        runtime_dependencies_digest="a" * 64,
        code_requirements_digest="a" * 64,
        resolved_config_digest="a" * 64,
        project_config_digest="a" * 64,
        universal_image=RuntimeImageRef(f"registry.example/base@sha256:{'b' * 64}"),
        kind_image=RuntimeImageRef(f"registry.example/kind@sha256:{'c' * 64}"),
        runtime_variant="supervised",
    )
    publication = ImagePublicationSpec("registry.example/posttrain-job")
    from posttrain.execution_pack import publication_key_for

    publication_key = publication_key_for(manifest, publication)
    record = PackageMaterializationRecord(
        package_key=manifest.package_key,
        context_digest="d" * 64,
        publication_key=publication_key,
        manifest=manifest,
        plan_key="f" * 64,
    )
    records = root / ".posttrain" / "state" / "packages" / "materializations"
    records.mkdir(parents=True)
    records.joinpath(f"{manifest.package_key}.json").write_bytes(record.to_bytes())
    image = PublishedJobImage(
        manifest.package_key,
        publication_key,
        RuntimeImageRef(f"registry.example/posttrain-job@sha256:{'e' * 64}"),
        manifest.kind_image,
        root / "receipt.json",
        True,
    )
    image.receipt.write_text("{}\n", encoding="utf-8")
    image.receipt.chmod(0o600)

    class Publisher:
        def resolve(self, request):
            assert request.publication_key == publication_key
            return image

    fake = type(
        "FakePlannedPackage",
        (),
        {
            "layout": layout,
            "pack_plan": type("Plan", (), {"plan_key": "f" * 64, "publication": publication})(),
            "_publisher": lambda self: Publisher(),
            "materialize": lambda self: (_ for _ in ()).throw(AssertionError("materialization was not skipped")),
        },
    )()

    packed = PlannedJobPackage.pack(cast(PlannedJobPackage, fake))

    assert packed.image.cache_hit
    assert packed.context.manifest == manifest
    assert not packed.context.root.exists()
