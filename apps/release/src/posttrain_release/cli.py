"""Framework-owner release command.

Not shipped to consumers: `posttrain` does not depend on this distribution.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer
from posttrain.runtime_images import cached_definition_root
from posttrain_execution_buildkit import BuildKitRuntimeBuilder

from .artifacts import (
    create_distribution_receipt,
    verify_distribution_receipt,
    verify_index_receipt,
    write_distribution_receipt,
)
from .candidate import fetch_simple_artifacts, next_candidate_version
from .fork_ledger import render_fork_ledger
from .promotion import create_promotion_receipt, write_promotion_receipt
from .publish import _release_image_plan, publish_release
from .readiness import run_readiness, verify_readiness_receipt, write_readiness_receipt
from .repository_audit import inspect_repository
from .runtime_lock import (
    compile_runtime_locks,
    export_runtime_workspace_lock,
    materialize_runtime_lock,
    synchronize_runtime_profile_pins,
)
from .versioning import check_release, load_release_manifest, lock_dependencies, prepare_release, stage_release

app = typer.Typer(help="publish framework runtime images and pin the release manifest")
images_app = typer.Typer(help="runtime image release operations")
app.add_typer(images_app, name="images")

_MANIFEST_RELATIVE = Path("packages/runtime-images/src/posttrain/runtime_images/published.toml")


@app.command("fork-ledger", help="render the maintained-fork release closure selected by this checkout")
def fork_ledger_cmd(
    repository_root: Annotated[
        Path,
        typer.Option("--repository-root", help="framework checkout to inspect"),
    ] = Path("."),
) -> None:
    print(json.dumps(render_fork_ledger(repository_root), indent=2, sort_keys=True))


@app.command("readiness", help="run deterministic release checks and write an immutable source-readiness receipt")
def readiness_cmd(
    destination: Annotated[Path, typer.Option("--destination", help="JSON receipt destination")],
    repository_root: Annotated[
        Path,
        typer.Option("--repository-root", help="framework checkout to verify"),
    ] = Path("."),
    allow_pending_runtime_lock: Annotated[
        bool,
        typer.Option(
            "--allow-pending-runtime-lock",
            help="allow a candidate branch to materialize the OCI lock after readiness",
        ),
    ] = False,
) -> None:
    receipt = run_readiness(
        repository_root,
        allow_pending_runtime_lock=allow_pending_runtime_lock,
    )
    write_readiness_receipt(receipt, destination)
    print(f"recorded readiness receipt: {destination}")


@app.command("readiness-check", help="verify a readiness receipt against this exact checkout")
def readiness_check_cmd(
    receipt: Annotated[Path, typer.Argument(help="readiness receipt JSON")],
    repository_root: Annotated[
        Path,
        typer.Option("--repository-root", help="framework checkout to verify"),
    ] = Path("."),
) -> None:
    verified = verify_readiness_receipt(receipt, repository_root)
    print(f"verified readiness receipt for {verified['source_sha']}")


@app.command("promotion-receipt", help="bind a candidate distribution receipt to the merged stable-release tree")
def promotion_receipt_cmd(
    candidate_receipt: Annotated[Path, typer.Argument(help="candidate Python distribution receipt")],
    destination: Annotated[Path, typer.Option("--destination", help="promotion receipt JSON destination")],
    candidate_run_id: Annotated[str, typer.Option("--candidate-run-id", help="successful candidate workflow run id")],
    candidate_source_sha: Annotated[str, typer.Option("--candidate-source-sha", help="candidate source commit")],
    candidate_source_tree: Annotated[str, typer.Option("--candidate-source-tree", help="candidate source tree")],
    merged_sha: Annotated[str, typer.Option("--merged-sha", help="merged default-branch commit")],
    merged_tree: Annotated[str, typer.Option("--merged-tree", help="merged default-branch tree")],
) -> None:
    receipt = create_promotion_receipt(
        candidate_receipt,
        candidate_run_id=candidate_run_id,
        candidate_source_sha=candidate_source_sha,
        candidate_source_tree=candidate_source_tree,
        merged_sha=merged_sha,
        merged_tree=merged_tree,
    )
    write_promotion_receipt(receipt, destination)
    print(f"recorded promotion receipt: {destination}")


@app.command("check", help="verify source templates and generated locks against the release manifest")
def check_cmd(
    repository_root: Annotated[
        Path,
        typer.Option("--repository-root", help="framework checkout to verify"),
    ] = Path("."),
    allow_pending_runtime_lock: Annotated[
        bool,
        typer.Option(
            "--allow-pending-runtime-lock",
            help="validate authored release inputs before the candidate materializes the runtime lock",
        ),
    ] = False,
) -> None:
    result = check_release(repository_root, allow_pending_runtime_lock=allow_pending_runtime_lock)
    print(f"authored version: release/manifest.toml = {result.version}")
    print(f"staged metadata: OK ({result.package_count} packages, {result.internal_pin_count} internal pins)")
    print("dependency locks: OK")
    if result.runtime_lock_pending:
        print("runtime dependency lock: pending candidate materialization")
        print("published images: current for the committed lock; candidate rebuild required")
    else:
        print("published images: OK")


@app.command(
    "lock-runtime-dependencies",
    help="materialize immutable internal-package wheel receipts into the OCI constraint lock",
)
def lock_runtime_dependencies_cmd(
    repository_root: Annotated[
        Path,
        typer.Option("--repository-root", help="framework checkout to update"),
    ] = Path("."),
    check: Annotated[
        bool,
        typer.Option("--check", help="report whether materialization is pending without writing"),
    ] = False,
) -> None:
    exported = export_runtime_workspace_lock(repository_root, check=check)
    materialized = materialize_runtime_lock(repository_root, check=check)
    compiled = compile_runtime_locks(repository_root, check=check)
    packages = ", ".join(materialized.packages) or "none"
    if check and (exported.changed or materialized.changed or compiled.changed_paths):
        raise typer.Exit(code=1)
    state = "updated" if exported.changed or materialized.changed or compiled.changed_paths else "current"
    paths = ", ".join(str(path) for path in compiled.changed_paths) or "none"
    print(
        f"runtime dependency locks {state}: {exported.path}; "
        f"internal receipts: {materialized.path} ({packages}); narrow closures: {paths}"
    )


@app.command(
    "sync-runtime-profile-pins",
    help="align internal runtime-profile pins with an already-resolved candidate lock",
)
def sync_runtime_profile_pins_cmd(
    repository_root: Annotated[
        Path,
        typer.Option("--repository-root", help="disposable framework checkout to update"),
    ] = Path("."),
) -> None:
    result = synchronize_runtime_profile_pins(repository_root)
    profiles = ", ".join(str(path) for path in result.changed_profiles) or "none"
    packages = ", ".join(result.packages) or "none"
    print(f"runtime profile pins synchronized: {profiles} ({packages})")


@app.command("prepare", help="set the one authored release version without rewriting package metadata")
def prepare_cmd(
    version: Annotated[str, typer.Argument(help="coordinated release version")],
    repository_root: Annotated[
        Path,
        typer.Option("--repository-root", help="framework checkout to update"),
    ] = Path("."),
) -> None:
    result = prepare_release(repository_root, version)
    print(f"prepared manifest {result.version}; source package templates are unchanged")


@app.command("stage", help="copy the repository and render static release metadata in the copy")
def stage_cmd(
    destination: Annotated[Path, typer.Argument(help="new directory for rendered release sources")],
    repository_root: Annotated[
        Path,
        typer.Option("--repository-root", help="framework checkout to stage"),
    ] = Path("."),
    version: Annotated[
        str | None,
        typer.Option("--version", help="candidate version rendered only in the staged copy"),
    ] = None,
) -> None:
    result = stage_release(repository_root, destination, version=version)
    print(f"staged {result.version}: {result.package_count} packages and {result.internal_pin_count} exact pins")


@app.command("candidate-version", help="allocate the next unused RC from the development index")
def candidate_version_cmd(
    simple_url: Annotated[
        str,
        typer.Option("--simple-url", help="PEP 503 page for the posttrain project on the development index"),
    ] = "https://pypi.lan/carbonteq/dev/+simple/posttrain/",
    repository_root: Annotated[
        Path,
        typer.Option("--repository-root", help="framework checkout containing release/manifest.toml"),
    ] = Path("."),
) -> None:
    target = load_release_manifest(repository_root.resolve()).version
    print(next_candidate_version(target, fetch_simple_artifacts(simple_url)))


@app.command("receipt", help="write a hash-addressed receipt for one built distribution set")
def receipt_cmd(
    distribution_root: Annotated[Path, typer.Argument(help="directory containing wheels and source distributions")],
    destination: Annotated[Path, typer.Option("--destination", help="receipt JSON destination")],
    version: Annotated[str, typer.Option("--version", help="version expected in every wheel")],
    revision: Annotated[str, typer.Option("--revision", help="source commit that produced the distributions")],
    repository_root: Annotated[Path, typer.Option("--repository-root", help="framework checkout")] = Path("."),
) -> None:
    root = repository_root.resolve()
    receipt = create_distribution_receipt(
        distribution_root.resolve(),
        version=version,
        revision=revision,
        uv_lock=root / "uv.lock",
        image_manifest=root / _MANIFEST_RELATIVE,
    )
    write_distribution_receipt(receipt, destination.resolve())
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, list):
        raise RuntimeError("release receipt writer returned an invalid artifact list")
    print(f"recorded {len(artifacts)} distributions in {destination}")


@app.command("receipt-check", help="verify local distribution bytes against a release receipt")
def receipt_check_cmd(
    receipt: Annotated[Path, typer.Argument(help="release receipt JSON")],
    distribution_root: Annotated[Path, typer.Option("--distribution-root", help="directory containing distributions")],
) -> None:
    result = verify_distribution_receipt(receipt.resolve(), distribution_root.resolve())
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, list):
        raise RuntimeError("release receipt verifier returned an invalid artifact list")
    print(f"verified {len(artifacts)} local distributions")


@app.command("index-check", help="download index artifacts and verify every byte against a release receipt")
def index_check_cmd(
    receipt: Annotated[Path, typer.Argument(help="release receipt JSON")],
    simple_base_url: Annotated[str, typer.Option("--simple-base-url", help="PEP 503 index base URL")],
) -> None:
    verify_index_receipt(receipt.resolve(), simple_base_url)
    print("verified every indexed distribution against the release receipt")


@app.command("lock-dependencies", help="regenerate the catalog dependency-lock table")
def lock_dependencies_cmd(
    repository_root: Annotated[
        Path,
        typer.Option("--repository-root", help="framework checkout to update"),
    ] = Path("."),
) -> None:
    digest = lock_dependencies(repository_root)
    print(f"generated catalog dependency lock: sha256:{digest}")


@app.command("repository-check", help="report repository ownership and local-documentation findings")
def repository_check_cmd(
    repository_root: Annotated[
        Path,
        typer.Option("--repository-root", help="framework checkout to inspect"),
    ] = Path("."),
    report_only: Annotated[
        bool,
        typer.Option("--report-only", help="explicitly request the non-failing migration inventory"),
    ] = False,
) -> None:
    # The command is intentionally report-only during the 0.3.0 migration.
    # Keep the explicit flag in the public invocation so promotion to a CI
    # gate is a conscious contract change rather than an accidental default.
    del report_only
    print(inspect_repository(repository_root).render())


@images_app.command("publish", help="build, push, and pin every image in this release")
def publish_cmd(
    registry: Annotated[
        str,
        typer.Option("--registry", help="registry prefix to push (normally registry.lan/carbonteq)"),
    ],
    framework_version: Annotated[
        str,
        typer.Option("--framework-version", help="version recorded in the manifest"),
    ],
    receipt_root: Annotated[
        Path,
        typer.Option("--receipt-root", help="directory retaining build receipts"),
    ],
    trust_bundle: Annotated[
        Path | None,
        typer.Option(
            "--trust-bundle",
            help=("machine-owned PEM CA bundle appended to the base runtime image for private HTTPS package indexes"),
        ),
    ] = None,
    default_prefix: Annotated[
        str | None,
        typer.Option(
            "--default-prefix",
            help="registry recorded in the manifest, when staging through another",
        ),
    ] = None,
    repository_root: Annotated[
        Path,
        typer.Option("--repository-root", help="framework checkout to rewrite"),
    ] = Path("."),
    variant: Annotated[
        list[str] | None,
        typer.Option(
            "--variant",
            help=(
                "rebuild only this job-kind (repeatable). Unlisted kinds are "
                "reused from the committed manifest only when their runtime "
                "inputs are unchanged"
            ),
        ),
    ] = None,
    base_image: Annotated[
        str | None,
        typer.Option(
            "--base-image",
            help=(
                "reuse this already-published base digest instead of rebuilding base (kinds FROM it from the registry)"
            ),
        ),
    ] = None,
    parallel: Annotated[
        bool,
        typer.Option(
            "--parallel/--no-parallel",
            help="build selected kind variants with bounded concurrency (default: parallel)",
        ),
    ] = True,
    revision: Annotated[
        str | None,
        typer.Option(
            "--revision",
            help="git commit recorded in image metadata; defaults to HEAD",
        ),
    ] = None,
    attestations: Annotated[
        bool,
        typer.Option(
            "--attestations/--no-attestations",
            help="emit provenance+SBOM attestation manifests (default: off)",
        ),
    ] = False,
    compression_level: Annotated[
        int,
        typer.Option(
            "--compression-level",
            help="zstd compression level for pushed layers (default: 1)",
        ),
    ] = 1,
    force_compression: Annotated[
        bool,
        typer.Option(
            "--force-compression/--no-force-compression",
            help="re-encode layers even when already compressed (default: off)",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="print the manifest instead of writing it"),
    ] = False,
) -> None:
    from posttrain.execution import RuntimeImageRef

    def _git(*arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repository_root), *arguments],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()

    resolved_revision = revision or _git("rev-parse", "HEAD")
    if not resolved_revision:
        raise typer.BadParameter(f"not a git checkout: {repository_root}")
    # The commit's own timestamp, never wall clock. CREATED is a build variable
    # and therefore part of the build key, so a clock reading would give every
    # publish a fresh key, defeat the receipt cache, and rebuild every image on
    # each run. Deriving it from the revision keeps publishing idempotent.
    created = _git("show", "-s", "--format=%cI", resolved_revision)
    if not created:
        raise typer.BadParameter(f"unknown revision: {resolved_revision}")
    reused_base = RuntimeImageRef(base_image) if base_image else None
    rendered = publish_release(
        prefix=registry,
        framework_version=framework_version,
        created=created,
        revision=resolved_revision,
        default_prefix=default_prefix,
        builder=BuildKitRuntimeBuilder(receipt_root=receipt_root.resolve()),
        variants=variant,
        base_image=reused_base,
        parallel=parallel,
        attestations=attestations,
        compression_level=compression_level,
        force_compression=force_compression,
        trust_bundle=trust_bundle.resolve() if trust_bundle is not None else None,
    )
    if dry_run:
        print(rendered, end="")
        return
    destination = (repository_root / _MANIFEST_RELATIVE).resolve()
    if not destination.parent.is_dir():
        raise typer.BadParameter(f"not a framework checkout: {repository_root}")
    destination.write_text(rendered, encoding="utf-8")
    print(f"pinned {destination}")


@images_app.command("plan", help="inspect immutable release-image inputs and print only the required actions")
def plan_cmd(
    registry: Annotated[
        str,
        typer.Option("--registry", help="destination registry prefix to inspect"),
    ],
    receipt_root: Annotated[
        Path,
        typer.Option("--receipt-root", help="builder receipt directory (not modified by planning)"),
    ],
    repository_root: Annotated[
        Path,
        typer.Option("--repository-root", help="framework checkout containing the installed definitions"),
    ] = Path("."),
    trust_bundle: Annotated[
        Path | None,
        typer.Option("--trust-bundle", help="machine-owned CA bundle whose bytes are part of the base identity"),
    ] = None,
    variant: Annotated[
        list[str] | None,
        typer.Option("--variant", help="force this job-kind to rebuild in the displayed plan (repeatable)"),
    ] = None,
) -> None:
    root = repository_root.resolve()
    if not root.is_dir():
        raise typer.BadParameter(f"framework checkout does not exist: {repository_root}")
    plan, _ = _release_image_plan(
        root=cached_definition_root(),
        prefix=registry.rstrip("/"),
        builder=BuildKitRuntimeBuilder(receipt_root=receipt_root.resolve()),
        trust_bundle=trust_bundle.resolve() if trust_bundle is not None else None,
        variants=variant,
    )
    document = {
        "blocked": plan.blocked,
        "known_transfer_bytes": plan.known_transfer_bytes,
        "has_unknown_transfer_bytes": plan.has_unknown_transfer_bytes,
        "nodes": [
            {
                "name": node.desired.name,
                "repository": node.desired.repository,
                "action": node.action,
                "reason": node.reason,
                "expected_digest": node.expected_digest,
                "estimated_transfer_bytes": node.estimated_transfer_bytes,
                "source_status": node.source.status,
                "destination_status": node.destination.status,
            }
            for node in plan.nodes
        ],
    }
    print(json.dumps(document, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    try:
        app(args=argv if argv is not None else sys.argv[1:], standalone_mode=False)
    except typer.Exit as exit_signal:
        return int(exit_signal.exit_code)
    except (typer.BadParameter, ValueError, RuntimeError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


__all__ = ["app", "main"]
