"""Framework-owner release command.

Not shipped to consumers: `posttrain` does not depend on this distribution.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer
from posttrain_execution_buildkit import BuildKitRuntimeBuilder

from .publish import publish_release

app = typer.Typer(help="publish framework runtime images and pin the release manifest")
images_app = typer.Typer(help="runtime image release operations")
app.add_typer(images_app, name="images")

_MANIFEST_RELATIVE = Path("packages/runtime-images/src/posttrain/runtime_images/published.toml")


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
                "reused from the committed manifest when their lock digest is "
                "unchanged"
            ),
        ),
    ] = None,
    base_image: Annotated[
        str | None,
        typer.Option(
            "--base-image",
            help=(
                "reuse this already-published base digest instead of rebuilding "
                "base (kinds FROM it from the registry)"
            ),
        ),
    ] = None,
    parallel: Annotated[
        bool,
        typer.Option(
            "--parallel/--no-parallel",
            help="build selected kind variants concurrently (default: parallel)",
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
    )
    if dry_run:
        print(rendered, end="")
        return
    destination = (repository_root / _MANIFEST_RELATIVE).resolve()
    if not destination.parent.is_dir():
        raise typer.BadParameter(f"not a framework checkout: {repository_root}")
    destination.write_text(rendered, encoding="utf-8")
    print(f"pinned {destination}")


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
