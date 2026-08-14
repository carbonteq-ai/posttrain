"""Runtime image commands.

These are consumer operations. Publishing a framework release is an owner
operation and lives outside the distribution a consumer installs.
"""

from __future__ import annotations

from typing import Annotated

import typer
from posttrain.common import ContractError
from posttrain.runtime_images import RUNTIME_VARIANTS
from posttrain.runtime_images.manifest import load_manifest
from posttrain_execution_buildkit import RuntimeImageInspector

from ..context import CliState
from ..execution_config import (
    REGISTRY_ENVIRONMENT_VARIABLE,
    load_local_execution_config,
)
from ..output import emit
from ..runtime_images import verify_registry


def _registry(state: CliState):
    layout = state.layout()
    configuration = load_local_execution_config(layout)
    if configuration.registry is None:
        raise ContractError(
            "no registry is configured: set "
            f"{REGISTRY_ENVIRONMENT_VARIABLE} in the protected project environment file, or declare "
            "[registry] in the execution configuration"
        )
    return configuration.registry


def register(app: typer.Typer) -> None:
    runtime_app = typer.Typer(help="framework-published runtime images")
    images_app = typer.Typer(help="inspect, verify, mirror, and build runtime images")
    runtime_app.add_typer(images_app, name="images")
    app.add_typer(runtime_app, name="runtime")

    @images_app.command("list", help="show the image identities this framework expects")
    def list_cmd(ctx: typer.Context) -> None:
        state: CliState = ctx.obj
        manifest = load_manifest()
        rows = [
            {
                "variant": variant,
                "repository": image.repository,
                "digest": image.digest,
                "lock_digest": image.lock_digest,
                "reference": image.reference(manifest.default_prefix),
            }
            for variant, image in sorted(manifest.kinds.items())
        ]
        payload = {
            "framework_version": manifest.framework_version,
            "default_prefix": manifest.default_prefix,
            "base": {
                "repository": manifest.base.repository,
                "digest": manifest.base.digest,
                "lock_digest": manifest.base.lock_digest,
                "reference": manifest.base.reference(manifest.default_prefix),
            },
            "kinds": rows,
        }
        if state.json_output:
            emit(state, payload, "")
            return
        print(f"framework {manifest.framework_version} publishes to {manifest.default_prefix}")
        print(f"  base  {manifest.base.repository}@{manifest.base.digest}")
        for row in rows:
            print(f"  kind  {row['variant']}: {row['repository']}@{row['digest']}")
            print(f"        lock {row['lock_digest']}")

    @images_app.command("verify", help="check the configured registry against this framework")
    def verify_cmd(
        ctx: typer.Context,
        variant: Annotated[
            list[str] | None,
            typer.Option("--variant", help="limit to one or more runtime variants"),
        ] = None,
    ) -> None:
        state: CliState = ctx.obj
        registry = _registry(state)
        results = verify_registry(registry, variants=variant or None)
        succeeded = all(result.ok for result in results)
        payload = {
            "ok": succeeded,
            "results": [
                {
                    "variant": result.variant,
                    "reference": result.reference,
                    "status": result.status,
                    "detail": result.detail,
                }
                for result in results
            ],
        }
        if state.json_output:
            emit(state, payload, "")
        else:
            for result in results:
                print(f"{result.status.upper():11} {result.variant}: {result.detail}")
        if not succeeded:
            raise typer.Exit(code=1)

    @images_app.command(
        "mirror",
        help="copy published images into another registry by digest",
    )
    def mirror_cmd(
        ctx: typer.Context,
        registry_prefix: Annotated[
            str,
            typer.Option("--registry", help="destination registry prefix"),
        ],
        source_prefix: Annotated[
            str | None,
            typer.Option(
                "--from",
                help="source registry prefix; defaults to the release registry",
            ),
        ] = None,
        variant: Annotated[
            list[str] | None,
            typer.Option("--variant", help="limit to one or more runtime variants"),
        ] = None,
    ) -> None:
        state: CliState = ctx.obj
        manifest = load_manifest()
        selected = sorted(set(variant)) if variant else sorted(manifest.kinds)
        unknown = [name for name in selected if name not in manifest.kinds]
        if unknown:
            raise ContractError(
                "this release does not publish: "
                + ", ".join(unknown)
                + "; published variants are "
                + ", ".join(sorted(manifest.kinds))
            )

        # A release is normally mirrored out of the framework's own registry,
        # but a release that was staged elsewhere has to be mirrored into it.
        # Both directions copy by digest, so identity is preserved either way.
        origin = (source_prefix or manifest.default_prefix).rstrip("/")
        inspector = RuntimeImageInspector()
        copied: list[dict[str, str | bool]] = []
        images = [("base", manifest.base)] + [(name, manifest.kinds[name]) for name in selected]
        # The destination is a tag because a digest cannot be pushed to. The
        # tag is derived from the release so it is stable and meaningful, and
        # the digest is then read back to confirm identity survived the copy.
        tag = f"v{manifest.framework_version}"
        for name, image in images:
            source = image.reference(origin)
            destination = f"{registry_prefix.rstrip('/')}/{image.repository}:{tag}"
            observed, transferred = inspector.ensure_copy(
                source,
                destination,
                expected_digest=image.digest,
            )
            if observed != image.digest:
                raise ContractError(
                    f"mirroring {name} changed its identity: expected {image.digest}, "
                    f"the destination reports {observed}"
                )
            copied.append(
                {
                    "variant": name,
                    "source": source,
                    "destination": f"{destination} ({image.digest})",
                    "transferred": transferred,
                }
            )

        payload = {"mirrored": copied, "registry": registry_prefix, "source": origin}
        if state.json_output:
            emit(state, payload, "")
            return
        for entry in copied:
            action = "mirrored" if entry["transferred"] else "reused"
            print(f"{action} {entry['variant']}: {entry['destination']}")
        print(
            "set [registry].mirror_prefix in the execution configuration to "
            f"{registry_prefix} so this project resolves images from it"
        )

    @images_app.command(
        "build",
        help="rebuild images from the shipped definitions (only where pulling is impossible)",
    )
    def build_cmd(
        ctx: typer.Context,
        variant: Annotated[
            list[str] | None,
            typer.Option("--variant", help="limit to one or more runtime variants"),
        ] = None,
        push: Annotated[
            bool,
            typer.Option("--push", help="push the rebuilt images to the configured registry"),
        ] = False,
    ) -> None:
        from ..runtime_image_builds import build_runtime_images, check_runtime_images

        state: CliState = ctx.obj
        registry = _registry(state)
        selected = sorted(set(variant)) if variant else list(RUNTIME_VARIANTS)

        if not push:
            # An image has no identity until it is published, so without
            # --push the only honest operation is validating the definitions.
            checked = check_runtime_images(registry, variants=selected)
            payload = {"checked": list(checked), "built": []}
            if state.json_output:
                emit(state, payload, "")
            else:
                for name in checked:
                    print(f"OK    {name}: definition resolves")
                print("no images were published; pass --push to build and publish")
            return

        results = build_runtime_images(registry, variants=selected)
        diverged = [result for result in results if not result.matches_published_digest]
        payload = {
            "built": [
                {
                    "variant": result.variant,
                    "image": result.image,
                    "lock_digest": result.lock_digest,
                    "matches_published_digest": result.matches_published_digest,
                }
                for result in results
            ],
            "unverified": [result.variant for result in diverged],
        }
        if state.json_output:
            emit(state, payload, "")
        else:
            for result in results:
                verified = "verified" if result.matches_published_digest else "UNVERIFIED"
                print(f"{verified:10} {result.variant}: {result.image}")
            if diverged:
                print(
                    "warning: "
                    + ", ".join(result.variant for result in diverged)
                    + " do not match the digests this release pins. They are locally "
                    "built images, not the published ones, and qualification evidence "
                    "produced on them is not comparable to published results."
                )
