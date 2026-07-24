"""Dataset commands."""

from __future__ import annotations

from typing import Annotated

import typer
from posttrain.common import CatalogRef, ContractError
from posttrain.data import DatasetLoadPlan, materialize_dataset

from ..constants import DATASET_FORMAT_CHOICES, DATASET_KIND_CHOICES, NEMO_FORMAT_CHOICES
from ..context import CliState
from ..output import emit
from ..overlay_write import ensure_overlay_file, overlay_directory, selection_revision, upsert_family_entry


def register(app: typer.Typer) -> None:
    dataset_app = typer.Typer(rich_markup_mode=None, no_args_is_help=True, help="resolve and validate project datasets")
    app.add_typer(dataset_app, name="dataset")

    @dataset_app.command("validate", help="materialize and adapter-validate one dataset selection")
    def dataset_validate_cmd(
        ctx: typer.Context,
        selection_id: Annotated[str, typer.Argument(metavar="id")],
    ) -> None:
        state: CliState = ctx.obj
        layout, catalog = state.open_catalog()
        resolved = catalog.resolve(CatalogRef("dataset", selection_id))
        if not isinstance(resolved.value, DatasetLoadPlan):
            raise ContractError(f"catalog dataset {selection_id!r} did not resolve to a dataset load plan")
        materialized = materialize_dataset(
            resolved.value,
            state_dir=layout.state,
            project_root=layout.root,
        )
        payload = {
            "id": materialized.selection_id,
            "revision": materialized.selection_revision,
            "source_layer": resolved.source_layer,
            "overlay_id": resolved.overlay_id,
            "source_kind": materialized.source_kind,
            "path": str(materialized.path),
            "manifest": str(materialized.manifest_path),
            "content_sha256": materialized.content_sha256,
            "examples": materialized.examples,
            "materialized": materialized.created,
        }
        action = "Materialized" if materialized.created else "Validated cached"
        emit(
            state,
            payload,
            f"{action} dataset {materialized.selection_id} ({materialized.examples} examples) at {materialized.path}",
        )

    add_app = typer.Typer(
        rich_markup_mode=None, no_args_is_help=True, help="write a dataset entry into the project overlay"
    )
    dataset_app.add_typer(add_app, name="add")

    @add_app.command("hf", help="register a Hugging Face dataset")
    def dataset_add_hf_cmd(
        ctx: typer.Context,
        selection_id: Annotated[
            str,
            typer.Option("--id", help="catalog selection id, for example datasets/support-sft@1"),
        ],
        repo: Annotated[str, typer.Option("--repo")],
        revision: Annotated[str, typer.Option("--revision", help="immutable Hub revision")],
        split: Annotated[str, typer.Option("--split")] = "train",
        config: Annotated[str | None, typer.Option("--config")] = None,
        format_kind: Annotated[str, typer.Option("--format")] = "messages",
        kind: Annotated[str, typer.Option("--kind")] = "supervised",
        file: Annotated[str, typer.Option("--file", help="overlay YAML filename")] = "datasets.yaml",
    ) -> None:
        _dataset_add(
            ctx,
            add_kind="hf",
            selection_id=selection_id,
            kind=kind,
            format_kind=format_kind,
            file=file,
            repo=repo,
            revision=revision,
            split=split,
            config=config,
        )

    @add_app.command("jsonl", help="register a project-relative JSONL dataset")
    def dataset_add_jsonl_cmd(
        ctx: typer.Context,
        selection_id: Annotated[str, typer.Option("--id")],
        path: Annotated[str, typer.Option("--path", help="path relative to the project root")],
        format_kind: Annotated[str, typer.Option("--format")] = "messages",
        kind: Annotated[str, typer.Option("--kind")] = "supervised",
        file: Annotated[str, typer.Option("--file")] = "datasets.yaml",
    ) -> None:
        _dataset_add(
            ctx,
            add_kind="jsonl",
            selection_id=selection_id,
            kind=kind,
            format_kind=format_kind,
            file=file,
            path=path,
        )

    @add_app.command("nemo", help="register a project-relative NeMo JSONL dataset")
    def dataset_add_nemo_cmd(
        ctx: typer.Context,
        selection_id: Annotated[str, typer.Option("--id")],
        path: Annotated[str, typer.Option("--path", help="path relative to the project root")],
        format_kind: Annotated[str, typer.Option("--format")] = "auto",
        kind: Annotated[str, typer.Option("--kind")] = "supervised",
        file: Annotated[str, typer.Option("--file")] = "datasets.yaml",
    ) -> None:
        _dataset_add(
            ctx,
            add_kind="nemo",
            selection_id=selection_id,
            kind=kind,
            format_kind=format_kind,
            file=file,
            path=path,
        )


def _dataset_add(
    ctx: typer.Context,
    *,
    add_kind: str,
    selection_id: str,
    kind: str,
    format_kind: str,
    file: str,
    **source_args: object,
) -> None:
    from posttrain.common.selections import validate_selection_id

    state: CliState = ctx.obj
    if kind not in DATASET_KIND_CHOICES:
        raise typer.BadParameter(f"invalid choice: {kind!r}", param_hint="--kind")
    if add_kind in {"hf", "jsonl"} and format_kind not in DATASET_FORMAT_CHOICES:
        raise typer.BadParameter(f"invalid choice: {format_kind!r}", param_hint="--format")
    if add_kind == "nemo" and format_kind not in NEMO_FORMAT_CHOICES:
        raise typer.BadParameter(f"invalid choice: {format_kind!r}", param_hint="--format")

    layout = state.layout()
    validated_id = validate_selection_id(selection_id, "dataset selection id")
    if add_kind == "hf":
        source: dict[str, object] = {
            "kind": "huggingface",
            "repo": source_args["repo"],
            "revision": source_args["revision"],
            "split": source_args["split"],
        }
        if source_args.get("config"):
            source["config"] = source_args["config"]
    elif add_kind == "jsonl":
        source = {"kind": "jsonl", "path": source_args["path"]}
    elif add_kind == "nemo":
        source = {"kind": "nemo", "path": source_args["path"]}
        if kind == "supervised" and format_kind not in {"auto", "messages"}:
            raise ContractError("nemo supervised format must be auto or messages")
        if kind == "preference" and format_kind not in {"auto", "nemo-ranked"}:
            raise ContractError("nemo preference format must be auto or nemo-ranked")
    else:
        raise ContractError(f"unsupported dataset add kind: {add_kind}")

    entry = {
        "revision": selection_revision(validated_id),
        "kind": kind,
        "source": source,
        "format": {"kind": format_kind},
    }
    overlay = overlay_directory(layout)
    path = ensure_overlay_file(overlay, file, layer_id=f"{layout.project_id}-v1")
    upsert_family_entry(path, family="dataset", entry_id=validated_id, entry=entry)
    _, catalog = state.open_catalog()
    resolved = catalog.resolve(CatalogRef("dataset", validated_id))
    if not isinstance(resolved.value, DatasetLoadPlan):
        raise ContractError(f"wrote dataset {validated_id!r} but it did not decode as a load plan")
    payload = {
        "id": validated_id,
        "path": str(path),
        "source_layer": resolved.source_layer,
        "overlay_id": resolved.overlay_id,
        "kind": kind,
        "source_kind": resolved.value.source_kind,
    }
    emit(state, payload, f"Added dataset {validated_id} to {path}")
