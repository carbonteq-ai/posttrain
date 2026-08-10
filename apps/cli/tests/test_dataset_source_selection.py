from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from posttrain.common import Catalog, CatalogLayer, CatalogRef, ContractError
from posttrain.data import DatasetLoadPlan
from posttrain.execution_pack import DatasetPackRequest, JobPackPlan
from posttrain_cli.execution_planning import _dataset_source_estimates, _reject_dataset_inputs_in_project_source


def _catalog() -> Catalog:
    local = DatasetLoadPlan(
        id="datasets/local@1",
        revision="1",
        kind="supervised",
        source={"kind": "nemo", "path": "data/local.jsonl"},
        format="messages",
    )
    remote = DatasetLoadPlan(
        id="datasets/remote@1",
        revision="1",
        kind="supervised",
        source={"kind": "fixture", "resource": "example:remote.jsonl"},
        format="messages",
    )
    return Catalog(
        CatalogLayer(
            "base",
            {
                CatalogRef("dataset", local.id): local,
                CatalogRef("dataset", remote.id): remote,
            },
        ),
        (),
        "test",
    )


@pytest.mark.parametrize("source_include", [".", "data", "data/local.jsonl"])
def test_rejects_catalog_dataset_copied_as_generic_project_source(source_include: str) -> None:
    with pytest.raises(ContractError, match="packaged only through the selected job's dataset seats"):
        _reject_dataset_inputs_in_project_source((source_include,), _catalog())


def test_allows_code_source_that_does_not_cover_catalog_dataset() -> None:
    _reject_dataset_inputs_in_project_source(("pyproject.toml", "src"), _catalog())


def test_dataset_source_estimates_are_seat_specific(tmp_path: Path) -> None:
    source = tmp_path / "data" / "local.jsonl"
    source.parent.mkdir()
    source.write_text("{}\n", encoding="utf-8")
    catalog = _catalog()
    plan = SimpleNamespace(
        spec=SimpleNamespace(
            datasets=(
                DatasetPackRequest(
                    "train",
                    cast(DatasetLoadPlan, catalog.resolve(CatalogRef("dataset", "datasets/local@1")).value),
                ),
            ),
        )
    )

    estimates = _dataset_source_estimates(tmp_path, cast(JobPackPlan, plan))

    assert estimates == (
        {
            "seat_name": "train",
            "selection_id": "datasets/local@1",
            "selection_revision": "1",
            "paths": ["data/local.jsonl"],
            "byte_count": 3,
            "materialization_estimate": "source-inputs",
        },
    )
