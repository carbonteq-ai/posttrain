"""Tests for declarative dataset catalog selections and materialization."""

import json
from pathlib import Path

import pytest
from posttrain.common import CatalogRef, ContractError
from posttrain.data import (
    DatasetLoadPlan,
    PreferenceDataset,
    SupervisedDataset,
    decode_dataset_selection,
    materialize_dataset,
    resolve_dataset_source,
)


def test_packaged_fixture_materialization_is_idempotent(tmp_path: Path) -> None:
    plan = decode_dataset_selection(
        CatalogRef("dataset", "datasets/posttrain-sft-smoke@1"),
        {
            "id": "datasets/posttrain-sft-smoke@1",
            "revision": "1",
            "kind": "supervised",
            "source": {
                "kind": "fixture",
                "resource": "posttrain.data.fixtures:sft_messages.jsonl",
            },
            "format": {"kind": "messages"},
        },
        {},
    )

    assert isinstance(plan, DatasetLoadPlan)
    first = materialize_dataset(plan, state_dir=tmp_path / "state", project_root=tmp_path)
    second = materialize_dataset(plan, state_dir=tmp_path / "state", project_root=tmp_path)

    assert first.created is True
    assert second.created is False
    assert first.path == second.path
    assert first.content_sha256 == second.content_sha256
    assert first.examples == second.examples == 2

    source = resolve_dataset_source(plan, state_dir=tmp_path / "state", project_root=tmp_path)
    assert isinstance(source, SupervisedDataset)
    assert source.id == "datasets/posttrain-sft-smoke"
    assert source.revision == "1"
    assert len(source.examples) == 2


def test_declared_builder_reuses_cache_and_rebuilds_when_an_input_changes(tmp_path: Path) -> None:
    data = tmp_path / "data" / "raw.txt"
    data.parent.mkdir()
    data.write_text("first\n", encoding="utf-8")
    builder = tmp_path / "datasets" / "build.py"
    builder.parent.mkdir()
    builder.write_text(
        """from pathlib import Path

def build():
    value = (Path(__file__).parent.parent / 'data/raw.txt').read_text(encoding='utf-8').strip()
    return [{'messages': [{'role': 'user', 'content': value}, {'role': 'assistant', 'content': 'ok'}]}]
""",
        encoding="utf-8",
    )
    plan = decode_dataset_selection(
        CatalogRef("dataset", "datasets/built@1"),
        {
            "id": "datasets/built@1",
            "revision": "1",
            "kind": "supervised",
            "source": {
                "kind": "built",
                "builder": {"kind": "python-file", "path": "datasets/build.py", "callable": "build"},
                "inputs": ["data/raw.txt"],
            },
            "format": {"kind": "messages"},
        },
        {},
    )
    first = materialize_dataset(plan, state_dir=tmp_path / "state", project_root=tmp_path)
    second = materialize_dataset(plan, state_dir=tmp_path / "state", project_root=tmp_path)
    data.write_text("second\n", encoding="utf-8")
    rebuilt = materialize_dataset(plan, state_dir=tmp_path / "state", project_root=tmp_path)

    assert first.created is True
    assert second.created is False
    assert rebuilt.created is True
    assert rebuilt.path != first.path


def test_nemo_supervised_catalog_source_materializes(tmp_path: Path) -> None:
    data_dir = tmp_path / "data" / "nemo"
    data_dir.mkdir(parents=True)
    (data_dir / "sft.jsonl").write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Cancel my order"},
                    {"role": "assistant", "content": "Done."},
                ],
                "tools": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    plan = decode_dataset_selection(
        CatalogRef("dataset", "datasets/nemo-sft@1"),
        {
            "id": "datasets/nemo-sft@1",
            "revision": "1",
            "kind": "supervised",
            "source": {"kind": "nemo", "path": "data/nemo/sft.jsonl"},
            "format": {"kind": "messages"},
        },
        {},
    )

    source = resolve_dataset_source(plan, state_dir=tmp_path / "state", project_root=tmp_path)

    assert isinstance(source, SupervisedDataset)
    assert source.id == "datasets/nemo-sft"
    assert source.metadata["source_kind"] == "nemo"
    assert source.examples[0].messages[-1]["content"] == "Done."
    assert source.examples[0].metadata["source_format"] == "messages"


def test_nemo_preference_catalog_source_materializes(tmp_path: Path) -> None:
    data_dir = tmp_path / "data" / "nemo"
    data_dir.mkdir(parents=True)
    (data_dir / "prefs.jsonl").write_text(
        json.dumps(
            {
                "context": [{"role": "user", "content": "Q"}],
                "completions": [
                    {"rank": 0, "completion": [{"role": "assistant", "content": "A"}]},
                    {"rank": 1, "completion": [{"role": "assistant", "content": "B"}]},
                ],
                "task_name": "math",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    plan = decode_dataset_selection(
        CatalogRef("dataset", "datasets/nemo-prefs@1"),
        {
            "id": "datasets/nemo-prefs@1",
            "revision": "1",
            "kind": "preference",
            "source": {"kind": "nemo", "path": "data/nemo/prefs.jsonl"},
            "format": {"kind": "nemo-ranked"},
        },
        {},
    )

    source = resolve_dataset_source(plan, state_dir=tmp_path / "state", project_root=tmp_path)

    assert isinstance(source, PreferenceDataset)
    assert source.metadata["source_kind"] == "nemo"
    assert source.examples[0].chosen[0]["content"] == "A"
    assert source.examples[0].rejected[0]["content"] == "B"
    assert source.examples[0].metadata["task_name"] == "math"
    # Materialized cache is HF-normalized TRL rows; NeMo layout is import-only.
    assert source.examples[0].metadata["source_format"] == "trl"


def test_nemo_source_rejects_incompatible_formats() -> None:
    with pytest.raises(ContractError, match="supervised dataset format"):
        decode_dataset_selection(
            CatalogRef("dataset", "datasets/nemo-bad@1"),
            {
                "id": "datasets/nemo-bad@1",
                "revision": "1",
                "kind": "supervised",
                "source": {"kind": "nemo", "path": "data/nemo/sft.jsonl"},
                "format": {"kind": "alpaca"},
            },
            {},
        )
