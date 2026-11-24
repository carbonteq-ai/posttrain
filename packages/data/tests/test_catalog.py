"""Tests for declarative dataset catalog selections and materialization."""

import json
from pathlib import Path

import pytest
from posttrain.common import CatalogRef, ContractError
from posttrain.data import (
    BuiltDatasetSource,
    DatasetLoadPlan,
    LocalDatasetInput,
    PreferenceDataset,
    PythonDatasetBuilder,
    SupervisedDataset,
    decode_dataset_selection,
    materialize_dataset,
    project_dataset_input_paths,
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


def test_project_dataset_input_paths_returns_only_declared_data_inputs() -> None:
    static = DatasetLoadPlan(
        id="datasets/static@1",
        revision="1",
        kind="supervised",
        source={"kind": "nemo", "path": "data/selected.jsonl"},
        format="messages",
    )
    typed = DatasetLoadPlan(
        id="datasets/typed@1",
        revision="1",
        kind="supervised",
        source=BuiltDatasetSource(
            builder=PythonDatasetBuilder("builder:build"),
            inputs={
                "second": LocalDatasetInput("data/second.jsonl"),
                "first": LocalDatasetInput("data/first.jsonl"),
            },
        ),
        format="messages",
    )
    fixture = DatasetLoadPlan(
        id="datasets/fixture@1",
        revision="1",
        kind="supervised",
        source={"kind": "fixture", "resource": "posttrain.data.fixtures:sft_messages.jsonl"},
        format="messages",
    )

    assert project_dataset_input_paths(static) == ("data/selected.jsonl",)
    assert project_dataset_input_paths(typed) == (
        "data/first.jsonl",
        "data/second.jsonl",
    )
    assert project_dataset_input_paths(fixture) == ()


def test_project_dataset_source_rejects_symlinked_input(tmp_path: Path) -> None:
    outside = tmp_path / "outside.jsonl"
    outside.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a"},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    data = tmp_path / "data"
    data.mkdir()
    (data / "selected.jsonl").symlink_to(outside)
    plan = DatasetLoadPlan(
        id="datasets/symlink@1",
        revision="1",
        kind="supervised",
        source={"kind": "nemo", "path": "data/selected.jsonl"},
        format="messages",
    )

    with pytest.raises(ContractError, match="cannot traverse symlinks"):
        materialize_dataset(plan, state_dir=tmp_path / "state", project_root=tmp_path)


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


def _agent_row() -> dict:
    """A tool-calling SFT row whose mappings are deliberately not in sorted key order."""

    return {
        "id": "rows/agent-key-order",
        "messages": [
            {"role": "system", "content": "You are an agent."},
            {"role": "user", "content": "Email the summary — café “quoted”."},
            {
                "role": "assistant",
                "content": "Reading the sheet and the inbox first.",
                "tool_calls": [
                    {
                        "id": "call_1_1",
                        "type": "function",
                        "function": {
                            "name": "sheets_get",
                            "arguments": {"spreadsheet": "ss_1", "includeGridData": True, "a": {"z": 1, "b": [2]}},
                        },
                    },
                    {
                        "id": "call_1_2",
                        "type": "function",
                        "function": {"name": "gmail_find", "arguments": {"query": "", "max_results": 20}},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_1_1", "name": "sheets_get", "content": '{"success": true}'},
            {"role": "tool", "tool_call_id": "call_1_2", "name": "gmail_find", "content": '{"success": true}'},
            {"role": "assistant", "content": "Sent."},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "sheets_get",
                    "description": "Get a sheet.",
                    "parameters": {
                        "type": "object",
                        "properties": {"spreadsheet": {"type": "string"}, "includeGridData": {"type": "boolean"}},
                        "required": ["spreadsheet"],
                    },
                },
            }
        ],
        "trainable_message_indices": [2, 5],
        "metadata": {"task": "t", "source": "unit"},
    }


def _jsonl_plan(path: str) -> DatasetLoadPlan:
    plan = decode_dataset_selection(
        CatalogRef("dataset", "datasets/key-order@1"),
        {
            "id": "datasets/key-order@1",
            "revision": "1",
            "kind": "supervised",
            "source": {"kind": "jsonl", "path": path},
            "format": {"kind": "messages"},
        },
        {},
    )
    assert isinstance(plan, DatasetLoadPlan)
    return plan


def test_materialization_keeps_row_key_order_and_round_trips_byte_identically(tmp_path: Path) -> None:
    row = _agent_row()
    # The adapter appends exactly these two metadata fields; with them present the row is a fixed point.
    expected = {**row, "metadata": {**row["metadata"], "source_row": 0, "source_format": "messages"}}
    (tmp_path / "rows.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    first = materialize_dataset(_jsonl_plan("rows.jsonl"), state_dir=tmp_path / "state", project_root=tmp_path)
    materialized = first.path.read_text(encoding="utf-8")
    assert materialized == json.dumps(expected) + "\n"
    for field in ("messages", "tools", "trainable_message_indices"):
        assert json.dumps(row[field]) in materialized
    loaded = json.loads(materialized)
    assert list(loaded["messages"][2]["tool_calls"][0]["function"]["arguments"]) == [
        "spreadsheet",
        "includeGridData",
        "a",
    ]
    assert list(loaded["tools"][0]["function"]["parameters"]["properties"]) == ["spreadsheet", "includeGridData"]

    # Materializing the materialized file again gives the same bytes.
    (tmp_path / "again").mkdir()
    (tmp_path / "again" / "rows.jsonl").write_text(materialized, encoding="utf-8")
    second = materialize_dataset(
        _jsonl_plan("rows.jsonl"), state_dir=tmp_path / "again" / "state", project_root=tmp_path / "again"
    )
    assert second.path.read_bytes() == first.path.read_bytes()
    assert second.content_sha256 == first.content_sha256

    # The loaded dataset (the training job's path) keeps the order too.
    source = resolve_dataset_source(_jsonl_plan("rows.jsonl"), state_dir=tmp_path / "state", project_root=tmp_path)
    assert isinstance(source, SupervisedDataset)
    message = source.examples[0].message_records()[2]
    assert list(message["tool_calls"][1]["function"]["arguments"]) == ["query", "max_results"]


def test_untyped_cache_fingerprint_names_the_row_serialization(tmp_path: Path) -> None:
    import hashlib

    from posttrain.data.catalog import MATERIALIZED_ROW_SERIALIZATION, _plan_json

    (tmp_path / "rows.jsonl").write_text(json.dumps(_agent_row()) + "\n", encoding="utf-8")
    plan = _jsonl_plan("rows.jsonl")
    identity = _plan_json(plan, project_root=tmp_path, input_root=tmp_path)
    assert MATERIALIZED_ROW_SERIALIZATION == "json-source-key-order@1"
    assert '"row_serialization":"json-source-key-order@1"' in identity
    # caches written by the earlier key-sorting code live under a fingerprint without the marker: never reused
    materialized = materialize_dataset(plan, state_dir=tmp_path / "state", project_root=tmp_path)
    assert materialized.path.parent.name == hashlib.sha256(identity.encode("utf-8")).hexdigest()


def test_untyped_cache_is_content_addressed_rewrite_under_same_revision_rebuilds(tmp_path: Path) -> None:
    """Rewriting the source file under the same selection revision must never reuse the stale materialization."""

    row = _agent_row()
    source = tmp_path / "rows.jsonl"
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")
    plan = _jsonl_plan("rows.jsonl")
    first = materialize_dataset(plan, state_dir=tmp_path / "state", project_root=tmp_path)
    cached = materialize_dataset(plan, state_dir=tmp_path / "state", project_root=tmp_path)
    assert first.created is True and cached.created is False  # unchanged bytes: still a cache hit
    assert cached.path == first.path

    changed = json.loads(json.dumps(row))
    changed["messages"][-1]["content"] = "A different final reply."
    source.write_text(json.dumps(changed) + "\n", encoding="utf-8")  # same path, same selection revision
    rebuilt = materialize_dataset(plan, state_dir=tmp_path / "state", project_root=tmp_path)
    assert rebuilt.created is True
    assert rebuilt.path != first.path
    assert rebuilt.content_sha256 != first.content_sha256
    assert "A different final reply." in rebuilt.path.read_text(encoding="utf-8")

    # Restoring the original bytes finds the original cache entry again.
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")
    again = materialize_dataset(plan, state_dir=tmp_path / "state", project_root=tmp_path)
    assert again.created is False and again.path == first.path


def test_untyped_cache_identity_with_missing_source_is_a_contract_error(tmp_path: Path) -> None:
    with pytest.raises(ContractError, match="not found"):
        materialize_dataset(_jsonl_plan("absent.jsonl"), state_dir=tmp_path / "state", project_root=tmp_path)
