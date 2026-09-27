from __future__ import annotations

import json
from pathlib import Path

import pytest
from posttrain.common import ContractError
from posttrain_cli.cli import main
from posttrain_cli.commands.query_cmd import build_query, parse_filter, parse_runs, render_csv, render_table

_RESULT = {
    "columns": [
        {"name": "run.id", "kind": "dimension"},
        {"name": "rollout_share", "kind": "metric"},
        {"name": "lr", "kind": "value"},
    ],
    "rows": [["run-a", 0.8964941, 5e-05], ["run-b", None, 0.0001]],
    "grain": "update",
    "sql": "SELECT r.`id` AS `run.id`\nFROM runs AS r",
    "engine": "doris",
    "truncated": True,
}


def _query(**overrides: object) -> dict[str, object]:
    options: dict[str, object] = {
        "measures": None,
        "by": None,
        "where": None,
        "runs": None,
        "order_by": None,
        "limit": None,
        "sql": None,
        "file": None,
    }
    options.update(overrides)
    return build_query(**options)  # type: ignore[arg-type]


def test_filters_parse_equality_lists_comparisons_and_wildcards() -> None:
    assert parse_filter("run.status=running") == ("run.status", "running")
    assert parse_filter("run.status=running,succeeded") == ("run.status", ["running", "succeeded"])
    assert parse_filter("update.step>=10") == ("update.step", ">= 10")
    assert parse_filter("run.work_package=train/lfm2.5-2.6b/*") == ("run.work_package", "train/lfm2.5-2.6b/*")
    with pytest.raises(ContractError):
        parse_filter("not a filter")


def test_runs_are_ids_or_filters_but_not_both() -> None:
    assert parse_runs(["a,b", "c"]) == ("a", "b", "c")
    assert parse_runs(["run.job_kind=train.sampo", "run.status=running,succeeded"]) == {
        "run.job_kind": "train.sampo",
        "run.status": ["running", "succeeded"],
    }
    assert parse_runs(None) is None
    with pytest.raises(ContractError):
        parse_runs(["run-a", "run.status=running"])


def test_semantic_and_sql_queries_from_options(tmp_path: Path) -> None:
    assert _query(measures=["entropy:last,kl"], by=["run.id"], where=["update.step>=5"], runs=["r1"], limit=5) == {
        "measures": ["entropy:last", "kl"],
        "by": ["run.id"],
        "where": {"update.step": ">= 5"},
        "runs": ["r1"],
        "limit": 5,
    }
    assert _query(sql="select 1", runs=["r1"]) == {"sql": "select 1", "runs": ("r1",)}
    assert _query(sql="select 1") == {"sql": "select 1"}
    with pytest.raises(ContractError, match="give --measures"):
        _query()
    path = tmp_path / "query.yaml"
    path.write_text("measures: [reward]\nby: [run.id]\n", encoding="utf-8")
    assert _query(file=path, runs=["r2"]) == {"measures": ["reward"], "by": ["run.id"], "runs": ("r2",)}


def test_table_keeps_small_values_and_reports_gaps() -> None:
    table = render_table(_RESULT)
    assert "5e-05" in table and "0.896494" in table
    assert "truncated: more rows exist" in table
    assert "SQL:\n  SELECT r.`id` AS `run.id`\n  FROM runs AS r" in table
    assert render_csv(_RESULT).splitlines() == ["run.id,rollout_share,lr", "run-a,0.8964941,5e-05", "run-b,,0.0001"]


def test_table_renders_an_empty_result() -> None:
    table = render_table({**_RESULT, "rows": [], "truncated": False})
    assert table.splitlines()[0].split() == ["run.id", "rollout_share", "lr"]


def test_query_command_sends_the_parsed_query_to_the_observatory(tmp_path: Path, capsys, monkeypatch) -> None:
    import posttrain_observatory
    from posttrain_observatory.semantic_layer import SemanticQuery, SemanticResult

    project = tmp_path / "example"
    assert main(["init", str(project)]) == 0
    capsys.readouterr()
    received: list[object] = []

    class FakeService:
        async def query_semantics(self, query: object) -> SemanticResult:
            received.append(query)
            return SemanticResult.model_validate_json(json.dumps(_RESULT))

    monkeypatch.setattr(posttrain_observatory, "create_service", lambda settings: FakeService())
    argv = ["--project-root", str(project), "query", "-m", "rollout_share", "--by", "run.id", "-r", "run-a,run-b"]
    assert main([*argv, "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["rows"][0][0] == "run-a"
    assert received == [SemanticQuery(measures=("rollout_share",), by=("run.id",), runs=("run-a", "run-b"))]
