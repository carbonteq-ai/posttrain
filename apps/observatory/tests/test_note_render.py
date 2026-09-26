from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from posttrain_observatory.note_render import (
    BlockSyntaxError,
    format_value,
    parse_block,
    render_note,
    split_blocks,
)
from posttrain_observatory.semantic_layer.query import QueryError, SemanticQuery, SemanticResult, SqlQuery


def _result(columns: list[tuple[str, str]], rows: list[list[Any]], grain: str = "run") -> SemanticResult:
    return SemanticResult.model_validate_json(
        json.dumps(
            {
                "columns": [{"name": name, "kind": kind} for name, kind in columns],
                "rows": rows,
                "grain": grain,
            }
        )
    )


class FakeLayer:
    def __init__(self) -> None:
        self.queries: list[SemanticQuery | SqlQuery] = []

    async def __call__(self, query: SemanticQuery | SqlQuery) -> SemanticResult:
        self.queries.append(query)
        if isinstance(query, SqlQuery):
            return _result([("n", "value")], [[3]], grain="sql")
        if query.measures == ("runs",):
            values = {"run.learning_rate": 5e-5, "run.kl_beta": 0.005, "run.algorithm": "olmo3"}
            if any(name not in values for name in query.by):
                raise QueryError(f"unknown dimension {next(n for n in query.by if n not in values)!r}")
            return _result(
                [(name, "dimension") for name in query.by] + [("runs", "measure")],
                [[*(values[name] for name in query.by), 1]],
            )
        if "missing_measure" in query.measures:
            raise QueryError("unknown measure or metric 'missing_measure'")
        if query.by == ("update.step",):
            return _result(
                [("update.step", "dimension"), ("entropy", "measure"), ("kl", "measure")],
                [[1, 0.18, 0.0004], [2, 0.21, 0.01], [3, 0.28, 0.049]],
                grain="update",
            )
        return _result(
            [("run.id", "dimension"), ("reward_last", "measure"), ("update_seconds_mean", "measure")],
            [["run-a", 0.62, 334.2], ["run-b", 0.55, 280.0]],
        )


def _render(body: str, layer: FakeLayer | None = None) -> Any:
    return asyncio.run(
        render_note(body, run_id="run-a", query=layer or FakeLayer(), link_for=lambda run: f"/runs/{run}")
    )


def test_block_parser_reads_scalars_lists_mappings_and_multiline_text() -> None:
    block = parse_block(
        "measures: [reward:last, update_seconds]\n"
        "by: run.id\n"
        "where: {update.step: '>= 10', run.status: [running, succeeded]}\n"
        "limit: 5\n"
        "# comments are ignored\n"
        "sql: |\n"
        "  select *\n"
        "  from runs\n"
    )
    assert block == {
        "measures": ["reward:last", "update_seconds"],
        "by": "run.id",
        "where": {"update.step": ">= 10", "run.status": ["running", "succeeded"]},
        "limit": 5,
        "sql": "select *\nfrom runs",
    }
    with pytest.raises(BlockSyntaxError):
        parse_block("measures [a]")
    with pytest.raises(BlockSyntaxError):
        parse_block("by: [a, b")


def test_views_references_and_links_render_from_data_blocks() -> None:
    layer = FakeLayer()
    rendered = _render(
        "# Result\n\n"
        "```data reward\nmeasures: [reward:last, update_seconds]\nby: run.id\nruns: [self, run-b]\n```\n\n"
        "Reward reached {{reward.reward_last | round 2}} at learning rate {{run.learning_rate | sci 0}} "
        "({{run.algorithm}}); run-b reached {{reward.reward_last where run.id = run-b}}. "
        "Compare with [[run:run-b|the baseline]].\n\n"
        "```value\ndata: reward\ncolumn: reward_last\nwhere: run.id = run-b\ncompare: run.id = run-a\nlabel: Reward\n```\n\n"
        "```table\ndata: reward\ncolumns: [run.id, update_seconds_mean]\n```\n\n"
        "```data curve\nmeasures: [entropy, kl]\nby: update.step\n```\n\n"
        "```chart\ndata: curve\nx: update.step\ny: [entropy, kl]\n```\n\n"
        "```python\nprint('{{reward.reward_last}} stays code')\n```\n",
        layer,
    )
    assert rendered.unresolved == ()
    assert "Reward reached 0.62 at learning rate 5e-05 (olmo3); run-b reached 0.55." in rendered.text
    assert "[the baseline](/runs/run-b)" in rendered.text
    assert "print('{{reward.reward_last}} stays code')" in rendered.text
    assert "```data" not in rendered.markdown and "```note-view 0" in rendered.markdown
    value, table, chart = rendered.views
    assert (value.formatted, value.compare_formatted, value.difference) == ("0.55", "0.62", "+0.07 (+12.7%)")
    assert "**Reward:** 0.55 → 0.62, change +0.07 (+12.7%)" in rendered.text
    assert "| run-b | 280 |" in rendered.text
    assert chart.result is not None and chart.query == "measures: [entropy, kl]\nby: update.step"
    assert "- entropy: `" in rendered.text and "0.18 → 0.28 over 3 points" in rendered.text
    first = layer.queries[0]
    assert isinstance(first, SemanticQuery) and first.runs == ("run-a", "run-b")
    assert sum(1 for query in layer.queries if isinstance(query, SemanticQuery) and query.measures == ("runs",)) == 1


def test_sql_data_blocks_default_to_the_note_run() -> None:
    layer = FakeLayer()
    rendered = _render("```data n\nsql: select count(*) as n from updates\n```\nUpdates: {{n.n}}.", layer)
    assert "Updates: 3." in rendered.text
    query = layer.queries[0]
    assert isinstance(query, SqlQuery) and query.runs == ("run-a",)


def test_everything_unresolvable_is_shown_not_blanked() -> None:
    rendered = _render(
        "```data broken\nmeasures: [missing_measure]\n```\n"
        "```data bad\nmesures: [x]\n```\n"
        "{{broken.x}} {{nothing.x}} {{run.no_such}} {{__import__('os').system('true')}} {{reward}}\n"
        "```value\ndata: broken\ncolumn: x\n```\n"
        "```chart\ndata: absent\nx: a\ny: b\n```\n"
        "```table\ncolumns: [a]\n```\n"
    )
    text = rendered.text
    assert "⟦unresolved: data broken — unknown measure or metric 'missing_measure'⟧" in text
    assert "⟦unresolved: data bad — unknown keys mesures⟧" in text
    assert "⟦unresolved: {{nothing.x}} — no data block named 'nothing'⟧" in text
    assert "⟦unresolved: {{run.no_such}} — unknown dimension 'run.no_such'⟧" in text
    assert "⟦unresolved: {{__import__('os').system('true')}}" in text
    assert "⟦unresolved: value — data broken failed" in text
    assert "⟦unresolved: chart — no data block named 'absent'⟧" in text
    assert "⟦unresolved: table — name its data block with 'data: <name>'⟧" in text
    assert len(rendered.unresolved) == 10


def test_formatting_filters() -> None:
    assert format_value(0.123456) == "0.1235"
    assert format_value(5e-05) == "5e-05"
    assert format_value(35183.2) == "35,183"
    assert format_value(0.8966, [("percent", None)]) == "89.7%"
    assert format_value(3725.0, [("duration", None)]) == "1h 02m"
    assert format_value(None, [("default", "n/a")]) == "n/a"
    assert format_value(True) == "yes"
    with pytest.raises(ValueError, match="unknown filter"):
        format_value(1.0, [("upper", None)])


def test_fences_close_only_on_a_matching_fence() -> None:
    parts = split_blocks("text\n````md\n```data x\n```\n````\nafter\n")
    assert [type(part).__name__ for part in parts] == ["_Text", "_Fenced", "_Text"]
