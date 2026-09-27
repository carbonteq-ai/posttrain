"""Render run notes: Markdown with data blocks, views, inline references and run links.

A note is plain Markdown plus four small elements, so it still reads sensibly
wherever they are not drawn:

- a data block, a fenced block whose info string is ``data <name>``, holding a
  semantic query (``measures``, ``by``, ``where``, ``runs``, ``order_by``,
  ``limit``) or read-only SQL (``sql``, ``runs``, ``load``) as ``key: value``
  lines; ``runs`` defaults to ``self``, the note's run;
- a view, a fenced ``chart``, ``value`` or ``table`` block naming a data block
  and how to show it;
- an inline reference, ``{{name.column}}`` (first row, or the row chosen by
  ``where column = value``) or ``{{run.<dimension>}}`` (the note's own run),
  optionally followed by filters: ``round 2``, ``percent``, ``duration``,
  ``sci``, ``default "text"``;
- a run link, ``[[run:<id>]]`` or ``[[run:<id>|label]]``.

Nothing in a note is evaluated as code: block bodies are read by a small
hand-written parser and references name data, never expressions. Anything
that cannot be resolved renders as a visible marker, never as a blank.
"""

from __future__ import annotations

import math
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import Field, ValidationError

from .models import JsonPayload, ObservatoryModel
from .semantic_layer.query import SemanticQuery, SemanticResult, SqlQuery

type QueryRunner = Callable[[SemanticQuery | SqlQuery], Awaitable[SemanticResult]]
type ViewKind = Literal["chart", "value", "table"]

VIEW_KINDS: tuple[ViewKind, ...] = ("chart", "value", "table")
VIEW_FENCE = "note-view"
_FENCE = re.compile(r"^(?P<indent> {0,3})(?P<fence>`{3,}|~{3,})(?P<info>[^`]*)$")
_REFERENCE = re.compile(r"\{\{(.*?)\}\}")
_LINK = re.compile(r"\[\[run:([A-Za-z0-9][A-Za-z0-9._:/-]*)(?:\|([^\]]*))?\]\]")
_INLINE_CODE = re.compile(r"(`+)(.+?)\1")
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")
_DATA_KEYS = frozenset({"measures", "by", "where", "runs", "order_by", "limit", "sql", "load"})
_VIEW_KEYS: Mapping[str, frozenset[str]] = {
    "chart": frozenset({"data", "x", "y", "series", "type", "title"}),
    "value": frozenset({"data", "column", "where", "compare", "format", "label"}),
    "table": frozenset({"data", "columns", "title"}),
}
MAX_DATA_BLOCKS = 12
MAX_BODY_CHARS = 200_000


def unresolved(element: str, reason: str) -> str:
    return f"⟦unresolved: {element} — {reason}⟧"


# ------------------------------------------------------------------ models


class RenderedView(ObservatoryModel):
    """One view with the data it displays, for the page to draw."""

    index: int = Field(ge=0)
    kind: ViewKind
    data: str | None = None
    options: dict[str, JsonPayload] = Field(default_factory=dict)
    query: str | None = None
    result: SemanticResult | None = None
    # For `value`: the chosen row's value and, with `compare`, the second value.
    value: JsonPayload = None
    compare_value: JsonPayload = None
    formatted: str | None = None
    compare_formatted: str | None = None
    difference: str | None = None
    error: str | None = None


class RenderedNote(ObservatoryModel):
    """A note ready to show.

    ``markdown`` keeps views as fenced ``note-view <index>`` blocks for the page
    to draw from ``views``; ``text`` is the same note with views written out as
    Markdown text, for the command line and agents.
    """

    markdown: str
    text: str
    views: tuple[RenderedView, ...] = ()
    data: dict[str, SemanticResult] = Field(default_factory=dict)
    unresolved: tuple[str, ...] = ()
    template: str | None = None


# ------------------------------------------------------------------ parsing


@dataclass(frozen=True, slots=True)
class _Text:
    text: str


@dataclass(frozen=True, slots=True)
class _Fenced:
    info: str
    body: str
    raw: str


def split_blocks(body_md: str) -> list[_Text | _Fenced]:
    """Split Markdown into text runs and fenced code blocks."""

    parts: list[_Text | _Fenced] = []
    lines = body_md.splitlines(keepends=True)
    text: list[str] = []
    index = 0
    while index < len(lines):
        match = _FENCE.match(lines[index].rstrip("\n"))
        if match is None:
            text.append(lines[index])
            index += 1
            continue
        fence = match.group("fence")
        closing = next(
            (
                end
                for end in range(index + 1, len(lines))
                if (line := lines[end].strip()).startswith(fence[0] * len(fence)) and set(line) == {fence[0]}
            ),
            None,
        )
        if closing is None:
            text.append(lines[index])
            index += 1
            continue
        if text:
            parts.append(_Text("".join(text)))
            text = []
        parts.append(
            _Fenced(
                info=match.group("info").strip(),
                body="".join(lines[index + 1 : closing]),
                raw="".join(lines[index : closing + 1]),
            )
        )
        index = closing + 1
    if text:
        parts.append(_Text("".join(text)))
    return parts


def data_blocks(body_md: str) -> list[tuple[str, str]]:
    """The ``(name, body)`` of each data block in a note, in order."""

    return [
        (part.info.split()[1], part.body)
        for part in split_blocks(body_md)
        if isinstance(part, _Fenced) and len(part.info.split()) == 2 and part.info.split()[0] == "data"
    ]


class BlockSyntaxError(ValueError):
    pass


def parse_block(body: str) -> dict[str, Any]:
    """Read ``key: value`` lines. ``key: |`` starts an indented multi-line value."""

    values: dict[str, Any] = {}
    lines = body.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        index += 1
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, separator, rest = line.partition(":")
        key = key.strip()
        if not separator or not re.fullmatch(r"[a-z_]+", key):
            raise BlockSyntaxError(f"line {index} must be 'key: value'")
        if key in values:
            raise BlockSyntaxError(f"{key!r} is given twice")
        rest = rest.strip()
        if rest == "|":
            block: list[str] = []
            while index < len(lines) and (not lines[index].strip() or lines[index][:1] in {" ", "\t"}):
                block.append(lines[index].strip())
                index += 1
            values[key] = "\n".join(block).strip()
        elif rest[:1] in {"[", "{"}:
            values[key] = parse_value(rest)
        else:
            # A plain value is the whole rest of the line, commas included.
            values[key] = _literal(rest[1:-1] if len(rest) >= 2 and rest[0] == rest[-1] and rest[0] in "'\"" else rest)
    return values


def parse_value(text: str) -> Any:
    value, rest = _value(text.strip(), 0)
    if text.strip()[rest:].strip():
        raise BlockSyntaxError(f"unexpected text after a value: {text.strip()[rest:].strip()!r}")
    return value


def _value(text: str, position: int) -> tuple[Any, int]:
    position = _skip(text, position)
    if position >= len(text):
        return None, position
    char = text[position]
    if char == "[":
        items: list[Any] = []
        position = _skip(text, position + 1)
        if position < len(text) and text[position] == "]":
            return items, position + 1
        while True:
            item, position = _value(text, position)
            items.append(item)
            position = _skip(text, position)
            if position < len(text) and text[position] == ",":
                position += 1
                continue
            if position < len(text) and text[position] == "]":
                return items, position + 1
            raise BlockSyntaxError("a list must look like [a, b]")
    if char == "{":
        mapping: dict[str, Any] = {}
        position = _skip(text, position + 1)
        if position < len(text) and text[position] == "}":
            return mapping, position + 1
        while True:
            key, position = _scalar(text, position, stop=":")
            position = _skip(text, position)
            if position >= len(text) or text[position] != ":":
                raise BlockSyntaxError("a mapping must look like {key: value}")
            item, position = _value(text, position + 1)
            mapping[str(key)] = item
            position = _skip(text, position)
            if position < len(text) and text[position] == ",":
                position += 1
                continue
            if position < len(text) and text[position] == "}":
                return mapping, position + 1
            raise BlockSyntaxError("a mapping must look like {key: value}")
    return _scalar(text, position, stop=",]}")


def _scalar(text: str, position: int, *, stop: str) -> tuple[Any, int]:
    position = _skip(text, position)
    if position < len(text) and text[position] in "\"'":
        quote = text[position]
        end = text.find(quote, position + 1)
        if end < 0:
            raise BlockSyntaxError("unterminated quoted string")
        return text[position + 1 : end], end + 1
    end = position
    while end < len(text) and text[end] not in stop:
        end += 1
    raw = text[position:end].strip()
    return _literal(raw), end


def _literal(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", "~", ""}:
        return None
    if re.fullmatch(r"[+-]?\d+", raw):
        return int(raw)
    if re.fullmatch(r"[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?", raw):
        return float(raw)
    return raw


def _skip(text: str, position: int) -> int:
    while position < len(text) and text[position] in " \t":
        position += 1
    return position


def build_query(block: Mapping[str, Any], run_id: str) -> SemanticQuery | SqlQuery:
    unknown = sorted(set(block) - _DATA_KEYS)
    if unknown:
        raise BlockSyntaxError(f"unknown keys {', '.join(unknown)}")
    runs = _runs(block.get("runs", "self"), run_id)
    try:
        if "sql" in block:
            if not isinstance(block["sql"], str):
                raise BlockSyntaxError("sql must be text")
            load = block.get("load")
            return SqlQuery(
                sql=block["sql"],
                runs=runs,
                load={str(key): tuple(_list(value)) for key, value in load.items()} if isinstance(load, dict) else None,
            )
        return SemanticQuery(
            measures=tuple(_list(block.get("measures"))),
            by=tuple(_list(block.get("by"))),
            where=dict(block.get("where") or {}),
            runs=runs,
            order_by=tuple(_list(block.get("order_by"))),
            limit=int(block.get("limit") or 1000),
        )
    except ValidationError as error:
        raise BlockSyntaxError("; ".join(item["msg"] for item in error.errors())) from None


def _runs(value: Any, run_id: str) -> tuple[str, ...] | dict[str, Any]:
    if isinstance(value, dict):
        return {key: (run_id if item == "self" else item) for key, item in value.items()}
    return tuple(run_id if item == "self" else str(item) for item in _list(value))


def _list(value: Any) -> list[Any]:
    if value is None:
        return []
    return list(value) if isinstance(value, list) else [value]


# ------------------------------------------------------------------ formatting


def format_value(value: Any, filters: Sequence[tuple[str, str | None]] = ()) -> str:
    """Format a value; filters apply in order (``round 2``, ``percent``, ``duration``, ``sci``, ``default``)."""

    for name, argument in filters:
        if name == "default":
            if value is None:
                return argument or ""
            continue
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"filter {name!r} needs a number, not {value!r}")
        if name == "round":
            digits = int(argument or 0)
            value = f"{value:,.{digits}f}"
        elif name == "percent":
            digits = int(argument) if argument else 1
            value = f"{value * 100:.{digits}f}%"
        elif name == "sci":
            digits = int(argument) if argument else 2
            value = f"{value:.{digits}e}"
        elif name == "duration":
            value = _duration(float(value))
        else:
            raise ValueError(f"unknown filter {name!r}")
    if value is None:
        raise ValueError("no value")
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        if value != 0 and (abs(value) < 1e-3 or abs(value) >= 1e7):
            return f"{value:.3g}"
        return f"{value:,.4g}" if abs(value) < 1000 else f"{value:,.0f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, second = divmod(round(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {second:02d}s"
    hours, minute = divmod(minutes, 60)
    return f"{hours}h {minute:02d}m"


def parse_filters(text: str) -> list[tuple[str, str | None]]:
    filters = []
    for part in text.split("|"):
        part = part.strip()
        if not part:
            raise ValueError("empty filter")
        name, _, argument = part.partition(" ")
        argument = argument.strip()
        if argument.startswith(("'", '"')) and argument.endswith(argument[0]) and len(argument) >= 2:
            argument = argument[1:-1]
        filters.append((name, argument or None))
    return filters


# ------------------------------------------------------------------ rows


def _column_index(result: SemanticResult, column: str) -> int:
    names = [item.name for item in result.columns]
    if column in names:
        return names.index(column)
    raise KeyError(f"no column {column!r} (columns: {', '.join(names)})")


def _matches(actual: Any, expected: Any) -> bool:
    if actual == expected:
        return True
    if isinstance(actual, bool) or isinstance(expected, bool):
        return str(actual).lower() == str(expected).lower()
    if isinstance(actual, int | float) and isinstance(expected, int | float):
        return math.isclose(actual, expected)
    return str(actual) == str(expected)


def parse_row_condition(text: str) -> tuple[str, Any]:
    column, separator, value = text.partition("=")
    if not separator or not column.strip():
        raise ValueError(f"a row condition must look like 'column = value', not {text!r}")
    return column.strip(), parse_value(value.strip())


def pick_row(result: SemanticResult, condition: str | None, *, run_id: str | None = None) -> tuple[Any, ...]:
    """The row a `where column = value` names; without one, the note's own run's row, else the first."""

    if not result.rows:
        raise LookupError("no rows")
    if condition is None:
        names = [column.name for column in result.columns]
        if run_id is not None and "run.id" in names:
            index = names.index("run.id")
            return next((row for row in result.rows if row[index] == run_id), result.rows[0])
        return result.rows[0]
    column, expected = parse_row_condition(condition)
    index = _column_index(result, column)
    for row in result.rows:
        if _matches(row[index], expected):
            return row
    raise LookupError(f"no row where {column} = {expected}")


# ------------------------------------------------------------------ render


async def render_note(
    body_md: str,
    *,
    run_id: str,
    query: QueryRunner,
    link_for: Callable[[str], str],
    template: str | None = None,
) -> RenderedNote:
    """Resolve a note's data blocks once each, then fill its views, references and links."""

    if len(body_md) > MAX_BODY_CHARS:
        body_md = body_md[:MAX_BODY_CHARS]
    parts = split_blocks(body_md)
    missing: list[str] = []
    data: dict[str, SemanticResult] = {}
    errors: dict[str, str] = {}
    queries: dict[str, str] = {}
    for part in parts:
        if not isinstance(part, _Fenced) or not part.info.startswith("data"):
            continue
        words = part.info.split()
        if words[0] != "data":
            continue
        name = words[1] if len(words) == 2 else ""
        if not _NAME.match(name):
            errors[name or "?"] = "a data block needs one name, like 'data reward'"
            continue
        if name in queries:
            errors[name] = "the name is used by another data block"
            continue
        queries[name] = part.body.strip()
        if len(queries) > MAX_DATA_BLOCKS:
            errors[name] = f"a note can hold at most {MAX_DATA_BLOCKS} data blocks"
            continue
        try:
            data[name] = await query(build_query(parse_block(part.body), run_id))
        except Exception as error:  # every failure is shown on the note, not raised
            errors[name] = str(error) or type(error).__name__

    run_values = await _run_references(body_md, run_id=run_id, query=query)

    markdown: list[str] = []
    text: list[str] = []
    views: list[RenderedView] = []
    for part in parts:
        if isinstance(part, _Text):
            resolved = _inline(run_id, part.text, data, errors, run_values, link_for, missing)
            markdown.append(resolved)
            text.append(resolved)
            continue
        words = part.info.split()
        kind = words[0] if words else ""
        if kind == "data":
            name = words[1] if len(words) == 2 else "?"
            if name in errors:
                marker = unresolved(f"data {name}", errors[name])
                missing.append(marker)
                markdown.append(marker + "\n")
                text.append(marker + "\n")
            continue
        if kind not in VIEW_KINDS:
            markdown.append(part.raw)
            text.append(part.raw)
            continue
        view = _view(len(views), kind, part.body, data, errors, queries, run_id)  # type: ignore[arg-type]
        views.append(view)
        if view.error is not None:
            marker = unresolved(kind, view.error)
            missing.append(marker)
        fence = f"```{VIEW_FENCE} {view.index}\n```\n"
        markdown.append(fence)
        text.append(view_text(view) + "\n")
    return RenderedNote(
        markdown="".join(markdown).strip("\n") + "\n",
        text="".join(text).strip("\n") + "\n",
        views=tuple(views),
        data=data,
        unresolved=tuple(missing),
        template=template,
    )


def _view(
    index: int,
    kind: ViewKind,
    body: str,
    data: Mapping[str, SemanticResult],
    errors: Mapping[str, str],
    queries: Mapping[str, str],
    run_id: str,
) -> RenderedView:
    try:
        options = parse_block(body)
    except BlockSyntaxError as error:
        return RenderedView(index=index, kind=kind, error=str(error))
    unknown = sorted(set(options) - _VIEW_KEYS[kind])
    name = options.get("data")
    base: dict[str, Any] = {
        "index": index,
        "kind": kind,
        "data": str(name) if name is not None else None,
        "options": {key: value for key, value in options.items() if key != "data"},
        "query": queries.get(str(name)) if name is not None else None,
    }
    if unknown:
        return RenderedView(**base, error=f"unknown keys {', '.join(unknown)}")
    if name is None:
        return RenderedView(**base, error="name its data block with 'data: <name>'")
    if name in errors:
        return RenderedView(**base, error=f"data {name} failed: {errors[name]}")
    result = data.get(str(name))
    if result is None:
        return RenderedView(**base, error=f"no data block named {name!r}")
    base["result"] = result
    try:
        if kind == "chart":
            for column in (options.get("x"), *_list(options.get("y")), options.get("series")):
                if column is not None:
                    _column_index(result, str(column))
            if options.get("x") is None or not _list(options.get("y")):
                raise ValueError("a chart needs x and y")
            if options.get("type", "line") not in {"line", "bar"}:
                raise ValueError("type must be line or bar")
            return RenderedView(**base)
        if kind == "table":
            for column in _list(options.get("columns")):
                _column_index(result, str(column))
            return RenderedView(**base)
        column = options.get("column")
        if column is None:
            raise ValueError("a value needs 'column'")
        filters = parse_filters(str(options["format"])) if options.get("format") else []
        where = options.get("where")
        row = pick_row(result, str(where) if where is not None else None, run_id=run_id)
        value = row[_column_index(result, str(column))]
        view: dict[str, Any] = {"value": value, "formatted": format_value(value, filters)}
        if options.get("compare") is not None:
            other = pick_row(result, str(options["compare"]))[_column_index(result, str(column))]
            view |= {"compare_value": other, "compare_formatted": format_value(other, filters)}
            if isinstance(value, int | float) and isinstance(other, int | float) and not isinstance(value, bool):
                view["difference"] = _difference(value, other, filters)
        return RenderedView(**base, **view)
    except (KeyError, LookupError, ValueError) as error:
        return RenderedView(**base, error=_reason(error))


def _reason(error: Exception) -> str:
    # KeyError quotes its message; the other errors carry it as written.
    return str(error.args[0]) if isinstance(error, KeyError) and error.args else str(error)


def _difference(value: float, other: float, filters: Sequence[tuple[str, str | None]] = ()) -> str:
    """The change from `value` to `other`, in the view's format, with the relative change."""

    change = other - value
    relative = f" ({change / value:+.1%})" if value else ""
    unit_filters = [item for item in filters if item[0] in {"round", "sci", "duration"}]
    if unit_filters:
        return f"{'+' if change >= 0 else '-'}{format_value(abs(change), unit_filters)}{relative}"
    return f"{change:+.4g}{relative}"


async def _run_references(body_md: str, *, run_id: str, query: QueryRunner) -> dict[str, Any] | str:
    """Values of the ``run.<dimension>`` references in a note, read in one query."""

    names: list[str] = []
    for part in split_blocks(body_md):
        if isinstance(part, _Text):
            for match in _REFERENCE.finditer(_INLINE_CODE.sub("", part.text)):
                path = match.group(1).split("|")[0].strip()
                if path.startswith("run.") and " " not in path and path not in names:
                    names.append(path)
    if not names:
        return {}
    try:
        result = await query(SemanticQuery(measures=("runs",), by=tuple(names), runs=(run_id,)))
    except Exception as error:
        return str(error) or type(error).__name__
    if not result.rows:
        return "the run was not found"
    return {name: result.rows[0][index] for index, name in enumerate(names)}


def _inline(
    note_run: str,
    text: str,
    data: Mapping[str, SemanticResult],
    errors: Mapping[str, str],
    run_values: Mapping[str, Any] | str,
    link_for: Callable[[str], str],
    missing: list[str],
) -> str:
    def reference(match: re.Match[str]) -> str:
        expression = match.group(1).strip()
        path, _, filter_text = expression.partition("|")
        path, _, where = path.strip().partition(" where ")
        try:
            filters = parse_filters(filter_text) if filter_text.strip() else []
            if path.startswith("run.") and not where:
                if isinstance(run_values, str):
                    raise ValueError(run_values)
                if path not in run_values:
                    raise ValueError("unknown run dimension")
                return format_value(run_values[path], filters)
            name, dot, column = path.partition(".")
            if not dot or not _NAME.match(name):
                raise ValueError("write {{data.column}} or {{run.<dimension>}}")
            if name in errors:
                raise ValueError(f"data {name} failed: {errors[name]}")
            result = data.get(name)
            if result is None:
                raise ValueError(f"no data block named {name!r}")
            row = pick_row(result, where.strip() or None, run_id=note_run)
            return format_value(row[_column_index(result, column)], filters)
        except (KeyError, LookupError, ValueError) as error:
            marker = unresolved("{{" + expression + "}}", _reason(error))
            missing.append(marker)
            return marker

    def link(match: re.Match[str]) -> str:
        run_id, label = match.group(1), match.group(2)
        return f"[{(label or run_id).replace(']', '')}]({link_for(run_id)})"

    pieces: list[str] = []
    position = 0
    for code in _INLINE_CODE.finditer(text):
        pieces.append(_LINK.sub(link, _REFERENCE.sub(reference, text[position : code.start()])))
        pieces.append(code.group(0))
        position = code.end()
    pieces.append(_LINK.sub(link, _REFERENCE.sub(reference, text[position:])))
    return "".join(pieces)


# ------------------------------------------------------------------ text views


_SPARK = "▁▂▃▄▅▆▇█"


def view_text(view: RenderedView) -> str:
    """A view written as Markdown text, for the command line and agents."""

    if view.error is not None:
        return unresolved(view.kind, view.error)
    result = view.result
    assert result is not None
    options = view.options
    if view.kind == "value":
        label = str(options.get("label") or options.get("column"))
        if view.compare_formatted is not None:
            difference = f", change {view.difference}" if view.difference else ""
            return f"**{label}:** {view.formatted} → {view.compare_formatted}{difference}"
        return f"**{label}:** {view.formatted}"
    if view.kind == "table":
        columns = [str(column) for column in _list(options.get("columns"))] or [c.name for c in result.columns]
        indexes = [_column_index(result, column) for column in columns]
        lines = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        ]
        for row in result.rows[:50]:
            lines.append("| " + " | ".join(_cell(row[index]) for index in indexes) + " |")
        if len(result.rows) > 50:
            lines.append(f"\n({len(result.rows) - 50} more rows)")
        title = f"**{options['title']}**\n\n" if options.get("title") else ""
        return title + "\n".join(lines)
    x = _column_index(result, str(options["x"]))
    series = _column_index(result, str(options["series"])) if options.get("series") else None
    title = f"**{options.get('title') or ', '.join(str(y) for y in _list(options['y']))}**"
    lines = [title]
    for y_name in _list(options["y"]):
        y = _column_index(result, str(y_name))
        groups: dict[Any, list[tuple[Any, float]]] = {}
        for row in result.rows:
            if isinstance(row[y], int | float) and not isinstance(row[y], bool):
                groups.setdefault(row[series] if series is not None else None, []).append((row[x], float(row[y])))
        for key, points in groups.items():
            points.sort(key=lambda point: (point[0] is None, point[0]))
            values = [value for _, value in points]
            name = f"{y_name}" + (f" ({key})" if key is not None else "")
            lines.append(
                f"- {name}: `{_sparkline(values)}` {format_value(values[0])} → {format_value(values[-1])}"
                f" over {len(values)} points"
            )
    return "\n".join(lines)


def _cell(value: Any) -> str:
    return "" if value is None else format_value(value).replace("|", "\\|")


def _sparkline(values: Sequence[float]) -> str:
    if len(values) > 40:
        step = len(values) / 40
        values = [values[int(index * step)] for index in range(40)]
    low, high = min(values), max(values)
    if high == low:
        return _SPARK[3] * len(values)
    return "".join(_SPARK[round((value - low) / (high - low) * (len(_SPARK) - 1))] for value in values)


__all__ = [
    "BlockSyntaxError",
    "RenderedNote",
    "RenderedView",
    "VIEW_FENCE",
    "data_blocks",
    "format_value",
    "parse_block",
    "render_note",
    "split_blocks",
    "unresolved",
    "view_text",
]
