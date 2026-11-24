"""The checker's own reading of N1 "why" clauses (engine-final specs, N1 (e)): the words are parsed back into
canonical predicate nodes and the contract's nodes are canonicalised by separate code, so a clause passes only
when the words say exactly what the contract says. Written independently of `abce.why` (the renderer).

Canonical nodes (hashable tuples):
  ("eq", F, V)  ("not", N)  ("in", F, frozenset[V])  ("includes", F, V)  ("mentions", F, V)  ("present", F)
  ("cmp", op, F, Decimal)  ("all", frozenset[N])  ("any", frozenset[N])  ("some"|"every", F, LEAF)
F is ("request", name) | ("effect", name) | ("cell", name) | ("item",); V is a JSON text of the literal.
"""

from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation
from typing import Any


class NotCanonical(Exception):
    pass


# ---------------------------------------------------------------------------------------------
# contract node -> canonical
# ---------------------------------------------------------------------------------------------
def _num(text: Any) -> Decimal:
    if isinstance(text, bool):
        raise NotCanonical("bool")
    if isinstance(text, (int, float)):
        return Decimal(str(text)).normalize()
    if isinstance(text, str):
        t = text.strip().replace(",", "").replace("$", "")
        if re.fullmatch(r"-?\d+(\.\d+)?", t):
            try:
                return Decimal(t).normalize()
            except InvalidOperation:
                pass
    raise NotCanonical("number")


def _val(v: Any) -> str:
    if isinstance(v, (dict, list)):
        raise NotCanonical("compound")
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return json.dumps(v)


def _field(node: Any) -> tuple:
    if not (isinstance(node, dict) and node.get("kind") == "field" and isinstance(node.get("path"), list)):
        raise NotCanonical("field")
    path = node["path"]
    if any(isinstance(p, int) for p in path):
        raise NotCanonical("index")
    if path == ["item"]:
        return ("item",)
    if path[0] == "request" and len(path) == 2:
        return ("request", str(path[1]))
    if path[0] == "effect":
        rest = list(path[1:])
        if rest[:1] == ["record"]:
            rest = rest[1:]
        if len(rest) == 2 and rest[0] == "after_cells":
            return ("cell", str(rest[1]))
        if len(rest) == 1 and rest[0] not in ("before", "before_cells", "after_cells"):
            return ("effect", str(rest[0]))
    raise NotCanonical("path")


def _lit(node: Any) -> Any:
    if isinstance(node, dict) and node.get("kind") == "literal":
        return node.get("value")
    raise NotCanonical("literal")


def canon(node: Any) -> tuple:
    if not isinstance(node, dict):
        raise NotCanonical("node")
    op = node.get("op")
    if op == "not":
        args = node.get("args") or [node.get("arg")]
        if len(args) != 1:
            raise NotCanonical("not")
        inner = canon(args[0])
        return inner[1] if inner[0] == "not" else ("not", inner)
    if op in ("all", "any"):
        members = [canon(a) for a in node.get("args") or []]
        if not members:
            raise NotCanonical("empty")
        if len(members) == 1:
            return members[0]
        return (op, frozenset(members))
    if op == "ne":
        return ("not", canon({**node, "op": "eq"}))
    if op in ("eq", "in"):
        left, right = node.get("left"), node.get("right")
        if op == "in" and isinstance(right, dict) and right.get("kind") == "field" and not (isinstance(left, dict) and left.get("kind") == "field"):
            return ("includes", _field(right), _val(_lit(left)))
        f = _field(left)
        v = _lit(right)
        if op == "in":
            if not isinstance(v, list) or not v:
                raise NotCanonical("in")
            if len(v) == 1:
                return ("eq", f, _val(v[0]))
            return ("in", f, frozenset(_val(x) for x in v))
        return ("eq", f, _val(v))
    if op == "mentions":
        if node.get("mode") not in (None, "words", "verbatim") or node.get("format") not in (None, "text"):
            raise NotCanonical("mode")
        v = _lit(node.get("value"))
        if not isinstance(v, str):
            raise NotCanonical("mention value")
        return ("mentions", _field(node.get("text")), _val(v))
    if op == "present":
        return ("present", _field(node.get("value")))
    if op in ("gt", "gte", "lt", "lte"):
        left, right = node.get("left") or {}, node.get("right") or {}
        expr = left.get("expression") or {} if isinstance(left, dict) else {}
        if left.get("kind") != "derived" or expr.get("kind") != "input" or "literal" in expr or not isinstance(expr.get("path"), list):
            raise NotCanonical("cmp left")
        if expr.get("format") not in ("decimal_string", "usd_string", "number"):
            raise NotCanonical("cmp format")
        f = _field({"kind": "field", "path": expr["path"]})
        if right.get("kind") == "literal":
            n = _num(right.get("value"))
        elif right.get("kind") == "derived" and (right.get("expression") or {}).get("kind") == "input" \
                and "literal" in right["expression"] and "path" not in right["expression"] \
                and right["expression"].get("format") in ("decimal_string", "usd_string", "number"):
            n = _num(right["expression"]["literal"])
        else:
            raise NotCanonical("cmp right")
        return ("cmp", op, f, n)
    if op in ("any_item", "all_items"):
        items = _field(node.get("items"))
        where = canon(node.get("where"))
        leaf = where[1] if where[0] == "not" else where
        if leaf[0] in ("all", "any", "some", "every") or len(leaf) < 2 or leaf[1] != ("item",):
            raise NotCanonical("item predicate")
        return ("some" if op == "any_item" else "every", items, where)
    raise NotCanonical(f"op {op}")


# ---------------------------------------------------------------------------------------------
# words -> canonical
# ---------------------------------------------------------------------------------------------
class _P:
    def __init__(self, text: str):
        self.s, self.i = text, 0

    def done(self) -> bool:
        return self.i >= len(self.s)

    def take(self, word: str) -> bool:
        if self.s.startswith(word, self.i):
            self.i += len(word)
            return True
        return False

    def need(self, word: str) -> None:
        if not self.take(word):
            raise NotCanonical(f"expected {word!r} at {self.s[self.i:self.i + 30]!r}")

    def quoted(self) -> str:
        if not self.s.startswith('"', self.i):
            raise NotCanonical("quote")
        dec = json.JSONDecoder()
        value, end = dec.raw_decode(self.s, self.i)
        self.i = end
        return value

    def value(self) -> Any:
        """A JSON literal (string, number, true/false/null) or a bare amount ("$75")."""
        if self.s.startswith('"', self.i):
            return self.quoted()
        m = re.compile(r"true|false|null|-?\$?[\d,]+(?:\.\d+)?").match(self.s, self.i)
        if not m:
            raise NotCanonical("value")
        self.i = m.end()
        t = m.group(0)
        if t in ("true", "false", "null"):
            return json.loads(t)
        if "$" in t or "," in t:
            return ("amount", t)
        return json.loads(t)


def _field_words(p: _P) -> tuple:
    if p.take("the row's ") or p.take("the record's "):
        return ("request", p.quoted())
    if p.take("its cell "):
        return ("cell", p.quoted())
    if p.take("its "):
        return ("effect", p.quoted())
    raise NotCanonical("field words")


def _predicate(p: _P, f: tuple) -> tuple:
    if p.take(" is not set"):
        return ("not", ("present", f))
    if p.take(" is set"):
        return ("present", f)
    for words, op in ((" is more than ", "gt"), (" is at least ", "gte"), (" is less than ", "lt"), (" is at most ", "lte")):
        if p.take(words):
            v = p.value()
            return ("cmp", op, f, _num(v[1] if isinstance(v, tuple) else v))
    for words, neg in ((" is one of ", False), (" is none of ", True)):
        if p.take(words):
            vals = [p.value()]
            while p.take(", "):
                vals.append(p.value())
            node = ("in", f, frozenset(_val(v) for v in vals))
            return ("not", node) if neg else node
    if p.take(" does not include "):
        return ("not", ("includes", f, _val(p.value())))
    if p.take(" includes "):
        return ("includes", f, _val(p.value()))
    if p.take(" does not mention "):
        return ("not", ("mentions", f, _val(p.value())))
    if p.take(" mentions "):
        return ("mentions", f, _val(p.value()))
    if p.take(" is not "):
        return ("not", ("eq", f, _val(p.value())))
    if p.take(" is "):
        return ("eq", f, _val(p.value()))
    raise NotCanonical("predicate")


def _leaf(p: _P) -> tuple:
    for words, kind in (("some entry of ", "some"), ("every entry of ", "every")):
        if p.take(words):
            f = _field_words(p)
            inner = _predicate(p, ("item",))
            return (kind, f, inner)
    f = _field_words(p)
    return _predicate(p, f)


def _alternative(p: _P) -> tuple:
    if p.take("("):
        node = _conjunction(p)
        p.need(")")
        return node
    return _leaf(p)


def _expr(p: _P) -> tuple:
    if p.take("not ("):
        node = _conjunction(p)
        p.need(")")
        return ("not", node)
    if p.take("either "):
        alts = [_alternative(p)]
        while p.take(" or "):
            alts.append(_alternative(p))
        if len(alts) < 2:
            raise NotCanonical("either")
        return ("any", frozenset(alts))
    return _leaf(p)


def _conjunction(p: _P) -> tuple:
    parts = [_expr(p)]
    while p.take(" and "):
        parts.append(_expr(p))
    return parts[0] if len(parts) == 1 else ("all", frozenset(parts))


def parse_clauses(text: str) -> list[tuple]:
    """The top-level conjuncts of a clause text ("A and B and either C or D")."""
    p = _P(text)
    parts = [_expr(p)]
    while p.take(" and "):
        parts.append(_expr(p))
    if not p.done():
        raise NotCanonical(f"trailing {p.s[p.i:p.i + 30]!r}")
    return parts


def balanced_segment(line: str, opener: str) -> tuple[int, int, str] | None:
    """(start, end, inner text) of the first "opener … )" segment, the parentheses balanced outside quotes."""
    start = line.find(opener)
    if start < 0:
        return None
    i, depth, in_q = start + len(opener), 1, False
    while i < len(line):
        ch = line[i]
        if in_q:
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                in_q = False
        elif ch == '"':
            in_q = True
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return start, i + 1, line[start + len(opener):i]
        i += 1
    return None
