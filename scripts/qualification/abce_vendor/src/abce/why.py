"""N1 "why" clauses (engine-final specs, section C, N1a/N1b; review changes R8, R9): a manifest rule's own
condition in plain words, rendered only from the compiled contract tree the Record carries
(`semantics.effect_match`, `semantics.condition`), through one closed grammar. Nothing outside the grammar is
rendered and nothing is paraphrased; the truth checker parses the words back (abce.checks.why_parse) and
compares them with the contract and evaluates the condition with the environment's own evaluator.

N1a (effect conditions): top-level conjuncts of `effect_match` (outside every `any`) that the compiler could not
express as address or content, and, for a forbid item, `prohibited_when` conjuncts that read the effect.
N1b (applicability): the non-selector conjuncts of `required_when` / `prohibited_when` over the candidate's
population member (`request.*`), plus request-only top-level conjuncts of `effect_match`; only with a candidate
label on the line and only when the renderer's own evaluator finds the whole condition true on the member.
"""

from __future__ import annotations

import contextvars
import json
import re
from decimal import Decimal, InvalidOperation
from typing import Any

from .record.model import Item

# set by the renderer and the checker when fix hints (version c) are on; the a/b/d contexts never carry N1
ENABLED: contextvars.ContextVar[bool] = contextvars.ContextVar("abce_why_enabled", default=False)
# whether the N1 clauses themselves are rendered when ENABLED is on (fix-hint kind "why_clause"); off, the version-(c)
# statement forms and checks stay and every line that would need a clause is rendered without one or withheld
CLAUSES: contextvars.ContextVar[bool] = contextvars.ContextVar("abce_why_clauses", default=True)
# the Record the statement being rendered belongs to (for the rule-2 test of a "forbidden when" clause)
RECORD: contextvars.ContextVar[Any] = contextvars.ContextVar("abce_why_record", default=None)


def needed_members(record: Any) -> tuple[set[str], set[str]]:
    """(candidate keys, address values) of the requirements that write something (any outcome, not reads)."""
    keys, values = set(), set()
    for i in getattr(record, "items", ()) or ():
        if i.polarity != "require" or i.type == "tool_error" or i.effect.kind == "read":
            continue
        if i.candidate and i.candidate.get("key") is not None:
            keys.add(str(i.candidate["key"]).strip().lower())
        for v in i.address.values():
            for x in (v.raw if isinstance(v.raw, list) else [v.raw]):
                if not isinstance(x, (dict, list)) and x is not None:
                    values.add(str(x).strip().lower())
    return keys, values


def condition_covers_required(item: Item, whole: list, record: Any) -> bool:
    """Rule 2 (fix round): a "forbidden when" condition that a population member satisfies whose row or record a
    requirement of the task needs written; or a member whose truth cannot be decided (unknown counts as covered)."""
    members = item.semantics.get("population_members")
    if not isinstance(members, dict) or record is None:
        return True
    keys, values = needed_members(record)
    for key, fields in members.items():
        if not isinstance(fields, dict):
            continue
        clean = {k: v for k, v in fields.items() if not str(k).startswith("_")}
        verdicts = [evaluate(c, clean) for c in whole]
        if False in verdicts:
            continue
        label = str(fields.get("_label", "")).strip().lower()
        if str(key).strip().lower() in keys or str(key).strip().lower() in values or (label and label in values):
            return True
    return False

ROW_ADAPTER = "google_sheets.rows@1"
RECORD_ADAPTER = "initial.records@1"
NUMERIC_FORMATS = ("decimal_string", "usd_string", "number")
CMP_WORDS = {"gt": "more than", "gte": "at least", "lt": "less than", "lte": "at most"}
MENTION_MODES = (None, "words", "verbatim")


class Unrenderable(Exception):
    pass


# ---------------------------------------------------------------------------------------------
# tree helpers
# ---------------------------------------------------------------------------------------------
def flatten_all(node: Any) -> list:
    """The top-level conjuncts of a predicate (an `all` flattened, anything else whole)."""
    if isinstance(node, dict) and node.get("op") == "all":
        out = []
        for arg in node.get("args") or []:
            out.extend(flatten_all(arg))
        return out
    return [node] if isinstance(node, dict) and node.get("op") else []


def _paths(node: Any) -> list[tuple]:
    out = []
    if isinstance(node, dict):
        if node.get("kind") in ("field", "input") and isinstance(node.get("path"), list):
            out.append(tuple(node["path"]))
        for v in node.values():
            out.extend(_paths(v))
    elif isinstance(node, list):
        for v in node:
            out.extend(_paths(v))
    return out


def reads(node: Any, root: str) -> bool:
    return any(p[:1] == (root,) for p in _paths(node))


def request_only(node: Any) -> bool:
    paths = _paths(node)
    return bool(paths) and all(p[:1] == ("request",) for p in paths)


def _field_path(node: Any) -> tuple | None:
    if isinstance(node, dict) and node.get("kind") == "field" and isinstance(node.get("path"), list):
        return tuple(node["path"])
    return None


def expressible(node: Any) -> bool:
    """What `abce.record.manifest.compile_predicate` turns into address or content (so the line already shows
    it): eq/in/ne with an effect field on one side, mentions/mentions_together on an effect text, a `not` of
    such, and an `any`/`all` made only of them."""
    if not isinstance(node, dict):
        return False
    op = node.get("op")
    if op in ("all", "any"):
        args = node.get("args") or []
        return bool(args) and all(expressible(a) for a in args)
    if op == "not":
        args = node.get("args") or [node.get("arg")]
        return bool(args) and all(expressible(a) for a in args)
    if op == "proven":
        return expressible(node.get("arg"))
    if op in ("eq", "in", "ne"):
        lp, rp = _field_path(node.get("left")), _field_path(node.get("right"))
        return bool((lp and lp[0] == "effect") or (rp and rp[0] == "effect"))
    if op in ("mentions", "mentions_together"):
        text = _field_path(node.get("text"))
        return bool(text and text[0] == "effect")
    return False


# ---------------------------------------------------------------------------------------------
# the grammar (tree -> words)
# ---------------------------------------------------------------------------------------------
def _field_words(path: tuple, adapter: str | None) -> str:
    if any(isinstance(p, int) for p in path):
        raise Unrenderable("index in path")
    if path[:1] == ("request",) and len(path) == 2:
        if adapter == ROW_ADAPTER:
            return f"the row's {json.dumps(str(path[1]))}"
        if adapter == RECORD_ADAPTER:
            return f"the record's {json.dumps(str(path[1]))}"
        raise Unrenderable("request field of another population")
    if path[:1] == ("effect",):
        rest = path[1:]
        if rest[:1] == ("record",):
            rest = rest[1:]
        if len(rest) == 2 and rest[0] == "after_cells":
            return f"its cell {json.dumps(str(rest[1]))}"
        if len(rest) == 1 and rest[0] not in ("before", "before_cells", "after_cells"):
            return f"its {json.dumps(str(rest[0]))}"
    raise Unrenderable(f"field {path}")


def _literal(node: Any) -> Any:
    """A literal operand's value; raises for anything else."""
    if isinstance(node, dict) and node.get("kind") == "literal":
        return node.get("value")
    raise Unrenderable("not a literal")


def _value_words(v: Any) -> str:
    if isinstance(v, bool) or v is None:
        return json.dumps(v)
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        return json.dumps(v)
    raise Unrenderable("compound literal")


def number_of(text: Any) -> Decimal | None:
    if isinstance(text, bool):
        return None
    if isinstance(text, (int, float)):
        return Decimal(str(text))
    if isinstance(text, str):
        t = text.strip().replace(",", "").replace("$", "")
        if re.fullmatch(r"-?\d+(\.\d+)?", t):
            try:
                return Decimal(t)
            except InvalidOperation:
                return None
    return None


def _number_words(node: Any) -> tuple[str, Decimal]:
    """A numeric right-hand side: a literal number, or a derived input literal (shown as written)."""
    if isinstance(node, dict) and node.get("kind") == "literal" and isinstance(node.get("value"), (int, float)) \
            and not isinstance(node.get("value"), bool):
        return str(node["value"]), Decimal(str(node["value"]))
    if isinstance(node, dict) and node.get("kind") == "derived":
        expr = node.get("expression") or {}
        if expr.get("kind") == "input" and "literal" in expr and expr.get("format") in NUMERIC_FORMATS and "path" not in expr:
            n = number_of(expr["literal"])
            if n is not None:
                return str(expr["literal"]), n
    raise Unrenderable("not a number")


def _derived_field(node: Any) -> tuple:
    if isinstance(node, dict) and node.get("kind") == "derived":
        expr = node.get("expression") or {}
        if expr.get("kind") == "input" and isinstance(expr.get("path"), list) and expr.get("format") in NUMERIC_FORMATS \
                and "literal" not in expr:
            return tuple(expr["path"])
    raise Unrenderable("not a numeric field")


def _leaf(node: dict, negated: bool, adapter: str | None, subject: str | None = None) -> str:
    """One comparison as words; `subject` replaces the field words (an item of a list: "some entry of F")."""
    op = node.get("op")
    if op == "ne":
        return _leaf({**node, "op": "eq"}, not negated, adapter, subject)
    if op == "eq" or op == "in":
        left, right = node.get("left"), node.get("right")
        lp, rp = _field_path(left), _field_path(right)
        if op == "in" and lp is None and rp is not None:  # in(L, F): the sequence F includes L
            value = _literal(left)
            field = subject or _field_words(rp, adapter)
            return f"{field} {'does not include' if negated else 'includes'} {_value_words(value)}"
        if lp is None:
            raise Unrenderable("eq without a field on the left")
        field = subject if subject is not None else _field_words(lp, adapter)
        value = _literal(right)
        if op == "in":
            if not isinstance(value, list) or not value:
                raise Unrenderable("in without a list")
            if len(value) == 1:
                value = value[0]
            else:
                words = ", ".join(_value_words(v) for v in value)
                return f"{field} is {'none' if negated else 'one'} of {words}"
        return f"{field} is {'not ' if negated else ''}{_value_words(value)}"
    if op == "mentions":
        if node.get("mode") not in MENTION_MODES or node.get("format") not in (None, "text"):
            raise Unrenderable("mention mode")
        tp = _field_path(node.get("text"))
        if tp is None:
            raise Unrenderable("mention text")
        field = subject if subject is not None else _field_words(tp, adapter)
        value = _literal(node.get("value"))
        if not isinstance(value, str):
            raise Unrenderable("mention value")
        return f"{field} {'does not mention' if negated else 'mentions'} {_value_words(value)}"
    if op == "present":
        vp = _field_path(node.get("value"))
        if vp is None:
            raise Unrenderable("present")
        field = subject if subject is not None else _field_words(vp, adapter)
        return f"{field} is {'not set' if negated else 'set'}"
    if op in CMP_WORDS:
        path = _derived_field(node.get("left"))
        field = subject if subject is not None else _field_words(path, adapter)
        words, _n = _number_words(node.get("right"))
        if negated:
            raise Unrenderable("negated comparison")
        return f"{field} is {CMP_WORDS[op]} {words}"
    raise Unrenderable(f"op {op}")


def render(node: Any, adapter: str | None, negated: bool = False, depth: int = 0) -> str:
    """Words for one conjunct (raises Unrenderable outside the grammar)."""
    if not isinstance(node, dict):
        raise Unrenderable("not a node")
    op = node.get("op")
    if op == "not":
        args = node.get("args") or [node.get("arg")]
        if len(args) != 1:
            raise Unrenderable("not with several args")
        inner = args[0]
        if isinstance(inner, dict) and inner.get("op") in ("all", "any"):
            if negated:
                raise Unrenderable("double negation")
            members = inner.get("args") or []
            if len(members) == 1:
                return render(members[0], adapter, True, depth)
            return f"not ({render(inner, adapter, False, depth + 1)})"
        return render(inner, adapter, not negated, depth)
    if op == "all":
        members = node.get("args") or []
        if not members:
            raise Unrenderable("empty all")
        if len(members) == 1:
            return render(members[0], adapter, negated, depth)
        if negated:
            raise Unrenderable("negated all")
        return " and ".join(render(m, adapter, False, depth + 1) for m in members)
    if op == "any":
        members = node.get("args") or []
        if not members:
            raise Unrenderable("empty any")
        if len(members) == 1:
            return render(members[0], adapter, negated, depth)
        if negated or depth > 1:
            raise Unrenderable("nested any")
        parts = []
        for m in members:
            if isinstance(m, dict) and m.get("op") == "any":
                raise Unrenderable("any in any")
            text = render(m, adapter, False, depth + 1)
            parts.append(f"({text})" if " and " in text or " or " in text else text)
        return "either " + " or ".join(parts)
    if op in ("any_item", "all_items"):
        items = _field_path(node.get("items"))
        where = node.get("where")
        if items is None or not isinstance(where, dict) or negated:
            raise Unrenderable("item quantifier")
        # the `where` leaf must be about the item itself
        refs = _paths(where)
        if not refs or any(r != ("item",) for r in refs):
            raise Unrenderable("item predicate")
        subject = ("some entry of " if op == "any_item" else "every entry of ") + _field_words(items, adapter)
        inner_neg = False
        leaf = where
        if leaf.get("op") == "not":
            args = leaf.get("args") or [leaf.get("arg")]
            if len(args) != 1:
                raise Unrenderable("item not")
            leaf, inner_neg = args[0], True
        if leaf.get("op") in ("all", "any", "not", "any_item", "all_items"):
            raise Unrenderable("item junction")
        return _leaf(leaf, inner_neg, adapter, subject)
    if op in ("eq", "ne", "in", "mentions", "present") or op in CMP_WORDS:
        return _leaf(node, negated, adapter)
    raise Unrenderable(f"op {op}")


# ---------------------------------------------------------------------------------------------
# the renderer's own evaluator (the checker uses the environment's instead)
# ---------------------------------------------------------------------------------------------
def _same(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    return type(a) is type(b) and a == b


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


def evaluate(node: Any, member: dict) -> bool | None:
    """True / False / None (unknown) for a request-only condition on one population member."""
    if not isinstance(node, dict):
        return None
    op = node.get("op")
    if op in ("all", "any"):
        values = [evaluate(a, member) for a in node.get("args") or []]
        if op == "all":
            return False if False in values else (None if None in values else True)
        return True if True in values else (None if None in values else False)
    if op == "not":
        args = node.get("args") or [node.get("arg")]
        v = evaluate(args[0], member) if len(args) == 1 else None
        return None if v is None else not v
    if op == "proven":
        return evaluate(node.get("arg"), member) is True

    def field(n: Any) -> tuple[bool, Any]:
        p = _field_path(n)
        if p is None or p[:1] != ("request",) or len(p) != 2:
            return False, None
        if p[1] not in member:
            return False, None
        return True, member[p[1]]
    if op in ("eq", "ne", "in"):
        lp, rp = _field_path(node.get("left")), _field_path(node.get("right"))
        if op == "in" and lp is None and rp is not None:
            ok, seq = field(node.get("right"))
            try:
                lit = _literal(node.get("left"))
            except Unrenderable:
                return None
            if not ok or not isinstance(seq, list):
                return None
            return any(_same(x, lit) for x in seq)
        ok, value = field(node.get("left"))
        try:
            lit = _literal(node.get("right"))
        except Unrenderable:
            return None
        if not ok:
            return None
        if op == "in":
            return isinstance(lit, list) and any(_same(value, x) for x in lit)
        return _same(value, lit) if op == "eq" else not _same(value, lit)
    if op == "mentions":
        ok, value = field(node.get("text"))
        try:
            lit = _literal(node.get("value"))
        except Unrenderable:
            return None
        if not ok or not isinstance(value, str) or not isinstance(lit, str):
            return None
        hay, needle = _words(value), _words(lit)
        if not needle:
            return None
        return any(hay[i:i + len(needle)] == needle for i in range(len(hay) - len(needle) + 1))
    if op == "present":
        p = _field_path(node.get("value"))
        if p is None or p[:1] != ("request",) or len(p) != 2:
            return None
        return p[1] in member and member[p[1]] is not None
    if op in CMP_WORDS:
        try:
            path = _derived_field(node.get("left"))
            _words_, n = _number_words(node.get("right"))
        except Unrenderable:
            return None
        if path[:1] != ("request",) or len(path) != 2 or path[1] not in member:
            return None
        x = number_of(member[path[1]])
        if x is None:
            return None
        return {"gt": x > n, "gte": x >= n, "lt": x < n, "lte": x <= n}[op]
    return None


# ---------------------------------------------------------------------------------------------
# per item: the N1a and N1b clauses
# ---------------------------------------------------------------------------------------------
def _selector(node: Any, key_fields: list[str]) -> bool:
    """`eq`/`in` on a population key field: it picks the candidate, it is not a reason."""
    if not isinstance(node, dict) or node.get("op") not in ("eq", "in"):
        return False
    lp = _field_path(node.get("left"))
    return bool(lp and lp[:1] == ("request",) and len(lp) == 2 and lp[1] in key_fields)


def _unresolved(item: Item) -> bool:
    unresolved = item.semantics.get("unresolved") or {}
    return bool(unresolved.get("address") or unresolved.get("content"))


def effect_clauses(item: Item) -> tuple[list[str], bool] | None:
    """N1a: (clauses, complete) or None when there is nothing to render. `complete` means no unstated condition
    is left (no unrenderable residual, no unresolved reference, no applicability conjunct left unsaid)."""
    if not ENABLED.get() or not CLAUSES.get() or item.origin != "manifest" or item.effect.kind == "read":
        return None
    sem = item.semantics
    adapter = sem.get("population_adapter")
    top = flatten_all(sem.get("effect_match"))
    residual_top = [c for c in top if not expressible(c)]
    condition = flatten_all(sem.get("condition"))
    role = sem.get("condition_role")
    effect_cond = [c for c in condition if reads(c, "effect") and not expressible(c)]
    clauses: list[str] = []
    complete = not _unresolved(item)
    for c in residual_top:
        if request_only(c):
            continue  # applicability, said by N1b (or left unsaid: then incomplete, see below)
        try:
            if c.get("op") == "any":
                raise Unrenderable("any at top")
            clauses.append(render(c, adapter))
        except Unrenderable:
            complete = False
    if role == "prohibited_when" and item.polarity == "forbid":
        for c in effect_cond:
            try:
                if c.get("op") == "any":
                    raise Unrenderable("any at top")
                clauses.append(render(c, adapter))
            except Unrenderable:
                complete = False
    elif effect_cond:
        complete = False  # a required_when conjunct on the effect is applicability, never "also required"
    if any(request_only(c) for c in residual_top) and rule_clauses(item) is None:
        complete = False  # request-only conjuncts of the effect match are left unsaid
    return list(dict.fromkeys(clauses)), complete


def rule_clauses(item: Item) -> list[str] | None:
    """N1b: the applicability clauses, or None (silent)."""
    if not ENABLED.get() or not CLAUSES.get() or item.origin != "manifest" or not item.candidate:
        return None
    from .render.plain import candidate_tied

    if not candidate_tied(item):
        return None  # without "for C" the reader cannot tell whose row or record the rule reads
    sem = item.semantics
    adapter = sem.get("population_adapter")
    if adapter not in (ROW_ADAPTER, RECORD_ADAPTER):
        return None
    member = item.candidate.get("member")
    if not isinstance(member, dict):
        return None
    condition = flatten_all(sem.get("condition"))
    extra = [c for c in flatten_all(sem.get("effect_match")) if not expressible(c) and request_only(c)]
    whole = condition + extra
    if not whole or any(not request_only(c) for c in whole):
        return None  # a condition that reads the effect (or another alias) is not the candidate's alone
    keys = list(sem.get("population_key_fields") or [])
    for c in whole:
        for p in _paths(c):
            if p[:1] == ("request",) and (len(p) != 2 or p[1] not in member):
                return None
    if any(evaluate(c, member) is not True for c in whole):
        return None  # the renderer's own evaluation must find the whole condition true for this candidate
    if item.polarity == "forbid" and condition_covers_required(item, whole, RECORD.get()):
        return None  # rule 2: the clause would cover a row or record a requirement needs written
    clauses = []
    for c in whole:
        if _selector(c, keys):
            continue
        try:
            clauses.append(render(c, adapter))
        except Unrenderable:
            return None
    clauses = list(dict.fromkeys(clauses))
    return clauses or None


MARKER = " (plus a further condition the task states)"


def goal_suffix(item: Item, has_residual: bool) -> str | None:
    """The text replacing the marker on a goal line: " (also required: …)" (complete) or
    " (also required: …; plus a further condition the task states)"; None to keep today's marker."""
    if not has_residual:
        return None
    found = effect_clauses(item)
    if found is None:
        return None
    clauses, complete = found
    if not clauses:
        return "" if complete else None  # everything left is said by the rule suffix (N1b), or nothing changes
    text = " and ".join(clauses)
    return f" (also required: {text})" if complete else f" (also required: {text}; plus a further condition the task states)"


def not_met_suffix(item: Item) -> str:
    found = effect_clauses(item)
    if found is None or not found[1] or not found[0]:
        return ""
    return f" (the task also requires: {' and '.join(found[0])})"


def harm_when(item: Item) -> str | None:
    """" when …" replacing " as made (a further condition the task states)" on a harm, only when complete."""
    found = effect_clauses(item)
    if found is None or not found[1] or not found[0]:
        return None
    return " when " + " and ".join(found[0])


def rule_suffix(item: Item) -> str:
    clauses = rule_clauses(item)
    if not clauses:
        return ""
    verb = "forbidden" if item.polarity == "forbid" else "required"
    return f" (the task's rule: {verb} when {' and '.join(clauses)})"


class enabled_as:
    """`with enabled_as(True): …` sets the switch for the block (renderer and checker alike); `clauses` (when given)
    also sets whether the N1 clauses are rendered (CLAUSES), else that stays as it is."""

    def __init__(self, value: bool, clauses: bool | None = None):
        self.value, self.token = bool(value), None
        self.clauses, self.clauses_token = (None if clauses is None else bool(clauses)), None

    def __enter__(self):
        self.token = ENABLED.set(self.value)
        if self.clauses is not None:
            self.clauses_token = CLAUSES.set(self.clauses)
        return self

    def __exit__(self, *exc):
        if self.clauses_token is not None:
            CLAUSES.reset(self.clauses_token)
        ENABLED.reset(self.token)
        return False
