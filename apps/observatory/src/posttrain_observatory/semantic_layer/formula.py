"""Metric formulas: arithmetic over aggregated measures, parsed without evaluating code.

Grammar::

    expr   := term (("+" | "-") term)*
    term   := factor (("*" | "/") factor)*
    factor := number | aggregation "(" measure ")" | "(" expr ")" | "-" factor

A missing value or a division by zero makes the whole result missing.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

_TOKEN = re.compile(r"\s*(?:(\d+(?:\.\d*)?|\.\d+)|([a-z][a-z0-9_]*)|(.))")
_AGGREGATIONS = frozenset({"last", "first", "min", "max", "mean", "sum", "count", "stddev"})

type Lookup = Callable[[str, str], float | None]


@dataclass(frozen=True, slots=True)
class _Number:
    value: float

    def evaluate(self, lookup: Lookup) -> float | None:
        return self.value

    def references(self) -> tuple[tuple[str, str], ...]:
        return ()


@dataclass(frozen=True, slots=True)
class _Reference:
    aggregation: str
    measure: str

    def evaluate(self, lookup: Lookup) -> float | None:
        return lookup(self.aggregation, self.measure)

    def references(self) -> tuple[tuple[str, str], ...]:
        return ((self.aggregation, self.measure),)


@dataclass(frozen=True, slots=True)
class _Negate:
    operand: Formula

    def evaluate(self, lookup: Lookup) -> float | None:
        value = self.operand.evaluate(lookup)
        return None if value is None else -value

    def references(self) -> tuple[tuple[str, str], ...]:
        return self.operand.references()


@dataclass(frozen=True, slots=True)
class _Binary:
    operator: str
    left: Formula
    right: Formula

    def evaluate(self, lookup: Lookup) -> float | None:
        left, right = self.left.evaluate(lookup), self.right.evaluate(lookup)
        if left is None or right is None:
            return None
        if self.operator == "+":
            return left + right
        if self.operator == "-":
            return left - right
        if self.operator == "*":
            return left * right
        return None if right == 0 else left / right

    def references(self) -> tuple[tuple[str, str], ...]:
        return (*self.left.references(), *self.right.references())


type Formula = _Number | _Reference | _Negate | _Binary


class FormulaError(ValueError):
    pass


def parse_formula(text: str) -> Formula:
    tokens: list[tuple[str, str]] = []
    position = 0
    while position < len(text):
        match = _TOKEN.match(text, position)
        if match is None or match.end() == position:
            break
        number, word, symbol = match.groups()
        if number is not None:
            tokens.append(("number", number))
        elif word is not None:
            tokens.append(("word", word))
        elif symbol is not None and symbol.strip():
            if symbol not in "+-*/()":
                raise FormulaError(f"unexpected character {symbol!r} in formula {text!r}")
            tokens.append(("symbol", symbol))
        position = match.end()
    parser = _Parser(tokens, text)
    result = parser.expression()
    if parser.index != len(tokens):
        raise FormulaError(f"unexpected {tokens[parser.index][1]!r} in formula {text!r}")
    return result


class _Parser:
    def __init__(self, tokens: list[tuple[str, str]], text: str) -> None:
        self.tokens = tokens
        self.text = text
        self.index = 0

    def _peek(self) -> tuple[str, str] | None:
        return self.tokens[self.index] if self.index < len(self.tokens) else None

    def _take(self, kind: str, value: str | None = None) -> str:
        token = self._peek()
        if token is None or token[0] != kind or (value is not None and token[1] != value):
            expected = value or kind
            raise FormulaError(f"expected {expected!r} in formula {self.text!r}")
        self.index += 1
        return token[1]

    def expression(self) -> Formula:
        node = self.term()
        while (token := self._peek()) is not None and token in (("symbol", "+"), ("symbol", "-")):
            self.index += 1
            node = _Binary(token[1], node, self.term())
        return node

    def term(self) -> Formula:
        node = self.factor()
        while (token := self._peek()) is not None and token in (("symbol", "*"), ("symbol", "/")):
            self.index += 1
            node = _Binary(token[1], node, self.factor())
        return node

    def factor(self) -> Formula:
        token = self._peek()
        if token is None:
            raise FormulaError(f"formula {self.text!r} ends early")
        if token == ("symbol", "-"):
            self.index += 1
            return _Negate(self.factor())
        if token == ("symbol", "("):
            self.index += 1
            node = self.expression()
            self._take("symbol", ")")
            return node
        if token[0] == "number":
            self.index += 1
            return _Number(float(token[1]))
        if token[0] == "word" and token[1] in _AGGREGATIONS:
            self.index += 1
            self._take("symbol", "(")
            measure = self._take("word")
            self._take("symbol", ")")
            return _Reference(token[1], measure)
        raise FormulaError(f"expected a number, aggregation or '(' in formula {self.text!r}")


__all__ = ["Formula", "FormulaError", "Lookup", "parse_formula"]
