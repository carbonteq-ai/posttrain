"""CLI error formatting helpers."""

from __future__ import annotations


def error_message(error: BaseException) -> str:
    if isinstance(error, KeyError) and error.args:
        return str(error.args[0])
    return str(error)
