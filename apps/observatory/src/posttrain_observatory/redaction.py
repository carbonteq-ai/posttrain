"""One recursive redaction policy used before every product serialization."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import cast

from posttrain.common import JsonValue


class RedactionPolicy:
    def __init__(
        self,
        patterns: tuple[str, ...] = (
            r"(^|[_-])(access|auth|bearer|refresh|session)?[_-]?token($|[_-])",
            "secret",
            "password",
            "api[_-]?key",
        ),
    ) -> None:
        self._patterns = tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)

    def apply(self, value: JsonValue) -> JsonValue:
        if isinstance(value, Mapping):
            return {
                str(key): "[REDACTED]" if self._sensitive(str(key)) else self.apply(cast(JsonValue, item))
                for key, item in value.items()
            }
        if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
            return [self.apply(cast(JsonValue, item)) for item in value]
        return value

    def mapping(self, value: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        return cast(dict[str, JsonValue], self.apply(dict(value)))

    def _sensitive(self, key: str) -> bool:
        return any(pattern.search(key) for pattern in self._patterns)


__all__ = ["RedactionPolicy"]
