"""Explicit, inert Python catalog providers.

Catalog providers are an authoring front-end for the normal composed catalog.
They return already-constructed selection values; they are not a registry and
must not execute dataset builders or perform materialization while loading.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from posttrain.common import CatalogRef, ContractError
from posttrain.common.selections import Selection


@dataclass(frozen=True, slots=True, init=False)
class CatalogEntries:
    """Complete typed entries returned by one explicit Python provider.

    The canonical form accepts a mapping indexed by ``CatalogRef``.  The
    ``datasets`` keyword is a small convenience for the common dataset case;
    each value supplies its catalog ID and is placed in the ``dataset``
    family.  No global registry or field-level merge is involved.
    """

    entries: Mapping[CatalogRef, Selection]

    def __init__(
        self,
        entries: Mapping[CatalogRef, Selection] | None = None,
        *,
        datasets: Iterable[Selection] = (),
    ) -> None:
        values = tuple(datasets)
        if entries is not None and values:
            raise ContractError("CatalogEntries accepts entries or datasets, not both")
        if entries is None:
            entries = {}
            for value in values:
                if not isinstance(value, Selection):
                    raise ContractError("Python catalog provider dataset entries must be selections")
                ref = CatalogRef("dataset", value.id)
                entries[ref] = value
            if len(entries) != len(values):
                raise ContractError("Python catalog provider returned duplicate dataset ids")
        validated: dict[CatalogRef, Selection] = {}
        for ref, value in entries.items():
            if not isinstance(ref, CatalogRef):
                raise ContractError("Python catalog provider entries must use CatalogRef keys")
            if not isinstance(value, Selection):
                raise ContractError(f"Python catalog provider entry {ref.family}/{ref.id} is not a selection")
            if value.id != ref.id:
                raise ContractError(
                    f"Python catalog provider key {ref.family}/{ref.id} does not match selection id {value.id!r}"
                )
            validated[ref] = value
        object.__setattr__(self, "entries", MappingProxyType(validated))


def load_python_catalog_provider(reference: str) -> CatalogEntries:
    """Import and invoke one explicit ``MODULE:CALLABLE`` provider.

    Only the provider target is imported.  Builders named by the returned
    values remain inert and are not imported by this function.
    """

    module_name, separator, attribute = reference.partition(":")
    if (
        not separator
        or not module_name
        or not attribute
        or ":" in attribute
        or not all(part.isidentifier() for part in module_name.split("."))
        or not all(part.isidentifier() for part in attribute.split("."))
    ):
        raise ContractError(f"invalid Python catalog provider reference: {reference!r}")
    try:
        module = importlib.import_module(module_name)
    except Exception as error:
        raise ContractError(f"could not import Python catalog provider {reference!r}: {error}") from error
    target: Any = module
    try:
        for part in attribute.split("."):
            target = getattr(target, part)
    except AttributeError as error:
        raise ContractError(f"Python catalog provider {reference!r} has no callable target") from error
    if not callable(target):
        raise ContractError(f"Python catalog provider {reference!r} target is not callable")
    try:
        result = target()
    except Exception as error:
        raise ContractError(f"Python catalog provider {reference!r} failed: {error}") from error
    if not isinstance(result, CatalogEntries):
        raise ContractError(
            f"Python catalog provider {reference!r} must return CatalogEntries, got {type(result).__name__}"
        )
    return result


__all__ = ["CatalogEntries", "load_python_catalog_provider"]
