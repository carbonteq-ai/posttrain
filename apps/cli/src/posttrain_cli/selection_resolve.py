"""Resolve catalog selections that may carry an ``@revision`` suffix."""

from __future__ import annotations

from posttrain.common import Catalog, CatalogRef, ContractError
from posttrain.common.selections import SelectionFamily


def resolve_selection(catalog: Catalog, family: SelectionFamily, selection_id: str):
    """Resolve ``family/selection_id``, accepting an optional ``@revision`` suffix.

    Catalog YAML keys for targets (and some other families) omit ``@revision``;
    the revision lives on the selection object. Callers often pass the printed
    ``id@revision`` form. Exact match wins first so selections whose id itself
    contains ``@`` (models, datasets) keep working.
    """
    ref = CatalogRef(family, selection_id)
    try:
        return catalog.resolve(ref)
    except KeyError:
        if "@" not in selection_id:
            raise
    bare, revision = selection_id.rsplit("@", 1)
    if not bare or not revision:
        raise KeyError(f"catalog selection not found: {family}/{selection_id}")
    resolved = catalog.resolve(CatalogRef(family, bare))
    actual = getattr(resolved.value, "revision", None)
    if actual is not None and str(actual) != revision:
        raise ContractError(
            f"{family} selection {bare!r} has revision {str(actual)!r}, not {revision!r}"
        )
    return resolved


__all__ = ["resolve_selection"]
