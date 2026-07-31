"""Composed base and overlay catalog for primitive selections."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError

from .artifacts import HubModelRef, LocalArtifactRef, TrackioArtifactRef
from .catalog_schema import ExecutionTargetSchema, InferenceBindingSchema, ModelVariantSchema, WorkloadSchema
from .errors import ContractError
from .models import ModelCapabilities, ModelVariant
from .selections import (
    ExecutionTarget,
    InferenceBinding,
    Selection,
    SelectionFamily,
    Workload,
    validate_selection_id,
)
from .variants import RENDERER_CONTRACTS

_CORE_FAMILY_ORDER: tuple[str, ...] = (
    "model",
    "target",
    "dataset",
    "environment",
    "evaluation",
    "workload",
    "training",
    "quantization",
    "recipe",
    "inference",
)
_FAMILY = re.compile(r"^[a-z][a-z0-9-]*$")


@dataclass(frozen=True, slots=True, order=True)
class CatalogRef:
    family: SelectionFamily
    id: str

    def __post_init__(self) -> None:
        if not _FAMILY.fullmatch(self.family):
            raise ContractError(f"catalog family must be a lowercase identifier: {self.family!r}")
        validate_selection_id(self.id, "catalog id")


@dataclass(frozen=True, slots=True)
class Resolved[SelectionT: Selection]:
    ref: CatalogRef
    value: SelectionT
    source_layer: str
    overlay_id: str | None = None

    def __post_init__(self) -> None:
        if self.source_layer not in {"base", "overlay"}:
            raise ContractError("source_layer must be 'base' or 'overlay'")
        if (self.source_layer == "overlay") != (self.overlay_id is not None):
            raise ContractError("overlay resolutions require exactly one overlay_id")


@dataclass(frozen=True, slots=True)
class CatalogLayer:
    id: str
    entries: Mapping[CatalogRef, Selection]

    def __post_init__(self) -> None:
        validate_selection_id(self.id, "catalog layer id")
        object.__setattr__(self, "entries", dict(self.entries))


type CatalogSource = CatalogLayer | Mapping[str, object] | Path | str
type SelectionDecoder = Callable[[CatalogRef, Mapping[str, object], Mapping[CatalogRef, Selection]], Selection]


class Catalog:
    """One project-scoped read view over a base catalog and its overlays."""

    def __init__(
        self,
        base: CatalogLayer,
        overlays: tuple[CatalogLayer, ...],
        scope: str,
        *,
        family_registry_lock: object | None = None,
    ) -> None:
        validate_selection_id(scope, "catalog scope")
        if len({overlay.id for overlay in overlays}) != len(overlays):
            raise ContractError("catalog overlay ids must be unique")
        self._base = base
        self._overlays = overlays
        self.scope = scope
        self.family_registry_lock = family_registry_lock

    @classmethod
    def open(
        cls,
        base: CatalogSource,
        overlays: Iterable[CatalogSource] = (),
        scope: str = "default",
        decoders: Mapping[SelectionFamily, SelectionDecoder] | None = None,
        family_registry_lock: object | None = None,
    ) -> Catalog:
        decoders = {} if decoders is None else decoders
        base_layer = _load_layer(base, default_id="base", known={}, decoders=decoders)
        known: dict[CatalogRef, Selection] = dict(base_layer.entries)
        overlay_layers: list[CatalogLayer] = []
        for index, source in enumerate(overlays):
            layer = _load_layer(source, default_id=f"overlay-{index + 1}", known=known, decoders=decoders)
            overlay_layers.append(layer)
            known.update(layer.entries)
        return cls(base_layer, tuple(overlay_layers), scope, family_registry_lock=family_registry_lock)

    def resolve(self, ref: CatalogRef) -> Resolved[Selection]:
        for overlay in reversed(self._overlays):
            if ref in overlay.entries:
                return Resolved(ref, overlay.entries[ref], "overlay", overlay.id)
        try:
            return Resolved(ref, self._base.entries[ref], "base")
        except KeyError as error:
            raise KeyError(f"catalog selection not found: {ref.family}/{ref.id}") from error

    @property
    def base_id(self) -> str:
        """Immutable release identity of the composed catalog's base layer."""

        return self._base.id

    @property
    def overlay_ids(self) -> tuple[str, ...]:
        """Overlay layer identities in application order."""

        return tuple(layer.id for layer in self._overlays)

    def contains(self, ref: CatalogRef) -> bool:
        return any(ref in layer.entries for layer in (*self._overlays, self._base))

    def list(self, family: SelectionFamily | None = None) -> list[CatalogRef]:
        refs = set(self._base.entries)
        for overlay in self._overlays:
            refs.update(overlay.entries)
        return sorted(ref for ref in refs if family is None or ref.family == family)

    def transitive_refs(self, roots: Iterable[CatalogRef]) -> tuple[CatalogRef, ...]:
        """Return resolved catalog entries reachable through selection values.

        Catalog decoders replace reference-shaped fields with the resolved
        selection object. Identity traversal therefore captures relationships
        between families without teaching the common catalog about every
        extension's schema.
        """

        values = {id(self.resolve(ref).value): ref for ref in self.list()}
        pending = list(roots)
        seen: set[CatalogRef] = set()
        while pending:
            ref = pending.pop()
            if ref in seen:
                continue
            self.resolve(ref)
            seen.add(ref)
            for value in _walk_values(self.resolve(ref).value):
                nested = values.get(id(value))
                if nested is not None and nested not in seen:
                    pending.append(nested)
        return tuple(sorted(seen))

    def refs_for_values(self, values: Iterable[object]) -> tuple[CatalogRef, ...]:
        """Return catalog references for resolved selection object identities."""

        known = {id(self.resolve(ref).value): ref for ref in self.list()}
        resolved = {
            ref
            for value in values
            for nested in _walk_values(value)
            if (ref := known.get(id(nested))) is not None
        }
        return tuple(sorted(resolved))


def _load_layer(
    source: CatalogSource,
    *,
    default_id: str,
    known: Mapping[CatalogRef, Selection],
    decoders: Mapping[SelectionFamily, SelectionDecoder],
) -> CatalogLayer:
    if isinstance(source, CatalogLayer):
        return source
    if isinstance(source, (str, Path)):
        path = Path(source)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ContractError(f"catalog file must contain an object: {path}")
        default_id = path.stem
    else:
        payload = dict(source)
    layer_id_value = payload.pop("layer_id", default_id)
    if not isinstance(layer_id_value, str):
        raise ContractError("catalog layer_id must be a string")
    raw_entries: dict[CatalogRef, object] = {}
    for family, family_entries in payload.items():
        if not isinstance(family, str) or not _FAMILY.fullmatch(family):
            raise ContractError(f"catalog family must be a lowercase identifier: {family!r}")
        if not isinstance(family_entries, Mapping):
            raise ContractError(f"catalog family {family!r} must contain an object")
        for selection_id, raw in family_entries.items():
            if not isinstance(selection_id, str):
                raise ContractError("catalog ids must be strings")
            ref = CatalogRef(family, selection_id)
            raw_entries[ref] = raw
    entries: dict[CatalogRef, Selection] = {}
    family_order = (*_CORE_FAMILY_ORDER, *sorted(set(ref.family for ref in raw_entries).difference(_CORE_FAMILY_ORDER)))
    for family in family_order:
        for ref, raw in raw_entries.items():
            if ref.family == family:
                entries[ref] = _decode_selection(ref, raw, {**known, **entries}, decoders)
    unsupported = [ref for ref in raw_entries if ref not in entries]
    if unsupported:
        ref = unsupported[0]
        raise ContractError(f"catalog loader for family {ref.family!r} is not available in slice 0")
    return CatalogLayer(layer_id_value, entries)


def _decode_selection(
    ref: CatalogRef,
    raw: object,
    known: Mapping[CatalogRef, Selection],
    decoders: Mapping[SelectionFamily, SelectionDecoder],
) -> Selection:
    if isinstance(raw, Selection):
        if raw.id != ref.id:
            raise ContractError(f"catalog key {ref.id!r} does not match selection id {raw.id!r}")
        return raw
    if not isinstance(raw, Mapping):
        raise ContractError(f"catalog entry {ref.family}/{ref.id} must be a selection or object")
    data = dict(raw)
    data.setdefault("id", ref.id)
    if decoder := decoders.get(ref.family):
        try:
            return decoder(ref, data, known)
        except ValidationError as error:
            raise ContractError(f"invalid catalog entry {ref.family}/{ref.id}: {error}") from error
    if ref.family == "model":
        return _decode_model(_validated(ModelVariantSchema, data, ref).model_dump())
    if ref.family == "target":
        payload = _validated(ExecutionTargetSchema, data, ref)
        return ExecutionTarget(**payload.model_dump())
    if ref.family == "workload":
        payload = _validated(WorkloadSchema, data, ref)
        return Workload(**payload.model_dump())
    if ref.family == "inference":
        payload = _validated(InferenceBindingSchema, data, ref)
        values = payload.model_dump()
        model = _linked(known, "model", values.pop("model"), ModelVariant)
        target = _linked(known, "target", values.pop("target"), ExecutionTarget)
        return InferenceBinding(model=model, target=target, **values)
    raise ContractError(f"catalog loader for family {ref.family!r} is not available in slice 0")


def _walk_values(value: object) -> Iterable[object]:
    """Yield nested values while preserving object identity relationships."""

    seen: set[int] = set()

    def visit(item: object) -> Iterable[object]:
        identifier = id(item)
        if identifier in seen:
            return
        seen.add(identifier)
        yield item
        if is_dataclass(item) and not isinstance(item, type):
            for field in fields(item):
                yield from visit(getattr(item, field.name))
        elif isinstance(item, Mapping):
            for key, nested in item.items():
                yield from visit(key)
                yield from visit(nested)
        elif isinstance(item, (tuple, list, set, frozenset)):
            for nested in item:
                yield from visit(nested)

    yield from visit(value)


def _linked[SelectionT: Selection](
    known: Mapping[CatalogRef, Selection],
    family: SelectionFamily,
    raw: object,
    expected: type[SelectionT],
) -> SelectionT:
    if isinstance(raw, expected):
        return cast(SelectionT, raw)
    selection_id = raw.get("id") if isinstance(raw, Mapping) else raw
    if not isinstance(selection_id, str):
        raise ContractError(f"{family} link must be a catalog id")
    ref = CatalogRef(family, selection_id)
    try:
        value = known[ref]
    except KeyError as error:
        raise ContractError(f"unresolved catalog link: {family}/{selection_id}") from error
    if not isinstance(value, expected):
        raise ContractError(f"catalog link {family}/{selection_id} has the wrong selection type")
    return value


def _validated[SchemaT: ModelVariantSchema | ExecutionTargetSchema | WorkloadSchema | InferenceBindingSchema](
    schema: type[SchemaT],
    data: dict[str, object],
    ref: CatalogRef,
) -> SchemaT:
    try:
        return schema.model_validate(data)
    except ValidationError as error:
        raise ContractError(f"invalid catalog entry {ref.family}/{ref.id}: {error}") from error


def _decode_model(data: dict[str, Any]) -> ModelVariant:
    renderer_id = data.pop("renderer_contract")
    if not isinstance(renderer_id, str):
        raise ContractError("model renderer_contract must be an id")
    try:
        renderer = RENDERER_CONTRACTS[renderer_id]
    except KeyError as error:
        raise ContractError(f"unknown renderer contract: {renderer_id!r}") from error
    base_data = data.pop("base", None)
    artifact = data.pop("artifact")
    if not isinstance(artifact, Mapping):
        raise ContractError("model artifact must be an object")
    artifact_data = dict(artifact)
    try:
        kind = artifact_data.pop("kind")
    except KeyError as error:
        raise ContractError("model artifact requires a kind") from error
    if kind == "hub":
        artifact_ref = HubModelRef(**artifact_data)
    elif kind == "local":
        artifact_ref = LocalArtifactRef(**artifact_data)
    elif kind == "trackio":
        artifact_ref = TrackioArtifactRef(**artifact_data)
    else:
        raise ContractError(f"unknown model artifact kind: {kind!r}")
    capabilities = data.pop("capabilities")
    if not isinstance(capabilities, Mapping):
        raise ContractError("model capabilities must be an object")
    capability_data = dict(capabilities)
    if base_data is not None:
        if not isinstance(base_data, Mapping):
            raise ContractError("model base must be a pinned Hub artifact")
        values = dict(base_data)
        if values.pop("kind", None) != "hub":
            raise ContractError("model base must be a pinned Hub artifact")
        base = HubModelRef(**values)
    else:
        base = artifact_ref if isinstance(artifact_ref, HubModelRef) else None
    if base is None:
        raise ContractError("catalog model variants currently require a pinned Hub base artifact")
    return ModelVariant(
        artifact=artifact_ref,
        capabilities=ModelCapabilities(**capability_data),
        renderer=renderer,
        base=base,
        **data,
    )


__all__ = ["Catalog", "CatalogLayer", "CatalogRef", "Resolved", "SelectionDecoder"]
