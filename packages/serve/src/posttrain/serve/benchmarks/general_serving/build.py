"""Build the checked-in ``general-serving-v1`` prompt population.

This module is the executable half of the corpus definition.  It is safe to
inspect :mod:`definition` during catalog loading without importing this module;
the builder is imported only for an explicit materialization or verification.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Iterable, Mapping
from importlib.resources import files
from typing import Any

GSM8K_REVISION = "e53f048856ff4f594e959d75785d2c2d37b678ee"
HUMANEVAL_REVISION = "e9b53e1677523f1e61e4d0960fd7502694a24bd4"
CORPUS_ID = "general-serving-v1"
CORPUS_REVISION = "1"


def _normalize(value: str) -> str:
    return unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n")).strip()


def _selection_hash(source_id: str, revision: str, key: str, prompt: str) -> str:
    identity = "\0".join((source_id, revision, key, _normalize(prompt)))
    return hashlib.sha256(identity.encode()).hexdigest()


def _select(
    rows: Iterable[Mapping[str, Any]],
    *,
    source_id: str,
    revision: str,
    key_field: str | None,
    prompt_field: str,
    count: int,
) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str, str]] = []
    for index, row in enumerate(rows):
        key = str(row[key_field]) if key_field is not None else str(index)
        prompt = _normalize(str(row[prompt_field]))
        candidates.append((_selection_hash(source_id, revision, key, prompt), key, prompt))
    return [(key, prompt) for _, key, prompt in sorted(candidates)[:count]]


def _record(
    *,
    record_id: str,
    prompt: str,
    category: str,
    source_id: str,
    source_revision: str,
    source_record_key: str,
    license_id: str,
    tools: tuple[dict[str, Any], ...] = (),
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": record_id,
        "messages": [{"role": "user", "content": prompt}],
        "tags": [category],
        "reasoning_mode": "native",
        "source_id": source_id,
        "source_revision": source_revision,
        "source_record_key": source_record_key,
        "license_id": license_id,
    }
    if tools:
        record["tools"] = list(tools)
    return record


def _first_party_records() -> tuple[Mapping[str, Any], ...]:
    resource = files("posttrain.serve.benchmarks.general_serving.resources").joinpath("first_party.json")
    try:
        raw = json.loads(resource.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid first-party prompt resource {resource}: {error}") from error
    if not isinstance(raw, list) or len(raw) != 32:
        raise RuntimeError(f"first-party prompt resource must contain 32 records: {resource}")
    records: list[Mapping[str, Any]] = []
    expected_categories = {"chat", "extraction", "structured-output", "tool-use"}
    category_counts: dict[str, int] = {}
    for index, value in enumerate(raw):
        if not isinstance(value, dict):
            raise RuntimeError(f"first-party prompt {index} must be an object: {resource}")
        category = value.get("category")
        prompt = value.get("prompt")
        tools = value.get("tools", [])
        if category not in expected_categories or not isinstance(prompt, str) or not prompt.strip():
            raise RuntimeError(f"first-party prompt {index} has invalid category or prompt: {resource}")
        if not isinstance(tools, list) or any(not isinstance(tool, dict) for tool in tools):
            raise RuntimeError(f"first-party prompt {index} has invalid tools: {resource}")
        category_counts[category] = category_counts.get(category, 0) + 1
        records.append(value)
    if category_counts != {category: 8 for category in sorted(expected_categories)}:
        raise RuntimeError(f"first-party prompt categories must contain eight records each: {resource}")
    return tuple(records)


def build_from_rows(gsm8k: Iterable[Mapping[str, Any]], humaneval: Iterable[Mapping[str, Any]]) -> tuple[str, str]:
    """Build canonical corpus bytes from already-resolved source rows.

    Keeping source resolution outside this function gives the generic
    materializer a deterministic, offline-testable builder boundary.
    """

    records: list[dict[str, Any]] = []
    for key, prompt in _select(
        gsm8k,
        source_id="openai/gsm8k",
        revision=GSM8K_REVISION,
        key_field=None,
        prompt_field="question",
        count=64,
    ):
        records.append(
            _record(
                record_id=f"gsm8k-train-{int(key):05d}",
                prompt=prompt,
                category="reasoning",
                source_id="openai/gsm8k",
                source_revision=GSM8K_REVISION,
                source_record_key=f"main/train/{key}",
                license_id="MIT",
            )
        )
    for key, prompt in _select(
        humaneval,
        source_id="openai/openai_humaneval",
        revision=HUMANEVAL_REVISION,
        key_field="task_id",
        prompt_field="prompt",
        count=32,
    ):
        records.append(
            _record(
                record_id=f"humaneval-{key.replace('/', '-').lower()}",
                prompt=prompt,
                category="code",
                source_id="openai/openai_humaneval",
                source_revision=HUMANEVAL_REVISION,
                source_record_key=key,
                license_id="MIT",
            )
        )
    first_party_index = 0
    for value in _first_party_records():
        category = str(value["category"])
        tools = tuple(value.get("tools", ()))
        records.append(
            _record(
                record_id=f"posttrain-{category}-{first_party_index % 8 + 1:02d}",
                prompt=_normalize(str(value["prompt"])),
                category=category,
                source_id="carbonteq-ai/posttrain",
                source_revision=CORPUS_REVISION,
                source_record_key=f"first-party/{first_party_index}",
                license_id="Apache-2.0",
                tools=tools,
            )
        )
        first_party_index += 1

    records_text = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n" for record in records
    )
    category_counts: dict[str, int] = {}
    for record in records:
        category = record["tags"][0]
        category_counts[category] = category_counts.get(category, 0) + 1
    manifest = {
        "schema_version": 1,
        "id": CORPUS_ID,
        "revision": CORPUS_REVISION,
        "digest": hashlib.sha256(records_text.encode()).hexdigest(),
        "record_count": len(records),
        "category_counts": category_counts,
        "sources": [
            {
                "id": "openai/gsm8k",
                "revision": GSM8K_REVISION,
                "config": "main",
                "split": "train",
                "license_id": "MIT",
            },
            {
                "id": "openai/openai_humaneval",
                "revision": HUMANEVAL_REVISION,
                "config": None,
                "split": "test",
                "license_id": "MIT",
            },
            {
                "id": "carbonteq-ai/posttrain",
                "revision": CORPUS_REVISION,
                "config": None,
                "split": "first-party",
                "license_id": "Apache-2.0",
            },
        ],
        "selection_algorithm": "lowest-sha256-over-source-revision-key-and-normalized-prompt-v1",
        "license_notices": [
            "GSM8K records are used under the MIT license.",
            "HumanEval records are used under the MIT license.",
            "First-party prompts are licensed under the repository Apache-2.0 license.",
        ],
    }
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    return records_text, manifest_text


def build() -> tuple[str, str]:
    """Resolve pinned public sources and return canonical corpus bytes."""

    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "corpus materialization requires Hugging Face datasets; run "
            "`uv run --with 'datasets>=4.6.1,<4.7' posttrain workload verify "
            "workloads/general-serving-32k-sweep@1`"
        ) from error

    gsm8k = load_dataset("openai/gsm8k", "main", split="train", revision=GSM8K_REVISION)
    humaneval = load_dataset("openai/openai_humaneval", split="test", revision=HUMANEVAL_REVISION)
    return build_from_rows(gsm8k, humaneval)


__all__ = [
    "CORPUS_ID",
    "CORPUS_REVISION",
    "GSM8K_REVISION",
    "HUMANEVAL_REVISION",
    "build",
    "build_from_rows",
]
