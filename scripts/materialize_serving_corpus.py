"""Rebuild the checked-in representative serving prompt corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
from collections.abc import Iterable
from pathlib import Path
from typing import Any

GSM8K_REVISION = "e53f048856ff4f594e959d75785d2c2d37b678ee"
HUMANEVAL_REVISION = "e9b53e1677523f1e61e4d0960fd7502694a24bd4"
CORPUS_ID = "general-serving-v1"
CORPUS_REVISION = "1"
OUTPUT_ROOT = (
    Path(__file__).resolve().parents[1]
    / "packages"
    / "serve"
    / "src"
    / "posttrain"
    / "serve"
    / "benchmarks"
    / "resources"
    / "corpora"
)


def _normalize(value: str) -> str:
    return unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n")).strip()


def _selection_hash(source_id: str, revision: str, key: str, prompt: str) -> str:
    identity = "\0".join((source_id, revision, key, _normalize(prompt)))
    return hashlib.sha256(identity.encode()).hexdigest()


def _select(
    rows: Iterable[dict[str, Any]],
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


def _tool(name: str, description: str, properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        },
    }


FIRST_PARTY_PROMPTS: dict[str, tuple[str, ...]] = {
    "chat": (
        "Explain the difference between throughput and per-request latency to a product manager in three short paragraphs.",
        "Draft a concise agenda for a 30-minute engineering retrospective after a production incident.",
        "A teammate says caching always makes software faster. Respond constructively and explain two important exceptions.",
        "Rewrite this status update to be direct and calm: We are kind of blocked because the vendor API keeps behaving strangely.",
        "Suggest a practical onboarding plan for an engineer joining a mature Python service with limited documentation.",
        "Compare queues and streams for a team choosing how to process asynchronous account events.",
        "Help a manager resolve a meeting conflict between a customer escalation and a planned architecture review.",
        "Explain why reproducible configuration matters when comparing machine-learning serving backends.",
    ),
    "extraction": (
        "Extract the order id, customer, currency, and total as JSON from: Order ORD-1042 for Northwind Labs totals EUR 8,430.50.",
        "Extract the incident severity, service, region, and start time as JSON from: SEV-2 on billing-api in eu-west began at 14:32 UTC.",
        "Extract the candidate name, role, interview date, and interviewer from: Mina Patel interviews for Staff Data Engineer on 18 August with Jorge.",
        "Extract the repository, pull request number, reviewer, and requested action from: carbonteq-ai/posttrain PR #17 needs Sara to re-run release checks.",
        "Extract the departure city, arrival city, date, and passenger count from: Three passengers fly from Lahore to Dubai on 12 September.",
        "Extract the invoice number, due date, supplier, and amount from: Invoice INV-88 from Acme Hosting is due 2026-08-01 for USD 1,240.",
        "Extract the model, context size, concurrency, and GPU from: Qwen 2B was tested at 32K context and concurrency 8 on one A6000.",
        "Extract the ticket id, priority, owner, and status from: SUP-451 is high priority, owned by Aisha, and waiting on the customer.",
    ),
    "structured-output": (
        "Return JSON with keys risks, mitigations, and owner for a database migration that must avoid write downtime.",
        "Return a JSON array of three deployment checks; each item must contain name, command, and expected_result.",
        "Produce JSON with summary, assumptions, and open_questions for planning a model-serving capacity test.",
        "Return JSON matching {title: string, decisions: string[], actions: [{owner: string, task: string}]} for a design review.",
        "Create JSON with primary_option, alternatives, and tradeoffs for choosing between batch and online inference.",
        "Return JSON with fields severity, user_impact, immediate_action, and follow_up for an expired TLS certificate incident.",
        "Produce a JSON object with cpu, memory_gb, gpu_count, and constraints for a single-node inference target.",
        "Return JSON containing accepted, reasons, and missing_evidence for a release-gate review with incomplete GPU benchmarks.",
    ),
    "tool-use": (
        "Check the weather in Islamabad tomorrow and report the expected high temperature.",
        "Find the first free 30-minute calendar slot after 2 PM next Tuesday.",
        "Look up current inventory for SKU GPU-A6000 and report whether four units are available.",
        "Open a high-priority support ticket for repeated payment timeouts affecting account ACME-42.",
        "Convert 1,250 EUR to USD using the available exchange-rate service.",
        "Find the fastest driving route from Lahore airport to Gulberg at 6 PM.",
        "Look up the CRM record for Northwind Labs and report the account owner.",
        "Search the internal knowledge base for the GPU benchmark release checklist.",
    ),
}

TOOL_SPECS = (
    _tool("get_weather", "Read a weather forecast.", {"location": {"type": "string"}, "date": {"type": "string"}}),
    _tool(
        "find_calendar_slot",
        "Find calendar availability.",
        {"after": {"type": "string"}, "duration_minutes": {"type": "integer"}},
    ),
    _tool("get_inventory", "Read inventory for a SKU.", {"sku": {"type": "string"}}),
    _tool(
        "create_ticket",
        "Create a support ticket.",
        {"account": {"type": "string"}, "priority": {"type": "string"}, "summary": {"type": "string"}},
    ),
    _tool(
        "convert_currency",
        "Convert between currencies.",
        {"amount": {"type": "number"}, "from": {"type": "string"}, "to": {"type": "string"}},
    ),
    _tool(
        "get_route",
        "Find a driving route.",
        {"origin": {"type": "string"}, "destination": {"type": "string"}, "departure": {"type": "string"}},
    ),
    _tool("lookup_crm", "Read a CRM account.", {"account": {"type": "string"}}),
    _tool("search_knowledge_base", "Search internal documentation.", {"query": {"type": "string"}}),
)


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


def build() -> tuple[str, str]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "corpus materialization requires Hugging Face datasets; run "
            "`uv run --with 'datasets>=4.6.1,<4.7' python scripts/materialize_serving_corpus.py`"
        ) from error

    gsm8k = load_dataset("openai/gsm8k", "main", split="train", revision=GSM8K_REVISION)
    humaneval = load_dataset("openai/openai_humaneval", split="test", revision=HUMANEVAL_REVISION)
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
    for category, prompts in FIRST_PARTY_PROMPTS.items():
        for category_index, prompt in enumerate(prompts):
            tools = (TOOL_SPECS[category_index],) if category == "tool-use" else ()
            records.append(
                _record(
                    record_id=f"posttrain-{category}-{category_index + 1:02d}",
                    prompt=prompt,
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if committed resources differ")
    args = parser.parse_args()
    records_text, manifest_text = build()
    outputs = {
        OUTPUT_ROOT / f"{CORPUS_ID}.jsonl": records_text,
        OUTPUT_ROOT / f"{CORPUS_ID}.manifest.json": manifest_text,
    }
    if args.check:
        mismatches = [
            str(path)
            for path, expected in outputs.items()
            if not path.is_file() or path.read_text(encoding="utf-8") != expected
        ]
        if mismatches:
            print("serving corpus is stale: " + ", ".join(mismatches), file=sys.stderr)
            return 1
        print(f"{CORPUS_ID}@{CORPUS_REVISION} is reproducible")
        return 0
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for path, content in outputs.items():
        path.write_text(content, encoding="utf-8")
    print(f"wrote {len(records_text.splitlines())} records to {OUTPUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
