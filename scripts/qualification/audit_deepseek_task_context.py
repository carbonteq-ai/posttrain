"""Read-only source audit and HTML review for a completed context-authoring batch."""

import argparse
import html
import json
import re
from collections import Counter
from pathlib import Path


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from strings(child)


def audit(root):
    summary = json.loads((root / "summary.json").read_text())
    inputs = json.loads((root / "inputs.json").read_text())
    rows = []
    for inp in inputs:
        directory = root / inp["task"]
        context = json.loads((directory / "context.json").read_text())
        response = json.loads((directory / "response.json").read_text())
        text = json.dumps(context)
        public = inp["public_request"].casefold()
        copied = []
        for value in set(strings(inp["goal_harm_contract"])):
            if len(value) < 4 or value.casefold() in public or value not in text:
                continue
            if (re.fullmatch(r"(?:ss_|ws_|ch_|C_)[\w-]+", value)
                    or re.fullmatch(r"00[0-9A-Za-z]{9,}", value)
                    or re.fullmatch(r"[^ @]+@[^ @]+\.[^ @]+", value)):
                copied.append(value)
        available = {t["function"]["name"] for t in inp["available_tools"]}
        named = set(re.findall(r"\b(?:gmail|slack|google_sheets|google_drive|salesforce|twilio|asana|airtable|zendesk)_[a-z_]+\b", text))
        unknown = sorted(named - available)
        rows.append({"task": inp["task"], "private_identifier_copies": sorted(copied),
                     "tool_like_names_not_in_schema": unknown, "schema_status": "passed",
                     "usage": response["usage"], "model": response["model"],
                     "provider": response["provider"], "checks": len(context["check_guidance"]),
                     "word_count": len(" ".join(strings(context)).split()),
                     "admission": "review_required" if copied or unknown else "not_semantically_qualified"})
    reviews = {
        "finance.year_end_audit_prep": {
            "verdict": "reject_as_is",
            "good": "Preserves in-scope ready/not-ready classification and explicitly identifies missing arithmetic/content coverage.",
            "defects": ["Copies ss_audit_prep/ws_audit_checklist, which are absent from the public request.",
                        "Requires sent-message read-back although available Gmail tool is send-only; this must be marked unavailable rather than mandatory."]},
        "hr.twilio_emergency_notification": {
            "verdict": "reject_as_is",
            "good": "Uses Office AND SMS Opt-In and keeps SMS, facilities email and Slack effects separate.",
            "defects": ["Copies ss_emergency/ws_phones/ws_office_contacts/C_ANNOUNCE private identifiers.",
                        "Tells teacher to retrieve a Slack channels list and inspect Twilio records, but neither capability exists in this task's tool list.",
                        "Correct field names alone do not establish that every required identifier/sender can be discovered."]},
        "marketing.email_campaign_analytics": {
            "verdict": "reject_as_is",
            "good": "Explicitly says the manifest checks posting destination rather than report quality or prior-month format.",
            "defects": ["Copies ch_email and asks for a request record whose public request contains no such record or target.",
                        "Requires retrieving prior Slack history/read-back although only channel listing and sending are available. Prior report might be in email, but no supported route is established here."]},
        "sales.soft_matching": {
            "verdict": "reject_as_is",
            "good": "Keeps phone-only update, title preservation, non-target exclusion and note-parent association separate; does not print the target ContactId.",
            "defects": ["Copies private expected note title 'Phone Update Processed' as a required literal instead of grounding it in retrieved policy.",
                        "Does not establish how to retrieve contact matching guidelines; says to compare against guidelines while admitting they are unknown.",
                        "Proposes inspecting changed_fields/update effects not exposed as an ordinary agent read tool; runtime audit evidence is distinct from solver observation."]},
        "hr.recruitment_dedup_alert": {
            "verdict": "reject_as_is",
            "good": "Retains candidate-to-row identity, assigned recruiter and verbatim Name/Email/Role association.",
            "defects": ["Incorrectly says the two duplicate emails are given in the public request; they occur only in the privileged check.",
                        "Suggests a separate email for each target despite admitting a combined message could satisfy multiple per-candidate checks.",
                        "Recommends sent-message read-back even though no Gmail read tool is supplied."]},
    }
    counts = Counter()
    for row in rows:
        counts["tasks_with_private_identifier_copies"] += bool(row["private_identifier_copies"])
        counts["tasks_with_tool_like_names_needing_review"] += bool(row["tool_like_names_not_in_schema"])
    report = {"batch": root.name, "tasks": len(rows), "actual_cost_usd": summary["actual_cost_usd"],
              "cost_coverage": summary["billed_cost_coverage"], "counts": dict(counts),
              "token_usage": {key: sum(r["usage"].get(key, 0) for r in rows)
                              for key in ["prompt_tokens", "completion_tokens", "total_tokens"]},
              "manual_source_reviews": reviews, "rows": rows,
              "conclusion": "32 contexts generated; all structurally valid; batch not qualified for teacher training. Five inspected contexts have substantive defects. No solver performance measured.",
              "limitations": "Identifier/tool-like flags are conservative partial checks, not complete semantic review. Tool-like strings may be manifest aliases rather than invented tools. Unflagged tasks are not passes. Costs are API-reported inference charges; taxes/credit purchase fees are not included. CostLedger uses conservative uncached token pricing and differs from cache-adjusted API billing."}
    (root / "source-review.json").write_text(json.dumps(report, indent=2) + "\n")
    esc = html.escape
    sections = []
    for inp, row in zip(inputs, rows, strict=True):
        context = json.loads((root / inp["task"] / "context.json").read_text())
        review = reviews.get(inp["task"], {"verdict": "not_manually_qualified"})
        sections.append(f'<section id="{esc(inp["task"])}"><h2>{esc(inp["task"])}</h2>'
                        f'<p>Cost ${row["usage"]["cost"]:.6f} · {row["checks"]} checks · {row["word_count"]} words</p>'
                        f'<h3>Source review</h3><pre>{esc(json.dumps(review, indent=2))}</pre>'
                        f'<h3>Automatic flags</h3><pre>{esc(json.dumps({k:row[k] for k in ["private_identifier_copies", "tool_like_names_not_in_schema"]}, indent=2))}</pre>'
                        f'<h3>Original request</h3><p>{esc(inp["public_request"])}</p>'
                        f'<details><summary>Available tools</summary><pre>{esc(json.dumps([t["function"]["name"] for t in inp["available_tools"]], indent=2))}</pre></details>'
                        f'<h3>Actual generated context</h3><pre>{esc(json.dumps(context, indent=2, ensure_ascii=False))}</pre></section>')
    navigation = " · ".join(f'<a href="#{esc(x["task"])}">{esc(x["task"])}</a>' for x in inputs)
    page = ('<!doctype html><meta charset="utf-8"><title>DeepSeek teacher contexts: 32-task review</title>'
            '<style>body{max-width:1100px;margin:40px auto;padding:0 24px;font:16px/1.55 system-ui;color:#192535;background:#f7f9fc}'
            'pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#fff;padding:18px;border:1px solid #dce3ee;border-radius:8px}'
            'section{margin:50px 0;border-top:2px solid #c9d3e0}a{color:#195ac4}h1{line-height:1.15}.warn{background:#fff0d3;padding:18px}</style>'
            '<h1>DeepSeek V4.1 Flash 1017 — 32 teacher contexts</h1>'
            f'<p>32 tasks across seven categories. Actual inference cost: <b>${summary["actual_cost_usd"]}</b>. '
            'No Luna executions or initial world state sent. Zero training updates and no solver evaluation.</p>'
            '<p class="warn"><b>Not qualified for training.</b> All32 pass structural checks, but five source-reviewed examples contain substantive defects. '
            'Private identifiers, inaccessible retrieval and public/private confusion remain. Unreviewed contexts are not passes.</p>'
            f'<details><summary>All32 tasks</summary><p>{navigation}</p></details>' + "".join(sections))
    (root / "context-review.html").write_text(page)
    print(json.dumps({k: v for k, v in report.items() if k not in ["rows", "manual_source_reviews"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    audit(parser.parse_args().directory)
