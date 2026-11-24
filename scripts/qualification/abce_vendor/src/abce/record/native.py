"""Native results -> Items, without probing (docs/design.md §2 step 1, polarity and classification).

No per-type table: the effect noun comes from the type-name prefix, polarity from the rubric's
negative marker (with a name heuristic as a cross-check), and parameter keys are sorted into
address / content / cardinality / control by one fixed key table. Keys the table does not know
are kept as `unclassified_keys` and counted by the coverage report.
"""

from __future__ import annotations

import re
from typing import Any

from .model import EffectRef, Item, Location, Value
from .observed import Corpus, value_forms

APPS = (
    "google_sheets", "google_calendar", "google_ads", "google_drive", "facebook_page", "facebook_lead_ads",
    "facebook_conversions", "linkedin_ads", "linkedin_conversions", "linkedin_leadgen_forms", "zoho_desk",
    "salesforce", "gmail", "slack", "hubspot", "jira", "asana", "docusign", "zoom", "twilio", "calendly",
    "mailchimp", "intercom", "freshdesk", "zendesk", "gorgias", "reamaze", "helpscout", "helpcrunch", "hiver",
    "airtable", "trello", "monday", "notion", "confluence", "buffer", "quickbooks", "xero", "wave", "chatgpt",
    "twitter", "instagram", "canva", "pipefy", "recruitee", "basecamp3", "bamboohr", "linkedin",
)

OBJECTS = (
    "draft_reply", "draft", "email", "message", "label", "row", "cell", "event_type", "event", "invitee",
    "export_job", "completion", "conversation", "cc", "envelope", "workspace", "photo", "post", "contact",
    "ticket", "campaign_member", "campaign", "conversion", "customer_list", "customer", "action", "company_post",
    "company", "invitation", "subscriber", "invoice", "sales_receipt", "bill", "payment", "purchase_order",
    "organization", "user", "account", "meeting", "registrant", "channel", "direct_message", "dm", "sms",
    "follow", "reply", "tweet", "media", "issue", "case", "lead", "note", "opportunity", "record", "task",
    "collection", "deal", "engagement", "product",
)

ADDRESS_KEYS = {
    "to", "cc", "bcc", "recipient", "recipients", "email", "to_number", "to_email", "channel", "channel_id",
    "channel_name", "user_id", "recipient_id", "spreadsheet_id", "spreadsheet", "worksheet_id", "worksheet",
    "row_id", "ticket_id", "conversation_id", "contact_id", "record_id", "parent_id", "message_id", "thread_id",
    "label_id", "label", "campaign_id", "envelope_id", "meeting_id", "profile_id", "profile_url",
    "recipient_profile_url", "signer_email", "recipient_email", "member_email", "customer_id", "invoice_number",
    "organization_id", "account_id", "related_to_id", "what_id", "assigned_to_id", "host_id", "event_id",
    "event_uri", "invitee_id", "design_id", "gclid", "page_id", "tweet_id", "in_reply_to_tweet_id", "username",
    "company_id", "list_id", "mailbox_id", "group_id", "project", "collection", "object_type", "applicationId",
    "tableName", "calendarid", "lead_id", "id", "field", "column", "property", "template_id", "action_key",
    "customer_name", "display_name", "contact_name", "lookup_column", "lookup_value", "target_column",
    "original_id", "sender_type", "author_type", "employee_id", "candidate_id", "issue_key", "board_id",
    "list", "card_id", "task_id", "deal_id", "opportunity_id", "case_id", "file_id", "folder_id",
    "purchase_order_number", "doc_number", "event_type_name", "template_name", "meeting_topic", "assignee_email",
    "to_contains", "worksheet_name", "calendar_id", "who_id",
}
CONTENT_KEYS = {
    "body_contains", "body_not_contains", "text_contains", "text_not_contains", "subject", "subject_contains",
    "cells", "cell_contains", "contains", "value", "value_contains", "expected_value", "content_contains",
    "summary", "summary_contains", "topic_contains", "title_contains", "caption", "caption_contains",
    "description_contains", "message_contains", "note_contains", "tag", "status", "priority", "stage", "text",
    "params", "fields", "criteria", "reason_contains", "phone", "order", "role", "event_name", "dealname",
    "dealstage", "name", "name_contains", "first_name", "last_name", "company", "domain", "issuetype",
    "settings_waiting_room", "start__dateTime", "start", "attendees", "attendee_email", "has_message",
    "due_date", "origin", "total", "topic", "prompt_contains", "additional_checks", "expected", "title", "label_name",
    "event_date", "category", "row_contains",
}
CARDINALITY_KEYS = {"count", "min_count"}
# Address keys that name the object itself (a row, a ticket, a contact), as opposed to where a new
# object goes (a channel, a spreadsheet, a recipient): a check with one of these is about an existing
# record, so "the attempt created no X" is never said of it.
OBJECT_ID_KEYS = {
    "row_id", "ticket_id", "conversation_id", "contact_id", "record_id", "native_record_id", "message_id", "thread_id",
    "envelope_id", "meeting_id", "invoice_number", "customer_id", "event_id", "event_uri", "invitee_id", "tweet_id",
    "in_reply_to_tweet_id", "lead_id", "id", "issue_key", "card_id", "task_id", "deal_id", "opportunity_id", "case_id",
    "file_id", "folder_id", "employee_id", "candidate_id", "purchase_order_number", "doc_number", "original_id",
    "design_id", "parent_id", "related_to_id", "what_id", "who_id", "account_id", "organization_id", "company_id",
    "profile_id", "profile_url", "recipient_profile_url", "campaign_id", "page_id", "gclid", "customer_name",
    "contact_name", "display_name", "label_id", "label",
}
# Keys whose values name *where* (a field, column, property, action) rather than *what*; they are
# schema names the request asks for and are treated as known to the attempt.
# A label's name is task data the attempt may never have seen (the label it was to create): not a schema name.
SCHEMA_KEYS = {"field", "property", "column", "action_key", "collection", "object_type", "lookup_column",
               "target_column", "worksheet", "worksheet_name", "tableName"}
CONTROL_KEYS = {"scored", "excluded"}

_NEGATIVE_NAME = re.compile(r"(^|_)(not|missing|never)(_|$)")


def split_type(type_name: str) -> tuple[str, str, str]:
    """(app, object, predicate) from a native type name; object may be '' when the name has none."""
    app = next((a for a in APPS if type_name.startswith(a + "_")), type_name.split("_", 1)[0])
    rest = type_name[len(app) + 1:] if type_name.startswith(app + "_") else type_name
    obj = next((o for o in OBJECTS if rest == o or rest.startswith(o + "_")), "")
    predicate = rest[len(obj) + 1:] if obj and rest != obj else (rest if not obj else "")
    return app, obj, predicate


def effect_kind(predicate: str, keys: set[str]) -> str:
    p = predicate
    if "count" in p:
        return "count"
    if "action" in p or "action_key" in keys:
        return "action"
    if any(w in p for w in ("sent", "posted", "published", "in_channel", "forwarded")):
        return "send"
    if any(w in p for w in ("field", "property", "cell_equals", "cell_not_equals", "equals", "changed")):
        return "field"
    if "updated" in p:
        return "update"
    if any(w in p for w in ("has_", "missing", "is_", "paused", "enabled", "followed", "liked", "archived",
                              "voided", "cancelled", "read", "no_show", "in_list", "from_template")):
        return "has"
    if "exists" in p or "created" in p:
        return "exist"
    if "contains" in p:
        return "content"
    return "other"


def negative_marker(type_name: str) -> bool | None:
    try:
        import automationbench.rubric  # noqa: F401  (registers handlers)
        from automationbench.rubric.registry import AssertionRegistry
    except ImportError:
        return None
    if type_name not in AssertionRegistry._handlers:
        return None
    return bool(AssertionRegistry.is_negative(type_name))


def classify_params(params: dict[str, Any], corpus: Corpus | None) -> tuple[dict, dict, dict | None, dict, tuple[str, ...]]:
    address, content, control = {}, {}, {}
    cardinality: dict | None = None
    unclassified = []

    def value(v: Any, key: str = "") -> Value:
        if key in SCHEMA_KEYS:
            return Value(raw=v, observed=True, written=False, forms=value_forms(v))
        return Value(raw=v, observed=corpus.is_observed(v) if corpus else False,
                     written=corpus.is_written(v) if corpus else False, forms=value_forms(v),
                     hidden_elements=corpus.hidden_elements(v) if corpus else ())

    for key, raw in params.items():
        if key in CONTROL_KEYS:
            control[key] = raw
        elif key in CARDINALITY_KEYS:
            cardinality = {**(cardinality or {}), key: raw}
        elif key in ADDRESS_KEYS:
            address[key] = value(raw, key)
        elif key in CONTENT_KEYS:
            content[key] = value(raw, key)
        else:
            unclassified.append(key)
            content[key] = value(raw, key)
    return address, content, cardinality, control, tuple(unclassified)


def native_item(index: int, result: dict, corpus: Corpus | None, initial_pass: bool | None = None) -> Item:
    """One Item from a recorded native result {type, passed, excluded, params}."""
    type_name = result["type"]
    params = result.get("params") or {}
    app, obj, predicate = split_type(type_name)
    marker = negative_marker(type_name)
    by_name = bool(_NEGATIVE_NAME.search(predicate)) if predicate else bool(_NEGATIVE_NAME.search(type_name))
    semantics: dict[str, Any] = {}
    if marker is None:
        negative, source = by_name, "name"
    else:
        negative, source = marker, "registry_marker"
        if marker != by_name:
            semantics["marker_name_disagreement"] = {"marker": marker, "name": by_name}
    if result.get("excluded"):
        outcome = "excluded"
    else:
        outcome = "passed" if result.get("passed") else "failed"
    polarity = "forbid" if negative else "require"
    if initial_pass is not None:
        semantics["initial_pass"] = initial_pass
    # A positive check that passed on the initial world and failed at the end is a record the attempt
    # broke: the world had to stay as it was ("keep"). A negative check that passed initially and
    # failed is simply a forbidden effect that was produced, so it stays "forbid".
    if initial_pass is True and outcome == "failed" and not negative:
        polarity = "keep"
        source = "initial_pass"
    address, content, cardinality, control, unclassified = classify_params(params, corpus)
    semantics["params"] = params  # the recorded parameters verbatim (raw render reproduces them)
    semantics["passed"] = bool(result.get("passed"))  # excluded checks keep their verdict too
    kind = effect_kind(predicate or type_name, set(params))
    return Item(
        id=f"native:{index}", origin="native", type=type_name, outcome=outcome, polarity=polarity,
        polarity_source=source, effect=EffectRef(app=app, object=obj or app, kind=kind), address=address,
        content=content, cardinality=cardinality, control=control, unclassified_keys=unclassified,
        location=Location(kind="unknown"), semantics=semantics, evidence=(f"native:{index}",),
    )
