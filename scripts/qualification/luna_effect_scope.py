"""Expose exact effect scope to generation; does not change native evaluation."""

import json
import re

SCOPE_RULES = """The effect-scope contracts identify what each check actually
requires or forbids. A generic source kind is an event classification, not a
service operation: use its collection and predicate. Prohibiting an update of an
existing record does NOT prohibit a requested creation of a separate record.
Do not replace unconditional requested actions with 'only if not found', 'stop
if found', or similar branches unless that condition is explicitly supported by
the public request or the check's actual predicate. Keep required creations and
preservation of old records as separate obligations. A historical example is not
authority to introduce new conditions. Cite exact check roots; preserve scope.
"""


def effect_scopes(contract):
    rows = []
    for check in contract["checks"]:
        source = contract.get("sources", {}).get(check.get("source"), {})
        rows.append({
            "check_id": check["check_id"], "role": check["role"], "operator": check["operator"],
            "source": check.get("source"), "source_adapter": source.get("adapter"),
            "effect_collection": source.get("collection"), "event_kind": source.get("kind"),
            "predicate": check.get("required_when", check.get("prohibited_when")),
            "scope_rule": "Only the matched effects in this collection are constrained; retain the full predicate.",
        })
    return rows


def private_email_domains(value, public):
    """Known source email domains, including those embedded in display strings."""
    domains = set(re.findall(r"@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", json.dumps(value)))
    return sorted(d[1:] for d in domains if d[1:] not in public)
