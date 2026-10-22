"""Schema-first API options and value-free SOQL structure for guidance prep."""

import json
import re
from datetime import datetime


def public_equivalence(value, public):
    """Recognize public date normalization/search composition, not hidden IDs."""
    if not isinstance(value, str):
        return None
    if value.strip('"').lower() in public.lower():
        return "public_text"
    if value.startswith(("{", "[")):
        try:
            parsed_json = json.loads(value)
        except ValueError:
            parsed_json = None

        def public_json(item):
            if isinstance(item, dict):
                return bool(item) and all(str(k).lower() in public.lower() and public_json(v)
                                          for k, v in item.items())
            if isinstance(item, list):
                return bool(item) and all(public_json(v) for v in item)
            return public_equivalence(str(item), public) is not None

        if parsed_json is not None and public_json(parsed_json):
            return "public_json_composition"
    months = "January|February|March|April|May|June|July|August|September|October|November|December"
    for match in re.finditer(rf"\b({months})\s+(\d{{1,2}}),?\s+(\d{{4}})\b", public, re.I):
        parsed = datetime.strptime(f"{match[1]} {match[2]} {match[3]}", "%B %d %Y")
        if value == parsed.date().isoformat():
            return "public_date_normalization"
    words = re.findall(r"[a-z0-9]+", value.lower())
    if (not any(c in value for c in "@/_") and len(words) >= 2
            and public_query(value, public)):
        return "public_word_search_composition"
    return None


def public_query(value, public):
    if not isinstance(value, str) or not value.strip():
        return False
    words = set(re.findall(r"[a-z0-9]+", value.lower()))
    public_words = set(re.findall(r"[a-z0-9]+", public.lower()))
    syntax = {"from", "to", "subject", "has", "in", "label", "and", "or", "not"}
    return bool(words - syntax) and words <= public_words | syntax


def query_structure(value):
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"\s*SELECT\s+([\w*,\s]+)\s+FROM\s+(\w+)\s+WHERE\s+(.+?)\s*", value, re.I)
    if not match:
        return None
    fields, table, where = match.groups()
    predicate = r'''(\w+)\s*(>=|<=|!=|=|>|<|LIKE)\s*(?:'[^']*'|"[^"]*"|\d+(?:\.\d+)?)'''
    parts = list(re.finditer(predicate, where, re.I))
    remainder = re.sub(predicate, "", where, flags=re.I)
    if not parts or re.sub(r"\b(?:AND|OR|NOT)\b|[\s()]", "", remainder, flags=re.I):
        return None
    return {"select": [f.strip() for f in fields.split(",")], "object": table,
            "predicates": [{"field": m[1], "operator": m[2].upper(),
                            "value": "derive from public request or observed tool results"} for m in parts],
            "boolean_structure": re.sub(predicate, "predicate", where, flags=re.I),
            "reference_literals_supplied": False}


def schema_role(material, tool_name, field, value):
    definitions = {t["function"]["name"]: t["function"] for t in material["tool_definitions"]}
    properties = definitions.get(tool_name, {}).get("parameters", {}).get("properties", {})
    enum = properties.get(field, {}).get("enum", [])
    if (field in {"object_type", "object", "searchField", "searchResults", "format", "max_results"}
            or any(type(value) is type(v) and value == v for v in enum)):
        return {"origin": "demonstrated_api_option", "value": value,
                "note": "API option precedes accidental substring overlap with observed content; not an email fact."}
    structure = query_structure(value) if field == "query" else None
    if structure:
        return {"origin": "demonstrated_query_structure", "structure": structure,
                "note": "Preserve fields/operators; derive query literals from public request or current observations."}
    if field == "query" and public_query(value, material["user_request"]):
        return {"origin": "public_request_search_composition", "value": value,
                "note": "All search terms occur in the public request; only query syntax was added."}
    return None
