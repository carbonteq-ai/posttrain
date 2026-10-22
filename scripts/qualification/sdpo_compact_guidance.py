"""Deterministic teacher constraint sheets; never generate an action procedure."""

import copy
import hashlib
import json
import re

VERSION = "compact-constraints-v4"
SUPPORTED = {"effects.required_when@1", "effects.prohibited_when@1",
             "sheets.retained_when@1", "records.retained_when@1",
             "objects.created_and_retained@1", "record.fields_equal@1"}


def render(value):
    """Small expression renderer; unsupported constructs remain explicit JSON."""
    if not isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if "field" in value:
        return ".".join(str(p) if isinstance(p, int) or str(p).isidentifier()
                        else json.dumps(p) for p in value["field"])
    if "observed_value" in value:
        return value["observed_value"]
    if "literal" in value:
        return json.dumps(value["literal"], ensure_ascii=False)
    op = value.get("op")
    if op in ("all", "any"):
        return "(" + (" AND " if op == "all" else " OR ").join(render(v) for v in value["args"]) + ")"
    if op == "not":
        return "NOT(" + render(value["arg"]) + ")"
    if op in ("eq", "ne", "in", "gt", "ge", "lt", "le"):
        if (op in ("eq", "ne") and value["left"] == value["right"]
                and any(k in value["left"] for k in ("literal", "observed_value"))):
            return "true" if op == "eq" else "false"
        symbol = {"eq": "=", "ne": "!=", "in": "IN", "gt": ">", "ge": ">=", "lt": "<", "le": "<="}[op]
        return f"{render(value['left'])} {symbol} {render(value['right'])}"
    if op == "mentions":
        return (f"{render(value['text'])} CONTAINS[{value.get('mode', 'words')}] "
                f"{render(value['value'])} EXCLUDING {json.dumps(value.get('excluding', []))}")
    if op == "mentions_together":
        terms = [f"{render(v['value'])}[{v.get('mode', 'words')};excluding={json.dumps(v.get('excluding', []))}]"
                 for v in value["terms"]]
        return f"{render(value['text'])} CONTAINS_TOGETHER[{value.get('scope')}] " + ", ".join(terms)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def compile_guidance(material):
    """Preserve every check in a compact IR, with private constants out of band.

    This is a contract representation, not proof that a hidden selection value
    is derivable. Unresolved bindings explicitly require observed evidence.
    """
    manifest = material["goal_manifest"]
    checks = [c for c in manifest["checks"] if c["role"] in ("goal", "harm")]
    if not checks or len({c["check_id"] for c in checks}) != len(checks):
        raise ValueError("missing or duplicate requirements")
    sources = manifest["sources"]
    public = material["user_request"]
    fields = set()

    def collect_fields(value):
        if isinstance(value, dict):
            if value.get("kind") == "field":
                fields.add(str(value["path"][-1]))
            for child in value.values():
                collect_fields(child)
        elif isinstance(value, list):
            for child in value:
                collect_fields(child)

    collect_fields(checks)
    private, value_keys, source_keys = {}, {}, {}
    used_sources = []

    def source_ref(name):
        if name not in sources:
            raise ValueError(f"unknown source: {name}")
        if name not in source_keys:
            source_keys[name] = f"S{len(source_keys) + 1}"
            used_sources.append(name)
        return source_keys[name]

    def literal(value):
        # Only public text, exact field names and protocol truth are executable
        # literals. Arbitrary private IDs/answers stay in the audit sidecar.
        if value is None or isinstance(value, bool):
            return {"literal": value}
        if isinstance(value, str) and value and (value.casefold() in public.casefold()
                or value in fields or value == "matched"):
            return {"literal": value}
        key = json.dumps(value, sort_keys=True)
        if key not in value_keys:
            ref = f"V{len(value_keys) + 1}"
            value_keys[key] = ref
            private[ref] = copy.deepcopy(value)
        return {"observed_value": value_keys[key]}

    def tree(value, population=None):
        if isinstance(value, list):
            return [tree(v, population) for v in value]
        if not isinstance(value, dict):
            return value
        if value.get("kind") == "literal":
            # Pinned contract dumps with exclude_none=True omit literal null's
            # value. This is distinct from a source field's copy_from binding.
            return literal(value.get("value"))
        if value.get("kind") == "field":
            path = value["path"]
            root = path[0]
            if root == "request":
                if population and sources[population]["adapter"] == "public.request@1":
                    declaration = sources[population]
                    if path[1:] == ["request_key"]:
                        return literal(declaration["member_key"])
                    binding = declaration.get("fields", {}).get(path[1]) if len(path) == 2 else None
                    if isinstance(binding, dict) and binding.get("copy_from"):
                        # Native schema dumps include value:null alongside a
                        # copy_from binding. Null is not the selected value.
                        resolved = {"task_evidence": {"initial": material["native_data"]["initial_state"],
                                                       "prompt": material["native_data"].get("prompt", [])}}
                        try:
                            for part in binding["copy_from"]:
                                resolved = resolved[part]
                        except (KeyError, IndexError, TypeError) as error:
                            raise ValueError("unresolved native copy_from authority") from error
                        return literal(resolved)
                    if isinstance(binding, dict) and "value" in binding:
                        return literal(binding["value"])
                root = "row" if population else "task_context"
            elif root == "candidate":
                root = "same_row_identity"
            return {"field": [root, *path[1:]]}
        return {k: (source_ref(v) if k == "source" and isinstance(v, str) else tree(v, population))
                for k, v in value.items()}

    constraints, audit_checks = [], []
    for check in checks:
        if check["operator"] not in SUPPORTED:
            raise ValueError(f"unsupported operator: {check['operator']}")
        population = check.get("population")
        if population:
            source_ref(population)
        source_ref(check["source"])
        ref = f"C{len(constraints) + 1}"
        semantics = {
            "effects.required_when@1": "Require an actual matching executed effect for each eligible source row.",
            "effects.prohibited_when@1": "Prohibit only matching executed effects where the declared condition applies.",
            "sheets.retained_when@1": "Require the corresponding sheet row to satisfy retained_when in final state; a successful write response alone is insufficient.",
            "records.retained_when@1": "Require the same native record identity to satisfy retained_when in final state, preserving declared unrelated fields and counts.",
            "objects.created_and_retained@1": "Require an object actually created by this attempt and still retained with the declared fields. Pre-existing matching objects alone do not satisfy creation.",
            "record.fields_equal@1": "Require the declared target record fields to equal expected values using each declared comparison mode. Other fields are not new write requirements.",
        }
        compiled = {"id": ref, "mode": "REQUIRE" if check["role"] == "goal" else "FORBID",
            "operator": check["operator"], "semantics": semantics[check["operator"]],
            "each_row_from": source_ref(population) if population else None,
            "operation_from": source_ref(check["source"]),
            "when": tree(check.get("required_when", check.get("prohibited_when", True)), population),
            "match": tree(check.get("effect_match", True), population)}
        # Preserve semantic modifiers/joins verbatim structurally; never silently
        # discard an unfamiliar condition. Metadata lives only in the sidecar.
        omitted = {"check_id", "signal_id", "role", "operator", "population", "source",
                   "required_when", "prohibited_when", "effect_match", "max_instances"}
        extra = {k: tree(v, population) for k, v in check.items() if k not in omitted}
        if check["operator"] == "record.fields_equal@1":
            extra["expected"] = [{**{k: v for k, v in expected.items() if k != "value"},
                                  "value": literal(expected["value"])} for expected in check["expected"]]
        if extra:
            compiled["constraints"] = extra
        visible = {k: v for k, v in compiled.items() if k != "constraints"}
        visible.update(when=render(compiled["when"]), match=render(compiled["match"]))
        if extra:
            modifiers = {k: v for k, v in extra.items() if v not in ([], {})}
            for key in ("effect_joins", "operational_dependencies"):
                if key in modifiers:
                    modifiers[key] = [{k: render(v) if k == "where" else v for k, v in join.items()}
                                      for join in modifiers[key]]
            visible["constraints"] = modifiers
        constraints.append(visible)
        audit_checks.append({"id": ref, "check_id": check["check_id"], "raw": copy.deepcopy(check), "compiled_ir": compiled})

    # Follow aliases added by joins. No initial data or concrete source IDs enter
    # the sheet. Exact adapter declarations remain in the private audit.
    source_rows, source_audit = {}, {}
    for name in used_sources:
        source = sources[name]
        ref = source_keys[name]
        path = source.get("path", [])
        service = source.get("service") or (path[2] if path[:2] == ["task_evidence", "initial"] and len(path) > 2 else None)
        source_rows[ref] = {"kind": source["adapter"].split("@")[0],
            "service": service, "operation": source.get("kind"),
            "records": source.get("collection") or (path[3:] if len(path) > 3 else None),
            "fields": source.get("required_fields", list(source.get("fields", {}))),
            "identity": source.get("identity_path", source.get("key_fields", []))}
        source_rows[ref] = {k: v for k, v in source_rows[ref].items() if v not in (None, [], {})}
        if source["adapter"] == "public.request@1":
            source_rows[ref] = {"kind": "single public task; bindings already resolved in constraints"}
        source_audit[ref] = {"name": name, "raw": copy.deepcopy(source)}

    reads = []
    for action in material["reference_actions"]:
        tool = action["call"]["name"]
        if any(word in tool for word in ("find", "get", "search", "query", "list")) and tool not in reads:
            reads.append(tool)
    value_origins = {}

    def locate(value, target, path=()):
        found = []
        if isinstance(value, dict):
            for key, child in value.items():
                found.extend(locate(child, target, (*path, key)))
        elif isinstance(value, list):
            for child in value:
                found.extend(locate(child, target, (*path, "*")))
        else:
            matches = value == target or (isinstance(value, str) and isinstance(target, str)
                                          and len(target) >= 4 and target in value)
            if not matches and isinstance(value, str) and isinstance(target, str):
                digits = re.sub(r"\D", "", target)
                if 7 <= len(digits) <= 15 and re.fullmatch(r"[+\d(). -]+", target):
                    pattern = r"(?<!\d)" + r"[(). +\-]*".join(digits) + r"(?!\d)"
                    matches = re.search(pattern, value) is not None
            if matches:
                found.append(".".join(map(str, path)))
        return found

    visible_json = json.dumps(constraints)
    for ref, value in private.items():
        if not re.search(r"\b" + ref + r"\b", visible_json):
            continue
        # Locations are schema paths, not source values or selected record indices.
        values = value if isinstance(value, list) else [value]
        paths = sorted({p for v in values for p in locate(material["native_data"]["initial_state"], v)})
        value_origins[ref] = paths or ["No exact initial-source location found; derive from actual public task/policy observations or leave unresolved."]
    tools = {t["function"]["name"]: t["function"] for t in material["tool_definitions"]}
    prerequisites = ["No S/V symbol is an executable argument. A constraint describes the observed result; "
                     "a tool may accept a public name and return its resolved ID.",
                     "If a retrieval is empty, broaden using public task words; do not invent sender addresses or domains."]
    if "google_sheets_get_spreadsheet_by_id" in tools and "google_drive_find_multiple_files" in tools:
        prerequisites.append("SHEETS: unless the public task supplies the spreadsheet ID, first discover the file "
            "with google_drive_find_multiple_files using task-topic words. Then pass its returned id to "
            "google_sheets_get_spreadsheet_by_id. Inspect returned worksheets/headers before reading rows. "
            "Use returned worksheet IDs with row tools. Never guess file/worksheet identifiers. "
            "A display label 'Title (id)' is not an ID; use only the returned bare id, not the whole label. "
            "Do not retry an unchanged failed identifier.")
    slack = tools.get("slack_send_channel_message", {})
    public_channels = re.findall(r"#[A-Za-z0-9_-]+", public)
    if public_channels and "Channel ID or name" in slack.get("description", ""):
        prerequisites.append("SLACK: this tool accepts channel ID or name. The public task supplies "
            + json.dumps(public_channels) + "; use the exact supplied #name, or an ID returned by an available "
            "lookup. Do not remove # and assume the bare name is an ID. Read the actual send result.")
    sheet = {"version": VERSION,
        "interpretation": "row means one record from each_row_from; same_row_identity is that record's native identity, NOT an email/API request ID. A FORBID constrains only its matched operation WHEN its condition applies. Read joins before dependent writes. Never invent extra actions.",
        "evidence": "V references are withheld instance values, not literal strings or tool arguments. Resolve them from current observations and the public task/policy. If ambiguous, retrieve evidence; do not guess. Source declarations are roles, not executed reads. Public IDs may be used directly.",
        "retrieval_prerequisites": prerequisites,
        "sources": source_rows, "constraints": constraints, "retrieve_values_from": value_origins,
        "historical_read_tools": reads,
        "read_scope": "Historical read-tool names indicate possible retrieval routes only; no historical arguments, outputs, chronology or successful-answer claims supplied."}
    return {"sheet": sheet, "audit": {"version": VERSION, "task": material["task"],
        "manifest_sha256": digest(manifest), "material_sha256": digest(material),
        "private_values": private, "sources": source_audit, "checks": audit_checks,
        "check_coverage": [c["check_id"] for c in checks], "sheet_sha256": digest(sheet),
        "qualification": "CPU compiled only; semantic interpretation/native teacher behavior not yet qualified"}}


def instructions(compiled):
    """Existing teacher interface consumes one compact JSON sheet."""
    return [json.dumps(compiled["sheet"], ensure_ascii=False, separators=(",", ":"))]
