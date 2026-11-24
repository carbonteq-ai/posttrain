"""Manifest (CarbonTeq contract) compiler: checks -> factored specs; findings -> Items.

The compiler is generic over the contract language (docs/survey.md §3): a check is
(population filter) -> (effect must / must not match). Identity conjuncts on `effect.*` become
`address` (every one of them: a second conjunct on the same field is `field#1`, so one message to
the employee *and* the manager is never rendered as a message to the manager), `mentions*`
conjuncts become `content`, and anything else is kept as `residual` operator names so the
renderer can say "a further condition" instead of inventing one. Effect nouns come from the
source adapter (24 adapter kinds, one table below).

Findings come from the environment's assessment receipts (guard, obligation, record evaluation);
their `occurrence` is a tool-server invocation id, which the Record maps to a call.
"""

from __future__ import annotations

import functools
import json
from typing import Any

from .model import CallRef, EffectRef, Item, Location, Value
from .observed import Corpus, value_forms

GUARD_OUTPUT = "automationbench.manifest_guard_result@1"
OBLIGATION_OUTPUT = "automationbench.manifest_obligation_result@1"
RECORD_OUTPUT = "automationbench.manifest_evaluation@1"

EFFECT_IDENTITY_FIELDS = {
    "recipients", "recipient", "to", "cc", "bcc", "channel_id", "channel", "channel_name", "recipient_user_id",
    "user_id", "record_id", "native_record_id", "row_id", "spreadsheet_id", "worksheet_id", "id", "issue_key",
    "ticket_id", "conversation_id", "contact_id", "message_id", "thread_id", "parent_id", "envelope_id",
    "meeting_id", "profile_id", "service", "collection", "tool_name", "object_type", "key",
}

POPULATION_ADAPTERS = {"public.request@1", "google_sheets.rows@1", "initial.records@1"}


def _singular(word: str) -> str:
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith("ses") or word.endswith("xes"):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def source_effect(source: dict) -> EffectRef:
    """(app, object, kind) of what a source observes; the only adapter table in the engine."""
    adapter = source.get("adapter") or ""
    kind = source.get("kind")
    name = adapter.split("@")[0]
    app, _, rest = name.partition(".")
    if adapter == "gmail.messages@1":
        return EffectRef(app="gmail", object="message", kind="send")
    if adapter == "slack.messages@1":
        return EffectRef(app="slack", object="dm" if kind == "direct_message" else "message", kind="send")
    if adapter == "google_sheets.row_writes@1":
        return EffectRef(app="google_sheets", object="row", kind="create" if kind == "append" else (kind or "update"))
    if adapter == "service.record_writes@1":
        collection = source.get("collection") or []
        obj = _singular(str(collection[-1])) if collection else "record"
        return EffectRef(app=str(source.get("service") or "service"), object=obj, kind=kind or "update")
    if rest.endswith("reads") or (kind or "").startswith("read"):
        obj = {"message_reads": "message", "reads": "sheet", "subscriber_reads": "subscriber", "list_reads": "list",
               "channel_reads": "channel", "user_reads": "user"}.get(rest, _singular(rest.replace("_reads", "")) or "record")
        return EffectRef(app=app, object=obj, kind="read")
    if adapter == "salesforce.record@1":
        return EffectRef(app="salesforce", object=str(source.get("object_type") or "record").lower(), kind="field")
    if adapter == "jira.issues@1":
        return EffectRef(app="jira", object="issue", kind="exist")
    if adapter == "hubspot.objects@1":
        return EffectRef(app="hubspot", object=_singular(str(source.get("collection") or "object")), kind="exist")
    if adapter in ("final.records@1", "final.created_records@1"):
        path = source.get("path") or []
        service = source.get("service") or (path[2] if len(path) > 2 else "records")
        coll = source.get("collection") or (path[3] if len(path) > 3 else "record")
        return EffectRef(app=str(service), object=_singular(str(coll)), kind="exist")
    if adapter == "asana.actions@1":
        return EffectRef(app="asana", object="task", kind=kind or "create")
    if adapter == "zendesk.ticket_updates@1":
        return EffectRef(app="zendesk", object="ticket", kind="update")
    if adapter == "assistant.outputs@1":
        return EffectRef(app="assistant", object="reply", kind="output")
    if adapter == "external.outputs@1":
        return EffectRef(app="external", object="text", kind="output")
    if adapter in POPULATION_ADAPTERS:
        return EffectRef(app=app, object="population", kind="population")
    return EffectRef(app=app or "unknown", object=rest or "record", kind=kind or "other")


# ---------------------------------------------------------------------------------------------
# Predicate compilation
# ---------------------------------------------------------------------------------------------
def _field_path(node: Any) -> tuple[str, ...] | None:
    if isinstance(node, dict) and node.get("kind") == "field" and isinstance(node.get("path"), list):
        return tuple(str(p) for p in node["path"])
    return None


def _operand(node: Any) -> dict[str, Any]:
    """A compact description of an operand: literal value or a reference."""
    if not isinstance(node, dict):
        return {"literal": node}
    kind = node.get("kind")
    if kind == "literal":
        return {"literal": node.get("value")}
    if kind == "field":
        return {"ref": ".".join(str(p) for p in node.get("path", []))}
    if kind == "input":
        return {"ref": ".".join(str(p) for p in node.get("path", [])), "format": node.get("format"),
                **({"literal": node["literal"]} if "literal" in node else {})}
    if kind == "derived":
        return {"derived": True}
    if kind == "conditional_number":
        return {"derived": True, "conditional": True}
    return {"opaque": kind}


PRECONDITION_PREFIXES = ("before", "before_cells")


def _new_out() -> dict:
    return {"address": {}, "content": [], "residual": [], "effect_residual": [], "preconditions": [], "groups": 0,
            "togethers": 0, "residual_nodes": []}


def _mentions_effect(node: Any) -> bool:
    """Whether a predicate subtree reads any `effect.*` field."""
    if isinstance(node, dict):
        if isinstance(node.get("path"), list) and node["path"][:1] == ["effect"]:
            return True  # a field or a typed input read from the effect
        return any(_mentions_effect(v) for v in node.values())
    if isinstance(node, list):
        return any(_mentions_effect(v) for v in node)
    return False


def _residual(out: dict, op: str, node: Any, negated: bool = False, group: int | None = None) -> None:
    out["residual"].append(op)
    if _mentions_effect(node):
        out["effect_residual"].append(op)
    # the raw subtree with its context (negation, the `any` it sits in): what a "why" clause may render
    out.setdefault("residual_nodes", []).append({"node": node, "negated": negated, "group": group,
                                                "effect": _mentions_effect(node)})


def add_address(address: dict, field: str, operand: dict) -> None:
    """Every identity conjunct is kept. A second conjunct on the same effect field (one message whose
    recipients include the employee *and* the manager) goes under `field#n`; it never replaces the
    first, which would render half the condition as if it were the whole of it."""
    existing = [k for k in address if k.split("#")[0] == field]
    if any(address[k] == operand for k in existing):
        return
    address[field if not existing else f"{field}#{len(existing)}"] = operand


def compile_predicate(node: Any, out: dict | None = None, negated: bool = False, group: int | None = None,
                      branch: Any = None) -> dict:
    """address: {effect field: operand}; content: [{field, value, mode, negated, group, branch}] where
    `group` numbers the `any` a conjunct belongs to (None = required outright) and `branch` the
    alternative of that `any` (the conjuncts of one branch must all hold: `any(all(A, B), C)` is "A and
    B, or C", never "A or B or C"); preconditions: conjuncts on the effect's `before` state (what the
    record was, never what the write produced); residual: operators the compiler cannot express as
    address or content (the renderer may only say "a further condition the task states" for them)."""
    out = out if out is not None else _new_out()
    if not isinstance(node, dict):
        return out
    op = node.get("op")
    if op == "all":
        for arg in node.get("args", []):
            compile_predicate(arg, out, negated, group, branch)
        return out
    if op == "not":
        for arg in node.get("args", []) or [node.get("arg")]:
            # not(any(A, B)) = neither A nor B: the negated conjuncts are required outright
            compile_predicate(arg, out, not negated, None if negated is False else group,
                              None if negated is False else branch)
        return out
    if op == "proven":
        compile_predicate(node.get("arg"), out, negated, group, branch)
        return out
    if op in ("eq", "in", "ne"):
        left, right = node.get("left"), node.get("right")
        lp, rp = _field_path(left), _field_path(right)
        effect_side = left if lp and lp[0] == "effect" else (right if rp and rp[0] == "effect" else None)
        other = right if effect_side is left else left
        if effect_side is not None:
            field = _field_path(effect_side)[1:]
            name = field[-1] if field else ""
            is_neg = negated != (op == "ne")
            if field and field[0] in PRECONDITION_PREFIXES:
                out["preconditions"].append({"field": ".".join(field), "value": _operand(other), "mode": op, "negated": is_neg})
            elif name in EFFECT_IDENTITY_FIELDS and not is_neg and group is None:
                add_address(out["address"], ".".join(field), _operand(other))
            else:
                out["content"].append({"field": ".".join(field), "value": _operand(other), "mode": op, "negated": is_neg,
                                       "group": group, "branch": branch})
            return out
        _residual(out, op, node, negated, group)
        return out
    if op in ("mentions", "mentions_together"):
        text = _field_path(node.get("text"))
        if not (text and text[0] == "effect"):
            _residual(out, op, node, negated, group)  # a mention in request data is a condition, not effect content
            return out
        field = ".".join(text[1:])
        values = node.get("values") or ([node.get("value")] if "value" in node else [])
        entries = [(v, node.get("mode") or op, negated, None) for v in values]
        together = None
        if node.get("terms") or node.get("excluding_values"):
            # mentions_together: every term on one readable line; excluding values: absent from that
            # same line (bound to the set, never a claim about the whole text)
            out["togethers"] += 1
            together = out["togethers"]
            entries = [(v, m, n, together) for v, m, n, _ in entries]
        for term in node.get("terms") or []:
            if isinstance(term, dict) and "value" in term:
                entries.append((term["value"], term.get("mode") or op, negated, together))
        excluded = []
        for term in node.get("excluding_values") or []:  # terms that must be absent from that line
            if isinstance(term, dict) and "value" in term:
                entries.append((term["value"], term.get("mode") or op, not negated, together))
                excluded.append(len(entries) - 1)
        for i, (value, mode, neg, tog) in enumerate(entries):
            out["content"].append({"field": field, "value": _operand(value), "mode": mode,
                                   "format": node.get("format"), "negated": neg, "group": group, "branch": branch,
                                   "together": tog, "scope": node.get("scope") or "line",  # where terms must appear together
                                   "excluded": i in excluded})  # absent from the unit the set's terms share
        return out
    if op == "any":
        # alternatives: the conjuncts of one branch share one group and one branch number (negated: see `not`)
        if negated:
            for arg in node.get("args", []) or []:
                compile_predicate(arg, out, negated, group, branch)
            return out
        out["groups"] += 1
        this = out["groups"]
        for index, arg in enumerate(node.get("args", []) or []):
            compile_predicate(arg, out, negated, this, index if branch is None else f"{branch}.{index}")
        return out
    if op in ("any_item", "all_items"):
        # item quantifiers over a list field with a `where` predicate on `item.*`: residual; the
        # children of list-shaped forms are compiled for their vocabulary only
        _residual(out, op, node, negated, group)
        for key in ("args", "items"):
            if isinstance(node.get(key), list):
                for arg in node[key]:
                    compile_predicate(arg, out, negated, group, branch)
        if isinstance(node.get("predicate"), dict):
            compile_predicate(node["predicate"], out, negated, group, branch)
        return out
    if op:
        _residual(out, op, node, negated, group)
    return out


# ---------------------------------------------------------------------------------------------
# Request population resolution
# ---------------------------------------------------------------------------------------------
def _resolve_path(path: list, task_initial_state: dict | None, prompt: list | None) -> Any:
    if not path or path[0] != "task_evidence":
        return None
    try:
        if path[1] == "initial":
            node: Any = task_initial_state
            for step in path[2:]:
                node = node[step]
            return node
        if path[1] == "prompt":
            node = prompt
            for step in path[2:]:
                node = node[step]
            return node
    except (KeyError, IndexError, TypeError):
        return None
    return None


def request_members(contract: dict, initial_state: dict | None, prompt: list | None) -> dict[str, dict]:
    """member_key -> {field: value} for every public.request@1 source."""
    members = {}
    for name, source in (contract.get("sources") or {}).items():
        if source.get("adapter") != "public.request@1":
            continue
        fields = {}
        for field, spec in (source.get("fields") or {}).items():
            if "value" in spec:
                fields[field] = spec["value"]
            elif "copy_from" in spec:
                fields[field] = _resolve_path(spec["copy_from"], initial_state, prompt)
        members[name] = fields
    return members


def _flat(record: Any) -> dict:
    if not isinstance(record, dict):
        return {}
    out = {}
    for k, v in record.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, list) and all(isinstance(x, str) for x in v):
            out[k] = v
        elif isinstance(v, dict):
            for k2, v2 in v.items():
                if isinstance(v2, (str, int, float, bool)):
                    out[f"{k}.{k2}"] = v2
    return out


def population_members(contract: dict, initial_state: dict | None, prompt: list | None) -> dict[str, dict[str, dict]]:
    """population source name -> {candidate label -> member fields} for every population adapter.

    Labels follow the environment's candidate identities: the member key for public requests, the
    row_id for sheet rows, the native id for initial records."""
    members: dict[str, dict[str, dict]] = {}
    for name, fields in request_members(contract, initial_state, prompt).items():
        key = (contract.get("sources") or {})[name].get("member_key") or name
        members[name] = {str(key): fields}
    for name, source in (contract.get("sources") or {}).items():
        adapter = source.get("adapter")
        if adapter == "google_sheets.rows@1" and initial_state:
            rows = ((initial_state.get("google_sheets") or {}).get("rows")) or []
            found = {}
            for row in rows:
                if not isinstance(row, dict):
                    continue
                if (row.get("spreadsheet_id"), row.get("worksheet_id")) != (source.get("spreadsheet_id"), source.get("worksheet_id")):
                    continue
                if row.get("row_id") is None:
                    continue
                cells = dict(row.get("cells") or {})
                keys = source.get("key_fields") or []
                if keys and keys[0] in cells:
                    cells["_label"] = str(cells[keys[0]])
                found[str(row["row_id"])] = cells
            members[name] = found
        elif adapter == "initial.records@1" and initial_state:
            path = source.get("path") or []
            node: Any = initial_state
            try:
                for step in path[2:]:
                    node = node[step]
            except (KeyError, IndexError, TypeError):
                node = []
            found = {}
            id_path = source.get("identity_path") or ["id"]
            for record in node if isinstance(node, list) else []:
                ident: Any = record
                try:
                    for step in id_path:
                        ident = ident[step]
                except (KeyError, IndexError, TypeError):
                    ident = None
                if ident is None:
                    continue
                found[str(ident)] = _flat(record)
            members[name] = found
    return members


def resolve_operand(operand: dict, member: dict | None, identity: list | None = None) -> Any:
    """A literal, a request field of the candidate member, or a candidate identity part; anything
    else (joins, lookups, selections, aggregates, derived values) is `{"unresolved": ...}` and is
    never rendered: the engine says nothing about a value it cannot establish."""
    if "literal" in operand:
        return operand["literal"]
    ref = operand.get("ref") or ""
    if member is not None and ref.startswith("request."):
        return member.get(ref[len("request."):], {"unresolved": ref})
    if identity and ref.startswith("candidate."):
        rest = ref[len("candidate."):]
        if rest == "native_record_id":
            return identity[-1]
        if rest.startswith("identity."):
            try:
                return identity[int(rest[len("identity."):])]
            except (ValueError, IndexError):
                return {"unresolved": ref}
    return {"unresolved": ref} if ref else {"unresolved": operand}


def is_unresolved(value: Any) -> bool:
    return isinstance(value, dict) and "unresolved" in value


def _dedupe_content(items: list[dict]) -> list[dict]:
    """Identical conjuncts collapse; a term of another "on one line" set is not identical (each
    alternative of `any(mentions_together(A, B), mentions_together(A, C))` keeps its own A)."""
    seen, out = set(), []
    for c in items:
        key = (c["field"], json.dumps(c["resolved"], sort_keys=True, default=str), c.get("negated"), c.get("mode"),
               c.get("group"), c.get("branch"), c.get("together"))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _fields_equal_content(expected: Any) -> list[dict]:
    """`record.fields_equal@1` expectations: a {field: value} mapping or a list of {field, value} rows."""
    rows = []
    if isinstance(expected, dict):
        rows = [(k, v) for k, v in expected.items()]
    elif isinstance(expected, list):
        for entry in expected:
            if isinstance(entry, dict) and "field" in entry and "value" in entry:
                rows.append((str(entry["field"]), entry["value"]))
    return [{"field": k, "value": {"literal": v}, "mode": "eq", "negated": False, "group": None, "branch": None} for k, v in rows]


@functools.lru_cache(maxsize=64)
def _compile_contract_cached(contract_json: str, initial_json: str | None, prompt_json: str | None) -> tuple:
    contract = json.loads(contract_json)
    initial_state = json.loads(initial_json) if initial_json else None
    prompt = json.loads(prompt_json) if prompt_json else None
    return tuple(_compile_contract(contract, initial_state, prompt))


def compile_contract(contract: dict, initial_state: dict | None = None, prompt: list | None = None) -> list[dict]:
    """One compiled spec per check: effect, population, address, content, residual, candidate fields.
    Cached per (contract, initial state, prompt): every episode of a task compiles the same specs."""
    try:
        key = (json.dumps(contract, sort_keys=True), json.dumps(initial_state, sort_keys=True) if initial_state else None,
               json.dumps(prompt, sort_keys=True) if prompt else None)
    except (TypeError, ValueError):
        return _compile_contract(contract, initial_state, prompt)
    return [dict(spec) for spec in _compile_contract_cached(*key)]


def _compile_contract(contract: dict, initial_state: dict | None, prompt: list | None) -> list[dict]:
    sources = contract.get("sources") or {}
    populations = population_members(contract, initial_state, prompt)
    specs = []
    for check in contract.get("checks") or []:
        spec = _compile_check(check, sources, populations)
        # a check may accept alternative effects from other sources ("a contact created, or a deal created
        # for the lead"): each is compiled as a variant, chosen per finding by the call it was located at
        spec["variants"] = []
        for alternative in check.get("alternatives") or []:
            if not isinstance(alternative, dict) or not alternative.get("source"):
                continue
            variant = _compile_check({**check, "source": alternative["source"], "effect_match": alternative.get("effect_match") or {}},
                                     sources, populations)
            variant["alias"] = alternative.get("alias")
            variant["variants"] = []
            spec["variants"].append(variant)
        specs.append(spec)
    return specs


def _compile_check(check: dict, sources: dict, populations: dict) -> dict:
    if True:
        source = sources.get(check.get("source") or "", {})
        population = check.get("population")
        candidates = populations.get(population, {}) if population else {}
        member = next(iter(candidates.values())) if len(candidates) == 1 else None
        predicate = check.get("effect_match") or check.get("expected") or {}
        compiled = compile_predicate(predicate) if isinstance(predicate, dict) and "op" in predicate else _new_out()
        if check.get("operator") == "record.fields_equal@1":
            compiled["content"] = _fields_equal_content(check.get("expected"))
        # Identity the source itself declares (which sheet, which record) is part of the address.
        for key in ("spreadsheet_id", "worksheet_id", "record_id", "object_type"):
            if source.get(key) is not None and key not in compiled["address"]:
                compiled["address"][key] = {"literal": source[key]}
        condition = check.get("required_when") or check.get("prohibited_when")
        cond = compile_predicate(condition) if isinstance(condition, dict) else None
        if cond is not None:
            # A condition stated on the effect itself ("prohibited when the row's batch_ref does not
            # mention X") is part of what the effect must look like: it joins the content, so a harm
            # names the offending content exactly instead of covering every write of that row.
            for key, operand in cond["address"].items():
                add_address(compiled["address"], key.split("#")[0], operand)
            for c in cond["content"]:
                compiled["content"].append({**c, "group": (c["group"] + compiled["groups"]) if c.get("group") is not None else None,
                                            "together": (c["together"] + compiled["togethers"]) if c.get("together") is not None else None})
            compiled["preconditions"].extend(cond["preconditions"])
        collection = source.get("collection")
        if isinstance(collection, list):
            collection = collection[-1] if collection else None
        if collection is None and source.get("adapter") in ("final.records@1", "final.created_records@1"):
            path = source.get("path") or []
            collection = path[3] if len(path) > 3 else None
        population_source = sources.get(population, {}) if population else {}
        population_collection = None  # the world collection the candidates are records of, when they are
        if population_source.get("adapter") == "initial.records@1":
            ppath = population_source.get("path") or []
            population_collection = ppath[3] if len(ppath) > 3 else None
        elif population_source.get("adapter") == "google_sheets.rows@1":
            population_collection = "rows"
        return {
            "check_id": check.get("check_id"), "signal_id": check.get("signal_id"), "role": check.get("role"),
            "collection": collection,  # the world collection the source observes, when it names one
            "population_collection": population_collection,
            "operator": check.get("operator"), "semantics": check.get("semantics"),
            "effect": source_effect(source) if source else EffectRef(app="unknown", object="record", kind="other"),
            "source": check.get("source"), "population": population,
            "population_adapter": sources.get(population, {}).get("adapter") if population else None,
            "member": member, "members": candidates,
            "raw_address": compiled["address"], "raw_content": compiled["content"],
            "preconditions": compiled["preconditions"],
            "address": {k: resolve_operand(v, member) for k, v in compiled["address"].items()},
            "content": _dedupe_content([{**c, "resolved": resolve_operand(c["value"], member)} for c in compiled["content"]]),
            "residual": sorted(set(compiled["residual"])),
            "residual_nodes": list(compiled.get("residual_nodes") or []),
            "condition": condition if isinstance(condition, dict) else None,
            "effect_match": predicate if isinstance(predicate, dict) else None,
            "condition_role": "required_when" if check.get("required_when") else ("prohibited_when" if check.get("prohibited_when") else None),
            "population_key_fields": list(population_source.get("key_fields") or []),
            "condition_residual": sorted(set(cond["residual"])) if cond else [],
            # condition operators that read the effect itself and could not be expressed as content
            "condition_effect_residual": sorted(set(cond["effect_residual"])) if cond else [],
            "joins": len(check.get("effect_joins") or []), "alternatives": len(check.get("alternatives") or []),
            "selections": len(check.get("selections") or []), "aggregates": len(check.get("aggregates") or []),
        }


# ---------------------------------------------------------------------------------------------
# Findings -> Items
# ---------------------------------------------------------------------------------------------
def findings(episode_batches: list[dict]) -> dict[str, list[dict]]:
    """Distinct receipt payloads per kind from complete assessment batches (rescoring de-duplicated)."""
    out: dict[str, list[dict]] = {GUARD_OUTPUT: [], OBLIGATION_OUTPUT: [], RECORD_OUTPUT: []}
    seen = set()
    for batch in episode_batches:
        run = batch.get("run") or {}
        if run.get("status") != "complete":
            continue
        for receipt in run.get("execution_evidence") or []:
            kind = receipt.get("kind")
            if kind not in out:
                continue
            payload_json = receipt.get("payload_json") or "{}"
            key = (kind, payload_json)
            if key in seen:
                continue
            seen.add(key)
            out[kind].append(json.loads(payload_json))
    return out


def _value(raw: Any, corpus: Corpus | None) -> Value:
    return Value(raw=raw, observed=corpus.is_observed(raw) if corpus else False,
                 written=corpus.is_written(raw) if corpus else False, forms=value_forms(raw),
                 hidden_elements=corpus.hidden_elements(raw) if corpus else ())


def resolve_spec(spec: dict, label: str | None, identity: list | None = None) -> tuple[dict, list[dict]]:
    """Address and content of a spec resolved for one candidate (by label and identity), or as compiled."""
    member = spec["members"].get(str(label)) if label is not None else None
    if member is None and not identity:
        return spec["address"], spec["content"]
    member = member if member is not None else spec["member"]
    address = {k: resolve_operand(v, member, identity) for k, v in spec["raw_address"].items()}
    content = _dedupe_content([{**c, "resolved": resolve_operand(c["value"], member, identity)} for c in spec["raw_content"]])
    return address, content


def _spec_values(spec: dict, corpus: Corpus | None, label: str | None = None,
                 identity: list | None = None) -> tuple[dict[str, Value], dict[str, Value], dict[str, Any]]:
    """(address, content, meta): values of a spec for one candidate; unresolvable operands are left
    out and listed in meta["unresolved"], negated/alternative conjuncts are listed in meta too."""
    raw_address, raw_content = resolve_spec(spec, label, identity)
    meta: dict[str, Any] = {"unresolved": {"address": [], "content": []}, "negated": [], "alternatives": {},
                            "together": {}, "together_scope": {}, "preconditions": [], "modes": {}, "excluded": [],
                            "branches": {}}
    address = {}
    for k, v in raw_address.items():
        if is_unresolved(v):
            meta["unresolved"]["address"].append(k)
            continue
        address[k] = _value(v, corpus)
    content = {}
    for i, c in enumerate(raw_content):
        if is_unresolved(c["resolved"]):
            meta["unresolved"]["content"].append(c["field"])
            continue
        key = c["field"] if c["field"] not in content else f"{c['field']}#{i}"
        content[key] = _value(c["resolved"], corpus)
        if c.get("negated"):
            meta["negated"].append(key)
        if c.get("group") is not None:
            meta["alternatives"][key] = c["group"]
            meta["branches"][key] = str(c.get("branch"))  # one alternative of that `any`: its conjuncts hold together
        if c.get("together") is not None:
            meta["together"][key] = c["together"]
            meta["together_scope"][str(c["together"])] = c.get("scope") or "line"
        if c.get("excluded"):
            meta["excluded"].append(key)
        meta["modes"][key] = c.get("mode")
    for p in spec.get("preconditions") or []:
        value = resolve_operand(p["value"], spec["members"].get(str(label)) if label is not None else spec["member"], identity)
        if not is_unresolved(value):
            meta["preconditions"].append({"field": p["field"], "value": value, "negated": bool(p.get("negated"))})
    return address, content, meta


def _candidate(spec: dict, identity: list | None, corpus: Corpus | None = None) -> dict | None:
    """Candidate record: the environment's identity, a display label (a key field when known) and
    whether the attempt observed or wrote that label (so renderers can hide it)."""
    if not identity:
        return None
    label = str(identity[-1])
    member = spec["members"].get(label) or {}
    display = member.get("_label") or label
    seen = True if corpus is None else (corpus.is_observed(display) or corpus.is_written(display))
    fields = [str(v) for k, v in member.items() if not str(k).startswith("_") and isinstance(v, (str, int, float)) and not isinstance(v, bool)]
    return {"identity": identity, "label": display, "key": label, "observed": seen, "fields": fields,
            "member": {str(k): v for k, v in member.items() if not str(k).startswith("_")}}


def variant_for_call(spec: dict, call) -> tuple[dict, bool]:
    """The spec variant (the check's own source or one of its alternatives) whose effect the located call
    produced, by the call's app and the object its tool names; (variant, ambiguous). With no alternatives
    the spec itself; ambiguous when the call fits none or several."""
    variants = [spec, *(spec.get("variants") or [])]
    if len(variants) == 1 or call is None:
        return spec, False

    def same_app(a: str, b: str) -> bool:
        a, b = a.replace("_", ""), b.replace("_", "")
        return bool(a and b) and (a == b or a.startswith(b) or b.startswith(a))

    tokens = set(call.tool.lower().split("_"))
    by_app = [v for v in variants if same_app(v["effect"].app, call.app)]
    by_object = [v for v in by_app if v["effect"].object.lower() in tokens or v["effect"].object.lower() + "s" in tokens]
    chosen = by_object if len(by_object) == 1 else (by_app if len(by_app) == 1 else [])
    return (chosen[0], False) if chosen else (spec, True)


def manifest_items(specs: list[dict], batches: list[dict], calls_by_invocation: dict, corpus: Corpus | None) -> list[Item]:
    """Items for every manifest check: outcome from findings when present, else abstained."""
    by_id = {s["check_id"]: s for s in specs}
    found = findings(batches)
    items: list[Item] = []

    def location_for(occurrence: str | None, kind: str, source: str) -> Location:
        call = calls_by_invocation.get(occurrence) if occurrence else None
        if call is None:
            return Location(kind="unknown", source=source if occurrence else None)
        return Location(kind=kind, turn=call.turn, call=CallRef(turn=call.turn, index=call.index), source=source)

    def base(spec: dict, instance: str | None, label: str | None = None, identity: list | None = None) -> dict:
        address, content, meta = _spec_values(spec, corpus, label, identity)
        polarity = {"harm": "forbid", "goal": "require", "diagnostic": "require"}.get(spec["role"], "require")
        if spec["operator"] in ("records.retained_when@1", "sheets.retained_when@1", "objects.created_and_retained@1"):
            polarity = "keep"
        return {"origin": "manifest", "type": spec["operator"], "check_id": spec["check_id"], "polarity": polarity,
                "polarity_source": "manifest_role", "effect": spec["effect"], "address": address, "content": content,
                "semantics": {"residual": spec["residual"], "condition_residual": spec["condition_residual"],
                              "condition_effect_residual": spec.get("condition_effect_residual", []),
                              "population_adapter": spec["population_adapter"], "signal_id": spec["signal_id"],
                              "collection": spec.get("collection"), "population_collection": spec.get("population_collection"),
                              "source": spec.get("source"), "population": spec.get("population"),
                              "unresolved": meta["unresolved"], "negated_keys": meta["negated"],
                              "alternatives": meta["alternatives"], "together": meta["together"],
                              "together_scope": meta["together_scope"], "modes": meta["modes"],
                              "excluded_keys": meta["excluded"], "branches": meta["branches"],
                              "preconditions": meta["preconditions"],
                              "residual_nodes": spec.get("residual_nodes") or [],
                              "condition": spec.get("condition"), "condition_role": spec.get("condition_role"),
                              "effect_match": spec.get("effect_match"),
                              "population_members": spec["members"] if spec.get("population_adapter") in (
                                  "google_sheets.rows@1", "initial.records@1") else None,
                              "population_key_fields": spec.get("population_key_fields") or []},
                "id": f"manifest:{spec['check_id']}" + (f":{instance[:8]}" if instance else "")}

    covered = set()
    for payload in found[GUARD_OUTPUT]:
        spec = by_id.get(payload.get("check_id"))
        if spec is None or payload.get("kind") != "finding":
            continue
        covered.add(spec["check_id"])
        status = payload.get("status")
        value = payload.get("value")
        outcome = "abstained" if status != "valid" else ("failed" if value == 1 else "passed")
        if outcome == "passed":
            continue  # a guard instance that matched nothing is not a fact worth an item
        candidate = payload.get("candidate_identity")
        location = location_for(payload.get("occurrence"), "at_call", "guard_occurrence")
        variant, ambiguous = variant_for_call(spec, calls_by_invocation.get(payload.get("occurrence")) if payload.get("occurrence") else None)
        b = base(variant, payload.get("instance_key"), str(candidate[-1]) if candidate else None, candidate)
        if ambiguous:
            b["semantics"]["source_ambiguous"] = True  # which of the check's alternative effects fired is not known
        if spec.get("variants"):
            b["semantics"]["alias"] = variant.get("alias")
        items.append(Item(**b, outcome=outcome, candidate=_candidate(spec, candidate, corpus), location=location,
                          evidence=(f"guard:{spec['check_id']}:{payload.get('occurrence')}",)))
    for payload in found[OBLIGATION_OUTPUT]:
        spec = by_id.get(payload.get("check_id"))
        if spec is None or payload.get("kind") != "finding":
            continue
        covered.add(spec["check_id"])
        status = payload.get("status")
        if status == "inapplicable" or payload.get("required") is False:
            continue
        value = payload.get("value")
        outcome = "abstained" if status != "valid" else ("passed" if value == 1 else "failed")
        witnesses = payload.get("witnesses") or []
        if isinstance(witnesses, str):
            witnesses = []
        earliest = min(witnesses, key=lambda w: (w.get("applied_revision", 0), w.get("occurrence", ""))) if witnesses else None
        candidate = payload.get("candidate_identity")
        b = base(spec, payload.get("instance_key"), str(candidate[-1]) if candidate else None, candidate)
        if spec.get("variants"):
            b["semantics"]["alternatives_allowed"] = len(spec["variants"])  # another effect the task allows would do
        candidate_info = _candidate(spec, candidate, corpus)
        loc = (location_for(earliest.get("occurrence"), "after_turn", "witness") if earliest and outcome == "passed"
               else Location(kind="end_of_attempt", source="unmet") if outcome == "failed" else Location(kind="unknown"))
        items.append(Item(**b, outcome=outcome, candidate=candidate_info,
                          location=loc, evidence=(f"obligation:{spec['check_id']}:{payload.get('instance_key')}",)))
    for payload in found[RECORD_OUTPUT]:
        for result in payload.get("results") or []:
            spec = by_id.get(result.get("check_id"))
            if spec is None:
                continue
            covered.add(spec["check_id"])
            outcome = "abstained" if result.get("status") != "valid" else ("passed" if result.get("value") == 1 else "failed")
            b = base(spec, None)
            items.append(Item(**b, outcome=outcome, location=Location(kind="end_of_attempt" if outcome == "failed" else "unknown"),
                              evidence=(f"record:{spec['check_id']}",)))
    for spec in specs:
        if spec["check_id"] in covered or spec["role"] == "diagnostic":
            continue
        b = base(spec, None)
        items.append(Item(**b, outcome="abstained", location=Location(kind="unknown"), evidence=()))
    return items
