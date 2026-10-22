"""Separate historical answer literals from same-model instruction generation."""

import json
import re


def argument_provenance(material, *, schema_roles=False):
    """Trace arguments to earlier observed data without exposing its values."""
    public = material["user_request"]
    static_fields = {
        "format",
        "label",
        "object",
        "searchField",
        "searchResults",
        "dealstage",
        "pipeline",
        "max_results",
    }

    def leaves(value, path=""):
        if isinstance(value, dict):
            for k, v in value.items():
                yield from leaves(v, path + "/" + k)
        elif isinstance(value, list):
            for i, v in enumerate(value):
                yield from leaves(v, path + f"/{i}")
        elif isinstance(value, (str, int, float)) and not isinstance(value, bool):
            yield path, value

    previous, rows = [], []
    for i, action in enumerate(material["reference_actions"]):
        arguments = {}
        for field, value in action["call"].get("arguments", {}).items():
            if schema_roles:
                from luna_argument_roles import schema_role

                role = schema_role(material, action["call"]["name"], field, value)
                if role is not None:
                    arguments[field] = role
                    continue
            matches = []
            if isinstance(value, (str, int, float)):
                for aid, result_path, observed in previous:
                    if (
                        str(value) == str(observed)
                        or isinstance(observed, str)
                        and len(str(value)) > 4
                        and str(value) in observed
                    ):
                        matches.append({"action_id": aid, "result_path": result_path})
            if matches:
                arguments[field] = {"origin": "prior_tool_result", "evidence": matches}
            elif str(value).strip('"') in public:
                arguments[field] = {"origin": "public_request", "value": value}
            elif field in static_fields:
                arguments[field] = {
                    "origin": "demonstrated_api_option",
                    "value": value,
                    "note": "Reference convention, not an email fact or user requirement.",
                }
            elif (
                field == "query"
                and not schema_roles
                and isinstance(value, str)
                and not any(c in value for c in ("@", "+"))
                and not any(c.isdigit() for c in value)
            ):
                arguments[field] = {"origin": "demonstrated_search_strategy", "value": value}
            else:
                arguments[field] = {
                    "origin": "requires_runtime_derivation",
                    "note": "Compose from observed data; no supplied answer literal.",
                }
        aid = f"a{i + 1}"
        rows.append({"action_id": aid, "tool": action["call"]["name"], "argument_provenance": arguments})
        previous.extend((aid, p, v) for p, v in leaves(action["result"].get("observed_result")))
    return rows


def separated_sources(material):
    public = material["user_request"] + json.dumps(material["tool_definitions"])
    bindings = {}

    def hidden(value, path):
        key = json.dumps(value, sort_keys=True)
        if key not in bindings:
            bindings[key] = {"symbol": f"retrieved_{len(bindings) + 1}", "retrieve_from": path}
        return bindings[key].copy()

    def walk(value, path, concrete=False):
        if isinstance(value, dict):
            return {
                k: walk(
                    v,
                    path + "/" + k,
                    concrete or k in ("arguments", "observed_result", "observed_error", "value", "record_id"),
                )
                for k, v in value.items()
            }
        if isinstance(value, list):
            return [walk(v, path + f"/{i}", concrete) for i, v in enumerate(value)]
        field = path.rsplit("/", 1)[-1]
        if concrete and isinstance(value, str) and value and value.strip('"') not in public:
            # Protocol flags remain readable; arbitrary names/IDs/body strings
            # and historical arguments cannot become shortcuts for the actor.
            if value not in ("matched", "returned", "success", "error"):
                return hidden(value, path)
        if concrete and isinstance(value, (float, int)) and not isinstance(value, bool):
            if field.endswith("id") and str(value) not in public:
                return hidden(value, path)
        return value

    contract = walk(material["goal_manifest"], "/manifest")
    actions = walk(material["reference_actions"], "/actions")
    return contract, actions, {"private_literal_count": len(bindings), "public_request_preserved": True}


def observation_bound_sources(material):
    """Replace opaque handles with retrieval operations grounded in saved reads."""
    contract, actions, receipt = separated_sources(material)
    reads = []
    protocol_fields = set()

    def leaves(value, path=""):
        if isinstance(value, dict):
            protocol_fields.update(k for k in value if not any(c.isdigit() for c in k))
            for k, v in value.items():
                yield from leaves(v, path + "/" + k)
        elif isinstance(value, list):
            for i, v in enumerate(value):
                yield from leaves(v, path + f"/{i}")
        elif isinstance(value, str):
            yield path, value

    for action in material["reference_actions"]:
        tool = action["call"]["name"]
        if any(kind in tool for kind in ("find", "get", "search", "lookup", "list")):
            reads.extend((tool, path, value) for path, value in leaves(action["result"].get("observed_result")))

    def describe(value):
        if isinstance(value, str):
            for exact in (True, False):
                for tool, path, observed in reads:
                    matches = value == observed if exact else len(value) > 4 and value in observed
                    if matches:
                        operation = ("read returned field" if exact
                                     else "extract requested value from returned text")
                        if (not exact and re.fullmatch(r"[A-Za-z0-9.-]+\.[A-Za-z]{2,}", value)
                                and "@" + value in observed):
                            operation = "extract the email sender domain (part after @), not the sender display name"
                        return {
                            "obtain_from_tool_result": tool,
                            "returned_field": ".".join(p for p in path.split("/") if p and not p.isdigit()),
                            "operation": operation,
                        }
        return {
            "obtain_at_runtime": "Identify the entity or value specified by the public request using declared tools; do not assume a hidden identifier."
        }

    def replace(original, masked):
        if (
            isinstance(original, dict)
            and original.get("kind") == "literal"
            and isinstance(original.get("value"), str)
            and original["value"] in protocol_fields
        ):
            return {**masked, "value": original["value"]}
        if isinstance(masked, dict) and set(masked) == {"symbol", "retrieve_from"}:
            return describe(original)
        if isinstance(masked, dict):
            return {k: replace(original[k], v) for k, v in masked.items() if k != "authority_paths"}
        if isinstance(masked, list):
            return [replace(a, b) for a, b in zip(original, masked, strict=True)]
        return masked

    contract = replace(material["goal_manifest"], contract)
    actions = replace(material["reference_actions"], actions)
    initial = material["native_data"]["initial_state"]
    for source_id, source in contract.get("sources", {}).items():
        original = material["goal_manifest"]["sources"][source_id]
        for field, binding in source.get("fields", {}).items():
            if not isinstance(binding, dict) or "copy_from" not in binding:
                continue
            original_binding = original["fields"][field]
            value = original_binding.get("value")
            path = original_binding.get("copy_from")
            if path and path[:2] == ["task_evidence", "initial"]:
                try:
                    value = initial
                    for part in path[2:]:
                        value = value[part]
                except (KeyError, IndexError, TypeError):
                    value = None
            source["fields"][field] = (
                {"runtime_binding": describe(value)}
                if value is None or str(value) not in material["user_request"]
                else {"public_value": value}
            )
        if source.get("adapter") == "initial.records@1":
            source.pop("path", None)
            source["retrieval_intent"] = (
                "Original records identified by the public request, accessed through declared tools."
            )
    for check in contract["checks"]:
        joins = check.pop("effect_joins", [])
        if not joins:
            continue

        def remove_join_predicate(expr):
            if not isinstance(expr, dict):
                return expr
            if expr.get("op") == "all":
                parts = [remove_join_predicate(v) for v in expr["args"]]
                return {**expr, "args": [p for p in parts if p is not None]}
            sides = [expr.get("left", {}), expr.get("right", {})]
            if any(s.get("kind") == "field" and s.get("path", [])[:1] == ["join"] for s in sides):
                if expr.get("op") != "eq" or not any(
                    s.get("kind") == "literal" and s.get("value") == "matched" for s in sides
                ):
                    raise ValueError("unsupported join predicate for operational compilation")
                return None
            return expr

        if "effect_match" in check:
            check["effect_match"] = remove_join_predicate(check["effect_match"])
        check["operational_dependencies"] = [
            {
                "source": j["source"],
                "matching_conditions": j["where"],
                "order": j.get("order", j.get("timing")),
                "meaning": "Do not issue a second matching effect in this execution."
                if check["role"] == "harm"
                else "The matching read must return before this action is decided and dispatched; this is ordering, not an extra tool argument.",
            }
            for j in joins
        ]
    return contract, actions, receipt
