"""Same-model task instruction preparation, separate from native execution.

The model interprets private source evidence; only reviewed instructions reach
the solver. Structural checks prove coverage and provenance, not semantic truth.
"""

import json
import re

from luna_effect_scope import SCOPE_RULES, effect_scopes, private_email_domains
from luna_policy_guidance import (
    POLICY_RULES,
    policy_retrieval_contracts,
    require_policy_retrieval,
    retrieval_only_review,
)
from luna_solver_prompts import check_hint_literals
from luna_solver_separation import argument_provenance, observation_bound_sources, separated_sources

REQUIREMENTS_PROMPT = """Interpret a task for a fresh tool-using solver. The formal
goal/harm manifest is authoritative; the demonstration can contain failed tries.
Return JSON {requirements:[{check_id,role,instruction,evidence}]} with exactly one
entry per goal/harm check. evidence is a nonempty list of JSON pointers into the
manifest, e.g. /checks/0. Preserve predicates, target field, occurrence bounds,
required read-before-dispatch joins and preservation constraints. Turn private
expected values into instructions to retrieve them, never reveal hidden IDs,
addresses, new field values or answer text absent from the public request.
Distinguish existing-record identity from a NEW value to be written. Do not
invent checks or assume an empty search achieved the goal. No reasoning prose.
"""

WORKFLOW_PROMPT = """Interpret the demonstrated tool workflow for a fresh solver.
Return JSON {actions:[{action_id,tool,disposition,instruction,depends_on}]}.
Include each supplied action once, in original order. disposition is reuse,
failed_attempt or uncertain. An empty result with success=true is a successful
API request, not successful retrieval. Explain how useful arguments are sourced:
public request, stable identity or a prior observed tool result. Preserve actual
result-field names. Do not search an existing record by its NEW field value.
depends_on lists earlier action IDs whose returned information is required.
Do not copy private literal values, target calls, or demonstration bookkeeping
into instructions. Treat failed attempts as pitfalls, not recommended actions.
"""

ASSEMBLY_PROMPT = """Create task-specific execution instructions from the supplied
requirements and demonstrated workflow. Return JSON
{steps:[{step_id,instruction,depends_on,check_ids,action_ids}]}.
Use sequential step IDs s1,s2,... and reference only earlier steps in depends_on.
Cover every goal and harm check, and cite relevant demonstrated action IDs.
Preserve retrieval-before-write dependencies. Explicitly wait for dependent tool
results; dependent calls must not share one parallel batch. Verify a write AFTER
it returns, and confirm creation before claiming it happened in a notification.
Use stable identity to find a record, not its proposed new value. Failed empty
queries may inform recovery but must not be required as a procedure. Do not
invent unavailable tools, facts, hidden values, or claim checks already passed.
Instructions should describe the task, not teachers, Luna, datasets or annotation.
"""

REVIEW_PROMPT = """Audit proposed solver instructions against the ORIGINAL sources.
Return JSON {checks:[{check_id,covered,reason}],workflow_faithful:boolean,
no_private_literals:boolean,approved:boolean,issues:[string]}.
Review every goal/harm. Reject lost predicates, wrong identities, searches by a
new update value, verification before write, invented API fields, false success
from empty responses, or lost read-before-dispatch ordering. Reject dependent
parallel calls and notifications claiming an unconfirmed write. Full legacy
score does not prove these instructions faithful. approved must be false for
any issue or unresolved ambiguity. No correction/retry or free-form reasoning.
"""


def _rows(value, field):
    rows = value.get(field) if isinstance(value, dict) else None
    if not isinstance(rows, list) or not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"invalid {field}")
    return rows


def _text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("missing instruction or evidence")
    return value


def _exact_ids(rows, key, expected):
    ids = [row.get(key) for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError(f"missing, duplicate or invented {key}")


def _pointer(document, pointer):
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ValueError("invalid manifest evidence pointer")
    value = document
    for part in pointer[1:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def _sensitive_values(value):
    """Screen concrete evidence values, not operator/tool/field vocabulary.

    This literal screen is deliberately only one gate; it cannot establish
    semantic non-disclosure or correctness of newly paraphrased facts.
    """
    if isinstance(value, str):
        if any(c.isdigit() for c in value) or "@" in value or " " in value:
            yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _sensitive_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _sensitive_values(child)


async def prepare_instructions(material, generate, record, variant="raw", *, schema_argument_roles=False,
                               repair_invalid_assembly=False):
    """generate(stage, prompt, data) uses the SAME pinned local model as solver.

    record is called before validation so rejected outputs remain inspectable.
    No hidden retries, output clipping, or fallback to lossy summaries.
    """
    contract = material["goal_manifest"]
    checks = [c for c in contract["checks"] if c["role"] in ("goal", "harm")]
    check_by_id = {c["check_id"]: c for c in checks}
    actions = [{"action_id": f"a{i + 1}", **action} for i, action in enumerate(material["reference_actions"])]
    action_by_id = {a["action_id"]: a for a in actions}
    public = material["user_request"] + json.dumps(material["tool_definitions"])
    private = {"manifest": contract, "actions": actions}
    sensitive = list(_sensitive_values(private))
    if schema_argument_roles:
        from luna_argument_roles import public_equivalence

        equivalents = {v: public_equivalence(v, material["user_request"]) for v in sensitive}
        record("public_equivalence_audit", {"input": {"task": material["task"]},
            "output": {v: reason for v, reason in equivalents.items() if reason}})
        sensitive = [v for v in sensitive if not equivalents[v]]
    policy = "policy" in variant
    policy_contracts = policy_retrieval_contracts(material) if policy else []
    generation_contract, generation_actions = contract, actions
    if variant != "raw":
        source_transform = observation_bound_sources if variant.endswith("v5") else separated_sources
        generation_contract, refs, separation = source_transform(material)
        generation_actions = [{"action_id": f"a{i + 1}", **a} for i, a in enumerate(refs)]
        record("source_separation", {"input": {"task": material["task"]}, "output": separation})
    if "provenance" in variant:
        for action, provenance in zip(generation_actions,
                                      argument_provenance(material, schema_roles=schema_argument_roles), strict=True):
            action["argument_provenance"] = provenance["argument_provenance"]
            if variant.endswith(("v4", "v5")):
                action["call"]["arguments"] = provenance["argument_provenance"]

        def retrieval_paths(value):
            if isinstance(value, dict):
                return {k: retrieval_paths(v) for k, v in value.items() if k != "authority_paths"}
            if isinstance(value, list):
                return [retrieval_paths(v) for v in value if not isinstance(v, int)]
            return value

        generation_contract = retrieval_paths(generation_contract)
        if variant.endswith(("v4", "v5")):
            for check in generation_contract["checks"]:
                if check["role"] == "harm":
                    check["execution_interpretation"] = {
                        "phase": "constraint on this execution",
                        "meaning": "Avoid the prohibited new effect while choosing actions. Do not invent tools to audit global history.",
                    }
                    if variant.endswith("v5"):
                        check["execution_interpretation"]["time_scope"] = (
                            "Earlier/later effect joins order effects of this execution, not global pre-existing records. "
                            "A no-duplicate creation guard means do not issue a second matching creation. "
                            "Track actions issued; do not infer global absence of writes from a lookup returning one record."
                        )
                elif check.get("operator") == "record.fields_equal@1":
                    check["execution_interpretation"] = {
                        "phase": "terminal state after write",
                        "meaning": "Check the returned updated record or an available read tool. This is not a precondition before updating.",
                    }
        # Quoting a public name is not private-answer disclosure. Keep the
        # original strict literal check for every actual nonpublic value.
        sensitive = [v for v in sensitive if v.strip('"') not in public]

    async def stage(name, prompt, data):
        if policy:
            if policy_contracts or not schema_argument_roles:
                prompt += "\n" + POLICY_RULES
            else:
                prompt += (
                    "\nRetrieve only the sources required by this task and its declared dependencies. "
                    "Do not import contact-matching, HR or note-title rules from another task. "
                    "No policy-title retrieval contract is declared for this task."
                )
            data = {**data, "policy_retrieval_contracts": policy_contracts}
        if variant.startswith("scope-"):
            prompt += "\n" + SCOPE_RULES
            prompt += "\nReturn entries only for eligible_check_ids. Diagnostic checks remain audit evidence, not goal/harm requirements."
            data = {**data, "effect_scope_contracts": effect_scopes(generation_contract),
                    "eligible_check_ids": list(check_by_id)}
        if variant != "raw":
            prompt += (
                "\nEvidence contains retrieval placeholders, not answer values. Describe how to retrieve them; "
                "do not put opaque symbols or source paths into solver instructions. "
                "Evidence pointers must be only the exact check roots /checks/0, /checks/1, etc. "
                "A record.fields_equal check describes the desired TERMINAL state, after the update."
            )
        if "provenance" in variant:
            prompt += (
                "\nargument_provenance is authoritative for each parameter's origin. "
                "A demonstrated_api_option is a reference convention, NOT an email field. "
                "Do not invent a deal stage inside an email or require unrelated message IDs to match. "
                "Initial collections describe retrieval intent, not direct hidden-state access. "
                "A prohibited new effect forbids extra actions in this execution, not pre-existing records. "
                "Do not invent tools for verification. Cover each check exactly once in requirement/review lists."
            )
        if variant.endswith("v5"):
            prompt += (
                "\nUse the observation-bound descriptions to write ordinary task instructions. "
                "Never invent an email address, record ID or placeholder handle. "
                "Describe a guard by limiting the actions issued in this execution; a lookup of one record "
                "does not prove global absence of changes. Do not add unavailable audit operations. "
                "Use returned-field names from argument provenance exactly, not a guessed body/subject field."
            )
        value = await generate(name, prompt, data)
        record(name, {"input": data, "output": value})
        return value

    common = {"public_request": material["user_request"], "tools": material["tool_definitions"]}
    grounded = "grounded" in variant or schema_argument_roles
    if grounded:
        sensitive.extend(private_email_domains(private, public))
    review_sources = {"manifest": generation_contract, "actions": generation_actions} if grounded else private

    def validate_requirements(value):
        rows = _rows(value, "requirements")
        _exact_ids(rows, "check_id", check_by_id)
        for row in rows:
            if row.get("role") != check_by_id[row["check_id"]]["role"]:
                raise ValueError("goal/harm role changed")
            _text(row.get("instruction"))
            evidence = row.get("evidence")
            if not isinstance(evidence, list) or not evidence:
                raise ValueError("missing manifest evidence")
            index = contract["checks"].index(check_by_id[row["check_id"]])
            if not any(p == f"/checks/{index}" or p.startswith(f"/checks/{index}/") for p in evidence):
                raise ValueError("requirement cites wrong check")
            for pointer in evidence:
                _pointer(contract, pointer)
        check_hint_literals([r["instruction"] for r in rows], public, sensitive)

    requirements = await stage("requirements", REQUIREMENTS_PROMPT, {**common, "manifest": generation_contract})
    validate_requirements(requirements)

    workflow = await stage("workflow", WORKFLOW_PROMPT, {**common, "actions": generation_actions})
    rows = _rows(workflow, "actions")
    _exact_ids(rows, "action_id", action_by_id)
    seen = set()
    for row, original in zip(rows, actions, strict=True):
        if row["action_id"] != original["action_id"] or row.get("tool") != original["call"]["name"]:
            raise ValueError("workflow reordered or invented a tool")
        if row.get("disposition") not in ("reuse", "failed_attempt", "uncertain"):
            raise ValueError("unknown action disposition")
        result = original.get("result", {})
        observed = result.get("observed_result")
        if row["disposition"] == "reuse" and (
            result.get("observed_error")
            or isinstance(observed, dict)
            and (observed.get("success") is False or observed.get("result_count") == 0)
        ):
            raise ValueError("failed or empty reference action endorsed for reuse")
        deps = row.get("depends_on")
        if not isinstance(deps, list) or not set(deps) <= seen:
            raise ValueError("workflow forward dependency")
        _text(row.get("instruction"))
        seen.add(row["action_id"])
    check_hint_literals([r["instruction"] for r in rows], public, sensitive)

    assembled = await stage("assembly", ASSEMBLY_PROMPT, {**common, "requirements": requirements, "workflow": workflow})
    def validate_assembly(plan):
        steps = _rows(plan, "steps")
        seen, covered = set(), set()
        lines = []
        for i, step in enumerate(steps):
            if step.get("step_id") != f"s{i + 1}":
                raise ValueError("invalid sequential step identity")
            for field, allowed in (("depends_on", seen), ("check_ids", check_by_id), ("action_ids", action_by_id)):
                values = step.get(field)
                if not isinstance(values, list) or not set(values) <= set(allowed):
                    raise ValueError(f"invalid step {field}")
            covered.update(step["check_ids"])
            instruction = _text(step.get("instruction"))
            barrier = f" Wait for completed steps {', '.join(step['depends_on'])} before dispatch." if step["depends_on"] else ""
            lines.append(f"{step['step_id']}: {instruction}{barrier}")
            seen.add(step["step_id"])
        if covered != set(check_by_id) and not variant.endswith(("v4", "v5")):
            raise ValueError("assembly omitted goal/harm coverage")
        if variant.endswith(("v4", "v5")):
            preserved = [f"{r['role']} requirement: {r['instruction']}" for r in requirements["requirements"]]
            lines = preserved + lines
            plan["required_outcomes_and_constraints"] = requirements["requirements"]
            record("preserved_requirements", {"input": {"task": material["task"]}, "output": {
                "all_check_ids_retained": sorted(check_by_id),
                "plan_uncited_check_ids": sorted(set(check_by_id) - covered),
            }})
        check_hint_literals(lines, public, sensitive)
        if variant.endswith("v5") and re.search(
            r"retrieved_\d+|task_evidence|copy_from|/manifest/|\bprior_read\b|\bprior_email_read\b|\bearlier_same\b",
            "\n".join(lines),
        ):
            raise ValueError("internal source bookkeeping in solver instructions")
        return lines

    def review_passed(review):
        rows = _rows(review, "checks")
        _exact_ids(rows, "check_id", check_by_id)
        return (all(r.get("covered") is True and _text(r.get("reason")) for r in rows)
                and all(review.get(k) is True for k in ("approved", "workflow_faithful", "no_private_literals"))
                and review.get("issues") == [])

    try:
        lines = validate_assembly(assembled)
    except ValueError as error:
        if not repair_invalid_assembly:
            raise
        issue = retrieval_only_review(str(error), sensitive)
        record("assembly_validation_failure", {"input": {"task": material["task"]}, "output": issue})
        assembled = await stage("assembly_repair", ASSEMBLY_PROMPT +
            "\nRepair the validator error. depends_on contains only earlier STEP IDs s1,s2,..., "
            "never action IDs a1,a2 or the current/future step. Keep all original source constraints. "
            "Return the whole corrected assembly, not a critique.",
            {**common, "requirements": requirements, "workflow": workflow,
             "previous_plan": retrieval_only_review(assembled, sensitive), "validation_error": issue})
        lines = validate_assembly(assembled)
    review = await stage("review", REVIEW_PROMPT, {**common, **review_sources, "proposed_instructions": assembled})
    approved = review_passed(review)
    if not approved and variant.startswith("scope-repair"):
        repair_review = retrieval_only_review(review, sensitive) if policy else review
        if grounded or policy:
            requirements = await stage("requirements_repair", REQUIREMENTS_PROMPT +
                                       "\nCorrect the previous requirement interpretations using the original "
                                       "observation-bound contract and review. Distinguish sender email domain "
                                       "from sender display name. Return all eligible checks once; no concrete "
                                       "historical values, invented conditions or new requirements.",
                                       {**common, "manifest": generation_contract,
                                        "previous_requirements": requirements, "review": repair_review})
            validate_requirements(requirements)
        assembled = await stage("assembly_repair", ASSEMBLY_PROMPT +
                                "\nRevise the previous plan using the cited review issues and original requirements. "
                                "Remove invented preconditions; preserve required actions and narrow guard scope. "
                                "Return the complete corrected plan, not a critique. No new requirements.",
                                {**common, "requirements": requirements, "workflow": workflow,
                                 "previous_plan": assembled, "review": repair_review})
        lines = validate_assembly(assembled)
        review = await stage("review_repair", REVIEW_PROMPT,
                             {**common, **review_sources, "proposed_instructions": assembled})
        approved = review_passed(review)
        if not approved:
            raise ValueError("bounded assembly repair rejected; retain evidence, no solver dispatch")
    if not approved:
        if variant.endswith(("v4", "v5")):
            record(
                "review_rejected_diagnostic_only",
                {
                    "input": {"task": material["task"]},
                    "output": {
                        "sft_qualified": False,
                        "native_execution_purpose": "Test critic judgement in simulated world; not SFT admission.",
                    },
                },
            )
            return lines
        raise ValueError("same-model instruction review rejected; retain evidence, no solver dispatch")
    if "natural" in variant:
        # Retain the checked structures while avoiding labels that invite the
        # actor to cite private preparation as evidence.
        lines = [r["instruction"] for r in requirements["requirements"]] + [
            "Follow the procedure in order; wait for observed tool results before dependent actions."
        ] + [step["instruction"] for step in assembled["steps"]]
        check_hint_literals(lines, public, sensitive)
        record("natural_actor_guidance", {"input": {"task": material["task"]}, "output": lines})
    if policy:
        require_policy_retrieval(lines, policy_contracts)
    return lines
