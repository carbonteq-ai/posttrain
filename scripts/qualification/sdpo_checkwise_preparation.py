"""Same4B check-by-check teacher briefing; no assembled action-ID DAG."""

import asyncio
import copy
import re

from luna_argument_roles import public_equivalence
from luna_effect_scope import effect_scopes
from luna_policy_guidance import retrieval_only_review
from luna_solver_prompts import check_hint_literals
from luna_solver_separation import argument_provenance, observation_bound_sources
from luna_solver_stages import _sensitive_values
from sdpo_effect_brief import operational_effect, population_binding

RULES = """Prepare teacher-only guidance for executing the public task with real tools.
Hidden evidence describes requirements, not observations already obtained.
Obtain current IDs, policy values and record contents through actual tools.
Evaluation predicates such as request_key, candidate, event collections and
effect joins describe evaluator scope; they are NOT additional API parameters
or instructions to read an invisible manifest. Explain their operational
meaning. Preserve required read-before-write dependencies and constraints on
actions in this execution. Do not invent global audit tools or new conditions.
For effects.required_when/prohibited_when, applicability and effect_match are
different clauses: the check concerns matching effects WHEN applicable, not
all effects whenever the selector is active. NOT(recipient equals target)
matches a WRONG recipient, not the correct requested recipient. Keep join
constraints too; a same-execution duplicate guard allows the first required
action and prohibits a second matching action.
Historical attempts guide procedure only; failed/empty attempts are not success.
An imperfect historical task outcome can include wrong classifications or
unnecessary notifications even when every API call returned success. Explicit
historical_reference_quality findings are counterexamples to copying those
actions, not desired outputs. Current requirements outrank the example.
Use documented canonical parameter names. Similar-looking names are not aliases
unless the current tool contract explicitly declares that equivalence.
A success:true response is operation status, not proof
that every requested field was applied. When returned records expose applied
fields, verify those actual fields before claiming the corresponding outcome.
historical_reference_scope is preparation provenance, not an actor task.
Never refer to a hidden binding, source contract, or record for this check.
When the public task names a recipient/channel without an ID, retrieve its ID;
do not claim the public request supplied that ID.
Never put its keys or state-match status into execution instructions; express
the required current-source retrieval instead of discussing the reference.
Do not quote internal check IDs in an instruction. Dates and search terms
derived from the public request are allowed; hidden answer literals are not.
Write ordinary task language: no references to teachers, annotation, datasets,
privileged context or Luna. Say returned rows or observed results instead.
Do not quote internal source-adapter identifiers or version tags.
In procedures use 'rows' or 'records', never the word 'dataset', even for a table.
Numeric thresholds in private examples or interpreted requirements are not fixed
execution rules. Read the applicable current policy, then compare against its
returned thresholds; never repeat private example thresholds in a procedure.
Never mention a request key, check key or check ID, even with spaces instead
of underscores. These are labels selecting this already identified task, not
conditions the executor must establish. Omit that selector clause entirely:
state the operational outcome directly, without a selector clause. Do not
borrow facts or operations from an example unrelated to this task. The check_id JSON field is only
for output association and must never appear in the instruction prose.
Use scoped_event as the authoritative operation being assessed. A read event
concerns retrieving the original record; it is not a send event and does not
require the newly sent message ID to equal the original message ID. Each check
brief explains only that check. Never add batch processing to a single reply.
Source feasibility matters: a private expected value is not evidence that an
available tool can retrieve it. Never invent a directory, policy document,
sheet, field or mailbox to justify that value. Distinguish the target message
from a separate policy message. For a policy-specified note title or threshold,
instruct retrieval and verbatim application of that policy, not a guessed label.
When a prohibition targets one record selected by its observed hold/eligibility
status, preserve that restriction; do not expand it to all summary records.
An evaluator's request object is not a runtime read source. Resolve named public
targets using actual exposed tools and their returned records only.
Where a check declares a population, its request variable means ONE RECORD from
that population. In a contacts population, request.id is the Salesforce contact
ID; it is not an email request ID. Compare effects to the same population record.
Never compare a person's name or email ID with a Salesforce contact ID.
Treat returned field names as exact contracts: do not invent aliases such as
Opted In for a returned SMS Opt-In header. State selection semantics and use
the actual headers returned at runtime. Keep different audiences/services
separate; facilities email does not authorize extra employee emails, and Gmail
sent-mail search cannot verify Slack delivery. Publicly provided IDs need not
be rediscovered. A reviewer must not reject their direct use as invented IDs.
"""


def preparation_rules(material):
    tools = {definition["function"]["name"] for definition in material["tool_definitions"]}
    if "asana_create_task" in tools:
        return RULES + ("\nFor asana_create_task, workspace_info does not establish the workspace field "
                        "merely because the tool accepted it. Use the documented applied workspace field.")
    return RULES


def normalize_instruction_wording(value):
    """Canonicalize one cosmetic noun; never edit sources, IDs or predicates."""
    if isinstance(value, list):
        return [normalize_instruction_wording(v) for v in value]
    if isinstance(value, dict):
        return {k: (re.sub(r"\bdatasets?\b", "records", v, flags=re.I)
                    if k == "instruction" and isinstance(v, str)
                    else normalize_instruction_wording(v)) for k, v in value.items()}
    return value


def validate_instruction_text(material, lines, sensitive):
    public = material["user_request"]
    declared_fields = set()
    for definition in material["tool_definitions"]:
        declared_fields.update(definition["function"].get("parameters", {}).get("properties", {}))
    text = "\n".join(lines)
    for field in ("request_key", "effect_match", "effect_join", "manifest"):
        pattern = field.replace("_", r"[\s_-]+")
        if (re.search(rf"\b{pattern}\b", text, re.I)
                and field not in declared_fields and field.lower() not in public.lower()):
            raise ValueError(f"evaluator bookkeeping used as an execution instruction: {field}; "
                             "omit the entire selector clause, retaining the operational outcome")
    if re.search(r"\b(?:request|check)\s+(?:with\s+)?(?:key|id)\b", text, re.I):
        raise ValueError("evaluator bookkeeping selector in prose; omit its entire clause")
    if re.search(r"\b(?:binding|source contract|for this check|current check)\b", text, re.I):
        raise ValueError("hidden source bookkeeping in execution prose; use actual runtime retrieval")
    try:
        check_hint_literals(lines, public, sensitive)
    except ValueError as error:
        if "generation bookkeeping" in str(error):
            words = re.findall(r"\b(?:teacher|annotation|dataset|privileged|luna)\b", text, re.I)
            raise ValueError(f"{error}; replace these words with ordinary records/rows: {words}") from error
        raise


def abstraction_text(text, sensitive, public):
    """Remove private instance strings from text-only rewrites, never typed ASTs."""
    identifiers = re.findall(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|\b(?:msg_|ws_|ss_|CPART|hv_|hu_)[\w-]+", text)
    identifiers += re.findall(r"\b(?:[A-Za-z0-9-]+\.)+(?:com|org|net|io|co|ai|edu|gov|example)\b", text)
    private = [*sensitive, *(value for value in identifiers if value not in public)]
    return retrieval_only_review(text, private)


def abstract_workflow_requirements(requirements, material, sensitive):
    """Hide privately observed numeric examples before procedural synthesis."""
    public = material["user_request"]
    public_numbers = set(re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])", public))
    numbers = set()

    def walk(value):
        if type(value) in (int, float) and (abs(value) >= 10 or value != int(value)):
            numbers.update((str(value), format(value, "g")))
        elif isinstance(value, str) and re.fullmatch(r"\d+(?:\.\d+)?", value):
            if float(value) >= 10 or float(value) != int(float(value)):
                numbers.update((value, format(float(value), "g")))
        elif isinstance(value, dict):
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(material["reference_actions"])
    result = []
    for requirement in requirements:
        text = abstraction_text(requirement["instruction"], sensitive, public)
        for number in sorted(numbers - public_numbers, key=len, reverse=True):
            text = re.sub(r"(?<![\w.])" + re.escape(number) + r"(?![\d.])",
                          "[value from the currently retrieved source/policy]", text)
        result.append({**requirement, "instruction": text})
    return result


def source_provenance(material, checks):
    """Private, exact source records for instance literals; no asserted gold actions."""
    literals = set()

    def collect(value):
        if isinstance(value, dict):
            if value.get("kind") == "literal" and isinstance(value.get("value"), str):
                if len(value["value"]) >= 5:
                    literals.add(value["value"])
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(checks)
    found, omitted = {}, 0

    def walk(value, path=(), parent=None):
        nonlocal omitted
        if isinstance(value, str) and any(s.casefold() in value.casefold() for s in literals):
            record = parent if isinstance(parent, dict) else value
            key = str(path[:-1])
            matches = {s for s in literals if s.casefold() in value.casefold()}
            if len(str(record)) > 6000:
                omitted += 1
            elif key in found:
                found[key]["matches"].update(matches)
            else:
                found[key] = {"source_path": list(path[:-1]), "record": copy.deepcopy(record), "matches": matches}
        elif isinstance(value, dict):
            for key, child in value.items():
                walk(child, (*path, key), value)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, (*path, index), value)

    walk(material["native_data"]["initial_state"])
    # Broad class literals can match many rows before a rare excluded entity.
    # Prefer coverage of rare bindings, rather than the first records in storage.
    from collections import Counter

    frequency = Counter(s for record in found.values() for s in record["matches"])
    remaining, selected, covered, used = list(found.values()), [], set(), 0
    while remaining and len(selected) < 12:
        ranked = sorted(remaining, key=lambda r: sum(1 / frequency[s] for s in r["matches"] - covered), reverse=True)
        record = next((r for r in ranked if used + len(str(r["record"])) <= 12000), None)
        if record is None or not record["matches"] - covered:
            break
        selected.append({k: v for k, v in record.items() if k != "matches"})
        covered.update(record["matches"])
        used += len(str(record["record"]))
        remaining.remove(record)
    return {"records": selected, "omitted_records": omitted + len(remaining),
        "unrepresented_literal_count": len(literals - covered),
        "scope": "Private exact initial-source evidence for interpreting entity roles and policy provenance. "
                 "Not current actor observations or permission to copy literal answers. "
                 "A value found only in Salesforce is not an ID provided by an email. "
                 "Keep hold/excluded-entity attributes distinct from all otherwise eligible employees."}


def grounded_review_payload(material, bundle):
    return {"public_request": material["user_request"], "tools": material["tool_definitions"],
        "source_checks": material["goal_manifest"]["checks"],
        "source_definitions": material["goal_manifest"]["sources"],
        "private_current_source_provenance": source_provenance(material, material["goal_manifest"]["checks"]),
        "retained_reference_actions": material["reference_actions"],
        "historical_reference_quality": material.get("historical_reference_quality"),
        "proposed": {"instructions": bundle["instructions"]}}


GROUNDED_REVIEW = """Review the actual COMPLETE guide against the exact sources.
Check every goal/harm role, applicability and read-before-write relation. Check
whether a prohibition for one excluded/held record was broadened to every eligible
record, contradicting a goal. Reject invented note titles, fixed policy thresholds,
unobserved IDs or source locations and any assumption that an email/public request
provides an ID absent there. Read source provenance to determine where each value
can actually be retrieved. A record ID in Salesforce is not an ID in an email.
The actor must retrieve and follow policy content, never invent a replacement.
Reject invisible collection readers, unknown tools, assumed standard locations
and known-address fallbacks. Require original-message/policy retrieval where the
source checks require it. Review semantic correctness, not style. Approve only
if every requirement is preserved with no unsupported addition. Approved requires
issues=[]; any substantive issue requires approved=false. At most three issues.
"""


def schema(stage, data):
    text = {"type": "string"}

    def obj(properties):
        return {"type": "object", "properties": properties, "required": list(properties),
                "additionalProperties": False}

    stage = stage.removesuffix("_repair")
    if stage in ("check_brief", "check_abstraction"):
        return obj({"check_id": {"type": "string", "enum": [data["check"]["check_id"]]},
                    "role": {"type": "string", "enum": [data["check"]["role"]]}, "instruction": text})
    if stage == "check_review":
        return obj({"approved": {"type": "boolean"}, "reason": text,
                    "issues": {"type": "array", "items": text, "maxItems": 3}})
    if stage == "workflow_brief":
        return obj({"steps": {"type": "array", "minItems": 1, "maxItems": 16,
            "items": obj({"tool": {"type": "string", "enum": data["allowed_tools"]}, "instruction": text})}})
    if stage in ("workflow_review", "bundle_review"):
        return obj({"approved": {"type": "boolean"}, "reason": text,
                    "issues": {"type": "array", "items": text, "maxItems": 3}})
    raise ValueError("unknown checkwise preparation stage")


def reference_scope(material, binding):
    from sdpo_guidance_context import digest

    if binding is not None:
        if (binding.get("current_native_state_sha256") != digest(material["native_data"]["initial_state"])
                or binding.get("current_manifest_sha256") != digest(material["goal_manifest"])
                or type(binding.get("historical_state_equal")) is not bool
                or type(binding.get("historical_assertions_equal")) is not bool):
            raise ValueError("historical reference binding differs from the current task source")
    return {"source_binding_verified": binding is not None,
            "state_matches_current_task": None if binding is None else binding["historical_state_equal"],
            "assertions_match_current_task": None if binding is None else binding["historical_assertions_equal"],
            "binding_sha256": None if binding is None else digest(binding),
            "authority": "Historical procedure example only, even when source states match. "
                         "It is not a current observation or a perfect-reference claim. "
                         "Current public instructions and the current source contract outrank it. "
                         "Do not transfer old entity values, policy answers, dates or selection rules. "
                         "Require runtime retrieval of current source facts."}


async def prepare(material, generate, record, *, private_preparation=False, historical_binding=None,
                  parallel_checks=False, abstract_private_briefs=False):
    if abstract_private_briefs and not private_preparation:
        raise ValueError("private abstraction requires private source interpretation")
    rules = preparation_rules(material)
    contract, actions, separation = observation_bound_sources(material)
    if private_preparation:
        # Interpret the real typed contract before redacting the OUTPUT. The
        # observation-bound view can mask nested literal.kind metadata or give
        # different records the same generic retrieval description. It is not
        # a faithful input AST for source/polarity interpretation.
        contract = copy.deepcopy(material["goal_manifest"])
        separation = {**separation, "preparation_model_receives_raw_contract": True,
                      "student_receives_raw_contract": False,
                      "output_boundary": "All compiled instructions still undergo private-literal and metadata checks."}
    provenance = argument_provenance(material, schema_roles=True)
    for action, origin in zip(actions, provenance, strict=True):
        action["call"]["arguments"] = origin["argument_provenance"]
    checks = [c for c in contract["checks"] if c["role"] in ("goal", "harm")]
    if not checks:
        raise ValueError("no source-backed goals/harms")
    sensitive = [v for v in _sensitive_values({"contract": material["goal_manifest"],
                 "actions": material["reference_actions"],
                 "reference_quality": material.get("historical_reference_quality", {})})
                 if not public_equivalence(v, material["user_request"])]
    public = material["user_request"]
    common = {"public_request": public, "tools": material["tool_definitions"],
              "source_bindings": contract.get("sources", {}), "historical_procedure": actions,
              "historical_reference_scope": reference_scope(material, historical_binding)}
    common["population_field_ownership"] = {
        check["check_id"]: population_binding(check, contract.get("sources", {}))
        for check in checks}
    if material.get("historical_reference_quality") is not None:
        quality = material["historical_reference_quality"]
        common["historical_reference_quality"] = (quality if private_preparation else {
            "partial_credit": quality.get("partial_credit"), "success": quality.get("success"),
            "excluded_assertions": quality.get("excluded_assertions"),
            "known_failed_assertion_count": len(quality.get("failed_assertions", [])),
            "scope": "Imperfect historical procedure, not a correct action label; private findings withheld."})
    if private_preparation:
        common["private_preparation_evidence"] = {
            "retained_reference_actions": material["reference_actions"],
            "scope": "Offline historical observations for the preparation model only; not current observations or student inputs.",
            "output_boundary": "Use this to understand required source retrieval and dependencies. "
                "Do not copy private IDs, policy values, answer text, status filters or date windows into the guide. "
                "Tell the actor to retrieve and apply the applicable source rules at runtime. "
                "The public task and current source contract outrank a historical procedure. "
                "Do not invent latest-email selection or infer a recency cutoff from today's date."}
    record("checkwise_source_separation", {"input": {"task": material["task"]}, "output": separation})
    record("historical_reference_binding", {"input": {"task": material["task"]},
                                           "output": common["historical_reference_scope"]})

    async def stage(name, prompt, data):
        if private_preparation and name.startswith(("check_review", "workflow_review")):
            prompt += (" Compare every proposed selection criterion with the retained evidence. "
                "Reject unsupported statuses, date comparisons, latest/first record assumptions, "
                "or copying a policy answer instead of retrieving the policy. A missing source rule "
                "must remain a runtime retrieval requirement, not a guessed business rule. "
                "Retrieving the applicable current policy and applying its criteria is the correct representation "
                "of a private criterion; do not demand copied private IDs or expected answer values. "
                "Reject substantive unsupported constraints, not stylistic preferences. Approved requires issues=[].")
        value = await generate(name, rules + "\n" + prompt, data)
        normalized = normalize_instruction_wording(value)
        if normalized != value:
            record("cosmetic_wording_normalization", {"input": value, "output": normalized})
            value = normalized
        record(name, {"input": data, "output": value})
        return value

    scopes = {r["check_id"]: r for r in effect_scopes(contract)}

    async def prepare_check(check):
        scope = scopes[check["check_id"]]
        source = contract.get("sources", {}).get(check.get("source"), {})
        data = {**common, "check": check, "effect_scope": scope,
                "scoped_event": {"kind": scope.get("event_kind", source.get("kind")),
                                 "rule": "Interpret effect fields as fields of THIS event, not a later action."},
                "operational_effect": operational_effect(check, contract.get("sources", {}))}
        if private_preparation:
            data["private_current_source_provenance"] = source_provenance(material, [check])
        brief = await stage("check_brief", "Return only a SHORT outcome or constraint for this ONE check, "
            "not a whole procedure or other checks. Use operational_effect to distinguish applicability "
            "from the actual required/prohibited effect, including negation. Preserve field, condition, "
            "scope and required order. Translate concrete instance bindings into runtime retrieval relationships: "
            "names, IDs, exact answer text and private values identify source records, but must NOT be repeated "
            "in the instruction. Explain which currently observed record supplies each value and how to identify "
            "the target from the public request. For policy-derived decisions, retrieve the policy and apply it; "
            "do not disclose the expected classification or policy answer. Do not add assumptions.", data)
        if abstract_private_briefs:
            # Source interpretation and executable abstraction have different
            # information boundaries. The rewrite cannot re-anchor on the full
            # raw contract/reference while removing its concrete instance facts.
            brief = await stage("check_abstraction", "Rewrite this private interpretation into ONE short "
                "runtime instruction. Keep its operation, fields, conditions, polarity and ordering, but replace "
                "instance IDs/names/addresses/answer values with how to retrieve and match the current source. "
                "Refer to the public task and observed records, never a check/source record for this check. "
                "No exact private source strings may appear in instruction. Do not invent new conditions. "
                "A policy answer becomes retrieve the applicable policy and apply it.",
                {"public_request": public, "check": {"check_id": check["check_id"], "role": check["role"]},
                 "private_interpretation": {**brief,
                    "instruction": abstraction_text(brief["instruction"], sensitive, public)},
                 "withheld_bindings": "Bracketed placeholders replace private instance values. "
                    "Do not reproduce placeholders; explain their runtime retrieval relationship.",
                 "allowed_tools": [t["function"]["name"]
                    for t in material["tool_definitions"]],
                 "polarity": ("FORBID the matched action; the applicability condition is not a requested action."
                              if check["role"] == "harm" else "REQUIRE the matched action when applicable."),
                 "field_ownership": {k: v for k, v in
                     population_binding(check, contract.get("sources", {})).items() if k != "source"},
                 "private_source_literals_to_remove": []})
        for attempt in range(2):
            if brief.get("check_id") != check["check_id"] or brief.get("role") != check["role"]:
                raise ValueError("check identity/role changed")
            instruction = brief.get("instruction")
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError("empty check instruction")
            try:
                validate_instruction_text(material, [instruction], sensitive)
            except ValueError as error:
                review = {"approved": False, "reason": "Literal/metadata validator rejected the instruction",
                          "issues": [retrieval_only_review(str(error), sensitive)]}
                record("check_validation_failure", {"input": {"task": material["task"],
                    "check_id": check["check_id"]}, "output": review})
            else:
                review = await stage("check_review", "Review this SINGLE interpretation. "
                    "An operational instruction need not expose evaluator bookkeeping fields or secret bindings. "
                    "Reject new facts, lost constraints or unsupported retrieval. Verify that the operation "
                    "matches scoped_event: read_message requires reading the original message, not matching "
                    "the ID of a newly sent reply. Reject unrelated batch-summary instructions. Be concise: one sentence "
                    "for reason, at most three brief issues. Approved requires no issues.", {**data, "proposed": brief})
            if review.get("approved") is True and review.get("issues") == []:
                break
            if attempt:
                raise ValueError(f"checkwise review rejected: {check['check_id']}")
            private_repair = ({"private_source_literals_to_remove": [value for value in sensitive
                if isinstance(value, str) and value and value.casefold() in instruction.casefold()],
                "boundary": "These exact strings identify copied private evidence in the previous instruction. "
                    "They are visible only to this preparation repair, never authorized guide content. "
                    "Replace each with its current-source retrieval relationship, not a different guessed value."}
                if private_preparation else {})
            previous, repair_review = brief, review
            if abstract_private_briefs:
                previous = {**brief, "instruction": abstraction_text(brief["instruction"], sensitive, public)}
                repair_review = {**review,
                    "reason": abstraction_text(review["reason"], sensitive, public),
                    "issues": [abstraction_text(v, sensitive, public) for v in review["issues"]]}
                private_repair = {"withheld_bindings": "Private strings have been replaced with retrieval placeholders. "
                    "Resolve them into runtime instructions; never copy placeholders or invent replacement IDs."}
            brief = await stage("check_brief_repair", "Correct the instruction using the review, "
                "preserving the original check. Remove evaluator bookkeeping/adapter labels; use "
                "ordinary task language and actual tool names. Remove EVERY private_source_literals_to_remove "
                "string from the instruction. Replace concrete names, IDs and expected values with instructions "
                "to retrieve and match the corresponding current source records. Never preserve those strings "
                "merely because the raw check contains them. Return a complete replacement instruction.",
                {**({"public_request": public, "check": {"check_id": check["check_id"], "role": check["role"]},
                     "allowed_tools": [t["function"]["name"] for t in material["tool_definitions"]]}
                    if abstract_private_briefs else data),
                 **private_repair, "previous": previous, "review": repair_review})
        return brief

    if parallel_checks:
        # Checks have independent source inputs; each still requires its own
        # review. Await every child, including failures, before rejecting the
        # task so no unobserved model calls outlive a rejected preparation.
        results = await asyncio.gather(*(prepare_check(check) for check in checks), return_exceptions=True)
        error = next((result for result in results if isinstance(result, BaseException)), None)
        if error is not None:
            raise error
        requirements = [result for result in results if isinstance(result, dict)]
        if len(requirements) != len(checks):
            raise ValueError("parallel check preparation lost a source requirement")
    else:
        requirements = [await prepare_check(check) for check in checks]
    allowed = [t["function"]["name"] for t in material["tool_definitions"]]
    procedural_requirements = (abstract_workflow_requirements(requirements, material, sensitive)
                               if abstract_private_briefs else requirements)
    data = {**({"public_request": public, "tools": material["tool_definitions"]}
               if abstract_private_briefs else common),
            "requirements": procedural_requirements, "allowed_tools": allowed}
    workflow = await stage("workflow_brief", "Give a concise ordered retrieval/execution procedure. "
        "Use native tools by name; no action-ID or step-ID dependency graph. "
        "Read returned results before dependent writes. All requirements remain authoritative.", data)
    for attempt in range(2):
        steps = workflow.get("steps")
        if (not isinstance(steps, list) or not steps
                or any(s.get("tool") not in allowed or not s.get("instruction") for s in steps)):
            raise ValueError("invalid workflow/tools")
        try:
            validate_instruction_text(material, [s["instruction"] for s in steps], sensitive)
        except ValueError as error:
            review = {"approved": False, "reason": "Literal/metadata validator rejected the procedure",
                      "issues": [retrieval_only_review(str(error), sensitive)]}
            record("workflow_validation_failure", {"input": {"task": material["task"]}, "output": review})
        else:
            review = await stage("workflow_review", "Review the procedure for invented operations, "
                "wrong scope, contradicted requirements or hidden-value shortcuts. "
                "It is not required to execute evaluator predicates or expose source IDs. "
                "One sentence reason and at most three brief issues; approved requires no issues.",
                {**data, "proposed": workflow})
        if review.get("approved") is True and review.get("issues") == []:
            break
        if attempt:
            raise ValueError("checkwise workflow review rejected")
        previous, repair_review = workflow, review
        if abstract_private_briefs:
            previous = {"steps": [{**s, "instruction": abstraction_text(s["instruction"], sensitive, public)}
                                  for s in steps]}
            repair_review = {**review, "reason": abstraction_text(review["reason"], sensitive, public),
                             "issues": [abstraction_text(v, sensitive, public) for v in review["issues"]]}
        workflow = await stage("workflow_brief_repair", "Repair only the cited procedure issues; "
            "preserve all requirements and native-tool constraints. Bracketed private-value placeholders "
            "must become runtime retrieval instructions, never guessed or copied identifiers.",
            {**data, "previous": previous, "review": repair_review})
    lines = [r["instruction"] for r in requirements] + [
        "Wait for each read or write result before issuing a dependent action."] + [s["instruction"] for s in steps]
    if private_preparation:
        # Individually approved interpretations can still contradict one another.
        # This opt-in candidate gate rejects the whole bundle rather than silently
        # deleting a constraint or repairing only the workflow around a bad check.
        review = await stage("bundle_review", "Independently review the COMPLETE assembled guide against "
            "the original source contract and retained historical evidence. Individual earlier approvals "
            "are not evidence of semantic correctness. Check every required/prohibited effect and its "
            "applicability condition; a prohibition on a decoy must not become a prohibition on the target. "
            "Reject mutually contradictory instructions, invented source-selection rules, statuses, dates "
            "or API parameters. Do not treat a bookkeeping selector as an execution instruction. "
            "If policy contents are private, require retrieving the applicable policy and following it, "
            "rather than guessing its rules. Approve only if the complete guide preserves every source "
            "requirement without adding an unsupported requirement. Retrieval of applicable current policy "
            "and application of its criteria represents private criteria without exposing their answers. "
            "Do not require copying concrete private IDs or example answers. Report substantive semantic "
            "defects only, not style or optional improvements. Approved MUST have issues=[]; any substantive "
            "issue requires approved=false. At most three concise issues.\n" + GROUNDED_REVIEW,
            {**common, "source_checks": checks, "requirements": requirements,
             "private_current_source_provenance": source_provenance(material, checks),
             "proposed": {"instructions": lines, "workflow": workflow}})
        if review.get("approved") is not True or review.get("issues") != []:
            raise ValueError("checkwise assembled source review rejected")
    return {"instructions": lines, "requirements": requirements, "review_approved": True,
            "historical_reference_scope": common["historical_reference_scope"]}
