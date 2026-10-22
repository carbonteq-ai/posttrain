"""Teacher-only manifest context and factual tool observations; no reward oracle."""

import hashlib
import json
import re


def authority_errors(manifest, initial_state, full_prompt):
    material = {'task_evidence': {'initial': initial_state, 'prompt': full_prompt}}
    errors = []
    for binding in manifest['bindings']:
        try:
            value = material
            for key in binding['path']:
                value = value[key]
            digest = hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                               separators=(',', ':')).encode()).hexdigest()
            if digest != binding['canonical_sha256']:
                errors.append({'path': binding['path'], 'reason': 'manifest_authority_digest_mismatch'})
        except (KeyError, IndexError, TypeError):
            errors.append({'path': binding['path'], 'reason': 'manifest_authority_path_unavailable'})
    return errors


def load_goal_context(task_name, initial_state=None, full_prompt=None):
    # Read the packaged, pinned environment's declarations, without changing or
    # running its evaluator or using heldout results.
    from importlib.resources import files

    from automationbench_v1.contracts.loader import load_task_contract

    root = files("automationbench_v1.contracts")
    catalog = json.loads(root.joinpath("catalog.json").read_text())
    relative = catalog["tasks"].get(task_name)
    if relative is None:
        raise ValueError(f"missing installed goal manifest: {task_name}")
    raw = root.joinpath(relative).read_bytes()
    manifest = json.loads(raw)
    load_task_contract(task_name)  # Reuse the pinned native schema/public-authority admission.
    errors = authority_errors(manifest, initial_state, full_prompt) if full_prompt is not None else []
    checks = [{"id": c["check_id"], "role": c["role"], "signal": c["signal_id"],
               "operator": c["operator"], "source": c.get("source"), "population": c.get("population"),
               "join_sources": [j["source"] for j in c.get("effect_joins", [])]}
              for c in manifest["checks"] if c["role"] in ("goal", "harm")]
    expected_senders = []
    read_sources = {name for name, spec in manifest['sources'].items()
                    if spec.get('adapter') == 'gmail.message_reads@1'}
    has_read_check = any(c.get('source') in read_sources or set(c.get('join_sources', [])) & read_sources
                         for c in checks)
    if initial_state is not None and has_read_check and not errors:
        material = {'task_evidence': {'initial': initial_state}}
        for source in manifest['sources'].values():
            if source.get('adapter') != 'public.request@1':
                continue
            for field in source.get('fields', {}).values():
                path = field.get('copy_from', [])
                if path[:4] != ['task_evidence', 'initial', 'gmail', 'messages'] or path[-1:] != ['from_']:
                    continue
                value = material
                for key in path:
                    value = value[key]
                if isinstance(value, str) and '@' in value:
                    expected_senders.append({'sender': value, 'authority_path': path})
    return {"task": task_name, "feedback_revision": "goals-observations-v4",
            "public_request": manifest["public_request"], "manifest_id": manifest["manifest_id"],
            "revision": manifest["revision"], "sha256": hashlib.sha256(raw).hexdigest(),
            "declared_checks": checks,
            "sources": {name: {k: spec[k] for k in ("adapter", "kind") if k in spec}
                        for name, spec in manifest["sources"].items()},
            "expected_source_senders": expected_senders,
            "authority_status": 'mismatch' if errors else 'verified' if full_prompt is not None else 'declarations_only',
            "authority_errors": errors,
            "assessment_status": "declarations only; not evaluated findings"}


def goal_links(tool, goals):
    """Map qualified tool families to manifest source roles, not goal completion."""
    families = {
        "gmail_find_email": ("gmail.message_reads@1", None),
        "gmail_get_email_by_id": ("gmail.message_reads@1", None),
        "gmail_reply_to_email": ("gmail.messages@1", "send"),
        "gmail_send_email": ("gmail.messages@1", "send"),
        "slack_send_channel_message": ("slack.messages@1", "channel_message"),
    }
    adapter, kind = families.get(tool, (None, None))
    matched = {name for name, source in goals.get("sources", {}).items()
               if adapter and source.get("adapter") == adapter and (kind is None or source.get("kind") == kind)}
    links = []
    for check in goals["declared_checks"]:
        relation = None
        if check.get("source") in matched:
            relation = "candidate_evidence_for_check"
        elif set(check.get("join_sources", [])) & matched:
            relation = "prerequisite_evidence"
        elif tool.startswith("google_sheets_get") and any(
            goals["sources"].get(s, {}).get("adapter") == "google_sheets.rows@1"
            for s in (check.get("source"), check.get("population"))
        ):
            relation = "prerequisite_input_lookup"
        if relation:
            links.append({"check_id": check["id"], "role": check["role"], "relation": relation,
                          "completion": "not evaluated"})
    return links


def repair_candidates(call, feedback, goals=None):
    """Conservative proposals derived from actual errors, never correctness labels."""
    responses = {item['tool']: json.loads(item['response']) for item in json.loads(feedback)
                 if item['response'].lstrip().startswith('{')}
    repairs = []
    for tool in call.get('message', {}).get('tool_calls', []):
        function = tool['function']
        args = json.loads(function['arguments'])
        response = responses.get(function['name'], {})
        corrected = dict(args)
        reason = None
        if function['name'] == 'gmail_find_email' and response.get('result_count') == 0:
            query = args.get('query', '')
            match = re.search(r'from:([A-Za-z0-9_.+-]+)@([A-Za-z0-9.-]+)', query)
            if match:
                # Assistant-generated guesses are not evidence of an address.
                evidence = '\n'.join(m.get('content') or '' for m in call.get('messages', [])
                                     if m.get('role') in ('user', 'tool'))
                if match.group(0)[5:] not in evidence:
                    senders = sorted({s['sender'] for s in (goals or {}).get('expected_source_senders', [])})
                    reference = senders[0] if len(senders) == 1 else match.group(1)
                    corrected['query'] = query[:match.start()] + 'from:' + reference + query[match.end():]
                    reason = ('manifest-bound source identity supplies the expected sender'
                              if len(senders) == 1 else
                              'empty search with an address unsupported by user/tool evidence; broaden name discovery')
        if function['name'].startswith('google_sheets_') and response.get('success') is False:
            error = str(response.get('error', ''))
            for stem, keys in [('ss_', ('spreadsheet', 'spreadsheet_id')), ('ws_', ('worksheet', 'worksheet_id'))]:
                identities = sorted(set(re.findall(r'\((' + stem + r'[A-Za-z0-9_-]+)\)', error)))
                if len(identities) == 1:
                    key = next((k for k in keys if k in args), None)
                    if key:
                        corrected[key] = identities[0]
                        reason = 'actual tool error lists one known raw resource ID; avoid the combined display label'
                        break
        if reason and corrected != args:
            repairs.append({'function': function['name'], 'arguments': corrected, 'basis': reason,
                            'status': 'candidate for this pre-action decision, not a scored success label'})
    return repairs


def teacher_message(call, feedback, goals=None):
    if goals is None:
        return f"Observed tool response to an attempted action: {feedback}\nChoose the action for the original request."
    observations = []
    for item in json.loads(feedback):
        try:
            response = json.loads(item["response"])
        except (ValueError, TypeError):
            continue
        if not isinstance(response, dict):
            continue
        if response.get("success") is False:
            observations.append({"tool": item["tool"], "status": "tool_error",
                                 "related_checks": goal_links(item["tool"], goals),
                                 "error": response.get("error")})
        if response.get("result_count") == 0 and response.get("messages") == []:
            attempted = {json.dumps(json.loads(t["function"]["arguments"]), sort_keys=True)
                         for t in call.get("message", {}).get("tool_calls", [])
                         if t["function"]["name"] == item["tool"]}
            previous = {}
            repeats = 0
            for message in call.get("messages", []):
                for tool in message.get("tool_calls", []):
                    function = tool["function"]
                    arguments = function["arguments"]
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                    previous[tool["id"]] = (function["name"], json.dumps(arguments, sort_keys=True))
                if message.get("role") == "tool" and previous.get(message.get("tool_call_id"), (None, None))[0] == item["tool"]:
                    try:
                        prior = json.loads(message["content"])
                    except (ValueError, TypeError):
                        continue
                    if (previous[message["tool_call_id"]][1] in attempted and isinstance(prior, dict)
                            and prior.get("result_count") == 0 and prior.get("messages") == []):
                        repeats += 1
            observations.append({"tool": item["tool"],
                                 "status": "repeated_empty_search" if repeats else "empty_search_result",
                                 "prior_identical_empty_results": repeats,
                                 "related_checks": goal_links(item["tool"], goals),
                                 "progress_this_action": "no matching message-read evidence produced",
                                 "meaning": "No matching message was retrieved. API success is not retrieval success."})
    related = {v['check_id']: v for item in json.loads(feedback) for v in goal_links(item['tool'], goals)}
    return (
        "Reconsider the SAME pre-action decision using hindsight, not the state after the trial.\n"
        "Related manifest goal/harm checks (candidate links, not scored outcomes):\n" + json.dumps(list(related.values()), sort_keys=True) +
        "\nManifest-bound source sender identities (teacher-only authority, not inferred from a guessed query):\n" +
        json.dumps(goals.get('expected_source_senders', []), sort_keys=True) +
        "\nObserved tool response to the trial:\n" + feedback +
        "\nFactual observation checks:\n" + json.dumps(observations, sort_keys=True) +
        "\nAn unfinished goal alone does not make an intermediate read wrong. "
        "An empty search did not find the requested evidence; repeating that unchanged query cannot establish it. "
        "Use names and identifiers supported by the request, history and tool schemas. Do not invent email domains. "
        "For resource-not-found errors, distinguish a displayed title-with-ID from a bare title or raw ID. "
        "When a supported next-call candidate is provided, prefer its repaired argument values over the failed values. "
        "Use the observation to favor a useful next action for the original request. "
        "Do not prepend a critique, continue after the trial, or treat a failed call as a correct label. "
        "Return the next assistant action in the existing native format.\n"
        "For this decision, the supported repaired call arguments are:\n" +
        json.dumps(repair_candidates(call, feedback, goals), sort_keys=True) +
        "\nUse these repaired argument values in the tool call rather than repeating the failed values."
    )


def requested_budget(body, reply_limit, context_limit, prefix_tokens):
    requested = body.get("max_completion_tokens")
    if requested is None:
        requested = body.get("max_tokens")
    if requested is None:
        requested = reply_limit
    if not isinstance(requested, int) or isinstance(requested, bool) or requested < 1:
        raise ValueError("positive integer completion budget required")
    budget = min(reply_limit, requested)
    return budget, prefix_tokens + budget <= context_limit
