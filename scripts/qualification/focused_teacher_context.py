"""Two-task deterministic presentation prototype, not a general manifest translator.

The original conversation remains the evidence authority. Reviewed task-scoped
rules explain relationships without exporting symbolic IR or private answers.
"""

import copy
import json
import re

from progress_teacher_context import observed_policy_candidates, retrieval_summary, tool_outcome

RULES = {
    'hr.offboarding_automation': [
        'Apply the policy already retrieved separately to each pending row: departure type, last day and existing status determine its branch. Use the simulated clock in the original system message.',
        'A voluntary farewell rule does not authorize a public announcement for an involuntary termination. When the retrieved involuntary policy requires immediate IT access revocation, request immediate revocation; do not substitute the employee last-day date. Apply that branch before choosing notification tools and recipients.',
        'Apply the observed future-date deferral rule before notifications. Leave already processed rows unchanged; distinguish similar names by the observed employee identity.',
        'Follow the retrieved Payroll/legal restriction for severance; the additional user request does not itself authorize HR Ops to process a payment.',
        'Copy employee values from the actual source row into notifications and attach each status write to that same row. Complete remaining work over later turns when necessary.',
    ],
    'sales.soft_matching': [
        'Use the retrieved contact-matching guidelines to connect the requested company/domain, account, role and name to the same active Contact. Similar names or successful API calls alone do not establish identity.',
        'Use only an explicit observed company alias as an alias; do not merge similarly named companies or treat every external HR email as automatically authorized.',
        'Do not infer the intended company from singular wording or hidden manifest target IDs. A public-evidence ambiguity remains unresolved; do not turn an expected answer into an identity rule.',
        'Change only the supported Contact phone. Preserve other contacts and unrelated fields, including titles.',
        'Follow the observed audit-note requirement on the same Contact as the phone update. Missing later work is not proof that an otherwise correct next action is wrong.',
    ],
}


def precise_feedback(task, call, assertions):
    """Ground deterministic diagnosis in observed records and selected calls.

    Assertion findings are explicitly evaluation hindsight, not public policy.
    Only identities already observed or proposed are exported, never unseen
    expected values. Unrelated later failures stay episode-level evidence.
    """
    records = []

    def walk(value):
        if isinstance(value, dict):
            records.append(value)
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    history = json.dumps(call['messages'], ensure_ascii=False)
    for message in call['messages']:
        if message.get('role') == 'tool':
            value = message.get('content')
            try:
                walk(json.loads(value) if isinstance(value, str) else value)
            except ValueError:
                continue
    output = []
    for number, tool in enumerate(call['message'].get('tool_calls') or [], 1):
        function = tool['function']
        args = function['arguments']
        args = json.loads(args) if isinstance(args, str) else args
        if task == 'hr.offboarding_automation' and function['name'] == 'slack_send_channel_message':
            text = args.get('text', '')
            for record in records:
                cells = record.get('cells', {})
                name, email = cells.get('Employee'), cells.get('Employee Email')
                if not name or name not in text:
                    continue
                if str(cells.get('Type', '')).startswith('Involuntary'):
                    # The business rule must actually have been retrieved.
                    if 'Do NOT post to any public Slack channel' in history:
                        output.append({'proposal_call':number,'problem':'public_involuntary_notice',
                            'observed_employee':name,'observed_type':cells['Type'],
                            'basis':'Retrieved policy forbids public Slack posts for involuntary termination; this proposed notice names that employee.',
                            'direction':'Omit this public notice; use the policy-authorized private actions.'})
                        if 'immediate access revocation' in history:
                            output.append({'proposal_call':number,'problem':'replacement_private_action_timing',
                                'observed_employee':name,
                                'exact_observed_policy_requirement':'Email IT requesting immediate access revocation',
                                'direction':'If replacing this public notice with the required IT notification for this involuntary departure, explicitly request immediate revocation. Do not copy the voluntary employee template saying as of last day.'})
                stated = re.search(r'Employee:\s*([^,\s]+@[^,\s]+)', text)
                if stated and email and stated.group(1) != email:
                    output.append({'proposal_call':number,'problem':'source_email_mismatch','observed_employee':name,
                        'proposed_email':stated.group(1),'observed_source_email':email,
                        'basis':'Exact employee row and proposed Employee field differ; preserve source values.'})
        if task == 'sales.soft_matching' and function['name'] == 'salesforce_contact_update':
            identity = args.get('id')
            for assertion in assertions:
                params = assertion.get('params', {})
                if (assertion.get('type') != 'salesforce_contact_phone_equals' or assertion.get('passed') is not False
                        or assertion.get('excluded') or params.get('contact_id') != identity):
                    continue
                found = next((r for r in records if r.get('Id') == identity and 'Phone' in r), None)
                if found is not None and 'phone' in args:
                    output.append({'proposal_call':number,'problem':'phone_preservation_check_failed',
                        'observed_contact_id':identity,'observed_phone_before':found['Phone'],'proposed_phone':args['phone'],
                        'basis':'This proposed contact is linked to a failed phone-equality check in the completed attempt. The before-value is from actual prior tool evidence, not a hidden expected value.',
                        'scope_boundary':'Evaluation feedback requests preserving this contact. This does not prove that singular wording or a company alias alone resolves public request scope.',
                        'direction':'Do not include this extra phone change in the corrected proposal.'})
    return output


def build(task, call, assertions, observations, *, include_precise=False):
    if task not in RULES:
        raise ValueError('No reviewed focused rule set; no generic fallback')
    if not any(m.get('role') == 'system' for m in call['messages']):
        raise ValueError('Original system restrictions required')
    tools = []
    for item in call['message'].get('tool_calls') or []:
        function = item['function']
        arguments = function.get('arguments')
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        tools.append({'tool': function['name'], 'arguments': copy.deepcopy(arguments)})
    feedback = []
    for observed in observations:
        outcome = tool_outcome(observed.get('response'))
        item = {'tool': observed['tool'], 'api_success': outcome['success']}
        if outcome['error']:
            item['api_error'] = outcome['error']
        if retrieval_summary(observed['tool'], observed.get('response'))['empty_returned_collection']:
            item['empty_read'] = True
        feedback.append(item)
    failed = sorted({a['type'] for a in assertions if a.get('passed') is False and not a.get('excluded')})
    policies = observed_policy_candidates(call['messages'])
    pointers = []
    for policy in policies:
        source = policy['source']
        label = source.get('subject') or source.get('Policy')
        if label and label not in pointers:
            pointers.append(label)
    payload = {
        'decision_rules': RULES[task],
        'policy_already_in_history': pointers,
        'proposed_next_action_untrusted': tools or {'assistant_text': call['message'].get('content')},
        'hindsight_for_this_proposal': feedback,
        'failed_checks_in_completed_attempt': failed,
    }
    if include_precise:
        payload['decision_specific_corrections'] = precise_feedback(task, call, assertions)
    text = ('Continue at the original decision, BEFORE the proposal below executes. The original system restrictions, public request, tool schemas and observed history remain authoritative.\n'
            + '\n' + json.dumps(payload, ensure_ascii=False, indent=2)
            + '\nThe proposal is a candidate to assess, not a successful demonstration. Hindsight API success is not business correctness; completed-attempt check failures do not by themselves identify this proposal as their cause. Use existing policy evidence without repeating discovery; retrieve only evidence genuinely needed for the next action. Perform a supported next action in native reasoning/tool format. Keep privileged feedback out of outgoing content.')
    audit = {
        'version': 'focused-two-task-v2-precise' if include_precise else 'focused-two-task-v1', 'task': task,
        'purpose_by_field': {
            'decision_rules': 'Explain task-specific identity, branch and preservation decisions.',
            'policy_already_in_history': 'Direct attention to retrieved policy without copying its body.',
            'proposed_next_action_untrusted': 'Expose the exact candidate being corrected without endorsing it.',
            'hindsight_for_this_proposal': 'Separate transport/tool outcomes from task correctness.',
            'failed_checks_in_completed_attempt': 'Retain diagnostic failure types without private expected parameters or false per-action credit.',
        },
        'omitted_from_teacher': {
            'symbolic_manifest_IR': 'No executable identity is established by S/V symbols.',
            'full_system_and_request_copy': 'Already present in original prefix; avoid duplicating it.',
            'full_observation_ledger_and_hashes': 'Original history is unchanged; hashes serve audit, not acting.',
            'policy_body_copy': 'Already present in original history.',
            'all_positive_excluded_assertions': 'Not a correctness label for this selected action.',
        },
        'policy_source_pointers_private_audit': [{'message_index':p['message_index'],'source_path':p['source_path']} for p in policies],
        'automatic_complete_manifest_semantics_qualified': False,
        'native_or_gradient_qualified': False,
    }
    if include_precise:
        audit['purpose_by_field']['decision_specific_corrections'] = (
            'Join observed source records and selected tool arguments to the specific mismatch; '
            'explicitly distinguish business-policy evidence from evaluation preservation feedback.')
    return text, audit
