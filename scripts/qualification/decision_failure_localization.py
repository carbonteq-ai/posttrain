"""Evidence-linked failure localization, not causal per-action reward credit."""

import json
import re

from progress_teacher_context import tool_outcome


def function(tool):
    return tool.get('function', tool)


def arguments(tool):
    value = function(tool).get('arguments', {})
    return json.loads(value) if isinstance(value, str) else value


def native_messages(episode):
    return [n['message'] for n in episode['traces'][0]['nodes'] if n.get('message')]


def localize(call, assertions, responses):
    """Export only selected arguments and actual error responses, not answers."""
    output = []
    records = []
    def walk(value):
        if isinstance(value, dict):
            records.append(value)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    for message in call.get('messages', []):
        if message.get('role') == 'tool':
            try:
                value = message.get('content')
                walk(json.loads(value) if isinstance(value, str) else value)
            except ValueError:
                pass
    channels = {r['id']:r['name'] for r in records if isinstance(r.get('id'), str)
                and r['id'].startswith('C') and isinstance(r.get('name'), str)}
    by_id = {m['tool_call_id']: m for m in responses if m.get('role') == 'tool' and m.get('tool_call_id')}
    for number, tool in enumerate(call['message'].get('tool_calls') or [], 1):
        fn, args = function(tool), arguments(tool)
        observed = by_id.get(tool.get('id'))
        if observed:
            error = tool_outcome(observed.get('content'))['error']
            if error:
                output.append({'proposal_call': number, 'kind': 'observed_execution_error',
                    'tool': fn['name'], 'observed_error': error,
                    'boundary': 'This tool outcome is directly observed; it does not explain every later episode failure.'})
        for assertion in assertions:
            if assertion.get('passed') is not False or assertion.get('excluded'):
                continue
            kind, params = assertion['type'], assertion.get('params', {})
            if 'not_' in kind and kind.split('_')[0] == fn['name'].split('_')[0]:
                terms = []
                for key in ('text_contains', 'body_contains', 'text_not_contains', 'body_not_contains'):
                    value = params.get(key, [])
                    terms.extend([value] if isinstance(value, str) else value if isinstance(value, list) else [])
                text = '\n'.join(str(args.get(k, '')) for k in ('text','body','body_text','subject','message','description'))
                matches = [t for t in terms if isinstance(t, str) and t and t in text]
                destination_matches = True
                if params.get('to'):
                    recipients = args.get('to', [])
                    recipients = [s.strip() for s in recipients.split(',')] if isinstance(recipients, str) else recipients
                    destination_matches = params['to'] in (recipients or [])
                if params.get('channel_name'):
                    channel = args.get('channel', args.get('channel_name', ''))
                    destination_matches = str(channels.get(channel, channel)).lstrip('#') == params['channel_name'].lstrip('#')
                if matches and destination_matches:
                    output.append({'proposal_call':number,'kind':'failed_outgoing_text_prohibition',
                        'tool':fn['name'],'text_already_in_proposal':matches,
                        'boundary':'Failed negative check matches this service, destination when declared, and proposed outgoing text; evaluation hindsight, not causal reward credit.'})
            if kind == 'gmail_message_not_sent_to' and fn['name'] == 'gmail_send_email':
                to = args.get('to', [])
                recipients = [x.strip() for x in to.split(',')] if isinstance(to, str) else to
                if params.get('to') and params.get('to') in (recipients or []):
                    output.append({'proposal_call': number, 'kind': 'failed_recipient_prohibition',
                        'tool': fn['name'], 'recipient_already_in_proposal': params['to'],
                        'boundary': 'Failed evaluation prohibition names this proposed recipient. This is hindsight supervision, not a fabricated public-policy explanation.',
                        'direction': 'Do not make this prohibited send; inspect the actual eligibility/exception policy already available or retrieve it through declared tools.'})
            if kind == 'google_sheets_row_not_updated' and fn['name'] == 'google_sheets_update_row':
                spreadsheet = args.get('spreadsheet_id', args.get('spreadsheet'))
                row = args.get('row_id', args.get('row'))
                cells = args.get('cells', {})
                if isinstance(cells, str):
                    try:
                        cells = json.loads(cells)
                    except ValueError:
                        cells = {}
                expected = params.get('cell_contains', {})
                if (spreadsheet == params.get('spreadsheet_id') and str(row) == str(params.get('row_id'))
                        and isinstance(cells, dict) and expected
                        and all(cells.get(k) == v for k, v in expected.items())):
                    output.append({'proposal_call': number, 'kind': 'failed_row_write_prohibition',
                        'tool': fn['name'], 'spreadsheet_already_in_proposal': spreadsheet,
                        'row_already_in_proposal': row, 'proposed_fields': {k: cells[k] for k in expected},
                        'boundary': 'Failed negative row-update check matches this exact proposed row and values; not a general ban on updating the tracker.',
                        'direction': 'Omit this prohibited row write; resolve its eligibility condition rather than applying the same values to every row.'})
            if kind == 'salesforce_contact_phone_equals' and fn['name'] == 'salesforce_contact_update':
                identity = args.get('id')
                before = next((r for r in records if r.get('Id') == identity and 'Phone' in r), None)
                if (identity == params.get('contact_id') and before and before['Phone'] == params.get('phone')
                        and args.get('phone') is not None and args['phone'] != before['Phone']):
                    output.append({'proposal_call':number,'kind':'failed_observed_field_preservation',
                        'tool':fn['name'],'observed_contact_id':identity,'observed_phone_before':before['Phone'],
                        'proposed_phone':args['phone'],
                        'boundary':'Failed field-equality check requires the already-observed before-value. No hidden expected value is exported; this is evaluation preservation feedback, not public-scope proof.'})
    return output


def workflow_gaps(episode, calls):
    """Absent operation families are episode omissions, not wrong tool calls."""
    assertions = episode['traces'][0].get('info', {}).get('automationbench', {}).get('assertions', [])
    seen = {function(t)['name'] for m in native_messages(episode) for t in m.get('tool_calls') or []}
    families = {
        'gmail_message_sent': ('gmail', ('send', 'reply')),
        'slack_message_exists': ('slack', ('send', 'post')),
    }
    gaps = []
    for check_prefix, (service, verbs) in families.items():
        failed = [a['type'] for a in assertions if a.get('passed') is False and not a.get('excluded')
                  and a['type'].startswith(check_prefix)]
        if failed and not any(service in n and any(v in n for v in verbs) for n in seen):
            gaps.append({'kind': 'missing_operation_family', 'operation_family': service + ' outgoing message',
                'failed_check_types': sorted(set(failed)),
                'boundary': 'No such call exists in the completed episode; this does not make an earlier read or valid reply incorrect.',
                'direction': 'Keep this still-required workflow effect in the remaining turn plan; no invented recipient, counts or summary content.'})
    return gaps


def select(episode, candidates):
    """Earliest linked prohibited write, then direct errors, then final omission frontier."""
    assertions = episode['traces'][0].get('info', {}).get('automationbench', {}).get('assertions', [])
    messages = native_messages(episode)
    complete = [r for r in candidates if r['call']['complete']]
    if not complete:
        raise ValueError('No complete original action; no truncation fallback')
    facts = {r['generation_index']: localize(r['call'], assertions, messages) for r in complete}
    prohibited = [r for r in complete if any(f['kind'].startswith('failed_') for f in facts[r['generation_index']])]
    if prohibited:
        selected = min(prohibited, key=lambda r: r['generation_index'])
        reason = 'earliest explicitly linked prohibited write, ahead of recovered discovery errors'
    else:
        gaps = workflow_gaps(episode, complete)
        if gaps:
            selected = max(complete, key=lambda r: r['generation_index'])
            reason = 'last complete original decision before terminal omission; current action is not labeled incorrect'
        else:
            errors = [r for r in complete if facts[r['generation_index']]]
            selected = min(errors, key=lambda r: r['generation_index']) if errors else min(complete, key=lambda r: r['generation_index'])
            reason = 'direct execution error' if errors else 'unlocalized; no fabricated corrective label'
    # Prefix/action/logprob arrays belong to the exact retained generation.
    call = selected['call']
    if len(call['action_ids']) != len(call['sampled_logprobs']):
        raise ValueError('Original action/logprob alignment differs')
    system = '\n'.join(m.get('content', '') for m in call['messages'] if m.get('role') == 'system')
    limit = re.search(r'at most (\d+) assistant turns', system)
    remaining = max(0, int(limit.group(1)) - sum(m.get('role') == 'assistant' for m in call['messages'])) if limit else None
    return selected, {'selection_reason': reason, 'selected_action_findings': facts[selected['generation_index']],
        'remaining_assistant_turns_including_selected': remaining,
        'workflow_omissions': workflow_gaps(episode, complete),
        'all_candidate_findings': [{'generation_index':r['generation_index'],'findings':facts[r['generation_index']]} for r in complete],
        'causal_credit_claimed': False, 'original_tokens_preserved': True}
