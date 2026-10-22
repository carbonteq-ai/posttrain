"""Same-state native comparison and explicit semantic warning checks, no rescoring of original episodes."""

import argparse
import json
import re
from pathlib import Path


def run(args):
    from qualify_combined_teacher_contexts import replay

    cases={c['row']['task']:c for c in json.loads((args.prepared/'source-cases.json').read_text())['cases']}
    if args.materials:
        materials={r['material']['task']:r for r in json.loads(args.materials.read_text())}
        if set(materials)!=set(cases):
            raise ValueError('Original material population differs')
        for name,case in cases.items():
            raw=materials[name]['raw_native_data']
            data=case['episode']['task']['data']
            if raw['assertions']!=data['assertions'] or [m['content'] for m in raw['prompt'] if m['role']=='user']!=[m['content'] for m in data['prompt'] if m['role']=='user']:
                raise ValueError('Original fixture assertions/public request differ')
            case['source_fixture_initial_state']=raw['initial_state']
    proposals=json.loads((args.run/'proposals.json').read_text())
    if len(proposals)!=124 or {p['task'] for p in proposals}!=set(cases):
        raise ValueError('Complete124native proposal population required')
    args.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for proposal in proposals:
        case=cases[proposal['task']]
        result={'task':proposal['task'],'complete':proposal['complete'],
                'finish_reason':proposal.get('finish_reason'),'native_error':proposal.get('native_error'),
                'semantic_warnings':[],'semantic_qualified':None}
        if proposal['complete'] and proposal.get('message'):
            try:
                # These are the TWO retained source representations, not guessed
                # IDs/fields or normalized replay outputs. Admit one only when
                # full official setup reproduces the native initial digest.
                forms=[('retained_native_task_data',case['episode']['task']['data']['initial_state'])]
                if 'source_fixture_initial_state' in case:
                    forms.append(('original_source_fixture',case['source_fixture_initial_state']))
                before=None
                failures=[]
                for form,initial in forms:
                    bound={**case,'setup_initial_state':initial}
                    try:
                        before=replay(bound,exact_captured_history=True,native_setup=True)
                        after=replay(bound,proposal['message'],exact_captured_history=True,native_setup=True)
                        result['native_setup_source_form']=form
                        result['native_initial_digest_verified']=True
                        break
                    except ValueError as error:
                        if not str(error).startswith('Native initial-world digest differs:'):
                            raise
                        failures.append({'source_form':form,'error':str(error)})
                if before is None:
                    raise ValueError(json.dumps({'documented_source_forms_failed':failures}))
            except Exception as error:
                result['independent_replay_error']={'type':type(error).__name__,'message':str(error)}
                rows.append(result)
                (args.output/'rows.json').write_text(json.dumps(rows,indent=2))
                continue
            if before['decision_state_sha256']!=after['decision_state_sha256']:
                raise ValueError('Same-state replay differs')
            result['same_decision_state']=True
            result['old_replay_error_preserved']=proposal.get('native_error')
            for label,observed in [('original_action',before),('teacher_action',after)]:
                score=observed['scoring_after_one_decision']
                result[label]={'partial_credit':score['partial_credit'],'task_success':score['task_completed_correctly'],
                    'failed_nonexcluded_checks':[a['type'] for a in score['assertion_results'] if a.get('passed') is False and not a.get('excluded')]}
            errors=[]
            for index,decision in enumerate(after['decisions'],1):
                value=decision.get('result')
                if decision.get('raised') or decision.get('strict_admission_rejected') or (isinstance(value,dict) and (value.get('success') is False or value.get('error'))):
                    errors.append({'call':index,'tool':decision['tool'],'raised':decision.get('raised'),
                                   'result':value,'strict_rejected':decision.get('strict_admission_rejected')})
            result['teacher_execution_errors']=errors
            history=json.dumps(case['call']['messages'],ensure_ascii=False)
            for index,decision in enumerate(after['decisions'],1):
                arguments=decision['arguments']
                # Explicit known policy failure, independent of the incomplete native scorer.
                if proposal['task']=='hr.offboarding_automation' and decision['tool']=='gmail_send_email':
                    body=arguments.get('body','')
                    if (arguments.get('to')=='it-ops@company.example.com' and 'Diana Reese' in body
                            and 'immediate access revocation' in history
                            and re.search(r'as of (?:last day|2026-03-14)',body,re.I)):
                        result['semantic_warnings'].append({'call':index,'kind':'immediate_revocation_replaced_by_future_date','body':body})
                # Pattern-limited provenance warning, not an exhaustive identity judge.
                for key,value in arguments.items():
                    if isinstance(value,str) and (key.endswith('_id') or key in ('id','spreadsheet','worksheet')):
                        if re.match(r'^(?:ss_|ws_|003[a-zA-Z0-9]|001[a-zA-Z0-9]|C_[A-Z])',value) and value not in history:
                            result['semantic_warnings'].append({'call':index,'kind':'identifier_not_in_original_observed_history','field':key,'value':value})
            result['native_replays_scope']='After one alternative action from the same state; not a complete new rollout or full causal goal credit.'
        rows.append(result)
        (args.output/'rows.json').write_text(json.dumps(rows,indent=2))
    result={'completed':True,'tasks':124,'optimizer_updates':0,'model_calls':0,
        'complete_teacher_actions':sum(r['complete'] for r in rows),
        'native_replay_pairs':sum(r.get('same_decision_state',False) for r in rows),
        'independent_replay_error_tasks':[r['task'] for r in rows if r.get('independent_replay_error')],
        'teacher_actions_with_execution_errors':[r['task'] for r in rows if r.get('teacher_execution_errors')],
        'semantic_warning_tasks':[r['task'] for r in rows if r['semantic_warnings']],
        'semantic_qualified':False,
        'scope':'Source/native diagnostic and limited warning detectors. Absence of warnings does not establish all task requirements, harm safety or correctness of a gradient reference.'}
    (args.output/'receipt.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','run','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--materials',type=Path)
    run(p.parse_args())
