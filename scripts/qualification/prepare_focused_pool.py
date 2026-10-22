"""Prepare source-bound focused candidate contexts without inference or updates."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict

from decision_failure_localization import select
from focused_corpus_context import build
from progress_teacher_context import selected_observations


def run(args):
    original=json.loads((args.reference/'source-cases.json').read_text())['cases']
    source_materials=json.loads(args.materials.read_text())
    materials={r['material']['task']:r['material'] for r in source_materials}
    raw_sources={r['material']['task']:r['raw_native_data']['initial_state'] for r in source_materials}
    expected={c['row']['task'] for c in original}
    assert len(original)==len(expected)==124 and set(materials)==expected
    id_owner={}
    selected_index={c['row']['generation_index']:c['row']['task'] for c in original}
    for case in original:
        for node in case['episode']['traces'][0]['nodes']:
            for tool in (node.get('message') or {}).get('tool_calls') or []:
                if tool['id'] in id_owner and id_owner[tool['id']] != case['row']['task']:
                    raise ValueError('Native tool-call identity collision')
                id_owner[tool['id']]=case['row']['task']
    calls=defaultdict(list)
    for index,line in enumerate(args.calls.open()):
        call=json.loads(line)
        owners={id_owner[t['id']] for m in call['messages']+[call['message']] for t in m.get('tool_calls') or [] if t['id'] in id_owner}
        if index in selected_index:
            owners.add(selected_index[index])
        if len(owners)>1:
            raise ValueError('Ambiguous native generation join')
        if owners:
            calls[next(iter(owners))].append({'generation_index':index,'call':call})
    output=args.output
    output.mkdir(exist_ok=False,parents=True)
    cases,summary=[],[]
    for old in original:
        task=old['row']['task']
        if old['row']['native_success']==1:
            chosen=next(r for r in calls[task] if r['generation_index']==old['row']['generation_index'])
            audit={'selection_reason':'retain original successful-episode decision','selected_action_findings':[],
                   'workflow_omissions':[],'remaining_assistant_turns_including_selected':None,'causal_credit_claimed':False}
        else:
            chosen,audit=select(old['episode'],calls[task])
        call=chosen['call']
        if not call['complete'] or len(call['action_ids'])!=len(call['sampled_logprobs']):
            raise ValueError('Incomplete/misaligned selected generation')
        tr=old['episode']['traces'][0]
        checks=tr.get('info',{}).get('automationbench',{}).get('assertions',[])
        messages=[n['message'] for n in tr['nodes'] if n.get('message')]
        observations=selected_observations(call,messages)
        candidate,context_audit=build(materials[task],call,checks,observations,workflow_findings=audit['workflow_omissions'])
        if audit['remaining_assistant_turns_including_selected'] is not None and audit['workflow_omissions']:
            candidate+='\nActual remaining assistant turns including this next action: '+str(audit['remaining_assistant_turns_including_selected'])+'. Plan remaining required effects within this budget.'
        # Isolate decision-local corrections from task/policy context for matched controls.
        start=candidate.index('{')
        payload,end=json.JSONDecoder().raw_decode(candidate[start:])
        baseline_payload={k:v for k,v in payload.items() if k not in ('precise_attempt_feedback','workflow_omissions_in_completed_attempt')}
        baseline=candidate[:start]+json.dumps(baseline_payload,ensure_ascii=False,indent=2)+candidate[start+end:]
        folder=output/task
        folder.mkdir()
        (folder/'candidate-supplement.txt').write_text(candidate)
        (folder/'baseline-supplement.txt').write_text(baseline)
        audit.update(task=task,old_generation_index=old['row']['generation_index'],new_generation_index=chosen['generation_index'],
            prefix_sha256=hashlib.sha256(json.dumps(call['prefix_ids']).encode()).hexdigest(),
            action_sha256=hashlib.sha256(json.dumps(call['action_ids']).encode()).hexdigest(),
            character_count=len(candidate),native_success=old['row']['native_success'],context_audit=context_audit)
        (folder/'audit.json').write_text(json.dumps(audit,indent=2))
        case={k:v for k,v in old.items() if k not in ('row','call','revised_teacher_prefix_ids','context_key')}
        case.update(call=call,row={'task':task,'episode_id':old['row']['episode_id'],'generation_index':chosen['generation_index'],
            'student_prefix_ids':call['prefix_ids'],'action_ids':call['action_ids'],'sampled_logprobs':call['sampled_logprobs'],
            'complete':call['complete'],'policy_version':call['policy_version'],'native_success':old['row']['native_success']})
        case['source_fixture_initial_state']=raw_sources[task]
        cases.append(case)
        summary.append(audit)
    (output/'source-cases.json').write_text(json.dumps({'cases':cases}))
    (output/'audit.json').write_text(json.dumps(summary,indent=2))
    receipt={'cpu_render_completed':True,'tasks':124,'source_check_count':sum(len(g['check_ids']) for r in summary for g in r['context_audit']['contract_coverage']),
        'selection_dispositions':dict(Counter(r['selection_reason'] for r in summary)),
        'changed_selected_decisions':sum(r['old_generation_index']!=r['new_generation_index'] for r in summary),
        'unlocalized_failed_tasks':[r['task'] for r in summary if r['native_success']==0 and not r['selected_action_findings'] and not r['workflow_omissions']],
        'largest_supplement_characters':max(r['character_count'] for r in summary),'model_calls':0,'optimizer_updates':0,
        'native_qualified':False,'semantic_qualified':False,'exact_template_fit_still_required':True}
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    from pathlib import Path
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('reference','materials','calls','output'):
        p.add_argument('--'+key,type=Path,required=True)
    run(p.parse_args())
