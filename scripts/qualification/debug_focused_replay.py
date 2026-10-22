"""Preserve exact historical replay mismatches without accepting or normalizing them."""

import argparse
import json
from pathlib import Path


def run(args):
    from qualify_combined_teacher_contexts import replay
    cases={c['row']['task']:c for c in json.loads((args.prepared/'source-cases.json').read_text())['cases']}
    proposals=json.loads((args.run/'proposals.json').read_text())
    args.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for proposal in proposals:
        if not proposal.get('native_error'):
            continue
        case=cases[proposal['task']]
        result={'task':proposal['task']}
        try:
            replay(case,exact_captured_history=True,diagnostic_differences=True)
            result['unexpected_pass']=True
        except Exception as error:
            try:
                result['difference']=json.loads(str(error))
            except ValueError:
                result['error']={'type':type(error).__name__,'message':str(error)}
        result['native_capture_events']=case['episode']['traces'][0]['info']['automationbench_capture']['events']
        rows.append(result)
        (args.output/'differences.json').write_text(json.dumps(rows,indent=2))
    (args.output/'receipt.json').write_text(json.dumps({'completed':True,'tasks':len(rows),'model_calls':0,'optimizer_updates':0,'mismatches_accepted':False},indent=2))
    print(json.dumps({'completed':True,'tasks':len(rows),'model_calls':0}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','run','output'):
        p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
