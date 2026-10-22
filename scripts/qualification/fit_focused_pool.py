"""Pinned tokenizer/template admission only; no model or inference allocation."""

import argparse
import hashlib
import json
from pathlib import Path


def run(args):
    from qualify_combined_teacher_contexts import native_messages
    from sdpo_prompting import prompt_ids
    from transformers import AutoTokenizer

    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    tokenizer._sdpo_model_family='qwen3.5'
    cases=json.loads((args.input/'source-cases.json').read_text())['cases']
    assert len(cases)==len({c['row']['task'] for c in cases})==124
    results=[]
    for c in cases:
        task=c['row']['task']
        call=c['call']
        original=prompt_ids(tokenizer,native_messages(call['messages']),call['tools'],thinking=True)
        candidate=(args.input/task/'candidate-supplement.txt').read_text()
        teacher=prompt_ids(tokenizer,native_messages(call['messages'])+[{'role':'user','content':candidate}],call['tools'],thinking=True)
        results.append({'task':task,'student_prefix_matches_original':original==call['prefix_ids'],
            'original_action_matches':c['row']['action_ids']==call['action_ids'],
            'logprob_count_matches':len(call['sampled_logprobs'])==len(call['action_ids']),
            'student_tokens':len(original),'teacher_tokens':len(teacher),'action_tokens':len(call['action_ids']),
            'whole_context_fits':len(teacher)+max(4096,len(call['action_ids']))<=32768,
            'context_sha256':hashlib.sha256(candidate.encode()).hexdigest()})
    args.output.mkdir(parents=True,exist_ok=False)
    failures=[r for r in results if not all(r[k] for k in ('student_prefix_matches_original','original_action_matches','logprob_count_matches','whole_context_fits'))]
    result={'cpu_template_qualified':not failures,'tasks':124,'model_calls':0,'optimizer_updates':0,
        'maximum_teacher_tokens':max(r['teacher_tokens'] for r in results),'failures':failures,'results':results,
        'scope':'Original-token/template/whole-context admission only, not semantic supervision or GPU behavior'}
    (args.output/'receipt.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='results'}),flush=True)
    if failures:
        raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',required=True)
    run(p.parse_args())
