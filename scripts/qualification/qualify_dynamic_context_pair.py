"""Two-task context-only teacher continuation and zero-update gradient diagnostic."""

import argparse
import asyncio
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


async def run(args):
    import torch
    from peft import LoraConfig, get_peft_model
    from probe_combined_teacher_tool_gradients import tool_positions
    from qualify_combined_teacher_contexts import native_messages, parsed_message, replay, replay_source_bound, write
    from sdpo_agent_ablation import SamplingRollout, loss_for, teacher_logits
    from sdpo_prompting import prompt_ids
    from sdpo_schema_parser import parser_class
    from transformers import AutoModelForImageTextToText, AutoTokenizer, set_seed

    args.output.mkdir(parents=True, exist_ok=False)
    source = json.loads((args.reference / 'source-cases.json').read_text())['cases']
    names = ([c['row']['task'] for c in source] if args.all_cases else
             ['hr.offboarding_automation', 'sales.soft_matching'])
    cases = [c for c in source if c['row']['task'] in names]
    assert len(cases) == len(names) == len(set(names))
    set_seed(18)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer._sdpo_model_family = 'qwen3.5'
    model = get_peft_model(AutoModelForImageTextToText.from_pretrained(args.model,
        dtype=torch.bfloat16, device_map={'': 'cuda:0'}, attn_implementation='sdpa', local_files_only=True),
        LoraConfig(r=16, lora_alpha=32, lora_dropout=0., target_modules='all-linear',
                   exclude_modules=r'.*visual.*', task_type='CAUSAL_LM'))
    rollout = await SamplingRollout.create(args.model, model, tokenizer, args.output,
        65536, 40, 16, speculative_config=None, prefix_caching=False, chunked_prefill=False)
    assert rollout.refreshes[0]['weights_sha256'] == 'e71dae81f1b4723c83aacd130bfbf60b072b3580212133a4fc45dcb9f403421c'
    rollout.reply_limit = 4096
    rollout.temperature = 0.
    parser_type, parser_receipt = parser_class()
    write(args.output / 'parser.json', parser_receipt)
    prefixes, bindings = {}, []
    for case in cases:
        row, call = case['row'], case['call']
        task = row['task']
        deterministic = case.get('revised_teacher_prefix_ids')
        if args.focused:
            baseline = (args.contexts / task / 'baseline-supplement.txt').read_text()
            deterministic = prompt_ids(tokenizer, native_messages(call['messages']) + [{'role':'user','content':baseline}], call['tools'], thinking=True)
        if deterministic is None:
            raise ValueError('No bound control prefix')
        supplement = (args.contexts / task / args.candidate_file).read_text()
        deepseek = prompt_ids(tokenizer, native_messages(call['messages']) + [{'role':'user','content':supplement}], call['tools'], thinking=True)
        for mode, prefix in [('deterministic', deterministic), ('deepseek', deepseek)]:
            assert len(prefix) + max(4096, len(row['action_ids'])) <= 32768
            prefixes[task, mode] = prefix
            folder = args.output / task
            folder.mkdir(exist_ok=True)
            (folder / f'{mode}-full-teacher-context.txt').write_text(tokenizer.decode(prefix, skip_special_tokens=False))
            bindings.append({'task':task,'mode':mode,'episode_id':row['episode_id'],
                'generation_index':row['generation_index'],'teacher_tokens':len(prefix),
                'student_tokens':len(row['student_prefix_ids']),'action_tokens':len(row['action_ids']),
                'prefix_sha256':hashlib.sha256(json.dumps(prefix).encode()).hexdigest(), 'truncated':False})
    write(args.output / 'token-bindings.json', bindings)
    proposals = []
    try:
        async def generate(case, mode):
            row, call = case['row'], case['call']
            prefix = prefixes[row['task'], mode]
            ids, complete, evidence = await rollout.generate(prefix, max_tokens=4096)
            raw = tokenizer.decode(ids, skip_special_tokens=True)
            result = {'task':row['task'],'mode':mode,'action_ids':ids,'raw':raw,'complete':complete,**evidence}
            if complete:
                try:
                    result['message'] = parsed_message(raw, tokenizer, call['tools'], parser_type, native_messages(call['messages']))
                    result['native_replay'] = (replay_source_bound(case, result['message'])
                                               if args.native_source_proof else replay(case, result['message']))
                except Exception as error:
                    result['native_error'] = {'type':type(error).__name__,'message':str(error)}
            return result
        modes = ['deepseek'] if args.candidate_only else ['deterministic','deepseek']
        # Keep actual request admission bounded by the qualified server slots.
        semaphore = asyncio.Semaphore(40)
        async def bounded(case, mode):
            async with semaphore:
                return await generate(case, mode)
        proposals = await asyncio.gather(*(bounded(case, mode) for case in cases for mode in modes))
        write(args.output / 'proposals.json', proposals)
        print(json.dumps({'stage':'four_teacher_continuations_complete','optimizer_updates':0}), flush=True)
    finally:
        await rollout.aclose()
    if args.native_only:
        write(args.output / 'receipt.json',{'runtime_completed':True,'semantic_qualified':None,'optimizer_updates':0,
            'tasks':names,'new_teacher_continuations':len(proposals),
            'mode_labels':{'deterministic':'focused before precise feedback','deepseek':'focused with deterministic precise feedback'} if args.focused else {},
            'scope':'Exact-state native next-decision replay only. No gradients, optimizer, full-episode reward gain or training approval.'})
        return
    model.load_adapter(args.output / 'rollout-policy-v0', adapter_name='sdpo_teacher', is_trainable=False)
    model.set_adapter('default')
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads()
    parameters = [p for name,p in model.named_parameters() if '.default.' in name and p.requires_grad]
    opening, closing = [tokenizer.convert_tokens_to_ids(t) for t in ('<tool_call>','</tool_call>')]
    directions = []
    for case in cases:
        task = case['row']['task']
        # A shared generated reference allows a paired geometric comparison.
        # It is not a gold target and correctness remains an independent review.
        proposal = next(p for p in proposals if p['task']==task and p['mode']=='deepseek')
        target = proposal['action_ids']
        try:
            positions = tool_positions(target, opening, closing) if proposal['complete'] and not proposal.get('native_error') else []
        except ValueError:
            positions = []
        for mode in (['deepseek'] if args.candidate_only else ['deterministic','deepseek']):
            row = copy.deepcopy(case['row'])
            row['teacher_prefix_ids'] = prefixes[task, mode]
            result = {'task':task,'mode':mode,
                'reference': ('same local Qwen teacher tool spans with focused deterministic context' if args.focused
                              else 'same local Qwen teacher tool spans with the candidate supplement'),
                'reference_semantic_qualified':None,'optimizer_updates':0}
            if not proposal['complete'] or proposal.get('native_error') or not positions:
                result.update(measurable=False,reason='No complete native-replayed tool reference')
            elif len(row['student_prefix_ids']) + len(target) > 32768:
                result.update(measurable=False,reason='Whole reference overlong; no truncation')
            else:
                teacher = teacher_logits(model, row)
                loss, _ = loss_for(model, row, SimpleNamespace(topk=20,alpha=1.), teacher=teacher)
                assert torch.isfinite(loss)
                gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
                result['loss'] = float(loss.detach())
                del loss,teacher
                ids = torch.tensor([row['student_prefix_ids']+target[:-1]], device='cuda')
                support = torch.tensor([len(row['student_prefix_ids'])-1+i for i in positions],device='cuda')
                selected = torch.tensor([target[i] for i in positions],device='cuda')
                values = model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=support).logits[0].float()
                logprob = values.log_softmax(-1).gather(-1,selected[:,None]).mean()
                reference = torch.autograd.grad(logprob, parameters, allow_unused=True)
                direction = sum(-float((g.float()*r.float()).sum()) for g,r in zip(gradients,reference,strict=True) if g is not None and r is not None)
                norm2 = sum(float(g.float().square().sum()) for g in gradients if g is not None)
                result.update(measurable=True,tool_tokens=len(positions),gradient_norm=norm2**.5,
                    negative_loss_gradient_dot_reference_tool_logprob_gradient=direction)
                del gradients,reference,values,logprob,ids,support,selected
                torch.cuda.empty_cache()
            directions.append(result)
            write(args.output / 'directions.json',directions)
            print(json.dumps(result),flush=True)
    write(args.output / 'receipt.json',{'runtime_completed':True,'semantic_qualified':None,'optimizer_updates':0,
        'tasks':names,'new_teacher_continuations':len(proposals),'directions':directions,
        'mode_labels': {'deterministic':'bound baseline supplement',
                        'deepseek':'bound candidate supplement; legacy loader label, not an external inference call'},
        'scope':'Same-state next decisions and isolated infinitesimal SGD geometry. No optimizer, Adam update, full episode rollout gain or training approval.'})


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--contexts',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--native-only',action='store_true')
    p.add_argument('--focused',action='store_true')
    p.add_argument('--all-cases',action='store_true')
    p.add_argument('--candidate-only',action='store_true')
    p.add_argument('--candidate-file',default='deepseek-supplement.txt')
    p.add_argument('--native-source-proof',action='store_true')
    asyncio.run(run(p.parse_args()))
