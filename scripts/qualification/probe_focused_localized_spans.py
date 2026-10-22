"""Zero-update gradient probe on original, explicitly localized bad tool spans.

Unlike a generated whole-reference projection, both context arms measure the
same retained student tokens. This remains infinitesimal SGD geometry, not Adam.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


def localized_terms(supplement):
    payload, _ = json.JSONDecoder().raw_decode(supplement[supplement.index('{'):])
    result = []
    for finding in payload.get('precise_attempt_feedback', []):
        if finding.get('kind') == 'failed_recipient_prohibition':
            term = finding['recipient_already_in_proposal']
            label = term.partition('@')[2].split('.')[0]
            result.append({'call': finding['proposal_call'], 'term': term,
                           'kind': 'explicitly_prohibited_recipient',
                           'focus_start': term.index('@') + 1, 'focus_end': term.index('@') + 1 + len(label),
                           'focus_basis': 'First hostname label from this same observed/proposed email; separates recipient identity from shared address syntax.'})
        if finding.get('problem') == 'phone_preservation_check_failed':
            term = finding['observed_contact_id']
            record = {'call': finding['proposal_call'], 'term': term,
                      'kind': 'contact_in_prohibited_extra_update'}
            candidates = payload.get('recorded_student_candidate', [])
            others = [t.get('arguments', t.get('candidate_arguments_for_review', {})).get('id') for t in candidates
                      if isinstance(t, dict) and t.get('tool') == 'salesforce_contact_update']
            others = [value for value in others if isinstance(value, str) and value != term and len(value) == len(term)]
            if len(others) == 1:
                other = others[0]
                start, end = 0, len(term)
                while start < end and term[start] == other[start]:
                    start += 1
                while end > start and term[end - 1] == other[end - 1]:
                    end -= 1
                if start >= len(term) // 2:
                    record.update(focus_start=start, focus_end=end,
                                  focus_basis='Differing characters against the other Contact ID in the original candidate; not a hidden target or proof the alternative is correct.')
            result.append(record)
        if finding.get('problem') == 'public_involuntary_notice':
            result.append({'call': finding['proposal_call'], 'term': finding['observed_employee'],
                           'kind': 'employee_in_prohibited_public_notice'})
    return result


def exact_positions(ids, tokenizer, terms):
    # Sampled BPE segments need not equal the tokenizer's canonical re-encoding.
    # Map the ORIGINAL token pieces, and fail closed if their decoding is not
    # compositional (e.g. a split UTF-8 sequence). Never substitute re-tokenized IDs.
    pieces = [tokenizer.decode([token], skip_special_tokens=False,
                               clean_up_tokenization_spaces=False) for token in ids]
    text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if ''.join(pieces) != text:
        raise ValueError('Original token decoding is noncompositional; exact span mapping unavailable')
    offsets, cursor = [], 0
    for piece in pieces:
        offsets.append((cursor, cursor + len(piece)))
        cursor += len(piece)
    opening = tokenizer.convert_tokens_to_ids('<tool_call>')
    closing = tokenizer.convert_tokens_to_ids('</tool_call>')
    groups, start = [], None
    for index, token in enumerate(ids):
        if token == opening:
            if start is not None:
                raise ValueError('Nested tool call')
            start = index
        if token == closing:
            if start is None:
                raise ValueError('Orphan tool close')
            groups.append((start, index + 1))
            start = None
    if start is not None:
        raise ValueError('Incomplete tool call')
    found = []
    for term in terms:
        if not 1 <= term['call'] <= len(groups):
            raise ValueError('Localized call number lacks original token span')
        left, right = groups[term['call'] - 1]
        positions, focused = set(), set()
        cursor = 0
        while (cursor := text.find(term['term'], cursor)) >= 0:
            end = cursor + len(term['term'])
            positions.update(i for i in range(left, right)
                             if offsets[i][0] < end and offsets[i][1] > cursor)
            if 'focus_start' in term:
                focus_left, focus_right = cursor + term['focus_start'], cursor + term['focus_end']
                focused.update(i for i in range(left, right)
                               if offsets[i][0] < focus_right and offsets[i][1] > focus_left)
            cursor = end
        found.append({**term, 'scope': 'whole_field', 'positions': sorted(positions)})
        if 'focus_start' in term:
            found.append({**term, 'scope': 'distinctive_field_part', 'positions': sorted(focused),
                          'focused_text': term['term'][term['focus_start']:term['focus_end']]})
    return found


def preflight(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    cases = json.loads((args.prepared / 'source-cases.json').read_text())['cases']
    output = []
    for case in cases:
        task = case['row']['task']
        old = (args.prepared / task / 'baseline-supplement.txt').read_text()
        new = (args.prepared / task / 'candidate-supplement.txt').read_text()
        terms = localized_terms(old)
        assert terms == localized_terms(new)
        if terms:
            spans = exact_positions(case['row']['action_ids'], tokenizer, terms)
            assert all(span['positions'] for span in spans), 'Exact bad-span support missing'
            output.append({'task': task, 'spans': spans})
    assert output, 'No localized diagnostic targets'
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'receipt.json').write_text(json.dumps({'passed': True, 'optimizer_updates': 0,
                                                       'native_token_spans': output}, indent=2))
    print(json.dumps(output))


def run(args):
    import torch
    from peft import PeftModel
    from qualify_combined_teacher_contexts import native_messages
    from sdpo_agent_ablation import loss_for, teacher_logits
    from sdpo_prompting import prompt_ids
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(18)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer._sdpo_model_family = 'qwen3.5'
    model = PeftModel.from_pretrained(AutoModelForImageTextToText.from_pretrained(args.model,
        dtype=torch.bfloat16, device_map={'': 'cuda:0'}, attn_implementation='sdpa', local_files_only=True),
        args.adapter, is_trainable=True)
    model.load_adapter(args.adapter, adapter_name='sdpo_teacher', is_trainable=False)
    model.set_adapter('default')
    model.eval()
    nonzero_dropout = [(name, module.p) for name, module in model.named_modules()
                       if isinstance(module, torch.nn.Dropout) and module.p != 0]
    if nonzero_dropout:
        raise ValueError(f'Nonzero dropout invalidates this paired diagnostic: {nonzero_dropout}')
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    parameters = [p for n, p in model.named_parameters() if '.default.' in n and p.requires_grad]
    cases = json.loads((args.prepared / 'source-cases.json').read_text())['cases']
    results = []
    for case in cases:
        task = case['row']['task']
        old = (args.prepared / task / 'baseline-supplement.txt').read_text()
        new = (args.prepared / task / 'candidate-supplement.txt').read_text()
        terms = localized_terms(old)
        if terms != localized_terms(new):
            raise ValueError('Context arms changed localized diagnostic targets')
        if not terms:
            continue
        spans = exact_positions(case['row']['action_ids'], tokenizer, terms)
        for arm, supplement in [('full_draft', old), ('no_copy', new)]:
            row = copy.deepcopy(case['row'])
            row['teacher_prefix_ids'] = prompt_ids(tokenizer, native_messages(case['call']['messages']) +
                [{'role': 'user', 'content': supplement}], case['call']['tools'], thinking=True)
            if max(len(row['teacher_prefix_ids']), len(row['student_prefix_ids'])) + len(row['action_ids']) > 32768:
                raise ValueError('Whole original action does not fit; no truncation permitted')
            logits = teacher_logits(model, row)
            loss, advantages = loss_for(model, row, SimpleNamespace(topk=20, alpha=1.), teacher=logits)
            assert torch.isfinite(loss)
            gradient = torch.autograd.grad(loss, parameters, allow_unused=True)
            del logits, loss
            for span in spans:
                result = {'task': task, 'arm': arm, 'diagnostic_target': span, 'optimizer_updates': 0}
                positions = span['positions']
                if not positions:
                    result.update(measurable=False, reason='No exact original-token match; missing, not zero')
                else:
                    result['teacher_minus_student_original_token_logprobs'] = [advantages[i] for i in positions]
                    result['mean_teacher_minus_student_original_token_logprob'] = sum(advantages[i] for i in positions) / len(positions)
                    ids = torch.tensor([row['student_prefix_ids'] + row['action_ids'][:-1]], device='cuda')
                    support = torch.tensor([len(row['student_prefix_ids']) - 1 + i for i in positions], device='cuda')
                    targets = torch.tensor([row['action_ids'][i] for i in positions], device='cuda')
                    values = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                                   logits_to_keep=support).logits[0].float()
                    logprob = values.log_softmax(-1).gather(-1, targets[:, None]).mean()
                    target_gradient = torch.autograd.grad(logprob, parameters, allow_unused=True)
                    dot = sum(-float((g.float() * r.float()).sum()) for g, r in
                              zip(gradient, target_gradient, strict=True) if g is not None and r is not None)
                    result.update(measurable=True, negative_loss_gradient_dot_bad_token_logprob_gradient=dot,
                                  interpretation='negative suppresses these original bad-span tokens locally; positive reinforces them')
                    del target_gradient, values, logprob, ids, support, targets
                results.append(result)
                (args.output / 'directions.json').write_text(json.dumps(results, indent=2))
                print(json.dumps(result), flush=True)
            del gradient
            torch.cuda.empty_cache()
    receipt = {'completed': True, 'optimizer_updates': 0, 'results': results,
               'nonzero_dropout_modules': nonzero_dropout,
               'initial_adapter_sha256': hashlib.sha256((args.adapter / 'adapter_model.safetensors').read_bytes()).hexdigest(),
               'scope': 'Same original localized bad-span tokens across context arms. Not an Adam update, causal decision-level correctness proof or training approval.'}
    (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('model', 'adapter', 'prepared', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    (preflight if args.preflight_only else run)(args)
