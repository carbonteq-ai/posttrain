"""Read-only failed-action signal checks on retained histories and checkpoint64."""

import argparse
import copy
import json
from pathlib import Path

import torch
from automationbench.tools import ALL_TOOLS
from automationbench_v1.world_codec import load_world
from peft import PeftModel
from self_feedback_native import prompt_ids
from self_feedback_signals import load_goal_context, teacher_message
from transformers import AutoModelForImageTextToText, AutoTokenizer


def select_cases(previous):
    folder = previous / 'step-0032/train'
    episodes = [json.loads(line) for line in (folder / 'native/traces.jsonl').read_text().splitlines()]
    by_tool = {n['message']['tool_call_id']: e for e in episodes for t in e['traces'] for n in t['nodes']
               if (n.get('message') or {}).get('role') == 'tool'}
    wanted = {
        'simple.gmail_reply_meeting_request': ('query', 'from:anika.patel@company.com', 'from:anika.patel'),
        'simple.slack_ceo_announcement': ('query', 'from:nathan.drake@company.com', 'from:nathan.drake'),
        'hr.offboarding_automation': ('spreadsheet', 'Employee Departures (ss_offboard)', 'ss_offboard'),
        'finance.credit_limit_review': ('spreadsheet', 'Credit Review (ss_credit_review)', 'ss_credit_review'),
    }
    found = {}
    for line in (folder / 'generation-calls.jsonl').read_text().splitlines():
        call = json.loads(line)
        tools = call['message'].get('tool_calls', [])
        if len(tools) != 1 or tools[0]['id'] not in by_tool:
            continue
        episode = by_tool[tools[0]['id']]
        name = episode['task']['data']['name']
        if name not in wanted or name in found or not call['complete']:
            continue
        key, bad, good = wanted[name]
        if len(tools) != 1 or json.loads(tools[0]['function']['arguments']).get(key) != bad:
            continue
        response = next(n['message']['content'] for t in episode['traces'] for n in t['nodes']
                        if (n.get('message') or {}).get('tool_call_id') == tools[0]['id'])
        context = load_goal_context(name, episode['task']['data']['initial_state'], episode['task']['data']['prompt'])
        if context['authority_status'] != 'verified':
            raise ValueError('signal probe manifest authority mismatch')
        senders = {s['sender'] for s in context.get('expected_source_senders', [])}
        if key == 'query' and len(senders) == 1:
            good = 'from:' + next(iter(senders))
        found[name] = {'name': name, 'episode': episode, 'call': call, 'parameter': key,
                       'bad_value': bad, 'good_value': good, 'response': response}
    if set(found) != set(wanted):
        raise ValueError('missing exact failure cases')
    return list(found.values())


def replay(case):
    functions = {f.__name__: f for f in ALL_TOOLS}
    world = load_world(case['episode']['task']['data']['initial_state'])
    for message in case['call']['messages']:
        for tool in message.get('tool_calls', []):
            f = tool['function']
            functions[f['name']](world, **json.loads(f['arguments']))
    function = case['call']['message']['tool_calls'][0]['function']
    arguments = json.loads(function['arguments'])
    original = json.loads(functions[function['name']](copy.deepcopy(world), **arguments))
    arguments[case['parameter']] = case['good_value']
    repaired = json.loads(functions[function['name']](copy.deepcopy(world), **arguments))
    observed = json.loads(case['response'])
    if original.get('success') != observed.get('success') or original.get('error') != observed.get('error'):
        raise ValueError('bad-call replay differs from retained feedback')
    if not repaired.get('success') or ('messages' in repaired and not repaired['messages']):
        raise ValueError('candidate correction did not retrieve evidence in the same pre-action state')
    return {'replayed_all_prior_calls': True, 'original': original, 'repaired': repaired,
            'repaired_function': {'name': function['name'], 'arguments': arguments},
            'scope': 'validated tool execution/retrieval, not complete task success'}


def prediction(model, prefix, action_prefix):
    ids = torch.tensor([prefix + action_prefix], device='cuda')
    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                   logits_to_keep=torch.tensor([ids.shape[1] - 1], device='cuda')).logits[0, 0]
    return logits.float().log_softmax(-1)


def compare(student, teacher, bad, good):
    kl = (student.exp() * (student - teacher)).sum().item()
    advantage_bad = (teacher[bad] - student[bad]).item() + kl
    advantage_good = (teacher[good] - student[good]).item() + kl
    gain = ((teacher[good] - teacher[bad]) - (student[good] - student[bad])).item()
    relative_descent = student[good].exp().item() * advantage_good - student[bad].exp().item() * advantage_bad
    return {'student_log_odds_correct_over_bad': (student[good] - student[bad]).item(),
            'teacher_log_odds_correct_over_bad': (teacher[good] - teacher[bad]).item(),
            'relative_log_odds_gain': gain, 'reverse_kl': kl,
            'bad_token_centered_teacher_advantage': advantage_bad,
            'descent_reduces_bad_logit': advantage_bad < 0,
            'descent_correct_vs_bad_log_odds_direction': relative_descent,
            'teacher_prefers_correct_token': bool(teacher[good] > teacher[bad]),
            'gate_passed': gain > 0 and bool(teacher[good] > teacher[bad]) and relative_descent > 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = select_cases(args.previous)
    validations = {case['name']: replay(case) for case in cases}
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    base = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16,
        device_map={'': 'cuda:0'}, attn_implementation='sdpa', local_files_only=True)
    checkpoint = args.previous / 'checkpoints/checkpoint-64'
    model = PeftModel.from_pretrained(base, checkpoint, is_trainable=False)
    model.load_adapter(checkpoint / 'self_teacher', adapter_name='self_teacher', is_trainable=False)
    model.eval()
    results = []
    with torch.no_grad():
        for case in cases:
            call = case['call']
            raw = tokenizer.decode(call['action_ids'], skip_special_tokens=False)
            if tokenizer.encode(raw, add_special_tokens=False) != call['action_ids']:
                raise ValueError('original action does not roundtrip native tokenizer')
            start = raw.index(f'<parameter={case["parameter"]}>')
            repaired_raw = raw[:start] + raw[start:].replace(case['bad_value'], case['good_value'], 1)
            repaired_ids = tokenizer.encode(repaired_raw, add_special_tokens=False)
            original = call['action_ids']
            divergence = next(i for i, (a, b) in enumerate(zip(original, repaired_ids, strict=False)) if a != b)
            bad, good = original[divergence], repaired_ids[divergence]
            function = call['message']['tool_calls'][0]['function']
            feedback = json.dumps([{'tool': function['name'], 'response': case['response']}], sort_keys=True)
            goals = load_goal_context(case['name'], case['episode']['task']['data']['initial_state'],
                                      case['episode']['task']['data']['prompt'])
            contexts = {}
            for name, context in [('minimal', None), ('goals-observations-v4', goals)]:
                messages = call['messages'] + [{'role': 'user', 'content': teacher_message(call, feedback, context)}]
                contexts[name] = prompt_ids(tokenizer, messages, call['tools'])
            if any(len(prefix) + divergence + 1 > 16384 for prefix in contexts.values()):
                raise ValueError('probe exceeds qualified context; no token truncation allowed')
            for weights in ['initial_base', 'checkpoint64']:
                if weights == 'initial_base':
                    with model.disable_adapter():
                        student = prediction(model, call['prefix_ids'], original[:divergence])
                        teachers = {name: prediction(model, prefix, original[:divergence])
                                    for name, prefix in contexts.items()}
                else:
                    model.set_adapter('default')
                    student = prediction(model, call['prefix_ids'], original[:divergence])
                    model.set_adapter('self_teacher')
                    teachers = {name: prediction(model, prefix, original[:divergence])
                                for name, prefix in contexts.items()}
                result = {'task': case['name'], 'weights': weights, 'position': divergence,
                          'bad_token': tokenizer.decode([bad]), 'correct_token': tokenizer.decode([good]),
                          'contexts': {name: compare(student, teacher, bad, good)
                                       for name, teacher in teachers.items()},
                          'replay_validation': validations[case['name']],
                          'prefix_lengths': {name: len(prefix) for name, prefix in contexts.items()}}
                results.append(result)
                print(json.dumps(result), flush=True)
    qualified = all(r['contexts']['goals-observations-v4']['gate_passed'] for r in results)
    receipt = {'qualified': qualified, 'results': results, 'external_model_calls': 0,
               'scope': 'first divergent tool-argument token, not full-sequence or task success proof',
               'checkpoint': str(checkpoint), 'optimizer_updates': 0,
               'gate': 'all four replay-validated alternatives favored over bad token and relative to student, with actual reverse-KL logit descent improving correct-vs-bad odds, at base and checkpoint64'}
    (args.output / 'teacher-signal.json').write_text(json.dumps(receipt, indent=2))
    if not qualified:
        raise ValueError('teacher signal gate failed; preserve evidence and do not launch training')


if __name__ == '__main__':
    main()
