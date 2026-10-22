"""Read-only original-token and teacher-only-context audit of applied updates."""

import argparse
import copy
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def restore_reasoning_alias(messages):
    """Invert the pinned request validator's proven string-field rename only."""
    restored = copy.deepcopy(messages)
    count = 0
    for message in restored:
        if message['role'] == 'assistant' and 'reasoning' in message:
            reasoning = message['reasoning']
            if not isinstance(reasoning, str):
                raise ValueError('unqualified non-string reasoning alias')
            if 'reasoning_content' in message and message['reasoning_content'] != reasoning:
                raise ValueError('conflicting reasoning aliases')
            message['reasoning_content'] = message.pop('reasoning')
            count += 1
    return restored, count


def audit(root, tokenizer, render):
    config = read(root / 'inputs.json')
    updates = read(root / 'updates.json') if (root / 'updates.json').exists() else []
    thinking = config['thinking']
    steps = []
    for update in updates:
        step = update['step']
        folder = root / f'step-{step:04d}' / 'online_train'
        selected = read(folder / 'selected-transitions.json')
        admission = read(folder / 'training-admission.json')
        indexed = {r['generation_index']: r for r in selected}
        if len(indexed) != 32 or len({r['task'] for r in selected}) != 32:
            raise ValueError('not32distinct selected tasks/actions')
        def identity(row):
            return row['task'], row['episode_id'], row['generation_index']
        if {identity(r) for r in selected} != {identity(r) for r in admission['selected']}:
            raise ValueError('selected token rows differ from admission evidence')
        if admission['truncated_training_sequences']:
            raise ValueError('training truncation recorded')
        episodes = {e['id']: e for e in (json.loads(line) for line in
                    (folder / 'native/traces.jsonl').read_text().splitlines() if line)}
        witnessed, reasoning, references, alias_histories, reasoning_tokens = set(), 0, 0, 0, 0
        with (folder / 'generation-calls.jsonl').open() as stream:
            for index, line in enumerate(stream):
                if index not in indexed:
                    continue
                call, row = json.loads(line), indexed[index]
                if (call['prefix_ids'] != row['student_prefix_ids']
                        or call['action_ids'] != row['action_ids']
                        or call['sampled_logprobs'] != row['sampled_logprobs']
                        or not call['complete'] or not row['complete']
                        or call['policy_version'] != step - 1 or row['policy_version'] != step - 1
                        or bool(call.get('thinking_enabled', False)) != thinking):
                    raise ValueError('original native prefix/action/policy/mode mismatch')
                restored, alias_count = restore_reasoning_alias(call['messages'])
                if render(tokenizer, restored, call['tools'], thinking=thinking) != row['student_prefix_ids']:
                    raise ValueError('student prompt differs from original native messages')
                teacher_messages = call['messages'] + [{'role': 'user', 'content': row['teacher_feedback']}]
                if render(tokenizer, teacher_messages, call['tools'], thinking=thinking) != row['teacher_prefix_ids']:
                    raise ValueError('teacher feedback prompt differs or was truncated')
                restored_teacher = restored + [{'role': 'user', 'content': row['teacher_feedback']}]
                if render(tokenizer, restored_teacher, call['tools'], thinking=thinking) != row['teacher_prefix_ids']:
                    raise ValueError('reasoning alias changes actual teacher conditioning')
                if (max(len(row['student_prefix_ids']), len(row['teacher_prefix_ids']))
                        + len(row['action_ids']) > config['context_limit']):
                    raise ValueError('overlong training transition admitted')
                episode = episodes[row['episode_id']]
                if episode['task']['data']['name'] != row['task']:
                    raise ValueError('selected native episode task mismatch')
                reference = row.get('reference_episode_id')
                if row['offline_luna_reference_used']:
                    raise ValueError('unexpected offline Luna in control/thinking audit')
                if reference is not None:
                    sibling = episodes[reference]
                    if (sibling['task']['data']['name'] != row['task'] or
                            sibling['traces'][0]['metrics'].get('task_completed_correctly') != 1):
                        raise ValueError('successful sibling source is not verified same-task success')
                    references += 1
                witnessed.add(index)
                reasoning += bool(call.get('reasoning_content_present', False))
                alias_histories += bool(alias_count)
                reasoning_tokens += call.get('reasoning_prefix_token_count', 0)
        if witnessed != set(indexed):
            raise ValueError('selected generation has no original native call')
        steps.append({'step': step, 'checked_actions': len(witnessed),
                      'actions_with_native_reasoning': reasoning, 'verified_successful_sibling_contexts': references,
                      'histories_with_reasoning_alias_restoration': alias_histories,
                      'native_reasoning_target_tokens': reasoning_tokens,
                      'selected_file_sha256': hashlib.sha256((folder / 'selected-transitions.json').read_bytes()).hexdigest()})
    return {'qualified_applied_update_lineage': bool(steps), 'thinking': thinking,
            'actual_updates': len(updates), 'steps': steps, 'GPU_used': False,
            'reasoning_alias_policy': 'Invert only the pinned native request string-field rename reasoning_content->reasoning. '
                                      'Restoration must reproduce original student tokens exactly and leave actual teacher tokens identical.',
            'scope': 'All selected applied actions preserve exact original native student messages/token IDs. '
                     'Feedback is added only to teacher context; successful siblings are native same-task successes. '
                     'This is provenance, not a proof that teacher preferences are correct or rewards improve.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from sdpo_prompting import prompt_ids
    from transformers import AutoTokenizer

    model = '/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    args.output.write_text(json.dumps(audit(args.root, tokenizer, prompt_ids), indent=2) + '\n')
