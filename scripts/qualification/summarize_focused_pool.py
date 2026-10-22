"""Missing-aware descriptive counts for teacher diagnostics, not learning gains."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def run(args):
    root = args.evidence
    proposals = json.loads((root / 'pool-r2-results/proposals.json').read_text())
    directions = json.loads((root / 'pool-r2-results/directions.json').read_text())
    cases = json.loads((root / 'prepared-v4/source-cases.json').read_text())['cases']
    old_cases = json.loads((root / 'prepared-v2/source-cases.json').read_text())['cases']
    names = {c['row']['task'] for c in cases}
    assert len(proposals) == len(names) == 124 and {p['task'] for p in proposals} == names
    before = {c['row']['task']: c for c in old_cases}
    for c in cases:
        old = before[c['row']['task']]
        assert c['row']['episode_id'] == old['row']['episode_id']
        assert c['row']['generation_index'] == old['row']['generation_index']
        assert c['row']['action_ids'] == old['row']['action_ids']
        assert c['row']['student_prefix_ids'] == old['row']['student_prefix_ids']
    assert {d['task'] for d in directions} == names
    errors, lengths, warnings = [], [], []
    for p in proposals:
        task = p['task']
        if not p['complete']:
            lengths.append({'task': task, 'finish_reason': p.get('finish_reason')})
            continue
        replay = p.get('native_replay')
        if not replay:
            errors.append({'task': task, 'observer_error': p.get('native_error')})
            continue
        assert replay['native_initial_digest_verified'] is True
        for action in replay['decisions']:
            result = action.get('result')
            if action.get('raised') or action.get('strict_admission_rejected') or (
                isinstance(result, dict) and (result.get('success') is False or result.get('error'))
            ):
                errors.append({'task': task, 'tool': action['tool'], 'result': result,
                               'raised': action.get('raised'),
                               'strict_admission_rejected': action.get('strict_admission_rejected')})
        if task == 'hr.offboarding_automation':
            for action in replay['decisions']:
                arguments = action.get('arguments', {})
                body = str(arguments.get('body', ''))
                if 'Diana Reese' in body and 'as of last day' in body:
                    warnings.append({'task': task, 'problem': 'Known immediate-revocation timing error', 'body': body})
    measurable = [d for d in directions if d.get('measurable')]
    result = {
        'tasks': 124, 'same_original_decisions_as_r1': True, 'optimizer_updates': 0,
        'complete_proposals': sum(p['complete'] for p in proposals),
        'incomplete_proposals': lengths, 'execution_or_observer_errors': errors,
        'error_tasks': sorted({e['task'] for e in errors}),
        'known_semantic_warnings': warnings,
        'measurable_projections': len(measurable),
        'projection_signs': dict(Counter('positive' if d['negative_loss_gradient_dot_reference_tool_logprob_gradient'] > 0
                                       else 'nonpositive' for d in measurable)),
        'semantic_qualified': False, 'training_approved': False,
        'limitations': [
            'No comprehensive semantic review is implied by execution without errors.',
            'Projection is whole-reference infinitesimal SGD geometry, not actual Adam or repaired-token evidence.',
            '15 retained selected failures remain explicitly unlocalized.',
            'Native next-decision scores are not full episode or learned student gains.',
        ],
        'artifact_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (root / 'pool-r2-results/proposals.json', root / 'pool-r2-results/directions.json')},
    }
    (root / 'pool-r2-summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ('artifact_sha256', 'execution_or_observer_errors')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    run(parser.parse_args())
