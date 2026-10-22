"""Render retained qualification artifacts; no inference or score imputation."""

import argparse
import html
import json
from pathlib import Path


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def escaped(value):
    return html.escape(value if isinstance(value, str) else json.dumps(value, indent=2, ensure_ascii=False))


def details(label, value):
    return f'<details><summary>{escaped(label)}</summary><pre>{escaped(value)}</pre></details>'


def run(args):
    root = args.evidence
    old = read(root / 'results/proposals.json', [])
    new = read(root / 'pool-r2-results/proposals.json', [])
    directions = {r['task']: r for r in read(root / 'pool-r2-results/directions.json', [])}
    old_observer = {r['task']: r for r in read(root / 'observer-r5/rows.json', [])}
    repairs = read(root / 'repair-four-r2-results/proposals.json', [])
    no_copy = read(root / 'no-copy-ten-r3-results/proposals.json', [])
    no_copy_review = {r['task']: r for r in read(root / 'no-copy-ten-r3-results/manual-review.json', {}).get('reviews', [])}
    localized = read(root / 'bad-spans-r7/receipt.json', {})
    rows = []
    for proposal in new or old:
        task = proposal['task']
        prefix = root / ('prepared-v4' if new else 'prepared-v2') / task
        replay = proposal.get('native_replay', {})
        direction = directions.get(task, {})
        metric = direction.get('negative_loss_gradient_dot_reference_tool_logprob_gradient')
        errors = []
        for action in replay.get('decisions', []):
            result = action.get('result')
            if action.get('raised') or action.get('strict_admission_rejected') or (
                isinstance(result, dict) and (result.get('success') is False or result.get('error'))
            ):
                errors.append(action)
        status = 'Incomplete reply' if not proposal['complete'] else (
            'Replay error' if proposal.get('native_error') else 'Execution error' if errors else 'Needs semantic review')
        audit = read(prefix / 'audit.json', {})
        body = details('Exact deterministic supplement', (prefix / 'candidate-supplement.txt').read_text())
        body += details('Actual teacher reasoning and tool calls', proposal.get('message') or proposal.get('raw'))
        body += details('Native next-decision execution and scoring', replay or proposal.get('native_error'))
        body += details('Gradient diagnostic (not an optimizer update)', direction or 'Not available for this run')
        body += details('Failure localization and source coverage', audit)
        body += details('Previous run independent replay audit', old_observer.get(task))
        rows.append(f'<article id="{escaped(task)}"><h3>{escaped(task)}</h3><p>{escaped(status)} · '
                    f'whole-reference gradient projection: {escaped(metric)}</p>{body}</article>')
    counts = {
        'old_pool_proposals': len(old), 'old_complete': sum(p['complete'] for p in old),
        'new_pool_proposals': len(new), 'new_complete': sum(p['complete'] for p in new),
        'new_replay_errors': sum(bool(p.get('native_error')) for p in new),
        'new_gradient_measurable': sum(bool(r.get('measurable')) for r in directions.values()),
        'new_gradient_positive': sum(r.get('negative_loss_gradient_dot_reference_tool_logprob_gradient', 0) > 0
                                     for r in directions.values()),
        'optimizer_updates': 0,
    }
    repair_body = ''.join(details(p['task'], p.get('message') or p.get('raw')) for p in repairs)
    no_copy_body = ''
    for p in no_copy:
        task = p['task']
        no_copy_body += ('<article><h3>' + escaped(task) + '</h3>' +
                         details('Manual review', no_copy_review.get(task)) +
                         details('Exact no-copy supplement', (root / 'no-copy-ten-r3' / task / 'candidate-supplement.txt').read_text()) +
                         details('Actual generated reasoning and calls', p.get('message') or p.get('raw')) +
                         details('Native tool results and next-decision scoring', p.get('native_replay') or p.get('native_error')) +
                         (f'<p><a href="no-copy-ten-r3-results/full-contexts/{escaped(task)}.txt">Exact serialized full teacher context</a></p>'
                          if (root / 'no-copy-ten-r3-results/full-contexts' / (task + '.txt')).exists() else '') + '</article>')
    localized_rows = ''
    for row in localized.get('results', []):
        target = row['diagnostic_target']
        if target['scope'] == 'distinctive_field_part' or target['kind'] == 'employee_in_prohibited_public_notice':
            localized_rows += ('<tr><td>' + escaped(row['task']) + '</td><td>' + escaped(row['arm']) +
                               '</td><td>' + escaped(target.get('focused_text', target['term'])) +
                               '</td><td>' + escaped(row.get('negative_loss_gradient_dot_bad_token_logprob_gradient')) +
                               '</td><td>' + escaped(row.get('mean_teacher_minus_student_original_token_logprob')) + '</td></tr>')
    signal_body = ('''<h2>Same original bad-token diagnostic</h2><p>Both arms measure exactly the same original sampled token IDs.
Negative projection locally suppresses the selected tokens; positive favors them under infinitesimal SGD.
This is not an actual Adam step, batch32 update or full-call decision probability. Small values require numerical caution.
The field-part view excludes shared address syntax and common record-ID prefixes. The teacher-minus-student
column measures conditional log-probability difference on these same tokens, not a reward.</p>
<table><tr><th>Task</th><th>Context</th><th>Observed target</th><th>SGD projection</th><th>Teacher − student logP</th></tr>'''
                   + localized_rows + '</table>' + details('Full native-token/gradient receipt', localized))
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Focused teacher qualification — retained evidence</title><style>
body{font:16px/1.55 system-ui,sans-serif;margin:auto;padding:32px;max-width:1200px;color:#172329;background:#f7f8f6}
h1,h2,h3{line-height:1.2}article{background:white;border:1px solid #ccd4d4;border-radius:8px;padding:20px;margin:18px 0}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.5 monospace;background:#eef2f2;padding:16px;max-height:650px;overflow:auto}
summary{cursor:pointer;padding:9px 0;font-weight:600}.warning{padding:18px;background:#fff0d6;border-left:4px solid #a86600}
</style><h1>Focused deterministic teacher: qualification evidence</h1>
<p class="warning">Training is blocked pending semantic and corrective-signal qualification. Process exit0,
positive gradient projection and native task success alone do not approve training. All runs here apply zero optimizer updates.</p>
<p>The124 training tasks remain intact, separate from heldout62. Original selected decisions, student tokens,
model, sampling and budgets are retained. The new run changes candidate wording, observed-policy timing,
resource-handle presentation and outgoing factual grounding together. This is a context-repair bundle.</p>
<h2>Artifact counts</h2>''' + details('Exact counts from retained files', counts) + '''
<h2>How to interpret the evidence</h2><p>These are single saved decisions, not complete student rollouts or
benchmark learning gains. A prerequisite read can legitimately score less immediately than a write.
Gradient projection measures local infinitesimal SGD geometry toward the generated teacher tool spans,
conditioned on that reference reasoning. It does not establish actual Adam movement, repaired-token causality
or correctness. Missing diagnostics remain missing. Legacy mode names containing “deepseek” are loader labels;
these runs use only deterministic context and local Qwen inference.</p>
<p>The first independent observer was repaired to reproduce the entire native initial-world digest using
only two retained source forms, then match historical responses. Observer-r5 closed121/121 complete proposals.
This is corrected measurement over retained outputs, not a new model run or pooled successes.</p>
<h2>Four-case intermediate repair</h2><p>HR immediate revocation, recruitment send retention and the
Sales contact update improved. Marketing completed but invented source claims; this intermediate repair
was rejected for semantic qualification despite four positive projections.</p>''' + repair_body + '''
<h2>No-copy ten-case diagnostic</h2><p>This is a diagnostic subset of the same124 training pool, not a replacement training corpus.
It omits draft prose from the review summary and emphasizes explicit localized prohibitions. Original student tokens
are unchanged. Revenue Weekly and invented statistics disappear from the generated Marketing call, but literal
placeholders, skipped policy retrieval and incorrect resources remain. This candidate is not approved for training.</p>''' + no_copy_body + signal_body + '<h2>Every task in pool r2</h2>' + ''.join(rows) + '</html>'
    args.output.write_text(page)
    print(json.dumps({'output': str(args.output), **counts}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
