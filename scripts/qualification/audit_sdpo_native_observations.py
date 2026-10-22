"""Stream native repetition/error/assessment evidence without changing rewards."""

import argparse
import json
from collections import Counter
from pathlib import Path


def audit(root):
    seen, populations = set(), []
    for path in sorted(root.glob('step-*/*/native/traces.jsonl')):
        counts = Counter()
        with path.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                episode = json.loads(line)
                if episode['id'] in seen:
                    raise ValueError('duplicate native episode ID')
                seen.add(episode['id'])
                counts['episodes'] += 1
                counts['episode_assessment_batch_records'] += len(episode.get('assessment_batches', []))
                calls, signatures, empty_seen = {}, [], set()
                for trace in episode['traces']:
                    counts['traces'] += 1
                    counts['trace_assessment_batch_records'] += len(trace.get('assessment_batches', []))
                    metric = trace.get('metrics', {}).get('tool_empty_results')
                    if metric is not None:
                        counts['native_empty_metric_trace_coverage'] += 1
                        counts['native_tool_empty_results_observed'] += metric
                    for node in trace.get('nodes', []):
                        message = node.get('message') or {}
                        for call in message.get('tool_calls') or []:
                            function = call.get('function', call)
                            arguments = function.get('arguments', {})
                            if isinstance(arguments, str):
                                try:
                                    arguments = json.loads(arguments)
                                except json.JSONDecodeError:
                                    counts['malformed_tool_argument_strings'] += 1
                            signature = json.dumps([function.get('name'), arguments], sort_keys=True)
                            if call['id'] in calls:
                                raise ValueError('duplicate native tool-call ID within episode')
                            calls[call['id']] = signature
                            signatures.append(signature)
                        if message.get('role') != 'tool':
                            continue
                        signature = calls.get(message.get('tool_call_id'))
                        counts['unmatched_tool_responses'] += signature is None
                        content = message.get('content', '')
                        if isinstance(content, str) and content.startswith('Error executing tool '):
                            counts['explicit_tool_error_responses'] += 1
                            continue
                        try:
                            response = json.loads(content) if isinstance(content, str) else content
                        except (ValueError, TypeError):
                            counts['unparsed_tool_response_contents'] += 1
                            continue
                        if not isinstance(response, dict):
                            continue
                        if response.get('success') is False:
                            counts['explicit_tool_error_responses'] += 1
                        empty = response.get('success') is True and (
                            response.get('messages') == [] and response.get('result_count') == 0 or
                            response.get('results') == [] and response.get('count') == 0)
                        if empty:
                            counts['API_successful_empty_retrieval_responses'] += 1
                            counts['repeated_identical_successful_empty_responses'] += (
                                signature is not None and signature in empty_seen)
                            if signature is not None:
                                empty_seen.add(signature)
                counts['attempted_tool_calls'] += len(signatures)
                counts['repeated_exact_tool_calls'] += len(signatures) - len(set(signatures))
        populations.append({'population': str(path.parents[1].relative_to(root)),
                            'closed': (path.parents[1] / 'outcomes.json').exists(), 'counts': dict(counts)})
    return {'unique_native_episodes': len(seen), 'populations': populations, 'retry_count': None,
            'scope': 'Identical calls and successful empty retrievals are observations, not necessarily mistakes. '
                     'Only explicit API-successful empty retrievals count as empty; errors are separate. '
                     'Response counts cover messages/result_count and results/count shapes; native empty metrics are separate. '
                     'Zero assessment batches means no provider-batch deduplication is applicable. '
                     'Native benchmark scores remain unchanged; no external calls or rescoring.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(audit(args.root), indent=2) + '\n')
