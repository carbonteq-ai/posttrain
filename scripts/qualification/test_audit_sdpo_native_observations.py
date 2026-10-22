"""Error payloads with empty result lists must not become successful empty searches."""

import json

import pytest
from audit_sdpo_native_observations import audit


def test_errors_are_separate_from_successful_empty_retrievals(tmp_path):
    nodes = []
    responses = [{'success': True, 'messages': [], 'result_count': 0},
                 {'success': False, 'results': [], 'count': 0},
                 {'success': True, 'messages': [], 'result_count': 0}]
    for i, response in enumerate(responses):
        nodes.extend([{'message': {'role': 'assistant', 'tool_calls': [
            {'id': str(i), 'name': 'search', 'arguments': '{"query":"x"}'}]}},
            {'message': {'role': 'tool', 'tool_call_id': str(i), 'content': json.dumps(response)}}])
    episode = {'id': 'one', 'assessment_batches': [], 'traces': [
        {'nodes': nodes, 'metrics': {'tool_empty_results': 2}, 'assessment_batches': []}]}
    path = tmp_path / 'step-0000/heldout62/native/traces.jsonl'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(episode) + '\n')
    (path.parent.parent / 'outcomes.json').write_text('{}')
    counts = audit(tmp_path)['populations'][0]['counts']
    assert counts['API_successful_empty_retrieval_responses'] == 2
    assert counts['explicit_tool_error_responses'] == 1
    assert counts['repeated_identical_successful_empty_responses'] == 1
    assert counts['repeated_exact_tool_calls'] == 2
    assert counts['native_empty_metric_trace_coverage'] == 1
    assert counts['native_tool_empty_results_observed'] == 2
    with path.open('a') as stream:
        stream.write(json.dumps(episode) + '\n')
    with pytest.raises(ValueError, match='duplicate native episode'):
        audit(tmp_path)
