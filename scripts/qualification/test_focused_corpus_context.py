import json

import pytest
from focused_corpus_context import (
    build,
    candidate_review_summary,
    linked_feedback,
    observed_resource_handles,
    policy_extracts,
)


def test_exact_policy_timing_retained_without_inventing_policy():
    passage='Do NOT post publicly. Email IT requesting immediate access revocation.'
    selected,_=policy_extracts([{'role':'tool','content':json.dumps({'cells':{
        'Policy':'Involuntary Termination','Steps':passage}})}])
    assert selected[0]['exact_passage']==passage
    assert policy_extracts([])[0]==[]


def test_resource_handles_only_from_observed_discovery_envelopes():
    messages = [{'role': 'user', 'content': '{"channels":[{"id":"unverified"}]}'},
                {'role': 'tool', 'content': json.dumps({'success': True, 'spreadsheet': {
                    'id': 'ss_seen', 'title': 'Demo Requests', 'worksheets': [
                        {'id': 'ws_seen', 'title': 'Requests'}]},
                    'rows': [{'id': 'business_row_not_a_resource'}]})},
                {'role': 'tool', 'content': json.dumps({'success': False, 'channels': [{'id': 'failed'}]})}]
    rows = observed_resource_handles(messages)
    assert [(r['kind'], r['id']) for r in rows] == [('spreadsheet', 'ss_seen'), ('worksheet', 'ws_seen')]
    assert rows[1]['spreadsheet_id'] == 'ss_seen'
    assert len(observed_resource_handles([messages[1], messages[1]])) == 2


def test_draft_summary_keeps_identity_without_exporting_or_mutating_unsourced_prose():
    call = {'message': {'tool_calls': [{'function': {'name': 'gmail_send_email', 'arguments':
            {'to': 'observed@example.com', 'body': 'UNSUPPORTED47%', 'subject': 'Pitch'}}}]}}
    summary = candidate_review_summary(call)
    assert summary[0]['candidate_arguments_for_review'] == {'to': 'observed@example.com', 'subject': 'Pitch'}
    assert summary[0]['draft_text_fields_intentionally_not_repeated'] == ['body']
    assert 'UNSUPPORTED47%' not in json.dumps(summary)
    assert call['message']['tool_calls'][0]['function']['arguments']['body'] == 'UNSUPPORTED47%'


def test_link_negative_feedback_only_to_matching_proposed_text():
    call={'message':{'tool_calls':[{'function':{'name':'gmail_send_email','arguments':{'body':'Excluded Building A'}}}]}}
    checks=[{'type':'gmail_message_not_sent','passed':False,'params':{'body_contains':'Building A','to':'PRIVATE_RECIPIENT'}}]
    facts=linked_feedback('operations.example',call,checks)
    assert facts[0]['proposal_call']==1
    assert 'PRIVATE_RECIPIENT' not in json.dumps(facts)
    assert linked_feedback('operations.example',call,[{**checks[0],'excluded':True}])==[]


def test_contract_audit_covers_each_check_without_exporting_hidden_constants():
    material={'task':'simple.example','goal_manifest':{'sources':{'x':{'adapter':'gmail.messages@1','kind':'send'}},
        'checks':[{'check_id':'c','source':'x','operator':'effects.required_when@1','role':'goal',
                   'private_expected':'SECRET_LITERAL'}]}}
    call={'messages':[{'role':'system','content':'Original restrictions'}],'message':{'content':'Next action'}}
    text,audit=build(material,call,[],[])
    assert audit['contract_coverage'][0]['check_ids']==['c']
    assert 'SECRET_LITERAL' not in text
    assert not audit['semantic_qualified']
    material['goal_manifest']['checks'][0]['operator']='unknown@1'
    with pytest.raises(ValueError,match='Unreviewed'):
        build(material,call,[],[])
