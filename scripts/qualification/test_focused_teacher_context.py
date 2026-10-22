import json

import pytest
from focused_teacher_context import build, precise_feedback


def case():
    return {'messages': [{'role':'system','content':'Do not name excluded rows'},
                         {'role':'user','content':'Use current policy'}],
            'message':{'tool_calls':[{'function':{'name':'salesforce_contact_update',
                        'arguments':json.dumps({'id':'OBSERVED_CONTACT','phone':'123'})}}]}}


def test_whole_proposal_and_scope_preserved_without_hidden_parameters():
    text, audit=build('sales.soft_matching',case(),
        [{'type':'phone_equal','passed':False,'params':{'id':'HIDDEN_EXPECTED_ID'}},
         {'type':'excluded_failure','passed':False,'excluded':True}],[])
    assert 'OBSERVED_CONTACT' in text and 'HIDDEN_EXPECTED_ID' not in text
    assert 'excluded_failure' not in text and 'phone_equal' in text
    assert 'BEFORE' in text and 'not by themselves' in text
    assert 'singular wording or hidden manifest target IDs' in text
    assert not audit['automatic_complete_manifest_semantics_qualified']


def test_no_repeated_history_or_guessed_evidence():
    text, _=build('hr.offboarding_automation',case(),[],[])
    assert 'Do not name excluded rows' not in text
    assert 'Payroll/legal' in text
    assert 'later turns' in text
    assert 'V1' not in text and 'response_sha256' not in text


def test_unknown_tasks_fail_closed():
    with pytest.raises(ValueError,match='No reviewed'):
        build('other.task',case(),[],[])


def test_sales_feedback_joins_selected_identity_without_exporting_expected_answer():
    call=case()
    call['messages'].append({'role':'tool','content':json.dumps({'results':[
        {'Id':'OBSERVED_CONTACT','Phone':'before'}]})})
    assertions=[{'type':'salesforce_contact_phone_equals','passed':False,
                 'params':{'contact_id':'OBSERVED_CONTACT','phone':'PRIVATE_EXPECTED'}}]
    result=precise_feedback('sales.soft_matching',call,assertions)
    assert result[0]['proposal_call']==1
    assert result[0]['observed_phone_before']=='before'
    assert 'PRIVATE_EXPECTED' not in json.dumps(result)
    assert not precise_feedback('sales.soft_matching',call,[{
        **assertions[0],'params':{'contact_id':'OTHER_CONTACT'}}])


def test_hr_feedback_requires_retrieved_policy_and_preserves_other_actions():
    call=case()
    call['messages'].append({'role':'tool','content':json.dumps({'rows':[
        {'cells':{'Employee':'Person A','Employee Email':'a@example.com','Type':'Involuntary - Position Eliminated'}},
        {'cells':{'Policy':'Involuntary Termination','Steps':'Do NOT post to any public Slack channel'}}]})})
    call['message']={'tool_calls':[{'function':{'name':'slack_send_channel_message','arguments':{
        'text':'Person A (Employee: wrong@example.com, Department: Test)','channel':'#departures'}}}]}
    result=precise_feedback('hr.offboarding_automation',call,[])
    assert {r['problem'] for r in result}=={'public_involuntary_notice','source_email_mismatch'}
    call['messages'].pop()
    assert not precise_feedback('hr.offboarding_automation',call,[])
