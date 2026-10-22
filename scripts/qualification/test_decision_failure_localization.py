import json

from decision_failure_localization import localize, select


def call(name, args):
    return {'messages': [], 'message': {'tool_calls':[{'id':'t','function':{'name':name,'arguments':json.dumps(args)}}]},
            'complete':True,'prefix_ids':[1,2],'action_ids':[3,4],'sampled_logprobs':[-1.,-1.]}


def test_exact_row_prohibition_is_not_broadened_to_other_rows():
    checks=[{'type':'google_sheets_row_not_updated','passed':False,'params':{
        'spreadsheet_id':'s','row_id':3,'cell_contains':{'Status':'Enrolled'}}}]
    c=call('google_sheets_update_row',{'spreadsheet':'s','row_id':'3','cells':'{"Status":"Enrolled"}'})
    assert localize(c,checks,[])[0]['kind']=='failed_row_write_prohibition'
    assert not localize(call('google_sheets_update_row',{'spreadsheet':'s','row_id':4,'cells':{'Status':'Enrolled'}}),checks,[])


def test_recipient_feedback_does_not_export_unrelated_private_parameters():
    checks=[{'type':'gmail_message_not_sent_to','passed':False,'params':{'to':'observed@example.com','secret':'HIDDEN'}}]
    result=localize(call('gmail_send_email',{'to':'observed@example.com'}),checks,[])
    assert result[0]['recipient_already_in_proposal']=='observed@example.com'
    assert 'HIDDEN' not in json.dumps(result)


def test_prohibited_write_outranks_earlier_recovered_error():
    checks=[{'type':'gmail_message_not_sent_to','passed':False,'params':{'to':'observed@example.com'}}]
    episode={'traces':[{'nodes':[{'message':{'role':'tool','tool_call_id':'t','content':'{"success":false,"error":"early discovery error"}'}}],
                       'info':{'automationbench':{'assertions':checks}}}]}
    candidates=[{'generation_index':1,'call':call('google_drive_find_multiple_files',{})},
                {'generation_index':9,'call':call('gmail_send_email',{'to':'observed@example.com'})}]
    chosen,audit=select(episode,candidates)
    assert chosen['generation_index']==9
    assert audit['original_tokens_preserved'] and not audit['causal_credit_claimed']


def test_missing_summary_is_workflow_omission_not_wrong_reply():
    episode={'traces':[{'nodes':[{'message':{'role':'assistant','tool_calls':[{'id':'t','name':'gorgias_create_ticket_message','arguments':'{}'}]}}],
                       'info':{'automationbench':{'assertions':[{'type':'slack_message_exists','passed':False}]}}}]}
    selected,audit=select(episode,[{'generation_index':1,'call':call('gorgias_create_ticket_message',{})},
                                 {'generation_index':5,'call':call('google_sheets_add_row',{})}])
    assert selected['generation_index']==5
    assert not audit['selected_action_findings']
    assert audit['workflow_omissions'][0]['operation_family']=='slack outgoing message'


def test_channel_scope_requires_observed_id_binding_and_correct_destination():
    c=call('slack_send_channel_message',{'channel':'C_OBS','text':'Excluded item'})
    c['messages']=[{'role':'tool','content':json.dumps({'channels':[{'id':'C_OBS','name':'team'}]})}]
    check={'type':'slack_message_not_exists','passed':False,'params':{'channel_name':'team','text_contains':'Excluded item'}}
    assert localize(c,[check],[])[0]['kind']=='failed_outgoing_text_prohibition'
    assert not localize(c,[{**check,'params':{**check['params'],'channel_name':'other'}}],[])


def test_field_preservation_uses_observed_before_value_only():
    c=call('salesforce_contact_update',{'id':'OBS','phone':'changed'})
    c['messages']=[{'role':'tool','content':json.dumps({'Id':'OBS','Phone':'before'})}]
    check={'type':'salesforce_contact_phone_equals','passed':False,'params':{'contact_id':'OBS','phone':'before'}}
    assert localize(c,[check],[])[0]['observed_phone_before']=='before'
    assert not localize(c,[{**check,'params':{'contact_id':'OBS','phone':'HIDDEN_DIFFERENT_EXPECTATION'}}],[])
