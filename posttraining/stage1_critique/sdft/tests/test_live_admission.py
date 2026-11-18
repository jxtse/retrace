"""Synthetic CPU fixtures for transport/split admission; not experiment output."""
import copy, hashlib, json, unittest
from admit_live_batch import verify_wire_record, select_indices, normalized_assistant, group_fixed_budget_attempts

MODEL='student-test'
def fixture():
    request={'model':MODEL,'messages':[{'role':'user','content':'test task'}],
             'tools':[],'max_tokens':32768,'reasoning_effort':'low','temperature':0.7}
    raw=json.dumps(request)
    return {'seq':1,'phase':'main','request_index':1,'call_id':'call-fixture','attempt':1,
            'input_message_count':1,'request':request,
            'request_sha256':hashlib.sha256(json.dumps(request,ensure_ascii=False,sort_keys=True).encode()).hexdigest(),
            'raw_request_body_utf8':raw,'raw_request_body_sha256':hashlib.sha256(raw.encode()).hexdigest(),
            'prepared_request_verified':True,'status_code':200,'outcome':'ok',
            'response':{'id':'response-fixture','model':MODEL,'choices':[{'message':{'role':'assistant','content':'OK'},'finish_reason':'stop'}]}}

class AdmissionTests(unittest.TestCase):
    def test_valid_wire(self):
        self.assertEqual(verify_wire_record(fixture(),MODEL)['model'], MODEL)
    def test_unverified_snapshot_rejected(self):
        r=fixture();r['prepared_request_verified']=False
        with self.assertRaises(ValueError):verify_wire_record(r,MODEL)
    def test_body_tamper_rejected(self):
        r=fixture();r['raw_request_body_utf8']+=' '
        with self.assertRaises(ValueError):verify_wire_record(r,MODEL)
    def test_wrong_response_model_rejected(self):
        r=fixture();r['response']['model']='teacher'
        with self.assertRaises(ValueError):verify_wire_record(r,MODEL)
    def test_removed_history_rejected(self):
        r=fixture();r['input_message_count']=2
        with self.assertRaises(ValueError):verify_wire_record(r,MODEL)
    def test_recovery_rejected(self):
        r=fixture();r['outcome']='recovery_ok'
        with self.assertRaises(ValueError):verify_wire_record(r,MODEL)
    def _truncated(self,seq,attempt,max_tokens=32768):
        r=fixture();r['seq']=seq;r['attempt']=attempt;r['outcome']='empty'
        r['request']['max_tokens']=max_tokens
        r['response']={'id':f'trunc-{seq}','model':MODEL,'choices':[{'message':{'role':'assistant','content':None,'reasoning_content':'...'},'finish_reason':'length'}]}
        return r
    def test_fixed_budget_identical_retry_keeps_final_attempt(self):
        final=fixture();final['seq']=2;final['attempt']=2
        finals=group_fixed_budget_attempts([self._truncated(1,1),final])
        self.assertEqual([f['seq'] for f in finals],[2])
        self.assertEqual(verify_wire_record(finals[0],MODEL)['max_tokens'],32768)
    def test_budget_growth_retry_rejected(self):
        final=fixture();final['seq']=2;final['attempt']=2;final['request']['max_tokens']=65536
        raw=json.dumps(final['request']);final['raw_request_body_utf8']=raw
        with self.assertRaises(ValueError):group_fixed_budget_attempts([self._truncated(1,1),final])
    def test_retry_after_visible_output_rejected(self):
        first=fixture();first['outcome']='empty';first['response']['choices'][0]['finish_reason']='stop'  # content 'OK' present
        final=fixture();final['seq']=2;final['attempt']=2
        with self.assertRaises(ValueError):group_fixed_budget_attempts([first,final])
    def test_retry_after_reasoning_only_stop_accepted(self):
        first=fixture();first['outcome']='empty';first['response']['choices'][0]['finish_reason']='stop';first['response']['choices'][0]['message']={'role':'assistant','content':None,'reasoning_content':'...'}
        final=fixture();final['seq']=2;final['attempt']=2
        self.assertEqual([f['seq'] for f in group_fixed_budget_attempts([first,final])],[2])
    def test_output_cap_rejected(self):
        r=fixture();r['response']['choices'][0]['finish_reason']='length'
        with self.assertRaises(ValueError):verify_wire_record(r,MODEL)
    def test_selection_is_fixed_not_score_based(self):
        self.assertEqual(select_indices(40),[0,13,26,39])
        self.assertEqual(select_indices(2),[0,1])
    def test_assistant_tool_arguments_object_equivalence(self):
        a={'role':'assistant','content':None,'tool_calls':[{'id':'t','type':'function','function':{'name':'f','arguments':'{"a":1}'}}]}
        b=copy.deepcopy(a);b['content']='';b['tool_calls'][0]['function']['arguments']={'a':1}
        self.assertEqual(normalized_assistant(a),normalized_assistant(b))

if __name__=='__main__':unittest.main()
