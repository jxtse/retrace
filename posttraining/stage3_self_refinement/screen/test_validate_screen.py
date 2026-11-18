"""Synthetic format-validation tests only; these are NOT chemistry verdicts."""
import unittest,copy
from validate_screen import validate_target
class TestValidator(unittest.TestCase):
 def setUp(self):
  self.sources={f'999_{rep}':({'success':True,'best_route':{'steps':['SYNTHETIC_A>>SYNTHETIC_B']},'route_text':'SYNTHETIC terminal'}, {'events':[{'content':'SYNTHETIC evidence text'}]}) for rep in ['rep1','rep2']}
  self.record={'target_id':999,'trajectories':[{'key':f'999_{rep}','verdict':'ADMIT_CANDIDATE','confidence':'medium','route_assessment_zh':'合成格式夹具，不是实际审查','process_assessment_zh':'格式测试','substantive_flags':[],'evidence':[{'source':'events','event_index':0,'step_number':None,'quote':'SYNTHETIC evidence','explanation_zh':'格式测试'}],'reading_scope':{'all_final_steps_read':True,'compact_process_read':True,'extra_event_indices':[]},'followup_zh':'格式测试'} for rep in ['rep1','rep2']],'preferred_key':'999_rep1','selection_reason_zh':'格式测试','needs_additional_rollout':False}
 def test_valid(self):validate_target(self.record,999,self.sources)
 def test_quote_not_literal_rejected(self):
  self.record['trajectories'][0]['evidence'][0]['quote']='SYNTHETIC ... text'
  with self.assertRaises(ValueError):validate_target(self.record,999,self.sources)
 def test_wrong_event_rejected(self):
  self.record['trajectories'][0]['evidence'][0]['event_index']=1
  with self.assertRaises(ValueError):validate_target(self.record,999,self.sources)
 def test_bool_index_rejected(self):
  self.record['trajectories'][0]['evidence'][0]['event_index']=False
  with self.assertRaises(ValueError):validate_target(self.record,999,self.sources)
 def test_native_failure_admission_rejected(self):
  self.sources['999_rep1'][0]['success']=False
  with self.assertRaises(ValueError):validate_target(self.record,999,self.sources)
 def test_wrong_preference_rejected(self):
  self.record['preferred_key']='123_rep1'
  with self.assertRaises(ValueError):validate_target(self.record,999,self.sources)
 def test_missing_repeat_rejected(self):
  self.record['trajectories'].pop()
  with self.assertRaises(ValueError):validate_target(self.record,999,self.sources)
 def test_conflicting_supplement_rejected(self):
  self.record['needs_additional_rollout']=True
  with self.assertRaises(ValueError):validate_target(self.record,999,self.sources)
if __name__=='__main__':unittest.main()
