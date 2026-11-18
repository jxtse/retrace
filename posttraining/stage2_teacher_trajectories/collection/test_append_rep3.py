"""Offline authored fixtures only. No provider or chemistry tool calls."""
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import append_rep3 as a


def put(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


class Rep3Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='authored-rep3-test-', dir=os.environ.get('TMPDIR'))
        self.root = Path(self.temp.name)
        self.original = (a.r.ROOT, a.r.M, a.r.CONFIG, a.PLAN)
        a.r.ROOT = self.root
        a.r.M = self.root/'migration'
        a.r.CONFIG = a.r.M/'config.json'
        a.PLAN = a.r.M/'COLLECTION_PLAN_V3.json'
        self.ids = list(range(130))
        self.split = {'train_stage2':self.ids,'dev':list(range(130,150)),
                      'test40_eval_only':list(range(150,190))}
        put(self.root/'split/SPLIT_STAGE2.json', self.split)
        put(a.r.CONFIG, {'fixture':True})
        self.plan = {'target_ids':self.ids,'replicates':['rep1','rep2','rep3'],
            'planned_total':390,'additional_trajectories':130,'workers':6,
            'teacher':'gpt-6-astra','scope':'train_only',
            'split_sha256':a.sha(self.root/'split/SPLIT_STAGE2.json'),
            'config_sha256':a.sha(a.r.CONFIG),'base_runner_sha256':a.sha(Path(a.r.__file__)),
            'predecessor_phase_receipts':['phase_a_EXIT.json','phase_b_EXIT.json'],
            'final_receipt':'COLLECTION_FINISHED_3REPS.json'}
        put(a.PLAN,self.plan)
        put(a.r.M/'CONTROLLER_START.json',{'pid':os.getpid(),'fixture':True})

    def tearDown(self):
        a.r.ROOT,a.r.M,a.r.CONFIG,a.PLAN=self.original
        self.temp.cleanup()

    def prefix(self):
        for rep in ['rep1','rep2']:
            for i in self.ids:
                put(a.r.record_path(rep,i),{'idx':i,'success':i!=7,'iterations':40,'fixture':True})
        put(a.r.M/'COLLECTION_FINISHED.json',{'terminal_records':260,'fixture':True})
        for name in self.plan['predecessor_phase_receipts']:
            put(a.r.M/name,{'exit_code':0,'fixture':True})

    def test_plan_accepts_exact_scope(self):
        self.assertEqual(a.read_plan()[1],self.ids)

    def test_config_drift_rejected(self):
        put(a.r.CONFIG,{'fixture':'changed'})
        with self.assertRaisesRegex(RuntimeError,'config drift'): a.read_plan()

    def test_heldout_overlap_rejected_even_with_new_hash(self):
        self.split['dev']=[0]
        put(self.root/'split/SPLIT_STAGE2.json',self.split)
        self.plan['split_sha256']=a.sha(self.root/'split/SPLIT_STAGE2.json')
        put(a.PLAN,self.plan)
        with self.assertRaisesRegex(RuntimeError,'held-out'): a.read_plan()

    def test_missing_predecessor_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'without completion'):a.check_predecessor(self.plan,self.ids)

    def test_failed_phase_rejected(self):
        self.prefix();put(a.r.M/'phase_a_EXIT.json',{'exit_code':1})
        with self.assertRaisesRegex(RuntimeError,'phase did not'):a.check_predecessor(self.plan,self.ids)

    def test_prefix_preserves_valid_negative_outcomes(self):
        self.prefix();rows=a.check_predecessor(self.plan,self.ids)
        self.assertEqual(len(rows),260)
        self.assertEqual(sum(x['success'] is False for x in rows),2)

    def test_missing_record_rejected(self):
        self.prefix();a.r.record_path('rep2',5).unlink()
        with self.assertRaisesRegex(RuntimeError,'coverage mismatch'):a.check_predecessor(self.plan,self.ids)

    def test_exact390_membership_and_identity(self):
        self.prefix()
        for i in self.ids:put(a.r.record_path('rep3',i),{'idx':i,'success':True,'iterations':40,'fixture':True})
        self.assertEqual(len(a.scan_records(self.ids,['rep1','rep2','rep3'])),390)
        put(a.r.record_path('rep3',5),{'idx':6,'fixture':True})
        with self.assertRaisesRegex(RuntimeError,'identity mismatch'):a.scan_records(self.ids,['rep1','rep2','rep3'])

    def monitor_setup(self):
        import monitor_three_reps as monitor
        self.plan['all_phase_receipts']=self.plan['predecessor_phase_receipts']+['rep3_EXIT.json']
        put(a.PLAN,self.plan)
        put(a.r.M/'ACTIVE_PHASE.json',{'phase':'finished'})
        return monitor

    def test_monitor_does_not_finish_at260(self):
        self.prefix();monitor=self.monitor_setup()
        obs=monitor.collect_status(self.root,owner_check=lambda *args:False)
        self.assertEqual((obs['terminal'],obs['expected']),(260,390))
        self.assertFalse(obs['complete'])

    def test_monitor_waiting_queue_bridges_owner_handoff(self):
        self.prefix();monitor=self.monitor_setup()
        put(a.r.M/'REP3_QUEUE_STATUS.json',{'pid':123456,'entrypoint':'fixture_append_rep3.py','status':'waiting_for_predecessor'})
        obs=monitor.collect_status(self.root,owner_check=lambda pid,entry:pid==123456)
        self.assertTrue(obs['controller_alive'])
        self.assertEqual(obs['queued_replicates'],['rep3'])
        self.assertFalse(obs['complete'])

    def test_monitor390_requires_final_receipt_exit_and_retired_owner(self):
        self.prefix();monitor=self.monitor_setup()
        for i in self.ids:put(a.r.record_path('rep3',i),{'idx':i,'success':True,'fixture':True})
        put(a.r.M/self.plan['final_receipt'],{'terminal_records':390})
        self.assertFalse(monitor.collect_status(self.root,owner_check=lambda *args:False)['complete'])
        put(a.r.M/'rep3_EXIT.json',{'exit_code':0})
        self.assertTrue(monitor.collect_status(self.root,owner_check=lambda *args:False)['complete'])
        self.assertFalse(monitor.collect_status(self.root,owner_check=lambda *args:True)['complete'])

    @unittest.skipUnless(sys.platform=='linux','production queue is Linux-only')
    def test_waiting_owner_cannot_dispatch_or_stop_predecessor(self):
        lock=(a.r.M/'controller.lock').open('a+')
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        before=(a.r.M/'CONTROLLER_START.json').read_bytes()
        script=f'''import sys\nfrom pathlib import Path\nsys.path.insert(0,{str(Path(a.__file__).parent)!r})\nimport append_rep3 as a\na.r.ROOT=Path({str(self.root)!r});a.r.M=a.r.ROOT/'migration';a.r.CONFIG=a.r.M/'config.json';a.PLAN=a.r.M/'COLLECTION_PLAN_V3.json'\ndef forbidden(*args): raise RuntimeError('MODEL_CALL_FORBIDDEN_IN_TEST')\na.r.run_phase=forbidden\ntry:a.main()\nexcept SystemExit as e:\n if a.owns_execution:a.r.cleanup_after_stop(e.code)\n if a.owns_queue:a.update_queue('stopped',exit_code=e.code)\n raise\n'''
        child=subprocess.Popen([sys.executable,'-c',script],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        try:
            deadline=time.monotonic()+5
            q=a.r.M/'REP3_QUEUE_STATUS.json'
            while not q.exists() and time.monotonic()<deadline:time.sleep(0.02)
            self.assertTrue(q.exists())
            self.assertEqual(json.loads(q.read_text())['status'],'waiting_for_predecessor')
            self.assertFalse((self.root/'main_collection/rep3').exists())
            child.send_signal(signal.SIGTERM);out,err=child.communicate(timeout=5)
            self.assertEqual(child.returncode,143,(out,err))
            self.assertEqual(before,(a.r.M/'CONTROLLER_START.json').read_bytes())
            self.assertFalse((a.r.M/'STOPPED.json').exists())
        finally:
            if child.poll() is None:child.kill();child.wait()
            lock.close()


if __name__=='__main__':unittest.main(verbosity=2)
