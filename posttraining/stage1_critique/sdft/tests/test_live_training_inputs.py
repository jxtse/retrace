"""CPU fixtures for the official trainer's admitted-input loader."""
import hashlib,json,tempfile,unittest
from pathlib import Path
from live_training_inputs import load_admitted_training

class InputTests(unittest.TestCase):
    def fixture(self,d):
        p=Path(d); data=p/'train.jsonl';tools=p/'tools.json'
        rows=[{'prompt':[{'role':'user','content':'synthetic fixture'}],'privileged_context':'fixture','meta':{'formal_training_admissible':True,'actual_wire_verified':True,'checkpoint_sha256':'base'}}]
        data.write_text(json.dumps(rows[0])+'\n');tools.write_text('[]')
        receipt={'status':'admitted','formal_training_admissible':True,'heldout_used':False,'rows':1,'checkpoint_sha256':'base','manifest_sha256':'manifest','data_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),'tools_sha256':hashlib.sha256(tools.read_bytes()).hexdigest()}
        (p/'receipt.json').write_text(json.dumps(receipt));return p,rows
    def test_accept_exact_new_capture(self):
        with tempfile.TemporaryDirectory() as d:
            p,rows=self.fixture(d);actual,_,_=load_admitted_training(p,'manifest','base');self.assertEqual(actual,rows)
    def test_reject_modified_data(self):
        with tempfile.TemporaryDirectory() as d:
            p,_=self.fixture(d);(p/'train.jsonl').write_text('{}\n')
            with self.assertRaises(ValueError):load_admitted_training(p,'manifest','base')
    def test_reject_wrong_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            p,_=self.fixture(d)
            with self.assertRaises(ValueError):load_admitted_training(p,'manifest','other')
    def test_reject_old_probe(self):
        with tempfile.TemporaryDirectory() as d:
            p,_=self.fixture(d);r=json.loads((p/'receipt.json').read_text());r['formal_training_admissible']=False;(p/'receipt.json').write_text(json.dumps(r))
            with self.assertRaises(ValueError):load_admitted_training(p,'manifest','base')

if __name__=='__main__':unittest.main()
