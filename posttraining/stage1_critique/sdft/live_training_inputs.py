"""Strict input loading only; trust is established by the separately pinned live verifier."""
import hashlib,json
from pathlib import Path

def load_admitted_training(directory, manifest_sha256, checkpoint_sha256):
    p=Path(directory);r=json.loads((p/'receipt.json').read_text())
    def check(ok,why):
        if not ok:raise ValueError(why)
    check(r.get('status')=='admitted' and r.get('formal_training_admissible') is True,'Not admitted')
    check(r.get('heldout_used') is False,'Heldout forbidden')
    check(r.get('manifest_sha256')==manifest_sha256,'Launch identity mismatch')
    check(r.get('checkpoint_sha256')==checkpoint_sha256,'Checkpoint mismatch')
    for file,key in [('train.jsonl','data_sha256'),('tools.json','tools_sha256')]:
        check(hashlib.sha256((p/file).read_bytes()).hexdigest()==r.get(key),'Dataset changed')
    rows=[json.loads(l) for l in (p/'train.jsonl').read_text().splitlines()]
    check(len(rows)==r['rows'] and len(rows)>0,'Empty or inconsistent dataset')
    for row in rows:
        meta=row['meta']
        check(meta.get('formal_training_admissible') is True and meta.get('actual_wire_verified') is True,'Old recorder/probe row')
        check(meta.get('checkpoint_sha256')==checkpoint_sha256,'Row checkpoint mismatch')
    return rows,json.loads((p/'tools.json').read_text()),r
