"""Seal completed acceptance, never release training."""
import hashlib,json
from pathlib import Path
O=Path(__file__).resolve().parent
r=json.loads((O/'NATIVE_READER_REPORT.json').read_text())
assert r['cases']==408 and r['unique_real_rows']==136
assert r['loss_tokens_by_control']=={'eligible_control':2008539,'pending_zero':0,'alternating_runs_control':1006944}
assert r['exact_full_tokens_and_masks'] and not r['model_constructed'] and not r['meta_model_constructed']
assert r['paired_padding_batches']==405 and r['v1_payload_hashes_verified_before_and_after']==2027
cases=list(map(json.loads,(O/'NATIVE_READER_CASES.jsonl').read_text().splitlines()))
assert len(cases)==408 and len({c['uid'] for c in cases})==136
assert all(c['exact_full_tokens'] and c['exact_labels'] and c['reasoning_loss_zero'] for c in cases)
log=(O/'TDD_GREEN_FINAL.log').read_text()
assert 'Ran 4 tests' in log and log.rstrip().endswith('OK')
assert (O/'train_selected.jsonl').read_bytes()==b''
gate=dict(engineering_native_reader='PASS',scientific_release='BLOCKED',training_ready=False,
          admitted_rows=0,admitted_selected_tokens=0,reason='No scientific route VALID or selector receipts',
          controls_only=True,original_bytes_preserved=True,final_test_suite='test_native (4 tests)',
          final_test_log='TDD_GREEN_FINAL.log',full_report='NATIVE_READER_REPORT.json')
(O/'RELEASE_GATE.json').write_text(json.dumps(gate,indent=2,sort_keys=True)+'\n')
files=[p for p in O.rglob('*') if p.is_file() and not any(x in p.parts for x in ['.datasets_cache','__pycache__']) and p.name not in ['ARTIFACT_HASHES.json','FINALIZATION.log']]
manifest={str(p.relative_to(O)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
(O/'ARTIFACT_HASHES.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
print(json.dumps({'sealed_files':len(manifest),'native_cases':408,'unique_real_rows':136,'admitted_rows':0,'training_ready':False},sort_keys=True))
