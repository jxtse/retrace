"""Freeze stage-2 Astra selected routes for the unchanged frozen Audit v1 judge.
Train-side data selection only. One judge call per unique (target, route) pair;
identical routes across replicates share one verdict."""
import json,hashlib,importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
# Digests of the audit modules as shipped in this directory; the freeze check fails if any change.
EXP={'audit_driver.py':'279f311d132402641e51aa80f62dd4107920f3ae50c09ed33581b6d8e7788f02','AUDIT_SPEC.md':'5fcc6c6712188bab612b25d166fdde12077fb0f15480bd4b650ef5247549c075'}
for n,h in EXP.items(): assert hashlib.sha256((HERE/n).read_bytes()).hexdigest()==h,n
spec=importlib.util.spec_from_file_location('g',HERE/'grounding_reference.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
src=HERE.parent/'best_routes_v1.json';B=json.loads(src.read_text())
routes={};members={}
for b in B:
    key=f"astra:{b['idx']}:{b['route_key']}"
    members.setdefault(key,[]).append(b['uid'])
    if key in routes: continue
    steps=[g.ground_step(i,s) for i,s in enumerate(b['steps'])]
    t=g.canon(b['target_smiles']);assert t
    flags={'any_unparseable':any(not s['parse_ok'] for s in steps),'target_produced':t in [g.canon(s['product']) if s['product'] else None for s in steps]}
    routes[key]={'uid':key,'cond':'astra','idx':b['idx'],'rank':1,'target_smiles':b['target_smiles'],'total_score':b['total_score'],'starting_materials':b['starting_materials'],'steps':steps,'flags':flags,'route_label':b['label']}
R=sorted(routes.values(),key=lambda r:(r['idx'],r['uid']))
(HERE/'audit_routes.json').write_text(json.dumps(R,indent=1,ensure_ascii=False))
(HERE/'route_members.json').write_text(json.dumps(members,indent=1))
(HERE/'PREPARATION.json').write_text(json.dumps({'protocol':'frozen audit@k v1.0 driver/prompts unchanged; used here for TRAIN-side data selection','frozen_hashes':EXP,'source':str(src),'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'n_trajectories_with_route':len(B),'n_unique_routes':len(R),'stage0_fail':sum(r['flags']['any_unparseable'] or not r['flags']['target_produced'] for r in R),'workers':1},indent=1))
print(len(B),len(R),sum(r['flags']['any_unparseable'] or not r['flags']['target_produced'] for r in R))
