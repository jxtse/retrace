"""Run frozen v4 selector (2 samples) on every VALID trajectory as audit verdicts arrive."""
import json,os,time,sys,concurrent.futures as cf
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import seg_select as ss
A=HERE.parent/'route_audit_v1'
def audit_done():
    st=json.loads((A/'status.json').read_text())
    try: os.kill(st['pid'],0);alive=True
    except OSError: alive=False
    return (not alive) and st['done']>=st['total'],st
def valid_uids():
    mem=json.loads((A/'route_members.json').read_text());out=[]
    for l in (A/'verdicts.jsonl').read_text().splitlines():
        v=json.loads(l)
        if v['route_verdict']=='VALID': out+=mem[v['uid']]
    return out
while True:
    done,st=audit_done()
    B,u2v,PF=ss.load();todo=[]
    for u in valid_uids():
        for s in ('final_v4_s1','final_v4_s2'):
            if not (HERE/s/(u.replace(':','_')+'.json')).exists(): todo.append((u,s))
    if todo:
        for s in {s for _,s in todo}: (HERE/s).mkdir(exist_ok=True)
        with cf.ThreadPoolExecutor(4) as ex:
            for r in ex.map(lambda t:ss.run(t[0],'v4',HERE/t[1],B,u2v,PF)+(t[1],),todo): print(time.strftime('%H:%M:%S'),*r,flush=True)
        continue
    if done:
        (HERE/'SELECTION_DONE.json').write_text(json.dumps({'at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'audit_status':st,'n_valid_traj':len(valid_uids())},indent=1));break
    time.sleep(300)
