"""Astra segment selector (Copilot :8080 Responses). Raw request/response saved; JSON validated."""
import json,sys,time,hashlib,urllib.request,urllib.error,re,argparse,concurrent.futures as cf
from pathlib import Path
HERE=Path(__file__).resolve().parent;D=HERE.parent;A=D/'route_audit_v1'
API=os.environ.get('ASTRA_BASE_URL', 'http://localhost:8080') + '/v1/responses';MODEL='gpt-6-astra'
def load():
    B={b['uid']:b for b in json.loads((D/'best_routes_v1.json').read_text())}
    mem=json.loads((A/'route_members.json').read_text())
    V={}
    for l in (A/'verdicts.jsonl').read_text().splitlines():
        v=json.loads(l);V[v['uid']]=v
    u2v={u:V.get(k) for k,us in mem.items() for u in us}
    PF={p['uid']:p for p in json.loads((D/'process_features_v1.json').read_text())}
    return B,u2v,PF
def build(uid,ver,B,u2v,PF):
    b=B[uid];v=u2v[uid];rep,idx=uid.split(':')
    traj=(D/'compact'/f"{rep}_{int(idx):03d}.txt").read_text()
    n=len(re.findall(r'^### A\d+$',traj,re.M))
    route='\n'.join(f"Step {i+1}: {s}" for i,s in enumerate(b['steps']))+f"\nLabel: {b['label']}  total_score: {b['total_score']}\nStarting materials: {b['starting_materials']}"
    judge='\n'.join(f"step {s['step_index']+1}: {s['verdict']} — {s['reason']}" for s in v['steps'])+'\nSummary: '+v.get('summary','')
    pf=PF[uid];flags=json.dumps({'final_route_had_scorer_warning_in_trajectory':pf['selected_route_flagged'],'scorer_warning_counts':pf['flag_kinds'],'n_score_route':pf['n_score'],'n_save_best_route':pf['n_save']})
    tpl=(HERE/f'prompt_{ver}.txt').read_text().replace('{sysprompt}',(D/'system_prompt_template.txt').read_text().replace('<TARGET>',b['target_smiles']))
    return tpl.replace('{target}',b['target_smiles']).replace('{route}',route).replace('{judge}',judge).replace('{flags}',flags).replace('{traj}',traj).replace('{n}',str(n)),n
def call(prompt,tag):
    body={'model':MODEL,'input':[{'role':'user','content':prompt}],'reasoning':{'effort':'high'},'max_output_tokens':32000}
    op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for att in range(1,5):
        try:
            r=urllib.request.Request(API,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
            t=time.time()
            with op.open(r,timeout=900) as resp: raw=resp.read().decode()
            (tag.parent/(tag.name+'.response.json')).write_text(raw);o=json.loads(raw)
            assert o.get('model','').startswith('gpt-6-astra'),o.get('model')
            assert o.get('status')=='completed',o.get('status')
            txt=''.join(c.get('text','') for it in o['output'] if it.get('type')=='message' for c in it.get('content',[]))
            return txt,round(time.time()-t,1),o.get('usage')
        except (urllib.error.URLError,ConnectionError,TimeoutError,AssertionError) as e:
            if att==4: raise
            time.sleep(10*att)
def parse(txt,n,ver='v3'):
    m=re.search(r'\{.*\}',txt,re.S);o=json.loads(m.group(0))
    seg=o['segments'];exp=1
    for s in seg:
        assert s['start']==exp and s['end']>=s['start'],('coverage',s,exp)
        for k in ('contribution','learnability','risk','overall'): assert isinstance(s[k],int) and 0<=s[k]<=10
        if ver>='v4': assert 'risky_turns' in s,'risky_turns missing'
        if 'risky_turns' in s:
            rt=s['risky_turns'];assert isinstance(rt,list)
            for t in rt:
                assert isinstance(t.get('turn'),int) and s['start']<=t['turn']<=s['end'],('risky turn out of segment',t,s['start'],s['end'])
                assert isinstance(t.get('risk'),int) and 3<=t['risk']<=10,('risky turn risk',t)
            assert len({t['turn'] for t in rt})==len(rt),'dup risky turn'
            assert s['risk']==(max(t['risk'] for t in rt) if rt else s['risk']) and (rt or s['risk']<=2),('segment risk/risky_turns mismatch',s['risk'],rt)
        exp=s['end']+1
    assert exp==n+1,('end',exp,n)
    return o
def run(uid,ver,out,B,u2v,PF):
    f=out/(uid.replace(':','_')+'.json')
    if f.exists(): return uid,'cached'
    prompt,n=build(uid,ver,B,u2v,PF);(out/'raw').mkdir(exist_ok=True)
    tag=out/'raw'/uid.replace(':','_');(tag.parent/(tag.name+'.prompt.txt')).write_text(prompt)
    last=None
    for att in range(2):
        txt,dt,usage=call(prompt,tag)
        try: o=parse(txt,n,ver);break
        except Exception as e: last=e;(tag.parent/(tag.name+f'.badparse{att}.txt')).write_text(txt)
    else: return uid,f'PARSE_FAIL {last}'
    o.update(uid=uid,prompt_version=ver,n_turns=n,elapsed_s=dt,usage=usage,prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
    f.write_text(json.dumps(o,indent=1,ensure_ascii=False));return uid,f'ok {dt}s'
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--ver');ap.add_argument('--out');ap.add_argument('--uids');ap.add_argument('--workers',type=int,default=4);a=ap.parse_args()
    B,u2v,PF=load();out=HERE/a.out;out.mkdir(exist_ok=True)
    uids=a.uids.split(',')
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for r in ex.map(lambda u:run(u,a.ver,out,B,u2v,PF),uids): print(*r,flush=True)
