"""Single-worker wrapper around the frozen Audit v1 driver for stage-2 TRAIN-side route selection.
Frozen prompts/driver/verdict rules unchanged. <=3 transport attempts, no schema rerolls, no model substitution.
Errors stay ERROR (unresolved), never counted as chemistry FAKE."""
import argparse,collections,datetime,fcntl,hashlib,json,os,time,urllib.error,urllib.request
from pathlib import Path
import audit_driver as frozen
HERE=Path(__file__).resolve().parent
LABELS={'VALID','SUSPECT','FAKE'};CANON='dev-anthropic-claude-opus-5'
def require(ok,msg):
    if not ok: raise ValueError(msg)
def validate_deep(a,indices):
    require(isinstance(a,dict),'deepread object missing');steps=a.get('steps')
    require(isinstance(steps,list) and len(steps)==len(indices),'step coverage mismatch')
    require(sorted(s.get('step_index') for s in steps)==sorted(indices),'step ids mismatch')
    for s in steps:
        require(s.get('verdict') in LABELS,'step verdict invalid');require(s.get('defect_code') in frozen.DEFECTS,'step defect invalid')
    nf=sum(s['verdict']=='FAKE' for s in steps);ns=sum(s['verdict']=='SUSPECT' for s in steps)
    require(a.get('route_verdict')==('FAKE' if nf else 'SUSPECT' if ns else 'VALID'),'route/step aggregation mismatch')
    require(a.get('worst_defect_code') in frozen.DEFECTS,'worst defect invalid')
def validate_adv(a):
    require(isinstance(a,dict) and type(a.get('fake_upheld')) is bool,'adversary flag');require(a.get('corrected_verdict') in LABELS,'corrected verdict')
    require(a['fake_upheld']==(a['corrected_verdict']=='FAKE'),'adversary inconsistent')
SCHEMA_ERRORS={'deepread object missing','step coverage mismatch','step ids mismatch','step verdict invalid','step defect invalid',
  'route/step aggregation mismatch','worst defect invalid','adversary flag','corrected verdict','adversary inconsistent'}
def is_schema_error(r): return r.get('route_verdict')=='ERROR' and r.get('error') in SCHEMA_ERRORS
def append(name,obj):
    with (HERE/name).open('a') as f: f.write(json.dumps(obj,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
def atomic(name,obj):
    t=HERE/(name+'.tmp');t.write_text(json.dumps(obj,ensure_ascii=False,indent=2));t.replace(HERE/name)
def make_caller(route):
    def call(prompt,retries=3):
        stage='deepread' if prompt.startswith('You are a senior') else 'adversary'
        payload={'model':frozen.MODEL,'messages':[{'role':'user','content':prompt}],'temperature':frozen.TEMP,'max_tokens':frozen.MAXTOK}
        pre=HERE/'raw'/(route['uid'].replace(':','_')+'.'+stage)
        req=Path(str(pre)+'.request.json');cache=Path(str(pre)+'.response.json')
        if req.exists(): require(json.loads(req.read_text())==payload,'request drift')
        else: req.write_text(json.dumps(payload,ensure_ascii=False))
        if cache.exists(): body=json.loads(cache.read_text())
        else:
            op=urllib.request.build_opener(urllib.request.ProxyHandler({}));body=None
            for att in range(1,4):
                t=time.monotonic()
                try:
                    r=urllib.request.Request(frozen.API,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
                    with op.open(r,timeout=600) as resp: raw=resp.read().decode();code=resp.status
                    Path(str(pre)+f'.attempt{att}.response.txt').write_text(raw)
                    append('requests.jsonl',{'uid':route['uid'],'stage':stage,'attempt':att,'http_status':code,'elapsed_s':round(time.monotonic()-t,2)})
                    cache.write_text(raw);body=json.loads(raw);break
                except urllib.error.HTTPError as e:
                    Path(str(pre)+f'.attempt{att}.error.txt').write_text(e.read().decode(errors='replace'))
                    append('requests.jsonl',{'uid':route['uid'],'stage':stage,'attempt':att,'http_status':e.code,'elapsed_s':round(time.monotonic()-t,2)})
                    if e.code in (400,401,403,404) or att==3: raise
                    time.sleep(30*att)
                except (TimeoutError,urllib.error.URLError,ConnectionError) as e:
                    append('requests.jsonl',{'uid':route['uid'],'stage':stage,'attempt':att,'error':str(e)[:200]})
                    if att==3: raise
                    time.sleep(5*att)
            require(body is not None,'provider exhausted')
        require(body.get('model')==CANON,'response model mismatch: %s'%body.get('model'))
        ch=body.get('choices') or [];require(len(ch)==1,'choices');require(ch[0].get('finish_reason')=='stop','finish_reason=%s'%ch[0].get('finish_reason'))
        text=ch[0]['message'].get('content');require(isinstance(text,str) and text.strip(),'empty')
        d=frozen.parse_json_reply(text)
        validate_deep(d,[s['i'] for s in route['steps']]) if stage=='deepread' else validate_adv(d)
        return text
    return call
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int,default=0);a=ap.parse_args()
    lock=(HERE/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prep=json.loads((HERE/'PREPARATION.json').read_text())
    for n,h in prep['frozen_hashes'].items(): require(hashlib.sha256((HERE/n).read_bytes()).hexdigest()==h,'frozen drift')
    routes=json.loads((HERE/'audit_routes.json').read_text());(HERE/'raw').mkdir(exist_ok=True)
    man={'pid':os.getpid(),'workers':1,'model':frozen.MODEL,'api':frozen.API,'temperature':frozen.TEMP,'max_tokens':frozen.MAXTOK,
         'routes_sha256':hashlib.sha256((HERE/'audit_routes.json').read_bytes()).hexdigest(),'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'purpose':'train-side selection only'}
    if not (HERE/'RUN_MANIFEST.json').exists(): atomic('RUN_MANIFEST.json',man)
    old=json.loads((HERE/'RUN_MANIFEST.json').read_text());require(old['routes_sha256']==man['routes_sha256'],'routes drift')
    recs=[json.loads(l) for l in (HERE/'verdicts.jsonl').read_text().splitlines()] if (HERE/'verdicts.jsonl').exists() else []
    # Amendment 1: schema-invalid judge output is a terminal ERROR (cached response would reproduce it; no rerolls).
    # Only transport/provider failures are retried on restart and trip the circuit breaker.
    done={r['uid'] for r in recs if r['route_verdict']!='ERROR' or is_schema_error(r)}
    t0=time.monotonic();errs=0;streak=0;added=0;schema_errs=0
    for route in routes:
        if route['uid'] in done: continue
        if streak>=3 or errs>=8: print('CIRCUIT_OPEN transport',flush=True);break
        if added>=40 and schema_errs/added>0.25: print('CIRCUIT_OPEN schema_rate',flush=True);break
        frozen.call_llm=make_caller(route)
        try: rec=frozen.audit_one(route);streak=0
        except Exception as e:
            rec={'uid':route['uid'],'idx':route['idx'],'route_verdict':'ERROR','error':str(e)[:400]}
            if is_schema_error(rec): schema_errs+=1;streak=0
            else: errs+=1;streak+=1
        append('verdicts.jsonl',rec);added+=1
        el=time.monotonic()-t0;n=len(done)+added
        atomic('status.json',{'pid':os.getpid(),'updated_at':datetime.datetime.now().astimezone().isoformat(),'done':n,'total':len(routes),'errors_this_run':errs,'schema_errors_this_run':schema_errs,'last':rec['uid'],'last_verdict':rec['route_verdict'],'eta_s':el/added*(len(routes)-n)})
        print(f"[{n}/{len(routes)}] {rec['uid']} {rec['route_verdict']} errs={errs} {el/60:.1f}m",flush=True)
        if a.limit and added>=a.limit: break
    recs=[json.loads(l) for l in (HERE/'verdicts.jsonl').read_text().splitlines()]
    last={}
    for r in recs: last[r['uid']]=r
    c=collections.Counter(r['route_verdict'] for r in last.values())
    atomic('exit_receipt.json',{'at':datetime.datetime.now().astimezone().isoformat(),'n_routes':len(routes),'n_judged':len(last),'labels':dict(c),'complete':len(last)==len(routes) and not c.get('ERROR'),'limited':bool(a.limit)})
    print(json.dumps(dict(c)),flush=True)
if __name__=='__main__': raise SystemExit(main())
