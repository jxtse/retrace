#!/usr/bin/env python3
"""Authorized enum-only scoring recovery. Original artifacts stay immutable.
# --- paths and endpoints come from the environment; defaults are placeholders ---
import os
RETRACE_SHARE = os.environ.get("RETRACE_SHARE", "/path/to/retrace-share")
RETRACE_WORKSPACE = os.environ.get("RETRACE_WORKSPACE", "/path/to/workspace")
RETRACE_ARCHIVE = os.environ.get("RETRACE_ARCHIVE", "/path/to/archive")
RETRACE_HOST = os.environ.get("RETRACE_HOST", "gpu-host")

Default: fully offline preflight. --execute: only missing base99/base110 adversaries.
"""
import argparse
import collections
import copy
import csv
import datetime
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
AUDIT = ROOT.parent
sys.path.insert(0, str(AUDIT))
import adapter as a
import audit_driver as frozen
from validators import validate_deep, validate_adv, response_text
import compare_s3_base as cmp

ARCHIVE = Path(RETRACE_ARCHIVE + '/Archive/stage3-endtoend')
MANIFESTS = {
    'base': ARCHIVE/'Test40-Baseline-Medium-Snapshot-v1/accepted40_base.json',
    's3': ARCHIVE/'Test40-Final-Snapshot-v2/accepted40_v2.json',
}
POLICY = {'version': 'enum-normalization-v1', 'mapping': {'CHEMOSELECTIVITY': 'OTHER'},
          'scope': 'all saved stage1 responses in both arms; no verdict or reason changes',
          'authorization': 'owner-approved enum normalization plus two missing adversarial defenses',
          'new_stage_allowlist': ['base:99:1', 'base:110:1'], 'workers': 1,
          'original_prompt_and_aggregation': 'unchanged'}
ALLOWED = set(POLICY['new_stage_allowlist'])
ERRORS = {'s3': {34,124}, 'base': {37,99,110,171}}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def dump(path, obj):
    a.atomic(path, a.encode(obj))


def original_hashes():
    return {str(p.relative_to(AUDIT)): sha(p.read_bytes())
            for p in AUDIT.rglob('*') if p.is_file() and ROOT not in p.parents
            and '__pycache__' not in p.parts and p.suffix != '.lock'}


def cached(camp, arm, route, stage, prompt):
    key = route['uid'].replace(':','_') + '.' + stage
    attempts = camp['stages'].get(key, [])
    expected_wire = a.encode(dict(model=frozen.MODEL, messages=[{'role':'user','content':prompt}],
                                 temperature=frozen.TEMP, max_tokens=frozen.MAXTOK))
    for row in attempts:
        wire = (AUDIT/f'campaign-{arm}'/row['wire']).read_bytes()
        assert sha(wire) == row['wire_sha256'] and wire == expected_wire, key
        if row['status'] != 'http200':
            if row['status'] in ('intent','unknown'):
                raise RuntimeError('Unknown original request: ' + key)
            continue
        raw = (AUDIT/f'campaign-{arm}'/row['response']).read_bytes()
        assert sha(raw) == row['response_sha256'], key
        body = a.decode(raw)
        text = response_text(body)
        assert not body['choices'][0]['message'].get('tool_calls')
        assert not body['choices'][0]['message'].get('function_call')
        parsed = frozen.parse_json_reply(text)
        return parsed, row
    return None, None


def normalize(obj, uid, edits):
    out = copy.deepcopy(obj)
    for step in out['steps']:
        code = step['defect_code']
        if code in POLICY['mapping']:
            edits.append(dict(uid=uid, field=f"steps[{step['step_index']}].defect_code", old=code, new=POLICY['mapping'][code]))
            step['defect_code'] = POLICY['mapping'][code]
    code = out['worst_defect_code']
    if code in POLICY['mapping']:
        edits.append(dict(uid=uid, field='worst_defect_code', old=code, new=POLICY['mapping'][code]))
        out['worst_defect_code'] = POLICY['mapping'][code]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--execute', action='store_true')
    args = ap.parse_args()
    a.verify_frozen()
    before = original_hashes()
    if (ROOT/'original-hashes.json').exists():
        assert json.loads((ROOT/'original-hashes.json').read_text()) == before, 'Original evidence changed since preflight'
    else:
        dump(ROOT/'original-hashes.json', before)
    dump(ROOT/'POLICY.json', POLICY)
    arms, contexts, edits, origins = {}, {}, [], []
    for arm,path in MANIFESTS.items():
        slots, routes, identity = a.load_manifest(path)
        camp = a.decode((AUDIT/f'campaign-{arm}'/'campaign.json').read_bytes())
        assert identity == camp['identity']
        assert {r['idx'] for r in camp['records'] if r['route_verdict']=='ERROR'} == ERRORS[arm]
        assert not [(k,t) for k,rows in camp['stages'].items() for t in rows if t['status'] in ('intent','unknown')]
        contexts[arm] = dict(slots=slots, routes=routes, identity=identity, campaign=camp)
        origins.append(dict(arm=arm, manifest=str(path), identity=identity))
    identity = dict(recovery_policy=POLICY, sources=origins, script_sha256=sha(Path(__file__).read_bytes()))
    new = a.Campaign(ROOT/'new-adversary', identity, execute=args.execute)
    missing = []
    reused = 0
    def run_arm(arm, ctx):
        nonlocal reused
        original = {r['uid']:r for r in ctx['campaign']['records']}
        final = []
        for route in ctx['routes']:
            uid = route['uid']
            local_edits = []
            def caller(prompt, retries=3):
                stage = 'deepread' if prompt.startswith('You are a senior') else 'adversary'
                obj, receipt = cached(ctx['campaign'], arm, route, stage, prompt)
                if obj is not None:
                    if stage == 'deepread':
                        obj = normalize(obj, uid, local_edits)
                        validate_deep(obj, [s['i'] for s in route['steps']])
                    else:
                        validate_adv(obj)
                    return json.dumps(obj, ensure_ascii=False)
                assert stage == 'adversary' and uid in ALLOWED, ('unauthorized missing stage', uid, stage)
                return new.caller(route)(prompt)
            old = frozen.call_llm
            frozen.call_llm = caller
            try:
                try:
                    rec = frozen.audit_one(route)
                except a.Pending as exc:
                    assert uid in ALLOWED
                    missing.append(dict(uid=uid, stage='adversary', reason=str(exc)))
                    rec = dict(uid=uid, cond=arm, idx=route['idx'], rank=1, route_verdict='UNRESOLVED')
            finally:
                frozen.call_llm = old
            if original[uid]['route_verdict'] != 'ERROR':
                assert not local_edits and rec == original[uid], ('Previously accepted verdict changed',uid)
                reused += 1
            edits.extend(local_edits)
            final.append(rec)
            if uid in ALLOWED:
                print(json.dumps(dict(uid=uid, final=rec['route_verdict'], defense=rec.get('adv_upheld'), reason=rec.get('adv_reason')),ensure_ascii=False),flush=True)
        return final
    with new.lock():
        if args.execute:
            new.start()
        for arm,ctx in contexts.items():
            arms[arm] = run_arm(arm,ctx)
    assert reused == 74
    assert {e['uid'] for e in edits} == {'s3:34:1','s3:124:1','base:37:1','base:99:1','base:110:1','base:171:1'}
    assert original_hashes() == before, 'Original evidence modified'
    dump(ROOT/'normalization-edits.json', edits)
    for arm,rows in arms.items():
        dump(ROOT/f'recovered-{arm}.json', rows)
    receipt = dict(original_artifacts_unchanged=True, original_files_verified=len(before),
                   source_manifest_and_snapshot_hashes_verified=True, frozen_prompts_and_stage_aggregation_replayed=True,
                   exact_expected_step_coverage_validated_from_original_routes=True,
                   unchanged_accepted_verdicts=reused, normalization_edits=len(edits), missing=missing,
                   new_request_stages=list((new.state or {}).get('stages',{})),
                   completed_at=datetime.datetime.now().astimezone().isoformat())
    dump(ROOT/'verification.json', receipt)
    if missing:
        print(json.dumps(dict(preflight='PASS', missing=missing, unchanged_accepted_verdicts=reused, normalization_edits=len(edits)),ensure_ascii=False),flush=True)
        return
    assert set((new.state or {}).get('stages',{})) == {'base_99_1.adversary','base_110_1.adversary'}
    result = dict(protocol='frozen Audit@1 + explicit enum-normalization-v1 recovery', n=40,
                  policy=POLICY, verification=receipt, arms={}, per_idx={}, unresolved=0)
    joined = {}
    for arm,rows in arms.items():
        by = {r['idx']:r for r in rows}
        assert set(by)==set(a.IDS) and len(rows)==40
        assert all(r['route_verdict'] in {'VALID','SUSPECT','FAKE'} for r in rows)
        joined[arm]=by
        c=collections.Counter(r['route_verdict'] for r in rows)
        native = sum(s['native_success'] for s in contexts[arm]['slots'])
        result['arms'][arm]=dict(native_success=native,labels=dict(c),strict=c['VALID'],lenient=c['VALID']+c['SUSPECT'],unresolved=0)
    for metric,ok in [('strict',{'VALID'}),('lenient',{'VALID','SUSPECT'})]:
        av=[int(joined['base'][i]['route_verdict'] in ok) for i in a.IDS]
        bv=[int(joined['s3'][i]['route_verdict'] in ok) for i in a.IDS]
        s3only=[i for i,x,y in zip(a.IDS,av,bv) if y and not x]
        baseonly=[i for i,x,y in zip(a.IDS,av,bv) if x and not y]
        result[metric]=dict(base=sum(av),s3=sum(bv),delta_pp=100*(sum(bv)-sum(av))/40,
                            s3_only=s3only,base_only=baseonly,both=sum(x and y for x,y in zip(av,bv)),
                            mcnemar_exact_p=cmp.mcnemar(len(s3only),len(baseonly)),
                            paired_bootstrap95_pp=[100*x for x in cmp.boot([y-x for x,y in zip(av,bv)])])
    for i in a.IDS:
        result['per_idx'][i]={arm:dict(verdict=joined[arm][i]['route_verdict'],
                                      native=next(s['native_success'] for s in contexts[arm]['slots'] if s['idx']==i)) for arm in ('base','s3')}
    dump(ROOT/'FINAL_COMPARISON.json',result)
    out=io.StringIO(); writer=csv.writer(out); writer.writerow(['idx','base_native','base_verdict','s3_native','s3_verdict'])
    for i,row in result['per_idx'].items():
        writer.writerow([i,int(row['base']['native']),row['base']['verdict'],int(row['s3']['native']),row['s3']['verdict']])
    a.atomic(ROOT/'FINAL_PER_TARGET.csv',out.getvalue().encode())
    lines=['# ReTRACE S3 vs 基座：test40@medium 最终修复版结果','',
           '协议：冻结 Audit@1，叠加明确版本的 enum-normalization-v1。不是未修改的原始冻结评分。',
           '所有 80 个条件×目标均已终判，未决 0。未重跑路线生成或 Stage 1 深读，只补基座 99、110 的 Stage 2 对抗辩护。','',
           '| 指标 | 基座 | S3 |','|---|---:|---:|']
    for key,label in [('native_success','Harness 完成路线'),('strict','严格：VALID'),('lenient','宽松：VALID + SUSPECT')]:
        x=result['arms']['base'][key]; y=result['arms']['s3'][key]
        lines.append(f'| {label} | {x}/40 ({100*x/40:.1f}%) | {y}/40 ({100*y/40:.1f}%) |')
    for label in ['VALID','SUSPECT','FAKE']:
        lines.append(f"| {label} 条数 | {result['arms']['base']['labels'].get(label,0)} | {result['arms']['s3']['labels'].get(label,0)} |")
    lines.extend(['','## 配对比较'])
    for metric in ['strict','lenient']:
        r=result[metric]; ci=r['paired_bootstrap95_pp']
        lines.append(f"- {metric}: S3 − 基座 {r['delta_pp']:+.1f} 个百分点，精确 McNemar p={r['mcnemar_exact_p']:.6g}，目标配对 bootstrap 95% CI [{ci[0]:+.1f}, {ci[1]:+.1f}] 个百分点。")
    lines.extend(['','## 6 条 ERROR 的恢复','', '| 条件 | 样本 | 原始阶段1 | 修复后终判 | 对抗辩护 |','|---|---:|---|---|---|'])
    for arm in ['s3','base']:
        for i in sorted(ERRORS[arm]):
            row=joined[arm][i]
            label='FAKE' if row.get('n_fake_steps',0) else ('SUSPECT' if row.get('n_suspect_steps',0) else 'VALID')
            defense='维持 FAKE' if row.get('adv_upheld') else ('推翻 FAKE' if row.get('adv_triggered') else '不触发')
            lines.append(f"| {arm} | {i} | {label} | {row['route_verdict']} | {defense} |")
    for i in [99,110]:
        row=joined['base'][i]
        lines.extend(['',f"### 基座 {i} 对抗辩护原文",row.get('adv_reason','')])
    lines.extend(['','## 验证与局限',
        '- 两组使用同一归一化规则：只将 CHEMOSELECTIVITY 代码转为已有 OTHER，不改原始 verdict、步骤理由或路线。逐条验证原始路线的步骤覆盖。',
        '- 74 条原先有效终判与原始记录逐字段一致；原始文件哈希前后相同。原始深读与既有辩护响应均复用，不重新采样。',
        '- 新增请求严格限于 base:99:1、base:110:1 对抗辩护，单 worker，同一冻结 prompt、模型、temperature 和 token budget。',
        '- S3 路线集包含基础设施补跑且原有受污染成功被保留，基座单轮。40/40 不是同条件单次成功率，也不能将差异单独归因于最后训练阶段。',
        '- 化学审核是模型判断，不是化学家真值；SUSPECT 不等于可靠可执行。',
        '- 比例分母均为冻结的 40 个目标；置信区间是按目标重采样，不覆盖审核模型随机性。',
        '', '机器可读文件：FINAL_COMPARISON.json、FINAL_PER_TARGET.csv、recovered-{base,s3}.json、normalization-edits.json、verification.json。'])
    a.atomic(ROOT/'FINAL_REPORT.md',('\n'.join(lines)+'\n').encode())
    print(json.dumps({'complete':True,'arms':result['arms'],'strict':result['strict'],'lenient':result['lenient'],'verification':receipt},ensure_ascii=False),flush=True)

if __name__=='__main__':
    main()
