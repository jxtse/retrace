#!/usr/bin/env python3
"""Paired S3 vs baseline (both test40 @ medium, same harness, same frozen Audit@1). Read-only.
Writes COMPARISON_S3_vs_BASE.json / .md next to this file."""
import json, math, random, collections, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
LAB = {'VALID', 'SUSPECT', 'FAKE'}


def load(cond, manifest):
    camp = json.load(open(HERE / f'campaign-{cond}' / 'campaign.json'))
    man = json.load(open(manifest))
    rec = {r['idx']: r for r in camp['records']}
    out = {}
    for e in man['accepted']:
        r = rec.get(e['idx'])
        v = r.get('route_verdict') if r else 'UNRESOLVED'
        out[e['idx']] = dict(native=e['success'], verdict=v, attempt=e['attempt_id'],
                             defect=(r or {}).get('worst_defect'))
    pending = [k for k, rows in camp['stages'].items() for a in rows if a['status'] in ('intent', 'unknown')]
    return out, pending


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def boot(diff, seed=42, B=10000):
    rng = random.Random(seed)
    n = len(diff)
    xs = sorted(sum(rng.choices(diff, k=n)) / n for _ in range(B))
    return [xs[int(0.025 * B)], xs[int(0.975 * B) - 1]]


def main(base_manifest, s3_manifest):
    s3, p3 = load('s3', s3_manifest)
    bs, pb = load('base', base_manifest)
    ids = sorted(s3)
    assert ids == sorted(bs) and len(ids) == 40
    res = dict(n=40, pending_s3=p3, pending_base=pb, arms={})
    for name, arm in (('base', bs), ('s3', s3)):
        c = collections.Counter(v['verdict'] for v in arm.values())
        res['arms'][name] = dict(native_success=sum(v['native'] for v in arm.values()), labels=dict(c),
                                 strict=c['VALID'], lenient=c['VALID'] + c['SUSPECT'],
                                 unresolved=sum(n for k, n in c.items() if k not in LAB))
    for metric, ok in (('strict', {'VALID'}), ('lenient', {'VALID', 'SUSPECT'})):
        a = [int(bs[i]['verdict'] in ok) for i in ids]
        b = [int(s3[i]['verdict'] in ok) for i in ids]
        s3_only = [i for i, x, y in zip(ids, a, b) if y and not x]
        base_only = [i for i, x, y in zip(ids, a, b) if x and not y]
        res[metric] = dict(base=sum(a), s3=sum(b), delta_pp=100 * (sum(b) - sum(a)) / 40,
                           s3_only=s3_only, base_only=base_only, both=sum(x and y for x, y in zip(a, b)),
                           mcnemar_exact_p=mcnemar(len(s3_only), len(base_only)),
                           paired_bootstrap95_pp=[100 * v for v in boot([y - x for x, y in zip(a, b)])])
    nat_a = [int(bs[i]['native']) for i in ids]; nat_b = [int(s3[i]['native']) for i in ids]
    res['native'] = dict(base=sum(nat_a), s3=sum(nat_b),
                         s3_only=[i for i, x, y in zip(ids, nat_a, nat_b) if y and not x],
                         base_only=[i for i, x, y in zip(ids, nat_a, nat_b) if x and not y])
    res['per_idx'] = {i: dict(base=bs[i], s3=s3[i]) for i in ids}
    json.dump(res, open(HERE / 'COMPARISON_S3_vs_BASE.json', 'w'), indent=1)
    L = ['# S3 vs baseline · test40 @ medium · frozen Audit@1 (rank1)', '',
         '| arm | native | VALID | SUSPECT | FAKE | unresolved | strict | lenient |', '|---|---|---|---|---|---|---|---|']
    for name in ('base', 's3'):
        a = res['arms'][name]; lb = a['labels']
        L.append(f"| {name} | {a['native_success']}/40 | {lb.get('VALID',0)} | {lb.get('SUSPECT',0)} | {lb.get('FAKE',0)} | {a['unresolved']} | {a['strict']}/40 | {a['lenient']}/40 |")
    L.append('')
    for m in ('strict', 'lenient'):
        r = res[m]
        L.append(f"- {m}: Δ={r['delta_pp']:+.1f} pp, S3-only {r['s3_only']}, base-only {r['base_only']}, both {r['both']}, McNemar p={r['mcnemar_exact_p']:.3g}, paired bootstrap95 [{r['paired_bootstrap95_pp'][0]:+.1f}, {r['paired_bootstrap95_pp'][1]:+.1f}] pp")
    L.append(f"- native: base {res['native']['base']}, S3 {res['native']['s3']}, S3-only {res['native']['s3_only']}, base-only {res['native']['base_only']}")
    if p3 or pb:
        L.append(f'- PENDING unknown intents: s3={p3} base={pb}')
    L += ['', '| idx | base native | base verdict | S3 native | S3 verdict |', '|---|---|---|---|---|']
    for i in ids:
        L.append(f"| {i} | {int(bs[i]['native'])} | {bs[i]['verdict']} | {int(s3[i]['native'])} | {s3[i]['verdict']} |")
    (HERE / 'COMPARISON_S3_vs_BASE.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L[:12]))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
