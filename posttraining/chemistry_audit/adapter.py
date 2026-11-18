#!/usr/bin/env python3
"""Source-bound, single-worker S3 Audit@1. Default operation is offline."""
import argparse
import collections
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import sys
import time
import urllib.error
import urllib.request

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
COND = 's3'
ADAPTER_SHA = 'e69aa10429dfd7aaa83b3a51201832e7380cd460f720eda52b9a169e62f9b146'
IDS = [10,13,24,33,34,37,52,53,55,56,58,60,66,80,87,89,90,94,99,102,
       104,105,108,110,113,115,121,124,125,132,136,147,154,156,158,171,179,181,182,189]
# Integrity pins: SHA-256 of the frozen audit sources as shipped in this directory.
# adapter.py refuses to run when any of them fails to match; SOURCE_PINS.json
# records where each file was extracted from.
FROZEN = {'audit_driver.py': '279f311d132402641e51aa80f62dd4107920f3ae50c09ed33581b6d8e7788f02',
          'AUDIT_SPEC.md': '5fcc6c6712188bab612b25d166fdde12077fb0f15480bd4b650ef5247549c075',
          'SPLIT_STAGE2.json': 'ea76470d8ae418e2d4b2a8cb9f0a203b5842362ba4a6d439714845fbd49273b6',
          'validators.py': '2a28aa94ee13b3f86c8e6214ade15c660d91de5c96462f535557e908b378d8e7',
          'grounding.py': '5c4c021acfa633a11ea1daf4f5f8b3edf66036099f1414a94b0eaa9da972d267'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encode(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key: ' + key)
        result[key] = value
    return result


def decode(data):
    return json.loads(data, object_pairs_hook=unique_object)


def verify_frozen(root=HERE):
    for name, expected in FROZEN.items():
        require(digest((root/name).read_bytes()) == expected, 'frozen source drift: ' + name)


verify_frozen()
import audit_driver as frozen
import grounding as g
from validators import validate_deep, validate_adv, response_text, LABELS


def source(base, ref):
    """Read one immutable local snapshot, validating the bytes actually parsed."""
    require(isinstance(ref, dict) and set(ref) == {'path', 'sha256', 'origin'}, 'source reference fields')
    require(isinstance(ref['origin'], str) and ref['origin'].startswith('/'), 'absolute source origin required')
    path = Path(ref['path'])
    require('://' not in str(path), 'local snapshots only')
    data = (base/path).read_bytes()
    require(digest(data) == ref['sha256'], 'source SHA mismatch: ' + str(path))
    return decode(data)


def rank1(case, transcript):
    require(type(case.get('success')) is bool, 'native success must be boolean')
    target = g.canon(case.get('smiles', ''))
    require(target and target == g.canon(transcript.get('target_smiles', '')), 'target mismatch')
    a, b = case.get('all_routes'), transcript.get('scratchpad', {}).get('routes')
    require(isinstance(a, list) and isinstance(b, list), 'explicit route arrays required')
    require(len(a) == len(b), 'route count mismatch')
    scores = []
    for i, (left, right) in enumerate(zip(a, b), 1):
        require(type(right.get('rank')) is int and right['rank'] == i, 'ambiguous scratchpad rank')
        require('rank' not in left or left['rank'] == i, 'result rank mismatch')
        require(isinstance(left.get('steps'), list) and all(isinstance(s, str) for s in left['steps']), 'literal steps required')
        require(left['steps'] == right.get('steps'), 'rank literal steps mismatch')
        require(left.get('total_score') == right.get('total_score'), 'rank score mismatch')
        score = left.get('total_score') or 0
        require(type(score) in (int, float) and math.isfinite(score), 'invalid ranking score')
        scores.append(score)
    require(scores == sorted(scores, reverse=True), 'routes not rank sorted')
    if not a:
        return None
    # Never substitute native best_route, even for native failures.
    route = a[0]
    steps = [g.ground_step(i, s) for i, s in enumerate(route['steps'])]
    return dict(uid=f"s3:{case['idx']}:1", cond='s3', idx=case['idx'], rank=1,
                target_smiles=case['smiles'], steps=steps,
                flags={'any_unparseable': any(not s['parse_ok'] for s in steps),
                       'target_produced': target in [g.canon(s['product']) if s['product'] else None for s in steps]},
                native_success=case['success'])


def load_manifest(path):
    """v2: accepted-final manifest (latest infra rerun per idx). Same rank1 extraction and frozen judge."""
    global COND
    path = Path(path).resolve()
    data = path.read_bytes()
    m = decode(data)
    require(m.get('schema') in ('s3.test40.accepted-final.v2', 'base.test40.accepted-final.v1'), 'manifest schema')
    COND = m.get('condition')
    require(COND in ('s3', 'base'), 'condition')
    if COND == 's3':
        require(m.get('adapter_sha256') == ADAPTER_SHA, 'S3 adapter lineage mismatch')
    entries = m.get('accepted', [])
    require(len(entries) == 40 and sorted(s['idx'] for s in entries) == IDS, 'exact frozen test40 IDs required, once each')
    slots, routes, origins = [], [], set()
    for entry in sorted(entries, key=lambda s: s['idx']):
        idx = entry['idx']
        case, tr, launch = [source(path.parent, entry[k]) for k in ('result', 'transcript', 'launch')]
        require(case.get('idx') == idx and case.get('finished_at') and not case.get('error'), 'unfinished/error result or target mismatch')
        require(case.get('session_dir') == tr.get('session_dir') and case.get('session_dir'), 'session mismatch')
        require(case.get('transcript_json') == entry['transcript']['origin'], 'transcript origin mismatch')
        require(entry['result']['origin'] not in origins, 'duplicate accepted artifact')
        origins.add(entry['result']['origin'])
        route = rank1(case, tr)
        if route is not None:
            route['uid'] = f"{COND}:{idx}:1"; route['cond'] = COND
        slots.append(dict(idx=idx, native_success=case['success'], rank1_present=route is not None))
        if route is not None:
            routes.append(route)
    import rdkit
    identity = {'manifest_sha256': digest(data), 'routes_sha256': digest(encode(routes)),
                'rdkit': rdkit.__version__, 'claim': 'production', 'condition': COND,
                'code_sha256': {n: digest((HERE/n).read_bytes()) for n in
                                ('adapter.py', 'validators.py', 'grounding.py', 'audit_driver.py', 'AUDIT_SPEC.md', 'SOURCE_PINS.json')}}
    return slots, routes, identity


def atomic(path, data):
    path = Path(path).resolve()
    require(HERE in path.parents, 'writes must remain inside adapter directory')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
    fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class Pending(RuntimeError):
    """Missing offline evidence or unknown in-flight request; not a verdict."""


class Stopped(RuntimeError):
    """Persisted campaign budget exhausted."""


def post(wire):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(frozen.API, data=wire, headers={'Content-Type': 'application/json'})
    with opener.open(request, timeout=600) as response:
        return response.status, response.read()


class Campaign:
    def __init__(self, root, identity, execute=False, transport=post, now=time.time, sleep=time.sleep):
        self.root, self.identity, self.execute = Path(root), identity, execute
        self.transport, self.now, self.sleep = transport, now, sleep
        self.state_path = self.root/'campaign.json'
        self.state = decode(self.state_path.read_bytes()) if self.state_path.exists() else None
        if self.state is not None:
            require(self.state['identity'] == identity, 'campaign identity drift')

    @contextmanager
    def lock(self):
        lock = self.root/'campaign.lock'
        if self.execute:
            require(HERE in lock.resolve().parents, 'campaign must be local')
            self.root.mkdir(parents=True, exist_ok=True)
        if not lock.exists() and not self.execute:
            yield
            return
        with lock.open('a' if self.execute else 'r') as f:
            fcntl.flock(f, (fcntl.LOCK_EX if self.execute else fcntl.LOCK_SH) | fcntl.LOCK_NB)
            try:
                # Re-read after locking: a competing runner may have finished.
                if self.state_path.exists():
                    self.state = decode(self.state_path.read_bytes())
                    require(self.state['identity'] == self.identity, 'campaign identity drift')
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def save(self):
        require(self.execute, 'offline is read-only')
        atomic(self.state_path, encode(self.state))

    def start(self):
        require(self.execute, 'execution requires --execute-owner')
        if self.state is None:
            self.state = dict(identity=self.identity, started_at=self.now(), stages={}, records=[])
            self.save()

    def check_budget(self):
        records = self.state['records']
        errors = sum(r['route_verdict'] == 'ERROR' for r in records)
        streak = 0
        for r in reversed(records):
            if r['route_verdict'] != 'ERROR':
                break
            streak += 1
        if errors >= 5 or streak >= 3 or self.now() >= self.state['started_at'] + 21600:
            raise Stopped('persisted 3 consecutive / 5 cumulative route errors or 6h deadline')

    def caller(self, route):
        def call(prompt, retries=3):
            stage = 'deepread' if prompt.startswith('You are a senior') else 'adversary'
            wire = encode(dict(model=frozen.MODEL, messages=[{'role': 'user', 'content': prompt}],
                               temperature=frozen.TEMP, max_tokens=frozen.MAXTOK))
            key = route['uid'].replace(':', '_') + '.' + stage
            attempts = (self.state or {}).get('stages', {}).get(key, [])
            for attempt in attempts:
                require(attempt['wire_sha256'] == digest(wire), 'request cache drift')
                require((self.root/attempt['wire']).read_bytes() == wire, 'wire bytes drift')
                cached = self.root/attempt['response']
                if cached.exists():
                    raw = cached.read_bytes()
                    if 'response_sha256' in attempt:
                        require(digest(raw) == attempt['response_sha256'], 'raw response drift')
                    # A durable .http200 body also closes a crash between cache and state writes.
                    return self.validate(raw, stage, route)
                if attempt['status'] in ('intent', 'unknown'):
                    raise Pending('interrupted intent: parent reconciliation required: ' + key)
                require(attempt['status'] != 'http200', 'cached raw missing')
                if attempt.get('http_status') in (400, 401, 403, 404):
                    raise ValueError('terminal HTTP error')
            if not self.execute:
                if len(attempts) >= 3:
                    raise ValueError('persisted transport attempts exhausted')
                raise Pending('offline raw missing: ' + key)
            self.check_budget()
            # Unknown requests anywhere in this campaign prevent further submissions.
            for rows in self.state['stages'].values():
                for a in rows:
                    if a['status'] in ('intent', 'unknown') and not (self.root/a['response']).exists():
                        raise Pending('unknown interrupted request; parent must reconcile')
            attempts = self.state['stages'].setdefault(key, [])
            while len(attempts) < 3:
                self.check_budget()
                prefix = f'raw/{key}.attempt{len(attempts)+1}'
                atomic(self.root/(prefix+'.wire.json'), wire)
                attempt = dict(status='intent', wire=prefix+'.wire.json', wire_sha256=digest(wire),
                               response=prefix+'.http200', endpoint=frozen.API, timeout=600,
                               headers={'Content-Type': 'application/json'}, intended_at=self.now())
                attempts.append(attempt)
                self.save()  # Durable intent and exact bytes BEFORE the only POST boundary.
                try:
                    code, raw = self.transport(wire)
                except urllib.error.HTTPError as exc:
                    code, raw = exc.code, exc.read()
                except urllib.error.URLError as exc:
                    if isinstance(exc.reason, ConnectionRefusedError):
                        attempt.update(status='transport_failure', error=str(exc))
                        self.save()
                        self.sleep(5 * len(attempts))
                        continue
                    attempt.update(status='unknown', error=str(exc))
                    self.save()
                    raise Pending('transport outcome unknown; parent reconciliation required') from exc
                except (TimeoutError, OSError) as exc:
                    attempt.update(status='unknown', error=str(exc))
                    self.save()
                    raise Pending('transport outcome unknown; parent reconciliation required') from exc
                if code == 200:
                    atomic(self.root/attempt['response'], raw)  # Preserve even malformed JSON/UTF-8.
                    attempt.update(status='http200', http_status=200, response_sha256=digest(raw))
                    self.save()
                    return self.validate(raw, stage, route)  # Never inside a retry catch.
                atomic(self.root/(prefix+'.http-error'), raw)
                attempt.update(status='http_error', http_status=code, error_sha256=digest(raw))
                self.save()
                if code in (400, 401, 403, 404):
                    raise ValueError(f'terminal HTTP {code}')
                if len(attempts) < 3:
                    self.sleep(30 * len(attempts))
            raise ValueError('persisted transport attempts exhausted')
        return call

    @staticmethod
    def validate(raw, stage, route):
        body = decode(raw)
        text = response_text(body)
        require(not body['choices'][0]['message'].get('tool_calls')
                and not body['choices'][0]['message'].get('function_call'), 'unexpected tools')
        parsed = frozen.parse_json_reply(text)
        if stage == 'deepread':
            validate_deep(parsed, [s['i'] for s in route['steps']])
        else:
            validate_adv(parsed)
        return text

    def audit(self, route):
        old = frozen.call_llm
        frozen.call_llm = self.caller(route)
        try:
            return frozen.audit_one(route)
        finally:
            frozen.call_llm = old

    def run(self, routes):
        self.start()
        self.replay(routes)  # Validate existing verdicts without making requests.
        done = {r['uid'] for r in self.state['records']}
        reason = None
        for route in routes:
            if route['uid'] in done:
                continue
            try:
                self.check_budget()
                rec = self.audit(route)
            except (Pending, Stopped) as exc:
                reason = str(exc)
                break
            except Exception as exc:
                rec = dict(uid=route['uid'], cond=COND, idx=route['idx'], rank=1,
                           route_verdict='ERROR', error=str(exc))
            self.state['records'].append(rec)
            self.save()
        return self.state['records'], reason

    def replay(self, routes):
        # Check retained evidence even for terminal ERROR rows; never hide byte drift.
        for key, attempts in (self.state or {}).get('stages', {}).items():
            require(1 <= len(attempts) <= 3, 'invalid persisted attempt count')
            require(key in {r['uid'].replace(':', '_') + '.' + s for r in routes
                            for s in ('deepread', 'adversary')}, 'unknown stage UID')
            for attempt in attempts:
                require(digest((self.root/attempt['wire']).read_bytes()) == attempt['wire_sha256'], 'wire bytes drift')
                if attempt['status'] == 'http200':
                    require(digest((self.root/attempt['response']).read_bytes()) == attempt['response_sha256'], 'raw response drift')
                if attempt['status'] == 'http_error':
                    error_path = attempt['wire'].removesuffix('.wire.json') + '.http-error'
                    require(digest((self.root/error_path).read_bytes()) == attempt['error_sha256'], 'raw error drift')
        saved = {r['uid']: r for r in (self.state or {}).get('records', [])}
        require(len(saved) == len((self.state or {}).get('records', [])), 'duplicate verdict UID')
        require(set(saved) <= {r['uid'] for r in routes}, 'unknown verdict UID')
        records = []
        was_execute, self.execute = self.execute, False
        try:
            for route in routes:
                old = saved.get(route['uid'])
                if old and old['route_verdict'] == 'ERROR':
                    records.append(old)  # Terminal unresolved error; no reroll on resume.
                    continue
                try:
                    rec = self.audit(route)
                except Pending:
                    require(old is None, 'saved verdict raw missing')
                    continue
                except Exception as exc:
                    require(old is None, 'saved verdict evidence invalid: ' + str(exc))
                    rec = dict(uid=route['uid'], route_verdict='ERROR', error=str(exc))
                require(old is None or old == rec, 'cached evidence verdict mismatch')
                records.append(rec)
        finally:
            self.execute = was_execute
        return records


def aggregate(slots, records, claim):
    require(len(slots) == 40 and sorted(s['idx'] for s in slots) == IDS, 'aggregate requires exact40 slots')
    by_uid = {r['uid']: r for r in records}
    require(len(by_uid) == len(records), 'duplicate verdict UID')
    require(set(by_uid) <= {f"{COND}:{s['idx']}:1" for s in slots if s['rank1_present']}, 'unknown verdict UID')
    labels = {s['idx']: by_uid.get(f"{COND}:{s['idx']}:1", {}).get('route_verdict', 'UNRESOLVED')
              if s['rank1_present'] else 'NO_ROUTE' for s in slots}
    require(all(v in LABELS | {'ERROR', 'UNRESOLVED', 'NO_ROUTE'} for v in labels.values()), 'invalid verdict')
    counts = collections.Counter(labels.values())
    unresolved = counts['ERROR'] + counts['UNRESOLVED']
    result = dict(condition=COND, claim=claim, denominator=40, labels=dict(counts), targets=labels,
                  unresolved=unresolved, complete=unresolved == 0, rates_are_lower_bounds=bool(unresolved),
                  native_success_n=sum(s['native_success'] for s in slots), paired_chemistry_delta=None)
    for metric, passed in [('strict', {'VALID'}), ('lenient', {'VALID', 'SUSPECT'})]:
        vector = [int(labels[s['idx']] in passed) for s in slots]
        intersection = [int(labels[s['idx']] in passed and s['native_success']) for s in slots]
        result[metric] = dict(n=sum(vector), rate=sum(vector)/40,
                              native_success_intersection_n=sum(intersection),
                              native_success_intersection_rate=sum(intersection)/40,
                              target_bootstrap95=None)
        if not unresolved and claim == 'production':
            rng = random.Random(42)
            boot = sorted(sum(rng.choices(vector, k=40))/40 for _ in range(10000))
            result[metric]['target_bootstrap95'] = [boot[249], boot[9749]]
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', choices=['verify', 'summarize', 'replay', 'run'], default='verify')
    parser.add_argument('--manifest', type=Path, default=HERE/'accepted40.json', help='v2 accepted-final manifest')
    parser.add_argument('--execute-owner', action='store_true', help='permit real judge POSTs; only valid with run and production manifest')
    args = parser.parse_args(argv)
    require(not args.execute_owner or args.action == 'run', '--execute-owner requires run')
    require(args.action != 'run' or args.execute_owner, 'run requires --execute-owner')
    slots, routes, identity = load_manifest(args.manifest)
    campaign = Campaign(HERE/('campaign-' + COND), identity, execute=args.execute_owner)
    with campaign.lock():
        if args.action == 'run':
            records, reason = campaign.run(routes)
        else:
            records, reason = campaign.replay(routes), None
        summary = aggregate(slots, records, identity['claim'])
        summary.update(identity=identity, n_routes=len(routes), stop_reason=reason)
        print(json.dumps(summary, indent=2))
        return 0 if args.action == 'verify' or summary['complete'] else 2


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print('BLOCKED: ' + str(exc), file=sys.stderr)
        raise SystemExit(2)
